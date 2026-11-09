"""Host offload of the wide residual inputs saved by whole-layer activation checkpointing.

Each Qwen3.8-FN decoder layer checkpoints its (1, T, hidden*hc_count) BF16 residual
stream input: at T=120k that is 2.46 GB/layer, 118 GB over 48 layers, which does
not fit next to the frozen base. Only forward saves of exactly that shape go to host
memory; everything else stays on the device. The checkpoint recompute's own saves of the
same shape stay on the device too: backward consumes them right away.

Host memory is a ``PinnedSlots`` pool: one exact-size page-locked buffer per offloaded
tensor, reused every step. (torch's ``pin_memory=True`` allocator rounds each request up to a
power of two and caches the blocks per size class; the soak held 329 GiB of host memory per
rank for 135 GB of offloaded data.)

``asynchronous=False``: the device-to-host copy runs on the compute stream during the
forward, the host-to-device copy when backward needs it.
``asynchronous=True``: the D2H copy runs on a side stream (the allocator keeps the source
alive via record_stream); when backward unpacks layer k it starts the H2D copy of the
previous, not yet used offloaded tensor on the side stream, so the copy overlaps layer k's
recompute and backward (``prefetch`` > 1: the previous ``prefetch`` tensors as well; QAD 10-16 saw no
step-time change at depth 3). The copy's destination is allocated on the compute stream (the side
stream allocates nothing: see _prefetch). A tensor unpacked again is copied again from its slot, which
is not reused before the next step.

``keep_bytes`` (asynchronous only): the newest forward saves stay on the device while they fit in ``keep_bytes``; a
save is copied out only when newer ones push it out of that window. The last layers' inputs are what backward needs
first, so they never make the round trip, and the copy volume drops by what is kept.
"""

import collections
import mmap
import weakref

import torch

# Debug only: per-token checksums of every synchronously offloaded tensor at pack time, checked at unpack.
VERIFY = {'on': False, 'packed': []}


class PinnedSlots:
    """Exact-size page-locked host buffers, one per offloaded tensor, handed out in order and reused every step.

    Slots are kept per dtype (FN offloads 51 BF16 and 2 FP32 tensors per step); each holds ``elements``
    values of its dtype. A slot is an anonymous mmap (page aligned, exactly sized) registered with
    cudaHostRegister. The pool grows to what one step offloads and then stays; the registration (~1.2 s per
    2.46 GB) is paid on the first step only.
    """

    def __init__(self, elements):
        self.elements = elements
        self.slots = {}  # dtype -> [(mmap, tensor)]; torch.frombuffer keeps each mmap alive while viewed
        self.used = {}

    @property
    def counts(self):
        return {str(dtype).removeprefix('torch.'): len(slots) for dtype, slots in self.slots.items()}

    @property
    def bytes(self):
        return sum(buffer.numel() * buffer.element_size() for slots in self.slots.values() for _, buffer in slots)

    def reset(self):
        torch.cuda.synchronize()  # no copy issued in the previous step may still touch a slot
        self.used = dict.fromkeys(self.slots, 0)

    def take(self, tensor):
        if tensor.numel() > self.elements:
            raise ValueError(f'{tuple(tensor.shape)} does not fit a {self.elements}-element slot')
        slots = self.slots.setdefault(tensor.dtype, [])
        used = self.used.get(tensor.dtype, 0)
        if used == len(slots):
            slots.append(self._allocate(tensor.dtype))
        self.used[tensor.dtype] = used + 1
        return slots[used][1][:tensor.numel()].view(tensor.shape)

    def _allocate(self, dtype):
        size = self.elements * torch.empty((), dtype=dtype).element_size()
        region = mmap.mmap(-1, size)
        buffer = torch.frombuffer(region, dtype=dtype)
        error = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), size, 0)
        if int(error) != 0 or not buffer.is_pinned():
            raise RuntimeError(f'cudaHostRegister of {size} bytes failed: {error}')
        return region, buffer

    def close(self):
        """Unregister every slot; an unmapped but still registered range breaks later CUDA copies.

        The mmaps are not closed here: torch.frombuffer keeps each one alive while any view of it
        exists, so it is unmapped when the last view goes, by then as plain pageable memory.
        """
        if not self.slots:
            return
        torch.cuda.synchronize()
        for slots in self.slots.values():
            for _, buffer in slots:
                error = torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
                if int(error) != 0:
                    raise RuntimeError(f'cudaHostUnregister failed: {error}')
        self.slots.clear()
        self.used = {}

    def __del__(self):
        self.close()


class _Offloaded:
    __slots__ = ('index', 'device', 'cpu', 'copied', 'gpu', 'ready', 'consumer', 'uses', 'resident', '__weakref__')

    def __init__(self, index, device, resident):
        self.index, self.device, self.resident = index, device, resident   # resident: still on the device
        self.cpu = self.copied = self.gpu = self.ready = self.consumer = None
        self.uses = 0


class CheckpointInputOffload:
    """Offload the wide residual inputs saved by whole-layer checkpointing for one step."""

    def __init__(self, length, width, slots, asynchronous=False, prefetch=1, keep_bytes=0):
        if length * width > slots.elements:
            raise ValueError(f'{length} tokens x {width} exceed the {slots.elements}-element host slots')
        if keep_bytes < 0 or (keep_bytes and not asynchronous):
            raise ValueError(f'keep_bytes {keep_bytes}: a non-negative budget, asynchronous mode only')
        slots.reset()
        self.length, self.width, self.slots = length, width, slots
        self.bytes = self.tensors = 0   # bytes copied out; saves handled (copied out or kept)
        self.asynchronous = asynchronous
        if prefetch < 1:
            raise ValueError(f'prefetch depth {prefetch} < 1')
        self.prefetch = prefetch
        self.stream = torch.cuda.Stream() if asynchronous else None
        # weak: the autograd graph owns each record, so a resident save is freed when its node is (as without offload)
        self.records = []
        self.keep_bytes = keep_bytes
        self.window = collections.deque()   # resident records during the forward, oldest first
        self.window_bytes = 0
        self.kept = 0   # saves left on the device by the forward

    def matches(self, tensor):
        return (tensor.device.type == 'cuda' and tensor.ndim in (3, 4)
                and tensor.shape[0] == 1 and tensor.shape[1] == self.length
                and tensor.numel() == self.length*self.width)

    def pack(self, tensor):
        if not self.matches(tensor) or torch._C._current_graph_task_id() != -1:
            return tensor.detach()  # not a layer input, or a checkpoint recompute save consumed right away
        self.tensors += 1
        if not self.asynchronous:
            self.bytes += tensor.numel()*tensor.element_size()
            cpu = self.slots.take(tensor)
            cpu.copy_(tensor.detach())
            if VERIFY['on']:
                # Same device and kernel on both sides: GPU reductions of identical bits give identical sums.
                VERIFY['packed'].append(tensor.detach().float().flatten(2).sum(-1).cpu())
                return tensor.device, cpu, len(VERIFY['packed']) - 1
            return tensor.device, cpu
        record = _Offloaded(len(self.records), tensor.device, tensor.detach())
        self.records.append(weakref.ref(record))
        self.window.append(record)
        self.window_bytes += tensor.numel()*tensor.element_size()
        while self.window_bytes > self.keep_bytes:
            self._copy_out(self.window.popleft())
        self.kept = len(self.window)
        return record

    def _copy_out(self, record):
        source, record.resident = record.resident, None
        size = source.numel()*source.element_size()
        self.window_bytes -= size
        self.bytes += size
        record.cpu = self.slots.take(source)
        produced = torch.cuda.Event()
        produced.record()
        self.stream.wait_event(produced)
        with torch.cuda.stream(self.stream):
            record.cpu.copy_(source, non_blocking=True)
            record.copied = torch.cuda.Event()
            record.copied.record(self.stream)
        source.record_stream(self.stream)

    def _prefetch(self, ref):
        record = ref()
        if record is None or record.resident is not None or record.gpu is not None or record.uses:
            return
        # The destination comes from the compute stream's pool. Allocated on the side stream it would come from that
        # stream's own pool (the caching allocator keeps one per stream; a new side stream every step), which the
        # compute stream cannot reuse: the side pools' free blocks pushed the reserved memory into the
        # --gpu-memory-fraction cap once per micro step, and the allocator then synchronized and unmapped its whole
        # cache (10-21 v3.1 p001: 14 GiB of a previous step's side pool still mapped, 1-2.7 s stalls). The copy
        # waits for the compute stream, which may still use the block's previous tenant.
        current = torch.cuda.current_stream()
        gpu = torch.empty_like(record.cpu, device=record.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(record.copied)
            gpu.copy_(record.cpu, non_blocking=True)
            record.ready = torch.cuda.Event()
            record.ready.record(self.stream)
        gpu.record_stream(self.stream)  # freed unconsumed (backward aborted), it outlives the copy
        record.gpu, record.consumer = gpu, current

    def unpack(self, saved):
        if isinstance(saved, tuple) and len(saved) == 3:
            device, cpu, index = saved
            gpu = cpu.to(device, non_blocking=True)
            live = VERIFY['packed'][index]
            restored = gpu.float().flatten(2).sum(-1).cpu()
            bad = (restored != live).nonzero()
            if len(bad):
                raise RuntimeError(f'offloaded tensor {index} restored differs from the live tensor at pack time on '
                                   f'{len(bad)} tokens, {bad[:, 1].min().item()}..{bad[:, 1].max().item()}')
            VERIFY['checked'] = VERIFY.get('checked', 0) + 1
            return gpu
        if isinstance(saved, tuple):
            device, cpu = saved
            return cpu.to(device, non_blocking=True)
        if not isinstance(saved, _Offloaded):
            return saved
        self.window.clear()   # backward has started: the graph alone keeps the resident saves
        current = torch.cuda.current_stream()
        # A saved residual can be unpacked more than once (checkpoint recompute and the outer op's
        # backward): the first use takes the prefetched copy, later ones copy again from the slot.
        if saved.resident is not None:
            gpu = saved.resident
        elif saved.gpu is not None:
            if current != saved.consumer:
                raise RuntimeError(f'offloaded tensor {saved.index} prefetched for stream {saved.consumer}, unpacked on '
                                   f'{current}')
            current.wait_event(saved.ready)
            gpu, saved.gpu = saved.gpu, None
        else:
            current.wait_event(saved.copied)
            gpu = saved.cpu.to(saved.device, non_blocking=True)
        saved.uses += 1
        for back in range(1, self.prefetch + 1):   # the next tensors backward will unpack (lower indices)
            if saved.index - back >= 0:
                self._prefetch(self.records[saved.index - back])
        return gpu

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)
