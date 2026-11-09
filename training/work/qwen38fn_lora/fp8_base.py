"""Direct FP8 checkpoint loading: no full BF16 expert/table copy, even on CPU."""

import json
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _lookup(W, Ids, Out, Scale, Count, Dim: tl.constexpr, Start, End, B: tl.constexpr):
    # int64 element index: all ranks' requests x Dim exceeds 2**31 once a rank's padded PLE batch passes
    # 209,715 tokens (4 ranks x 16 heads x 160). The int32 version wrapped negative, its `i < Count * Dim`
    # mask let the store through, and it overwrote ~1.46M elements before Out (2026-10-06, on an 8×B200 node).
    # Count/Start/End are runtime values: as constexprs every new request count recompiled the kernel.
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = Count.to(tl.int64) * Dim
    row = i // Dim
    token = tl.load(Ids + row, i < total, other=-1)
    value = tl.load(
        W + (token - Start) * Dim + i % Dim, (i < total) & (token >= Start) & (token < End), other=0.0
    ).to(tl.float32)
    scale = tl.load(Scale).to(tl.float32)
    tl.store(Out + i, value * scale, i < total)


@triton.jit
def _lookup_int5(W, Ids, Out, Scale, Count, Dim: tl.constexpr, RowBytes: tl.constexpr, Start, End, B: tl.constexpr):
    # Mirror of sglang arc_ple_int5's gather (qwen4_exp._gather_ple_embedding_from_pinned_kernel, is_int5) plus the
    # per-tensor weight_scale multiply the server does after the gather. A row of W (uint8) is Dim * 5 / 8 code bytes
    # (element i = the 5-bit field at bit 5 i, little-endian, stored as code + 16) then Dim / 16 e4m3 scales.
    # int64 element index for the same reason as _lookup.
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = Count.to(tl.int64) * Dim
    row = i // Dim
    col = i % Dim
    token = tl.load(Ids + row, i < total, other=-1)
    valid = (i < total) & (token >= Start) & (token < End)
    base = W + (token - Start) * RowBytes
    bit = col * 5
    lo = tl.load(base + (bit >> 3), valid, other=0).to(tl.int32)
    # the field straddles two bytes at most; the byte after the last code byte is the row's first scale byte
    hi = tl.load(base + (bit >> 3) + 1, valid, other=0).to(tl.int32)
    unsigned = ((lo | (hi << 8)) >> (bit & 7)) & 0x1F
    group_scale = (
        tl.load(base + Dim * 5 // 8 + col // 16, valid, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    )
    # |code| <= 15 times an e4m3 scale has at most 8 significant bits: exact in bf16 (the server stores it as bf16)
    value = ((unsigned - 16).to(tl.float32) * group_scale).to(tl.bfloat16).to(tl.float32)
    value = tl.where(valid, value, 0.0)
    scale = tl.load(Scale).to(tl.float32)
    tl.store(Out + i, value * scale, i < total)


@triton.jit
def _lookup_nvfp4(W, S, Ids, Out, Global, Count, Dim: tl.constexpr, Start, End, B: tl.constexpr):
    # NVIDIA ModelOpt NVFP4 rows (Qwen3.8-Flash-Next-FP4-Local's PLE): W [rows, Dim / 2] uint8, element 2j in the low
    # nibble of byte j (sign bit 3, E2M1 magnitude bits 0-2), S [rows, Dim / 16] e4m3 block scales as uint8, Global the fp32
    # weight_scale_2. code x block scale is exact in fp32; times the global scale, rounded once to bf16.
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    total = Count.to(tl.int64) * Dim
    row = i // Dim
    col = i % Dim
    token = tl.load(Ids + row, i < total, other=-1)
    valid = (i < total) & (token >= Start) & (token < End)
    local = token - Start
    byte = tl.load(W + local * (Dim // 2) + col // 2, valid, other=0).to(tl.int32)
    nibble = (byte >> ((col & 1) * 4)) & 0xF
    exponent = (nibble >> 1) & 3
    half = (nibble & 1).to(tl.float32) * 0.5
    magnitude = tl.where(exponent == 0, half, tl.exp2((exponent - 1).to(tl.float32)) * (1.0 + half))
    code = tl.where((nibble & 8) != 0, -magnitude, magnitude)
    block = tl.load(S + local * (Dim // 16) + col // 16, valid, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    value = tl.where(valid, code * block * tl.load(Global), 0.0)
    tl.store(Out + i, value, i < total)


class RowShardedPLE(nn.Module):
    """Rows [start, end) of a ``total``-row PLE table live on this rank (a vocabulary shard, already in the stored
    codec). forward: every rank's ids are all-gathered, each rank looks up the ids inside its rows (zeros elsewhere),
    and a reduce-scatter (disjoint contributions, so exact) hands each rank its own vectors. ``scale`` is the
    checkpoint's per-tensor weight_scale. Subclasses implement ``_local_lookup``."""

    def __init__(self, local_weight, scale, start, end, total, embedding_dim):
        super().__init__()
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        self.start, self.end, self.num_embeddings = start, end, total
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", local_weight)
        self.register_buffer("scale", torch.tensor(scale, device=local_weight.device, dtype=torch.bfloat16))

    def _local_lookup(self, gathered, vectors):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, ids):
        count = torch.tensor([ids.numel()], device=ids.device)
        counts = [torch.empty_like(count) for _ in range(self.world)]
        dist.all_gather(counts, count)
        width = max(int(x) for x in counts)
        request = torch.zeros(width, dtype=torch.long, device=ids.device)
        request[: ids.numel()] = ids.flatten()
        gathered = torch.empty(self.world * width, dtype=torch.long, device=ids.device)
        dist.all_gather_into_tensor(gathered, request)
        vectors = torch.empty((gathered.numel(), self.embedding_dim), dtype=torch.bfloat16, device=ids.device)
        self._local_lookup(gathered, vectors)
        output = torch.empty((width, self.embedding_dim), dtype=torch.bfloat16, device=ids.device)
        dist.reduce_scatter_tensor(output, vectors)
        return output[: ids.numel()].reshape(*ids.shape, self.embedding_dim)


class FP8RowShardedEmbedding(RowShardedPLE):
    def __init__(self, local_weight, scale, start, end, total):
        if local_weight.dtype != torch.float8_e4m3fn or local_weight.shape[0] != end - start:
            raise ValueError("invalid FP8 table shard")
        super().__init__(local_weight, scale, start, end, total, local_weight.shape[1])

    def _local_lookup(self, gathered, vectors):
        _lookup[(triton.cdiv(vectors.numel(), 1024),)](
            self.weight, gathered, vectors, self.scale, gathered.numel(), self.embedding_dim, self.start, self.end, 1024
        )


class Int5RowShardedEmbedding(RowShardedPLE):
    """The deployed server's INT5 PLE table (sglang ``ARC_PLE_INT5=1``, arc_ple_int5), byte for byte: ``weight`` is
    [rows, width * 5 / 8 + width / 16] uint8, a row being its packed 5-bit codes then its e4m3 group scales. Built by
    ``quantize_int5_rows`` from the FP8 table; lookups equal the server's bit for bit."""

    def __init__(self, local_weight, scale, start, end, total, embedding_dim):
        row_shape = (end - start, int5_row_bytes(embedding_dim))
        if local_weight.dtype != torch.uint8 or tuple(local_weight.shape) != row_shape:
            raise ValueError("invalid INT5 table shard")
        super().__init__(local_weight, scale, start, end, total, embedding_dim)

    def _local_lookup(self, gathered, vectors):
        _lookup_int5[(triton.cdiv(vectors.numel(), 1024),)](
            self.weight,
            gathered,
            vectors,
            self.scale,
            gathered.numel(),
            self.embedding_dim,
            self.weight.shape[1],
            self.start,
            self.end,
            1024,
        )


class NVFP4RowShardedEmbedding(RowShardedPLE):
    """An NVFP4 PLE table taken from another checkpoint (``--ple-codec nvfp4:<dir>``; NVIDIA ModelOpt layout, see
    ``_lookup_nvfp4``). ``weight`` = packed codes, ``block_scale`` = e4m3 per 16, ``global_scale`` = fp32
    weight_scale_2; the base class ``scale`` is 1 (the lookup applies the fp32 global itself)."""

    def __init__(self, packed, block_scale, global_scale, start, end, total):
        n, half = packed.shape
        if (packed.dtype != torch.uint8 or n != end - start or half * 2 % 16
                or tuple(block_scale.shape) != (n, half * 2 // 16)):
            raise ValueError("invalid NVFP4 table shard")
        if block_scale.dtype != torch.float8_e4m3fn or global_scale.dtype != torch.float32 or global_scale.numel() != 1:
            raise ValueError("NVFP4 block scales must be e4m3 and the global scale one fp32")
        super().__init__(packed, 1.0, start, end, total, half * 2)
        self.register_buffer("block_scale", block_scale)
        self.register_buffer("global_scale", global_scale.reshape(1))

    def _local_lookup(self, gathered, vectors):
        _lookup_nvfp4[(triton.cdiv(vectors.numel(), 1024),)](
            self.weight, self.block_scale.view(torch.uint8), gathered, vectors, self.global_scale, gathered.numel(), self.embedding_dim,
            self.start, self.end, 1024,
        )


INT5_BLOCK = 16  # elements per e4m3 scale
INT5_QMAX = 15
INT5_ZERO = 16  # stored unsigned code = signed code + 16, in [1, 31]
INT5_E4M3_MIN_NORMAL = 2.0**-6
INT5_CHUNK_ROWS = 1 << 18  # rows per GPU quantization step (fp32 copy 160 MiB at width 160)


def _int5_check_width(width):
    if width <= 0 or width % INT5_BLOCK:
        raise ValueError(f"INT5 PLE needs a width divisible by {INT5_BLOCK}, got {width}")


def int5_row_bytes(width):
    """Bytes per stored row: width * 5 / 8 of packed codes, then width / 16 e4m3 scales."""
    _int5_check_width(width)
    return width * 5 // 8 + width // INT5_BLOCK


@torch.no_grad()
def quantize_int5_rows(x):
    """[n, width] FP8 rows (before the per-tensor weight_scale) -> ([n, int5_row_bytes] uint8, blocks with a subnormal
    scale). Same arithmetic, in the same torch ops, as sglang arc_ple_int5.quantize_rows + pack_rows: symmetric INT5
    per 16 elements, ``scale = e4m3(amax / 15)``, ``code = clamp(round(x / scale), -15, 15)`` (half to even) chosen
    against the *rounded* scale, a group whose scale rounds to zero stores code 0."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError(f"INT5 PLE quantizes FP8 rows only, got {x.dtype}")
    n, width = x.shape
    _int5_check_width(width)
    blocks = x.float().view(n, width // INT5_BLOCK, INT5_BLOCK)
    amax = blocks.abs().amax(dim=-1)
    if not bool(torch.isfinite(amax).all()):
        raise ValueError("INT5 PLE: non-finite rows in the FP8 table")
    scale = (amax / INT5_QMAX).to(torch.float8_e4m3fn)
    s = scale.float().unsqueeze(-1)
    q = torch.where(s > 0, blocks / s, torch.zeros_like(blocks))
    code = q.round().clamp_(-INT5_QMAX, INT5_QMAX).to(torch.int32) + INT5_ZERO
    # eight 5-bit codes -> one little-endian 40-bit word -> five bytes
    fields = code.to(torch.int64).view(n, width // 8, 8)
    word = torch.zeros((n, width // 8), dtype=torch.int64, device=x.device)
    for i in range(8):
        word |= fields[..., i] << (5 * i)
    packed = torch.stack([(word >> (8 * k)) & 0xFF for k in range(5)], dim=-1).to(torch.uint8).view(n, width * 5 // 8)
    subnormal = int(((amax > 0) & (amax / INT5_QMAX < INT5_E4M3_MIN_NORMAL)).sum())
    return torch.cat([packed, scale.view(torch.uint8)], dim=1), subnormal


def dequant_reference(weight, scale):
    """Explicit BF16 reference mode only; never called by the FP8 compute path."""
    e, n, k = weight.shape
    return (
        (weight.float().view(e, n // 128, 128, k // 128, 128) * scale.float()[:, :, None, :, None])
        .reshape(e, n, k)
        .bfloat16()
    )


def install_expert_backend(model, base_storage):
    from types import MethodType

    if base_storage == "fp8":
        from fp8_moe import FrozenFP8Experts

        for layer in model.model.layers:
            if not isinstance(layer.mlp.experts, FrozenFP8Experts):
                raise RuntimeError("FP8 backend requires directly loaded frozen experts")
            layer.mlp.experts.forward = MethodType(FrozenFP8Experts.forward, layer.mlp.experts)
    elif base_storage == "int3":
        from int3_moe import FrozenInt3Experts

        for layer in model.model.layers:
            if not isinstance(layer.mlp.experts, FrozenInt3Experts):
                raise RuntimeError("INT3 backend requires directly loaded frozen experts")
            layer.mlp.experts.forward = MethodType(FrozenInt3Experts.forward, layer.mlp.experts)
    else:
        from frozen_moe import install_frozen_moe

        install_frozen_moe(model)


PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
PLE_CODECS = ("fp8", "int5")


def _checkpoint_ple_rows(handle, shard_keys, shapes, start, end, max_rows=None):
    """Yield (first global row, FP8 [n, width] CPU rows) covering rows [start, end) of the checkpoint's PLE shards in
    order; ``max_rows`` bounds n (None: one piece per checkpoint shard overlap)."""
    cursor = 0
    for key, shape in zip(shard_keys, shapes, strict=True):
        lo, hi = max(start, cursor), min(end, cursor + shape[0])
        if lo < hi:
            step = hi - lo if max_rows is None else max_rows
            for a in range(lo, hi, step):
                value = handle(key).get_slice(key)[a - cursor : min(a + step, hi) - cursor]
                if value.dtype != torch.float8_e4m3fn:
                    raise RuntimeError("PLE source is not FP8")
                yield a, value
        cursor += shape[0]


def build_ple_embedding(handle, index, expected_rows, device, codec="fp8"):
    """This rank's rows of the layer-1 PLE table (read from the FP8 checkpoint, split evenly over the ranks of the
    default group) as a row-sharded lookup module on ``device``. ``handle(key)`` opens the safetensors file holding
    ``key``; ``index`` is the checkpoint's weight_map; ``expected_rows`` the native model's PLE vocabulary.

    codec "fp8": the rows stay FP8; a lookup is fp8 x the per-tensor weight_scale.
    codec "int5": every row is re-quantized on the GPU, chunk by chunk, to the INT5 table the deployed server builds
    with ``ARC_PLE_INT5=1`` (see ``quantize_int5_rows``); the FP8 rows are never all resident. A lookup equals the
    server's bit for bit (INT5 value to bf16, times the same weight_scale).

    codec "nvfp4:<dir>": the table comes from another checkpoint's NVFP4 PLE (build_nvfp4_ple), not from this one.

    Returns (module, weight_scale as float, report)."""
    if codec.startswith("nvfp4:"):
        return build_nvfp4_ple(Path(codec.split(":", 1)[1]), expected_rows, device)
    if codec not in PLE_CODECS:
        raise ValueError(f"ple codec {codec!r}: expected one of {PLE_CODECS}")
    shard_keys = sorted(
        [k for k in index if k.startswith(PLE_PREFIX + "shard_") and k.endswith(".weight")],
        key=lambda k: int(k.split("shard_")[1].split(".")[0]),
    )
    shapes = [handle(k).get_slice(k).get_shape() for k in shard_keys]
    total = sum(s[0] for s in shapes)
    if total != expected_rows:
        raise RuntimeError("PLE checkpoint/native vocabulary shape mismatch")
    width = shapes[0][1]
    if any(s[1] != width for s in shapes):
        raise RuntimeError("PLE shards differ in width")
    rank, world = dist.get_rank(), dist.get_world_size()
    start, end = total * rank // world, total * (rank + 1) // world
    scale = float(handle(PLE_PREFIX + "weight_scale").get_tensor(PLE_PREFIX + "weight_scale").float().item())
    if codec == "fp8":
        local = torch.empty((end - start, width), device=device, dtype=torch.float8_e4m3fn)
        for lo, value in _checkpoint_ple_rows(handle, shard_keys, shapes, start, end):
            local[lo - start : lo - start + value.shape[0]].copy_(value)
        embedding = FP8RowShardedEmbedding(local, scale, start, end, total)
        report = {"ple_storage": "float8_e4m3fn"}
    else:
        local = torch.empty((end - start, int5_row_bytes(width)), device=device, dtype=torch.uint8)
        loaded = subnormal = 0
        for lo, value in _checkpoint_ple_rows(handle, shard_keys, shapes, start, end, INT5_CHUNK_ROWS):
            packed, blocks = quantize_int5_rows(value.to(device))
            local[lo - start : lo - start + value.shape[0]].copy_(packed)
            loaded += value.shape[0]
            subnormal += blocks
        if loaded != end - start:
            raise RuntimeError(f"INT5 PLE: quantized {loaded} rows, this rank holds {end - start}")
        embedding = Int5RowShardedEmbedding(local, scale, start, end, total, width)
        report = {"ple_storage": "int5_g16_e4m3", "ple_subnormal_scale_blocks": subnormal}
    report.update(
        ple_codec=codec,
        ple_local_weight_bytes=local.numel() * local.element_size(),
        ple_fp8_local_weight_bytes=(end - start) * width,
        ple_rows=total,
    )
    return embedding, scale, report


def build_nvfp4_ple(root, expected_rows, device):
    """This rank's rows of the layer-1 NVFP4 PLE table of the checkpoint at ``root`` (NVIDIA ModelOpt: per shard a
    ``.weight`` [rows, width / 2] uint8 and ``.weight_scale`` [rows, width / 16] e4m3, one fp32 ``weight_scale_2``),
    split evenly over the default group like build_ple_embedding. Returns (module, 1.0, report)."""
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def handle(key):
        if index[key] not in handles:
            handles[index[key]] = safe_open(root / index[key], framework="pt", device="cpu")
        return handles[index[key]]

    keys = sorted([k for k in index if k.startswith(PLE_PREFIX + "shard_") and k.endswith(".weight")],
                  key=lambda k: int(k.split("shard_")[1].split(".")[0]))
    shapes = [handle(k).get_slice(k).get_shape() for k in keys]
    total = sum(s[0] for s in shapes)
    if total != expected_rows:
        raise RuntimeError(f"NVFP4 PLE has {total} rows, the model {expected_rows}")
    half = shapes[0][1]
    if any(s[1] != half for s in shapes):
        raise RuntimeError("NVFP4 PLE shards differ in width")
    global_key = PLE_PREFIX + "weight_scale_2"
    global_scale = handle(global_key).get_tensor(global_key)
    rank, world = dist.get_rank(), dist.get_world_size()
    start, end = total * rank // world, total * (rank + 1) // world
    packed = torch.empty((end - start, half), device=device, dtype=torch.uint8)
    block = torch.empty((end - start, half * 2 // 16), device=device, dtype=torch.float8_e4m3fn)
    cursor = loaded = 0
    for key, shape in zip(keys, shapes, strict=True):
        lo, hi = max(start, cursor), min(end, cursor + shape[0])
        if lo < hi:
            codes = handle(key).get_slice(key)[lo - cursor : hi - cursor]
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            scales = handle(scale_key).get_slice(scale_key)[lo - cursor : hi - cursor]
            if codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
                raise RuntimeError(f"{key}: expected uint8 codes and e4m3 scales, got {codes.dtype} / {scales.dtype}")
            packed[lo - start : hi - start].copy_(codes)
            block[lo - start : hi - start].copy_(scales)
            loaded += hi - lo
        cursor += shape[0]
    if loaded != end - start:
        raise RuntimeError(f"NVFP4 PLE: loaded {loaded} rows, this rank holds {end - start}")
    embedding = NVFP4RowShardedEmbedding(packed, block, global_scale.to(device=device, dtype=torch.float32),
                                         start, end, total)
    report = {"ple_storage": "nvfp4_g16_e4m3", "ple_codec": f"nvfp4:{root}", "ple_rows": total,
              "ple_global_scale": float(global_scale.item()),
              "ple_local_weight_bytes": packed.numel() + block.numel()}
    return embedding, 1.0, report


def open_native_model(model_path):
    """Meta-device native Qwen4Exp text model, its text config and a lazy safetensors handle per shard (FP8 and INT3)."""
    from safetensors import safe_open
    from transformers import AutoConfig, __version__
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

    if __version__ != "5.16.1":
        raise RuntimeError("direct FP8/INT3 loader requires audited transformers 5.16.1")
    root = Path(model_path)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    config = AutoConfig.from_pretrained(root, local_files_only=True).text_config
    config.use_cache = False
    config._attn_implementation = "sdpa"
    config._experts_implementation = "grouped_mm"
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM(config)
    model.model.rotary_emb = Qwen4ExpTextRotaryEmbedding(config, device="cpu")
    handles = {}

    def handle(key):
        filename = index[key]
        if filename not in handles:
            handles[filename] = safe_open(root / filename, framework="pt", device="cpu")
        return handles[filename]

    return index, config, model, handle


def install_ple(model, index, handle, device, ple_codec):
    """The single PLE construction site of both loaders (build_ple_embedding); returns (weight_scale, report)."""
    embedding = model.model.layers[1].ple.ple_embedding
    ple, scale, report = build_ple_embedding(handle, index, embedding.ngram_embedding.num_embeddings, device, ple_codec)
    embedding.ngram_embedding = ple
    return scale, report


def load_dense_tensors(model, index, handle):
    """Every tensor still on meta (dense, shared expert, router, norms) from the checkpoint; returns the count."""
    # Native text names differ only in their root; packed experts and PLE are handled by the caller.
    loaded_dense = 0
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if not value.is_meta:
            continue
        source = "model.language_model." + name[len("model.") :] if name.startswith("model.") else name
        if source in index:
            actual = handle(source).get_tensor(source)
        elif name.endswith("ngram_heads_vocab_sizes"):
            actual = torch.tensor(model.model.layers[1].ple.ple_embedding.head_vocab_sizes)
        elif name.endswith("ngram_heads_offsets"):
            actual = torch.tensor(model.model.layers[1].ple.ple_embedding.head_offsets)
        else:
            raise RuntimeError(f"unhandled native checkpoint tensor: {name}")
        if actual.shape != value.shape:
            raise RuntimeError(f"shape mismatch: {name}")
        if actual.dtype == torch.float8_e4m3fn:
            # a block-FP8 dense weight would be cast to BF16 without its scale
            raise RuntimeError(f"dense tensor {source} is FP8; this loader needs BF16 dense weights")
        if actual.is_floating_point():
            actual = actual.to(torch.bfloat16)
        owner, leaf = name.rsplit(".", 1)
        module = model.get_submodule(owner)
        setattr(module, leaf, nn.Parameter(actual, requires_grad=False) if isinstance(value, nn.Parameter) else actual)
        loaded_dense += 1
    return loaded_dense


def load_fp8_cpu_model(model_path, device, keep_experts_fp8=True, ple_codec="fp8"):
    from fp8_moe import FrozenFP8Experts

    started = time.perf_counter()
    index, config, model, handle = open_native_model(model_path)
    scale, ple_report = install_ple(model, index, handle, device, ple_codec)
    for layer in model.model.layers:
        layer.mlp.experts = FrozenFP8Experts(config.num_experts, config.hidden_size, config.moe_intermediate_size)
    expert_pattern = re.compile(
        r"model.language_model.layers.(\d+).mlp.experts.(\d+).(gate|up|down)_proj.(weight|weight_scale_inv)$"
    )
    loaded_expert = 0
    for key in index:
        match = expert_pattern.fullmatch(key)
        if match is None:
            continue
        layer, e, proj, suffix = match.groups()
        expert = model.model.layers[int(layer)].mlp.experts
        weight = handle(key).get_tensor(key)
        if suffix == "weight":
            target = expert.down_proj if proj == "down" else expert.gate_up_proj
            width = config.moe_intermediate_size
            if weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError("source experts not E4M3")
        else:
            target = expert.down_scale if proj == "down" else expert.gate_up_scale
            width = config.moe_intermediate_size // 128
        if proj == "down":
            target.data[int(e)].copy_(weight)
        else:
            offset = 0 if proj == "gate" else width
            target.data[int(e), offset : offset + width].copy_(weight)
        loaded_expert += 1
    if loaded_expert != config.num_hidden_layers * config.num_experts * 3 * 2:
        raise RuntimeError("incomplete FP8 expert weights/scales")
    loaded_dense = load_dense_tensors(model, index, handle)
    if any(v.is_meta for v in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("unmaterialized FP8 base tensor")
    model._arc_fp8_base = True
    report = {
        "expert_tensors_loaded": loaded_expert,
        "dense_tensors_loaded": loaded_dense,
        "expert_storage": "float8_e4m3fn",
        "full_bf16_weight_copies": 0,
        "persistent_transposed_weight_copies": 0,
        **ple_report,
    }
    if not keep_experts_fp8:
        for layer in model.model.layers:
            expert = layer.mlp.experts
            expert.gate_up_proj = nn.Parameter(
                dequant_reference(expert.gate_up_proj, expert.gate_up_scale), requires_grad=False
            )
            expert.down_proj = nn.Parameter(dequant_reference(expert.down_proj, expert.down_scale), requires_grad=False)
        report.update(expert_storage="bfloat16_reference", full_bf16_weight_copies=1)
    return model, time.perf_counter() - started, report, scale


INT3_EXPERT_PREFIX = "model.language_model.layers."


def int3_expert_contract(experts, hidden, inter):
    """name -> (shape, safetensors dtype) of the packed INT3 routed experts of one layer."""
    from int3_moe import GROUP

    return {
        "w13_qweight": ((experts, 2 * inter, hidden * 3 // 32), "I32"),
        "w13_scales": ((experts, 2 * inter, hidden // GROUP), "BF16"),
        "w2_qweight": ((experts, hidden, inter * 3 // 32), "I32"),
        "w2_scales": ((experts, hidden, inter // GROUP), "BF16"),
    }


def check_int3_expert_keys(index, layers):
    """Exactly the four packed tensors per layer: no per-expert, FP8 or otherwise unknown routed-expert key."""
    names = int3_expert_contract(1, 128, 128)
    expected = {f"{INT3_EXPERT_PREFIX}{i}.mlp.experts.{name}" for i in range(layers) for name in names}
    present = {k for k in index if k.startswith(INT3_EXPERT_PREFIX) and ".mlp.experts." in k}
    if present != expected:
        raise RuntimeError(
            f"INT3 expert tensors differ from the contract: missing {sorted(expected - present)[:3]}, "
            f"unknown {sorted(present - expected)[:3]}"
        )


def load_int3_cpu_model(model_path, device, ple_codec="fp8"):
    """gEAN320 INT3 package with BF16 dense weights -> native model whose routed experts are FrozenInt3Experts.

    Experts: the packed uint3 codes are unpacked on this rank's GPU, layer by layer, into exact E4M3 `(code - 4)` that
    is copied to the CPU parameter (FSDP shards it like the FP8 experts); the signed BF16 group scales are kept as
    they are. PLE is the FP8 table (``ple_codec`` as in load_fp8_cpu_model); dense / shared expert / router / vision
    are BF16 from the same directory.
    """
    from int3_moe import FrozenInt3Experts, codes_to_e4m3, unpack_uint3

    started = time.perf_counter()
    index, config, model, handle = open_native_model(model_path)
    scale, ple_report = install_ple(model, index, handle, device, ple_codec)
    hidden, inter, experts, layers = (
        config.hidden_size,
        config.moe_intermediate_size,
        config.num_experts,
        config.num_hidden_layers,
    )
    check_int3_expert_keys(index, layers)
    contract = int3_expert_contract(experts, hidden, inter)
    for layer_index, layer in enumerate(model.model.layers):
        layer.mlp.experts = module = FrozenInt3Experts(experts, hidden, inter)
        tensors = {}
        for name, (shape, dtype) in contract.items():
            key = f"{INT3_EXPERT_PREFIX}{layer_index}.mlp.experts.{name}"
            piece = handle(key).get_slice(key)
            if tuple(piece.get_shape()) != shape or piece.get_dtype() != dtype:
                raise RuntimeError(f"{key}: {piece.get_shape()} {piece.get_dtype()}, expected {shape} {dtype}")
            tensors[name] = handle(key).get_tensor(key)
        module.gate_up_proj.data.copy_(codes_to_e4m3(unpack_uint3(tensors["w13_qweight"].to(device), hidden)))
        module.down_proj.data.copy_(codes_to_e4m3(unpack_uint3(tensors["w2_qweight"].to(device), inter)))
        module.gate_up_scale.copy_(tensors["w13_scales"])
        module.down_scale.copy_(tensors["w2_scales"])
    loaded_dense = load_dense_tensors(model, index, handle)
    if any(v.is_meta for v in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("unmaterialized INT3 base tensor")
    model._arc_int3_base = True
    report = {
        "expert_tensors_loaded": layers * len(contract),
        "dense_tensors_loaded": loaded_dense,
        "expert_storage": "float8_e4m3fn (uint3 code - 4) + signed bf16 g128 scales",
        "full_bf16_weight_copies": 0,
        "persistent_transposed_weight_copies": 0,
        **ple_report,
    }
    return model, time.perf_counter() - started, report, scale
