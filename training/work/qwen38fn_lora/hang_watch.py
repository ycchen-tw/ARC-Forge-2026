"""Hang evidence for the RL trainer service (train_rl_service.py --hang-dump-seconds).

10-21 a16t08 (node1, W3A16 + trainer fast kernels): twice the three ranks sat in the same FSDP backward all-gather until
NCCL's 600 s watchdog took the process down, and neither replay of the packets hung. The watchdog's own report only names
the collective. A phase (one micro-step, the optimizer step, publish, save) that runs longer than ``seconds`` therefore
writes, once, ``<out>/hang/rank<r>-<utc>.txt`` while the ranks are still stuck:

- the phase label and how long it has run;
- every OS thread's state and wchan (``/proc/self/task``);
- this rank's CUDA caching-allocator counters (host-side, read for this rank's device);
- every Python thread's stack (faulthandler);
- ``cuda-gdb -p <pid>`` (only with ``cuda_gdb``; train_rl_service.py --hang-cuda-gdb, off by default since 10-26: it
  never printed within 240 s): ``info cuda kernels`` / ``info cuda blocks``, the kernels still resident on this rank's GPU
  (``rank<r>-<utc>.cuda-gdb.txt``). yama ptrace_scope 1 lets a process trace only its descendants, so
  ``allow_ptrace`` opts the trainer in (PR_SET_PTRACER_ANY): cuda-gdb, or py-spy from a shell, can attach to it.

The NCCL flight recorder (train_rl_service.sh: TORCH_NCCL_TRACE_BUFFER_SIZE, TORCH_NCCL_ENABLE_TIMING,
TORCH_NCCL_DUMP_ON_TIMEOUT, TORCH_FR_DUMP_TEMP_FILE) adds each rank's last collectives with scheduled / started / completed
state at the timeout, and PYTHONFAULTHANDLER=1 the Python stacks on its abort.
"""

import ctypes
import faulthandler
import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PR_SET_PTRACER = 0x59616D61          # <linux/prctl.h>, yama
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value
CUDA_GDB = '/usr/local/cuda/bin/cuda-gdb'
CUDA_GDB_COMMANDS = ('info cuda devices', 'info cuda kernels', 'info cuda blocks')


def allow_ptrace():
    """Let any process of this user attach (yama ptrace_scope 1 otherwise allows only ancestors of the target)."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, ctypes.c_ulong(PR_SET_PTRACER_ANY), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f'prctl(PR_SET_PTRACER, ANY): {os.strerror(err)}')


def thread_table():
    """``tid comm state wchan`` for every OS thread of this process."""
    lines = []
    for task in sorted(Path('/proc/self/task').iterdir(), key=lambda p: int(p.name)):
        comm = (task / 'comm').read_text().strip()
        state = (task / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        lines.append(f'{task.name} {comm} {state} {(task / "wchan").read_text().strip()}')
    return lines


class HangWatch:
    """``with watch.phase(label):`` around work that must finish; a daemon thread dumps once when the innermost open
    phase has run past ``seconds``. ``seconds`` 0 turns the thread off (phases still nest-check)."""

    def __init__(self, rank, device, out, seconds, cuda_gdb=CUDA_GDB, gdb_timeout=240, poll=5.0):
        if seconds < 0:
            raise ValueError(f'hang dump seconds {seconds} < 0')
        self.rank, self.device, self.dir, self.seconds = rank, device, Path(out) / 'hang', seconds
        self.cuda_gdb, self.gdb_timeout, self.poll = cuda_gdb, gdb_timeout, poll
        self.lock = threading.Lock()
        self.stack = []                  # [label, started (monotonic), dumped]
        self.dumps = []                  # paths written, newest last
        if seconds > 0:
            threading.Thread(target=self._loop, name='hang-watch', daemon=True).start()

    @contextmanager
    def phase(self, label):
        entry = [label, time.monotonic(), False]
        with self.lock:
            self.stack.append(entry)
        try:
            yield
        finally:
            with self.lock:
                if not self.stack or self.stack[-1] is not entry:
                    raise RuntimeError(f'hang watch: phase {label!r} closed out of order: {[e[0] for e in self.stack]}')
                self.stack.pop()

    def _due(self):
        with self.lock:
            if not self.stack:
                return None
            entry = self.stack[-1]
            elapsed = time.monotonic() - entry[1]
            if entry[2] or elapsed <= self.seconds:
                return None
            entry[2] = True
            return entry[0], elapsed, [e[0] for e in self.stack]

    def _loop(self):
        while True:
            time.sleep(self.poll)
            due = self._due()
            if due is not None:
                self.dump(*due)

    def dump(self, label, elapsed, stack):
        import torch

        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime())
        path = self.dir / f'rank{self.rank}-{stamp}.txt'
        print(f'[rank{self.rank}] hang watch: {label!r} has run {elapsed:.0f} s (> {self.seconds:.0f} s); '
              f'evidence -> {path}', flush=True)
        with open(path, 'w') as f:
            f.write(f'rank {self.rank} pid {os.getpid()} device {self.device} utc {stamp}\n'
                    f'phase {label!r} running {elapsed:.1f} s; open phases {stack}\n\n== threads: tid comm state wchan\n')
            f.write('\n'.join(thread_table()) + '\n\n== CUDA caching allocator (this rank)\n')
            f.write(json.dumps({'allocated': torch.cuda.memory_allocated(self.device),
                                'reserved': torch.cuda.memory_reserved(self.device),
                                'max_allocated': torch.cuda.max_memory_allocated(self.device),
                                'max_reserved': torch.cuda.max_memory_reserved(self.device)}) + '\n\n== Python stacks\n')
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
        self.dumps.append(path)
        if self.cuda_gdb is not None:
            self.attach_cuda_gdb(path.with_suffix('.cuda-gdb.txt'))

    def attach_cuda_gdb(self, path):
        """cuda-gdb attaches (every thread of this process stops, this one included), prints the resident kernels and
        blocks, detaches. Nothing in this process can time it out while it is attached, so coreutils ``timeout`` (a
        separate process) interrupts it after ``gdb_timeout`` s (SIGINT: gdb detaches) and kills it 30 s later."""
        cmd = ['timeout', '-k', '30', '-s', 'INT', str(self.gdb_timeout), self.cuda_gdb, '-nx', '-batch',
               '-p', str(os.getpid())]
        for command in CUDA_GDB_COMMANDS:
            cmd += ['-ex', command]
        started = time.monotonic()
        with open(path, 'w') as f:
            f.write(' '.join(cmd) + '\n')
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            f.write(f'\n[hang watch] rc {rc} (124 = timeout) after {time.monotonic() - started:.0f} s\n')
        print(f'[rank{self.rank}] hang watch: cuda-gdb rc {rc} -> {path}', flush=True)
