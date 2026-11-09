"""GPU time per QAD backward section (trainer --time-sections)."""
import contextlib

import torch

# Per-section GPU time of the QAD-specific backward work (``--time-sections``): CUDA event pairs collected during a
# step and summed (one synchronize) when the trainer reads them. Off by default: no events, no cost.
TIMING = {"on": False, "events": []}


@contextlib.contextmanager
def section(name):
    if not TIMING["on"]:
        yield
        return
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        TIMING["events"].append((name, start, end))


def collect():
    """{section: seconds} since the last collect (synchronizes)."""
    torch.cuda.synchronize()
    out = {}
    for name, start, end in TIMING["events"]:
        out[name] = out.get(name, 0.0) + start.elapsed_time(end) / 1000.0
    TIMING["events"].clear()
    return out
