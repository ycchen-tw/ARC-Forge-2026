"""Opt-in final-GR side channel for a dedicated, single-request SGLang worker.

No installed files are changed. Import-time installation is repeated in spawned
schedulers. The original model forward, QSA, GDN and PLE remain intact.
"""

import functools
import inspect
import json
import os
import sys
import time
from pathlib import Path


def install():
    import torch
    from safetensors.torch import save_file
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from sglang.srt.runtime_context import get_parallel

    root = Path(os.environ["MTP3_CAPTURE_ROOT"])
    cls = Qwen4ExpForConditionalGeneration
    original = cls.forward
    if "hc_hidden_states = self.model.last_hc_hidden_states" not in inspect.getsource(original):
        raise RuntimeError("unrecognized native final-GR handoff; re-audit runtime")

    @functools.wraps(original)
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        output = original(self, input_ids, positions, forward_batch, **kwargs)
        active = root / "active.json"
        if get_parallel().tp_rank != 0 or not active.is_file() or input_ids.numel() == 0:
            return output
        job = json.loads(active.read_text())
        if not forward_batch.forward_mode.is_extend():
            raise RuntimeError("capture requires exact-token prefill, not generated replay")
        if positions.ndim != 1 or input_ids.shape != positions.shape:
            raise RuntimeError("capture expects a single unpacked text request")
        h = self.model.last_hc_hidden_states
        if h is None or h.shape != (input_ids.numel(), 10240) or h.dtype != torch.bfloat16:
            raise RuntimeError("final pre-collapse GR handoff shape/dtype mismatch")
        directory = root / job["job_id"] / "parts"
        destination = directory / f"{time.time_ns()}-{os.getpid()}.safetensors"
        temp = destination.with_suffix(".partial")
        save_file(
            {
                "input_ids": input_ids.detach().cpu().contiguous(),
                "positions": positions.detach().cpu().contiguous(),
                "target_gr": h.detach().view(-1, 4, 2560).cpu().contiguous(),
            },
            str(temp),
        )
        temp.rename(destination)
        return output

    cls.forward = forward


install()

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(sys.argv[1:])
    if args.max_running_requests != 1 or not args.disable_radix_cache or not args.disable_cuda_graph:
        raise ValueError("capture isolation requires max-running-requests=1, no radix and no CUDA graphs")
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
