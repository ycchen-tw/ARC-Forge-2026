"""Opt-in token/M-RoPE capture alongside the existing GPTQ Hessian hook.

Set ARC_SGLANG_LAUNCH_PY to this file, TITO_CAPTURE_ROOT to a fresh local
directory, and use serve_gptq_calibration.sh with MAXREQ=1 and
--disable-overlap-schedule --disable-prefill-cuda-graph. The driver deletes
each verified job's GR; this wrapper does not collect REAP saliency.
"""
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import capture_tito_runtime as rt


def check_calls(calls, n_layers):
    expected = {str(layer): {"gateup": 1, "down": 1} for layer in range(n_layers)}
    if calls != expected:
        raise RuntimeError(f"native Hessian calls must occur once per layer: {calls}")


def apply_control(hess, control, previous, run_id):
    if control["run_id"] != run_id or control["action"] not in ("arm", "freeze_dump"):
        raise RuntimeError("Hessian file control belongs to another run or has an invalid action")
    if control["token"] == previous:
        return previous, None
    if control["action"] == "arm":
        if hess._STATE.get("H"):
            raise RuntimeError("refusing to arm a new epoch over existing Hessians")
        hess.arm()
    else:
        hess._STATE["armed"] = False
        hess._STATE["dump_requested"] = False
    return control["token"], control["action"]


def hessian_status(hess, rank, run_id, token, window):
    lo, hi = map(int, window.split(":"))
    banks = {}
    for layer in range(lo, hi):
        banks[str(layer)] = {}
        for bank in ("gateup", "down"):
            key = ("target", layer, bank)
            if key in hess._STATE.get("H", {}):
                samples = hess._STATE["nsamp"][key]
                banks[str(layer)][bank] = {"nsamp_sum": float(samples.sum()),
                    "n_hit": int((samples > 0).sum()), "shape": list(hess._STATE["H"][key].shape)}
    return {"rank": rank, "pid": os.getpid(), "run_id": run_id, "token": token,
            "armed": bool(hess._STATE.get("armed")), "banks": banks}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value) + "\n")
    tmp.replace(path)


def install_hessian_hooks():
    """The pinned fork has native LoRA hooks, not the old environment-based hook patch."""
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from sglang.srt.layers.moe.moe_runner.triton import TritonRunnerCore
    from sglang.srt.lora.lora_moe_runners import LoRAHooks
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    path = Path(os.environ["ARC_SALIENCY_HOOK_PY"])
    spec = importlib.util.spec_from_file_location("arc_int3_mm_hessian", path)
    hess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hess)
    root = Path(os.environ[rt.ROOT_ENV])
    n_layers = int(os.environ["ARC_HESS_N_LAYERS"])
    run_id = os.environ["ARC_HESS_RUN_ID"]
    window = os.environ["ARC_HESS_LAYERS"]
    state = {"calls": None, "control_token": None}
    original_forward = Qwen4ExpForConditionalGeneration.forward
    original_run = TritonRunnerCore.run

    @functools.wraps(original_forward)
    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return original_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        if state["calls"] is not None or forward_batch.batch_size != 1:
            raise RuntimeError("Hessian bridge requires one non-overlapping prefill")
        control = root / "hessian_control.json"
        action = None
        if control.is_file():
            state["control_token"], action = apply_control(hess, json.loads(control.read_text()),
                                                         state["control_token"], run_id)
        calls = {}
        state["calls"] = calls
        try:
            output = original_forward(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            state["calls"] = None
        check_calls(calls, n_layers)
        rank = get_tensor_model_parallel_rank()
        status = hessian_status(hess, rank, run_id, state["control_token"], window)
        if action == "freeze_dump":
            # This same closure owns make_hooks; no OS signal or final-layer dispatch.
            hess.dump_dir(os.environ["ARC_SALIENCY_OUT"])
            status["dump_complete"] = True
        atomic_json(root / "hessian_status" / f"rank{rank}.json", status)
        active = root / "active.json"
        if active.is_file():
            job = json.loads(active.read_text())
            prefix = int(forward_batch.extend_prefix_lens_cpu[0])
            n = int(forward_batch.extend_seq_lens_cpu[0])
            out = root / job["job_id"] / "hesshooks" / f"rank{rank}"
            out.mkdir(parents=True, exist_ok=True)
            dest = out / f"{prefix:012d}.json"
            if dest.exists():
                raise RuntimeError("duplicate Hessian hook chunk receipt")
            dest.write_text(json.dumps({"rank": rank, "prefix": prefix, "tokens": n, "calls": calls}) + "\n")
        return output

    @functools.wraps(original_run)
    def run(self, runner_input, quant_info, running_state, hooks=None):
        if hooks is not None:
            raise RuntimeError("native Hessian bridge found an existing MoE hook (double collection / LoRA)")
        cfg = self.config
        if cfg.layer_id is None or cfg.num_local_experts is None or cfg.num_experts is None:
            raise RuntimeError("native Hessian bridge requires explicit layer id and expert layout")
        if cfg.num_fused_shared_experts:
            raise RuntimeError("native Hessian bridge does not collect fused shared experts")
        layer = str(int(cfg.layer_id))
        if not 0 <= int(cfg.layer_id) < n_layers:
            raise RuntimeError("unexpected MoE layer in target-only calibration")
        hooks_h = hess.make_hooks(cfg, kind_override="target")
        if hooks_h is None:
            raise RuntimeError("GPTQ Hessian hook is not enabled")

        def invoke(bank, callback, *args):
            if state["calls"] is not None:
                counts = state["calls"].setdefault(layer, {"gateup": 0, "down": 0})
                counts[bank] += 1
            return callback(*args)

        bridge = LoRAHooks(after_gate_up=functools.partial(invoke, "gateup", hooks_h.after_gate_up),
                           after_down=functools.partial(invoke, "down", hooks_h.after_down))
        return original_run(self, runner_input, quant_info, running_state, hooks=bridge)

    Qwen4ExpForConditionalGeneration.forward = forward
    TritonRunnerCore.run = run


def main():
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    root = Path(os.environ[rt.ROOT_ENV])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "active.json").exists():
        raise RuntimeError("capture root has an active job; inspect it before restarting")
    if not os.environ.get("ARC_HESS_RUN_ID") or not os.environ.get("ARC_HESS_MASK_SHA256"):
        raise RuntimeError("Hessian run identity and target mask SHA256 are required")
    if Path(os.environ["ARC_SALIENCY_HOOK_PY"]).name != "gptq_hess_hook.py":
        raise RuntimeError("this launcher requires the GPTQ Hessian hook")
    args = prepare_server_args(sys.argv[1:])
    checks = {
        "one request": args.max_running_requests == 1,
        "radix off": args.disable_radix_cache,
        "graphs off": args.disable_cuda_graph and args.disable_prefill_cuda_graph,
        "overlap off": args.disable_overlap_schedule,
        "spec off": args.speculative_algorithm is None,
        "EP=TP": args.ep_size == args.tp_size and args.ep_size > 1,
        "triton MoE hook": args.moe_runner_backend == "triton",
        "unfused MoE": os.environ.get("SGLANG_CI_DISABLE_MOE_FUSED_FUNC") == "1",
        "Hessian source": os.environ.get("ARC_SALIENCY_MODEL") == args.model_path,
    }
    broken = [name for name, ok in checks.items() if not ok]
    if broken:
        raise RuntimeError(f"MM calibration isolation failed: {broken}")
    config = Path(args.model_path) / "config.json"
    n_layers = json.loads(config.read_text())["text_config"]["num_hidden_layers"]
    os.environ["ARC_HESS_N_LAYERS"] = str(n_layers)
    import sglang

    tree = Path(sglang.__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    expected = os.environ.get("ARC_MM_CALIB_EXPECT_SGLANG_SHA")
    if expected and commit != expected:
        raise RuntimeError(f"calibration runtime {commit} != pinned {expected}")
    info = {
        "launcher_pid": os.getpid(), "served_model_name": args.served_model_name,
        "model_path": args.model_path, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "port": args.port, "n_ranks": args.tp_size,
        "n_experts": json.loads(config.read_text())["text_config"]["num_experts"],
        "run_id": os.environ["ARC_HESS_RUN_ID"], "mask_sha256": os.environ["ARC_HESS_MASK_SHA256"],
        "mtp_mask_sha256": os.environ.get("ARC_HESS_MTP_MASK_SHA256", ""),
        "layers": os.environ["ARC_HESS_LAYERS"], "isolation": checks,
        "sglang_commit": commit, "argv": sys.argv[1:],
        "ple_mm_placeholder_ids": "--ple-mm-placeholder-ids" in sys.argv[1:],
        "max_nsamp": int(os.environ["ARC_HESS_MAX_NSAMP"]),
        "n_layers": n_layers, "hessian_bridge": "native-triton-file-control-v2",
    }
    (root / "mm_calib_server.json").write_text(json.dumps(info, indent=2) + "\n")
    rt.install()
    install_hessian_hooks()
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
elif __name__ == "__mp_main__":
    rt.install()
    install_hessian_hooks()
