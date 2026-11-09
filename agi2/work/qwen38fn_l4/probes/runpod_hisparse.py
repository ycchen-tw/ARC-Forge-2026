"""Experimental L4x4 HiSparse launcher: two attention-DP pools, opt-in guarded MTP.

Readiness validates admission receipts and allocated pools separately. It does not prove
session correctness, graph replay, long-context residency, quality, or throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from runpod_baseline import gpu_probe, require_free_port, stop_owned
from runpod_runtime import (REPO, MODEL_IDENTITIES, notebook_metadata, runtime_env, safe_relative, sha256_file, verify_model,
                            write_json)

STARTUP_MARKER = "ARC_HISPARSE_STARTUP="
HOST_POOL_MARKER = "ARC_NATIVE_MTP_KV_HOST_POOL="


@dataclass(frozen=True)
class Profile:
    running: int
    logical_tokens: int
    host_reserve_gib: float
    hot_blocks: int = 512
    context: int = 131072
    snapshot_interval: int = 64
    snapshot_staging_slots: int = 4
    prefill_chunk: int = 2048
    mem_fraction: float = 0.90
    eager: bool = False
    index_fp8: bool = False
    dist_timeout: int | None = None
    prefill_max_requests: int | None = None
    mtp: bool = False
    debug_checks: bool = True
    sparse_graphs: bool = False
    draft_fp8_direct: bool = False
    fold_in_verify: bool = False
    # h24 packed GDN state (ARC_GDN_STATE_FMT i10h / i12h) on the h20 fold-in-verify; on this distributed shape the
    # fork also needs ARC_GDN_STATE_FMT_DISTRIBUTED=1 (qsa_hisparse_distributed.packed_gdn_state_admitted)
    gdn_state_fmt: str | None = None
    mtp_rejection_sampling: bool = False
    native_mtp_host_kv: bool = False
    cpu_affinity: bool = False
    custom_all_reduce: bool = False
    spec_phase_packed: bool = False
    mtp_steps: int = 3
    attn_dp: int = 2  # 1 = pure TP4 (KV heads replicated), gated by ARC_HISPARSE_TP4 in the source
    swap_in_block_size: int | None = None  # HiSparse swap-in CTA width; default 960 fails some hot sizes on SM89
    # GDN snapshot host buffers per rank (hisparse_config qsa_snapshot_host_buffers): None keeps this launcher's
    # 2 x (R/DP + 16) (the fork's anchor sizing); the fork's own default without anchors is R/DP + 16 -- one buffer
    # per session plus slack, and a shortage only evicts the LRU parked record (h09)
    snapshot_host_buffers: int | None = None
    # h23 resume anchors (hisparse_config qsa_session_anchor): each parked session record also keeps its resume point
    # as a second snapshot buffer, so a request that agrees only up to the previous turn (a retry, a dropped last
    # generation) truncates to it instead of prefilling the whole context again; a buffer shortage drops anchors first
    session_anchor: bool = False
    deterministic: bool = False  # quality A/B: exact QSA top-k, torch HC mix, non-fused MoE finalize
    # MTP5-DFlash drafter (work/qwen38fn_dflash/l4) in place of the native MTP draft: export dir and keep set
    # (a dense export, config ffn_type dense_shared, has no experts to keep: dflash_dense, no keep set)
    dflash_drafter: str | None = None
    dflash_keep: str | None = None
    dflash_dense: bool = False
    # --enable-prefill-delayer (DP attention): a prefill on one DP waits up to max_passes scheduler passes for the other
    # DP, so both prefill in one forward instead of one idling; queue ratio / max delay ms set the optional queue trigger
    prefill_delayer: bool = False
    prefill_delayer_max_passes: int | None = None
    prefill_delayer_queue_ratio: float | None = None
    prefill_delayer_max_delay_ms: float | None = None

    def __post_init__(self):
        # the native MTP is reviewed at NEXTN1/3; the DFlash drafter proposes K-1 = 1..7 (verify width 2..8)
        allowed = tuple(range(1, 8)) if self.dflash_drafter is not None else (1, 3)
        if type(self.mtp_steps) is not int or self.mtp_steps not in allowed:
            raise ValueError(f"MTP steps must be explicitly one of {allowed}")
        if self.mtp_steps != 3 and not self.mtp:
            raise ValueError("nondefault MTP steps requires MTP")
        if self.mtp_steps == 1 and not self.mtp_rejection_sampling:
            raise ValueError("NEXTN1 requires native MTP rejection sampling")
        if self.attn_dp not in (1, 2):
            raise ValueError("attention DP must be 1 (pure TP4) or 2")
        if self.attn_dp == 1 and self.mtp and self.dflash_drafter is None:
            # 10-09 l4opt: DFlash on pure TP4 (fork exp/l4-tp4-attn: h29 protocol channels and the drafter's attention
            # split over the TP4 group); the native MTP draft was never ported
            raise ValueError("pure TP4 HiSparse spec decoding is DFlash only")
        if self.running <= 0 or self.running % 2:
            raise ValueError("global running must be positive and even for DP2")
        if self.logical_tokens < self.context or self.logical_tokens % 64:
            raise ValueError("logical tokens are per DP, must cover context, and must align to page64")
        if self.context < 131072 or self.context % 64:
            raise ValueError("context must be >=131072 and page64 aligned")
        if not math.isfinite(self.host_reserve_gib) or self.host_reserve_gib < 0:
            raise ValueError("explicit finite host reserve >=0 GiB required")
        if self.hot_blocks < 512 or self.hot_blocks % 16:
            raise ValueError("hot blocks must cover top512 and align to16")
        if self.snapshot_interval <= 0 or self.snapshot_interval % 64:
            raise ValueError("snapshot interval must be a positive multiple of64")
        if self.snapshot_staging_slots <= 0:
            raise ValueError("snapshot staging slots must be positive")
        if self.snapshot_host_buffers is not None and self.snapshot_host_buffers < self.running // self.attn_dp + 16:
            raise ValueError("snapshot host buffers below the fork's no-anchor default R/DP + 16")
        if type(self.session_anchor) is not bool:
            raise ValueError("session anchor must be a bool")
        if self.prefill_chunk <= 0 or self.prefill_chunk % 128:
            raise ValueError("global prefill chunk must be positive and align to128 for DP2")
        if not 0 < self.mem_fraction < 1:
            raise ValueError("memory fraction must be between0and1")
        if self.dist_timeout is not None and self.dist_timeout <= 0:
            raise ValueError("distributed timeout must be positive when explicitly set")
        if self.prefill_max_requests is not None and self.prefill_max_requests <= 0:
            raise ValueError("prefill request limit must be positive when explicitly set")
        if self.prefill_delayer and self.attn_dp != 2:
            raise ValueError("the prefill delayer pairs the two attention-DP schedulers' prefills (attn_dp 2)")
        if not self.prefill_delayer and (self.prefill_delayer_max_passes is not None
                                         or self.prefill_delayer_queue_ratio is not None
                                         or self.prefill_delayer_max_delay_ms is not None):
            raise ValueError("prefill delayer options without prefill_delayer")
        if self.prefill_delayer_max_passes is not None and self.prefill_delayer_max_passes < 2:
            raise ValueError("prefill delayer max passes must be >= 2 (it delays at most max_passes - 1)")
        if self.mtp and self.hot_blocks < self.verify_tokens * 512:
            raise ValueError(f"NEXTN{self.mtp_steps}/verify{self.verify_tokens} needs hot blocks >={self.verify_tokens * 512} to cover the selected-block union")
        if self.sparse_graphs and self.eager:
            raise ValueError("sparse graph capture requires CUDA graphs, not eager mode")
        if self.sparse_graphs and self.mtp and self.snapshot_staging_slots != 4:
            raise ValueError("sparse MTP graphs require the reviewed four snapshot staging slots")
        if self.draft_fp8_direct and not self.mtp:
            raise ValueError("draft FP8 direct attention requires MTP")
        if self.fold_in_verify and not self.mtp:
            raise ValueError("fold in verify requires MTP ReplaySSM-spec")
        if self.gdn_state_fmt is not None and (self.gdn_state_fmt not in ("i10h", "i12h") or not self.fold_in_verify):
            raise ValueError(f"packed GDN state {self.gdn_state_fmt!r}: i10h or i12h, on fold in verify only")
        if self.mtp_rejection_sampling and not self.mtp:
            raise ValueError("MTP rejection sampling requires MTP")
        if self.native_mtp_host_kv and not self.mtp:
            raise ValueError("native MTP host KV requires MTP")
        if self.spec_phase_packed and not self.mtp:
            raise ValueError("phase-packed agreement requires MTP")
        if self.dflash_drafter is None and (self.dflash_keep is not None or self.dflash_dense):
            raise ValueError("a DFlash keep set / dense flag without a drafter export")
        if self.dflash_drafter is not None and (self.dflash_keep is None) != self.dflash_dense:
            raise ValueError("a MoE DFlash drafter needs its keep set; a dense one takes none")
        if self.dflash_drafter is not None:
            if not (self.mtp and self.mtp_rejection_sampling):
                raise ValueError("DFlash rides the NEXTN verify path with rejection sampling (--mtp --mtp-rejection-sampling)")
            if self.native_mtp_host_kv or self.draft_fp8_direct:
                raise ValueError("native-MTP draft options do not apply to the DFlash drafter")

    @property
    def verify_tokens(self) -> int:
        return self.mtp_steps + 1 if self.mtp else 1

    @property
    def per_dp(self) -> int:
        return self.running // self.attn_dp

    @property
    def mamba_global(self) -> int:
        # Live state + permanently reserved snapshot staging + one spare per DP.
        return self.running + self.attn_dp * (self.snapshot_staging_slots + 1)

    @property
    def graph_batches(self) -> list[int]:
        if not self.sparse_graphs:
            return list(range(1, self.per_dp + 1))
        # h17's bounded list, with the exact endpoint for non-power-of-two caps.
        # Unlisted tail batches pad into a captured shape; req row 0 and GDN
        # slot -1 retain their runtime null-row semantics. Do not disable padding.
        sizes = [1, 2, 4, 8, 16, 32] + list(range(48, self.per_dp + 1, 16))
        return sorted({size for size in sizes if size <= self.per_dp} | {self.per_dp})

    def hisparse_config(self) -> dict:
        return {"qsa_hot_blocks": self.hot_blocks, "qsa_logical_tokens": self.logical_tokens,
                "qsa_session_resume": True, "qsa_session_replay": True,
                "qsa_snapshot_interval": self.snapshot_interval,
                "qsa_snapshot_staging_slots": self.snapshot_staging_slots,
                "qsa_snapshot_host_buffers": (2 * (self.per_dp + 16) if self.snapshot_host_buffers is None
                                              else self.snapshot_host_buffers),
                **({"qsa_session_anchor": True} if self.session_anchor else {}),
                "qsa_session_idle_timeout_s": 0, "host_reserve_gib": self.host_reserve_gib,
                **({"swap_in_block_size": self.swap_in_block_size} if self.swap_in_block_size else {})}


def server_args(python: str, model: Path, port: int, profile: Profile) -> list[str]:
    args = [python, "-m", "sglang.launch_server", "--model-path", str(model),
            "--tokenizer-path", str(model), "--served-model-name", "tir", "--host", "127.0.0.1",
            "--port", str(port), "--tp", "4", "--ep-size", "4", "--dp-size", str(profile.attn_dp),
            *(["--enable-dp-attention"] if profile.attn_dp == 2 else []),
            "--quantization", "modelopt_mixed", "--fp4-gemm-backend", "marlin", "--moe-runner-backend", "marlin",
            "--attention-backend", "triton", "--linear-attn-prefill-backend", "triton",
            "--linear-attn-decode-backend", "triton", "--ple-offload-embedding",
            "--json-model-override-args", json.dumps({"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}}),
            "--dtype", "bfloat16", "--kv-cache-dtype", "nvfp4", "--page-size", "64",
            "--context-length", str(profile.context), "--chunked-prefill-size", str(profile.prefill_chunk),
            "--max-running-requests", str(profile.running), "--max-mamba-cache-size", str(profile.mamba_global),
            "--mamba-ssm-dtype", "float16", "--mamba-radix-cache-strategy", "extra_buffer_lazy",
            "--mamba-max-states-per-path", "1", "--disable-radix-cache", "--enable-session-radix-cache",
            "--radix-cache-backend", "qsa_hisparse_session", "--enable-hisparse",
            "--hisparse-config", json.dumps(profile.hisparse_config()), "--mem-fraction-static", str(profile.mem_fraction),
            "--cuda-graph-max-bs-decode", str(profile.per_dp), "--disable-prefill-cuda-graph",
            "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
            "--image-processor-backend", "pil",
            "--enable-cache-report", "--enable-metrics", "--decode-log-interval", "10", "--random-seed", "7300",
            "--model-loader-extra-config", json.dumps({"enable_multithread_load": True, "num_threads": 2})]
    if not profile.custom_all_reduce:
        args.append("--disable-custom-all-reduce")
    if profile.dist_timeout is not None:
        args += ["--dist-timeout", str(profile.dist_timeout)]
    if profile.prefill_max_requests is not None:
        args += ["--prefill-max-requests", str(profile.prefill_max_requests)]
    if profile.prefill_delayer:
        args.append("--enable-prefill-delayer")
        if profile.prefill_delayer_max_passes is not None:
            args += ["--prefill-delayer-max-delay-passes", str(profile.prefill_delayer_max_passes)]
        if profile.prefill_delayer_queue_ratio is not None:
            args += ["--prefill-delayer-queue-min-ratio", str(profile.prefill_delayer_queue_ratio)]
        if profile.prefill_delayer_max_delay_ms is not None:
            args += ["--prefill-delayer-max-delay-ms", str(profile.prefill_delayer_max_delay_ms)]
    if profile.mtp:
        args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", str(profile.mtp_steps),
                 "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(profile.verify_tokens),
                 "--speculative-moe-runner-backend", "triton",
                 "--speculative-draft-kv-cache-dtype", "fp8_e4m3", "--speculative-attention-mode", "prefill",
                 "--enable-linear-replayssm-spec", "--linear-replayssm-cache-len", "16"]
        if profile.mtp_rejection_sampling:
            args.append("--speculative-use-rejection-sampling")
    if profile.eager:
        args.append("--disable-cuda-graph")
    else:
        args += ["--cuda-graph-bs-decode", *[str(n) for n in profile.graph_batches]]
    # Experiment-only passthrough (e.g. ["--enable-dp-lm-head"]); recorded in launch.json argv.
    extra = json.loads(os.environ.get("ARC_EXTRA_SERVER_ARGS", "[]"))
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ValueError("ARC_EXTRA_SERVER_ARGS must be a JSON list of strings")
    return args + extra


def verify_source_manifest(source: Path, manifest: Path) -> str:
    text = manifest.read_text()
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if match is None:
            raise RuntimeError("invalid HiSparse source manifest row")
        expected, rel = match.groups()
        safe_relative(rel)
        if sha256_file(source / "sglang" / rel) != expected:
            raise RuntimeError(f"HiSparse source manifest mismatch: {rel}")
        rows.append(rel)
    if not rows or rows != sorted(set(rows)) or "srt/mem_cache/qsa_hisparse_distributed.py" not in rows:
        raise RuntimeError("HiSparse source manifest must include h25 distributed helper and sorted unique paths")
    return hashlib.sha256(text.encode()).hexdigest()


def profile_env(env: dict, manifest: Path, profile: Profile, cache: Path) -> dict:
    result = {**env, "ARC_HISPARSE_DISTRIBUTED": "1", "ARC_HISPARSE_QSA_MANIFEST": str(manifest),
              "SGLANG_SET_CPU_AFFINITY": "1" if profile.cpu_affinity else "0",
              "ARC_HISPARSE_DISTRIBUTED_SPEC": "1" if profile.mtp else "0",
              "ARC_HISPARSE_TP4": "1" if profile.attn_dp == 1 else "0",
              "ARC_HISPARSE_MTP_STEPS": str(profile.mtp_steps),
              "ARC_QSA_NVFP4": "1", "ARC_FULL_NVFP4": "0", "ARC_QSA_FP4_FUSED": "1",
              "ARC_QSA_NVFP4_PREFILL": "fp8", "SGLANG_QSA_DIRECT_DECODE": "1", "SGLANG_QSA_DIRECT_MIN_ROWS": "1",
              "ARC_QSA_INDEX_FP8": "1" if profile.index_fp8 else "0",
              "ARC_NATIVE_MTP_KV_HOST": "1" if profile.native_mtp_host_kv else "0",
              "ARC_HISPARSE_SPEC_PHASE_PACKED": "1" if profile.spec_phase_packed else "0",
              "ARC_QSA_FP8_DIRECT": "1" if profile.draft_fp8_direct else "0",
              "ARC_GDN_FOLD_IN_VERIFY": "1" if profile.fold_in_verify else "0",
              "ARC_HISPARSE_QSA_DEBUG": "1" if profile.debug_checks else "0",
              "ARC_QSA_TOPK_EXACT": "1" if profile.deterministic else "0",
              "ARC_HC_MIX_DETERMINISTIC": "1" if profile.deterministic else "0",
              "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0" if profile.deterministic else "1", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "0",
              "SGLANG_QWEN4_MTP_DISTRIBUTED_TOPK1": "0",
              "SGLANG_JIT_CACHE_DIR": str(cache / "jit"), "TRITON_CACHE_DIR": str(cache / "triton"),
              "CUDA_CACHE_PATH": str(cache / "nv"), "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer")}
    # h24 off unless requested (a made-up "fp16" format would fail its parser)
    result.pop("ARC_GDN_STATE_FMT", None)
    result.pop("ARC_GDN_STATE_FMT_DISTRIBUTED", None)
    if profile.gdn_state_fmt is not None:
        result["ARC_GDN_STATE_FMT"] = profile.gdn_state_fmt
        result["ARC_GDN_STATE_FMT_DISTRIBUTED"] = "1"
    result.pop("ARC_ONLINE_FP8", None)
    for key, default in {"SGLANG_NUMA_BIND_V2": "1", "SGLANG_AUTO_NUMA_BIND": "1",
                         "SGLANG_CRASH_ON_NUMA_BIND_FAILURE": "0"}.items():
        result.setdefault(key, default)
    if profile.custom_all_reduce:
        # Legacy CUDA IPC cannot export PyTorch expandable-segment graph buffers.
        result["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
        result["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    if profile.mtp:
        result["SGLANG_RAGGED_VERIFY_MODE"] = "static"
    if profile.dflash_drafter is not None:
        # the drafter worker is in the sglang-arc-2 source (speculative/arc_dflash); sm_89 takes the Marlin experts
        result["ARC_DFLASH_DRAFTER"] = profile.dflash_drafter
        if profile.dflash_keep is not None:
            result["ARC_DFLASH_KEEP"] = profile.dflash_keep
        else:  # dense: the worker refuses any keep set, an inherited one included
            result.pop("ARC_DFLASH_KEEP", None)
    return result


def recorded_env(env: dict, manifest: Path, profile: Profile, cache: Path) -> dict:
    keys = set(profile_env({}, manifest, profile, cache)) | {
        "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LIBRARY_PATH", "TORCH_CUDA_ARCH_LIST",
        "FLASHINFER_CUDA_ARCH_LIST", "MAX_JOBS", "FLASHINFER_NVCC_THREADS", "PYTORCH_ALLOC_CONF"}
    record = {k: env[k] for k in keys}
    if env.get("CUDA_COMPAT_DIR"):
        record["CUDA_COMPAT_DIR"] = env["CUDA_COMPAT_DIR"]
    record.update({k: v for k, v in env.items() if k.startswith("NCCL_")
                   or k in {"ARC_FLASHINFER_PCIE_VENDOR", "ARC_FLASHINFER_PCIE_REPORT",
                            "ARC_ATTN_TP_CUSTOM_AR", "PYTORCH_CUDA_ALLOC_CONF",
                            "ARC_FLASHINFER_PCIE_ATTN_VENDOR", "ARC_FLASHINFER_PCIE_ATTN_REPORT",
                            "ARC_FLASHINFER_PCIE_DIAGNOSTICS"}})
    return record


def verify_h25_shape(python: str, env: dict, source: Path, profile: Profile) -> dict:
    verify_mtp_steps_support(source, profile)
    verify_rs_relay_support(source, profile)
    # Installed helper is stdlib-only until a consensus or host-memory function is called.
    code = '''import json, runpy, sys
from types import SimpleNamespace
module = runpy.run_path(sys.argv[1])
assert module["distributed_hisparse_enabled"]() is True
view = SimpleNamespace(**json.loads(sys.argv[2]))
checks = module["distributed_shape_needs"](view)
assert all(checks.values()), checks
print(json.dumps(checks))
'''
    view = {"tp_size": 4, "ep_size": 4, "dp_size": profile.attn_dp, "enable_dp_attention": profile.attn_dp == 2,
            "nnodes": 1,
            "speculative_algorithm": "EAGLE" if profile.mtp else None,
            "speculative_num_steps": profile.mtp_steps if profile.mtp else None,
            "speculative_num_draft_tokens": profile.verify_tokens if profile.mtp else None,
            "speculative_eagle_topk": 1 if profile.mtp else None,
            "speculative_draft_kv_cache_dtype": "fp8_e4m3" if profile.mtp else None,
            "speculative_attention_mode": "prefill" if profile.mtp else None,
            "speculative_use_rejection_sampling": profile.mtp_rejection_sampling,
            "disable_overlap_schedule": False, "speculative_adaptive": False, "enable_multi_layer_eagle": False,
            "linear_replayssm_cache_len": 16,
            "mamba_ssm_dtype": "float16", "dtype": "bfloat16",
            "enable_linear_replayssm_spec": profile.mtp,
            "max_running_requests": profile.running, "radix_cache_backend": "qsa_hisparse_session",
            "hisparse_config": json.dumps(profile.hisparse_config())}
    output = subprocess.check_output([python, "-c", code,
                                      str(source / "sglang/srt/mem_cache/qsa_hisparse_distributed.py"), json.dumps(view)],
                                     env=env, text=True, timeout=30)
    return json.loads(output)


def config_preflight(python: str, env: dict, command: list[str], out: Path) -> dict:
    """Run the installed CLI parser and resolution pipeline, with CUDA tensor ops blocked."""
    code = '''import json, sys, torch
from torch.utils._python_dispatch import TorchDispatchMode

def cuda_value(value):
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, torch.device):
        return value.type == "cuda"
    if isinstance(value, str):
        return value == "cuda" or value.startswith("cuda:")
    if isinstance(value, dict):
        return any(cuda_value(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(cuda_value(item) for item in value)
    return False

class NoCudaTensorOps(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if cuda_value(args) or cuda_value(kwargs):
            raise RuntimeError("CLI preflight attempted a CUDA tensor operation: " + str(func))
        return func(*args, **kwargs)

argv = json.loads(sys.argv[1])
before = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
with NoCudaTensorOps():
    from sglang.srt.server_args import prepare_server_args
    config = prepare_server_args(argv)
    config.resolve_once()
    config.check_server_args()
    resolved = config.resolved_dict()
    assert resolved["speculative_use_rejection_sampling"] == (
        "--speculative-use-rejection-sampling" in argv
    ), "resolved rejection sampling differs from requested argv"
    if "--speculative-algorithm" in argv:
        assert resolved["speculative_moe_runner_backend"] == "triton", "FP8 MTP draft requires Triton MoE"
    from sglang.srt.arg_groups.overrides import model_config_of
    model = model_config_of(config)
    assert model.is_multimodal is True and model.is_lm_only is False
    vision = model.hf_config.vision_config
    assert vision is not None, "checkpoint has no vision_config"
after = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
assert before == after == [0] * len(before), (before, after)
for field in ("language_model_only", "language_only", "encoder_only"):
    assert resolved[field] is False, (field, resolved[field])
print("ARCFORGE_CONFIG_PREFLIGHT=" + json.dumps({
    "argv": argv, "resolved": resolved, "cuda_allocated_bytes_before": before,
    "cuda_allocated_bytes_after": after, "server_launched": False,
    "cuda_tensor_ops_blocked": True, "model_is_multimodal": model.is_multimodal,
    "vision_config_present": True, "vision_config": vision.to_dict(),
    "vision_http_validation": "pending"}, default=str))
'''
    result = subprocess.run([python, "-c", code, json.dumps(command[3:])], env=env,
                            capture_output=True, text=True, timeout=180)
    (out / "config_preflight.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("installed SGLang CLI/resolution preflight failed; see config_preflight.log")
    marker = "ARCFORGE_CONFIG_PREFLIGHT="
    rows = [line.split(marker, 1)[1] for line in result.stdout.splitlines() if marker in line]
    if len(rows) != 1:
        raise RuntimeError("CLI preflight completed without one configuration receipt")
    record = json.loads(rows[0])
    if record["argv"] != command[3:] or record["server_launched"] is not False:
        raise RuntimeError("CLI preflight did not inspect the exact requested argv")
    write_json(out / "config_preflight.json", record)
    return record


def validate_receipts(log: str, profile: Profile) -> list[dict]:
    receipts = []
    decoder = json.JSONDecoder()
    for line in log.splitlines():
        if STARTUP_MARKER in line:
            receipt, _ = decoder.raw_decode(line.split(STARTUP_MARKER, 1)[1].lstrip())
            receipts.append(receipt)
    ranks = [r["global_rank"] for r in receipts]
    if len(receipts) != 4 or set(ranks) != {0, 1, 2, 3}:
        raise RuntimeError(f"need exactly four distinct h25 rank receipts, got {ranks}")
    receipts.sort(key=lambda row: row["global_rank"])
    common = ("kv_host_bytes_all_ranks", "snapshot_host_bytes_all_ranks", "host_reserve_bytes",
              "host_available_bytes_before_pin", "cgroup_limit_bytes")
    first = receipts[0]
    for receipt in receipts:
        rank = receipt["global_rank"]
        width = 4 // profile.attn_dp
        dp = rank // width
        required = {"attn_tp_ranks": list(range(width * dp, width * dp + width)),
                    "attn_tp_leader_global_rank": width * dp,
                    "attn_dp_rank": dp, "namespace": f"attention-dp-{dp}",
                    "max_running_requests_per_dp": profile.per_dp,
                    "logical_tokens_per_dp": profile.logical_tokens, "hot_blocks": profile.hot_blocks}
        if profile.mtp and profile.mtp_steps == 1:
            required["verify_tokens"] = profile.verify_tokens
        for name, expected in required.items():
            if receipt[name] != expected:
                raise RuntimeError(f"h25 rank{rank} receipt mismatch: {name}={receipt[name]!r}, expected {expected!r}")
        for name in common:
            if receipt[name] != first[name]:
                raise RuntimeError(f"h25 aggregate admission differs across ranks: {name}")
        for name in ("kv_host_bytes_per_rank", "snapshot_host_bytes_per_rank", "host_reserve_bytes",
                     "host_available_bytes_before_pin", "cgroup_current_bytes", "proc_available_bytes"):
            if not isinstance(receipt[name], int) or receipt[name] < 0:
                raise RuntimeError(f"invalid h25 host accounting: {name}")
    for per_rank, aggregate in (("kv_host_bytes_per_rank", "kv_host_bytes_all_ranks"),
                                ("snapshot_host_bytes_per_rank", "snapshot_host_bytes_all_ranks")):
        if sum(r[per_rank] for r in receipts) != first[aggregate]:
            raise RuntimeError(f"h25 aggregate does not sum rank allocations: {aggregate}")
    if first["host_reserve_bytes"] < int(profile.host_reserve_gib * (1 << 30)):
        raise RuntimeError("h25 host reserve below requested amount")
    demand = first["kv_host_bytes_all_ranks"] + first["snapshot_host_bytes_all_ranks"] + first["host_reserve_bytes"]
    if profile.native_mtp_host_kv:
        expected = (profile.logical_tokens + 64) * 512  # Pinned REAP320 attention TP2: one 256-dim KV head, FP8 K/V.
        for receipt in receipts:
            if receipt["draft_host_bytes_per_rank"] != expected or receipt["draft_host_bytes_all_ranks"] != 4 * expected:
                raise RuntimeError("native MTP host admission differs from full-history draft geometry")
            if receipt["planned_pinned_bytes_all_ranks"] != first["kv_host_bytes_all_ranks"] + first["snapshot_host_bytes_all_ranks"] + 4 * expected:
                raise RuntimeError("native MTP host admission omitted or double-counted pinned bytes")
        demand += 4 * expected
    else:
        for receipt in receipts:
            if "draft_host_bytes_per_rank" in receipt and receipt["draft_host_bytes_per_rank"] != 0:
                raise RuntimeError("unrequested native MTP host KV appeared in admission")
    if demand > first["host_available_bytes_before_pin"]:
        raise RuntimeError("h25 whole-node admission exceeded available RAM")
    return receipts


def verify_native_host_support(source: Path, profile: Profile) -> None:
    if not profile.native_mtp_host_kv:
        return
    files = {"srt/mem_cache/qsa_kv_pool.py": "class NativeMTPMappedHostFP8Pool",
             "srt/mem_cache/kv_cache_configurator.py": HOST_POOL_MARKER[:-1]}
    for relative, marker in files.items():
        if marker not in (source / "sglang" / relative).read_text():
            raise RuntimeError("native MTP host KV requires a source candidate containing h35")


def verify_phase_packed_support(source: Path, profile: Profile) -> None:
    if profile.spec_phase_packed:
        for relative, marker in {
            "srt/mem_cache/qsa_hisparse_distributed.py": "def spec_phase_to_cpu(",
            "srt/managers/qsa_hisparse_coordinator.py": "if spec_phase_packed_enabled():",
        }.items():
            if marker not in (source / "sglang" / relative).read_text():
                raise RuntimeError("phase-packed agreement requires a source candidate containing h36")


def verify_mtp_steps_support(source: Path, profile: Profile) -> None:
    if profile.mtp and profile.mtp_steps == 1:
        path = source / "sglang/srt/mem_cache/qsa_hisparse_distributed.py"
        if "def distributed_mtp_steps(" not in path.read_text():
            raise RuntimeError("NEXTN1 requires explicitly pinned h38 source support")


def verify_dflash_support(source: Path, profile: Profile) -> None:
    if profile.dflash_drafter is None:
        return
    for relative, marker in {"srt/speculative/eagle_worker_v2.py": "arc_dflash.worker import DFlashDraftWorker",
                             "srt/speculative/arc_dflash/worker.py": "class DFlashDraftWorker:",
                             "srt/managers/overlap_utils.py": "def dflash_drafter_enabled(",
                             "srt/model_executor/pool_configurator.py": "arc_dflash.enabled()"}.items():
        if marker not in (source / "sglang" / relative).read_text():
            raise RuntimeError("DFlash requires a sglang-arc-2 source with speculative/arc_dflash")
    assets = [Path(profile.dflash_drafter) / "config.json"] + ([] if profile.dflash_dense else [Path(profile.dflash_keep)])
    for path in assets:
        if not path.is_file():
            raise RuntimeError(f"DFlash asset missing: {path}")


def verify_rs_relay_support(source: Path, profile: Profile) -> None:
    if profile.mtp_rejection_sampling:
        path = source / "sglang/srt/managers/overlap_utils.py"
        if not path.is_file() or "def require_real_draft_probs(" not in path.read_text():
            raise RuntimeError("RS1 requires pinned h39 real-q relay support; earlier HiSparse RS1 sources are invalid")


def validate_host_pool_receipts(log: str, profile: Profile, admissions: list[dict]) -> list[dict]:
    decoder = json.JSONDecoder()
    pools = [decoder.raw_decode(line.split(HOST_POOL_MARKER, 1)[1].lstrip())[0]
             for line in log.splitlines() if HOST_POOL_MARKER in line]
    if not profile.native_mtp_host_kv:
        if pools:
            raise RuntimeError("unrequested mapped-host draft pool appeared in allocated receipts")
        return []
    if len(pools) != 4 or {pool["global_rank"] for pool in pools} != {0, 1, 2, 3}:
        raise RuntimeError("native MTP host KV needs four distinct allocated rank receipts")
    pools.sort(key=lambda row: row["global_rank"])
    by_rank = {row["global_rank"]: row for row in admissions}
    index_cell = 33 if profile.index_fp8 else 64
    expected_index = ((profile.logical_tokens + 64) * index_cell
                      + (profile.per_dp + 1) * (4 + profile.verify_tokens) * (128 * 2 + 3 * 8))
    for pool in pools:
        required = {"logical_tokens": profile.logical_tokens, "kv_heads": 1, "head_dim": 256,
                    "v_head_dim": 256, "gpu_kv_payload_bytes": 0, "gpu_index_payload_bytes": expected_index,
                    "host_kv_payload_bytes": by_rank[pool["global_rank"]]["draft_host_bytes_per_rank"],
                    "full_history": True, "dtype": "fp8_e4m3", "is_draft_worker": True}
        for name, expected in required.items():
            if pool[name] != expected:
                raise RuntimeError(f"native MTP host pool {name}={pool[name]!r}, expected {expected!r}")
    return pools


def validate_server_info(info: dict, profile: Profile) -> list[dict]:
    required = {"tp_size": 4, "ep_size": 4, "dp_size": profile.attn_dp, "enable_dp_attention": profile.attn_dp == 2,
                "enable_hisparse": True, "kv_cache_dtype": "nvfp4", "dtype": "bfloat16",
                "mamba_ssm_dtype": "float16", "context_length": profile.context,
                "disable_radix_cache": True, "enable_session_radix_cache": True,
                "radix_cache_backend": "qsa_hisparse_session", "enable_linear_replayssm": False,
                "enable_linear_replayssm_spec": profile.mtp,
                "disable_custom_all_reduce": not profile.custom_all_reduce}
    # Validated G4 Qwen4Exp profile uses neither architecture/disaggregation language flag.
    for name, expected in (("language_model_only", False), ("language_only", False), ("encoder_only", False)):
        if name in info:
            required[name] = expected
    if profile.dist_timeout is not None:
        required["dist_timeout"] = profile.dist_timeout
    if profile.prefill_max_requests is not None:
        required["prefill_max_requests"] = profile.prefill_max_requests
    if profile.prefill_delayer:
        required["enable_prefill_delayer"] = True
    elif info.get("enable_prefill_delayer"):  # older trees do not report the field at all
        raise RuntimeError("prefill delayer enabled on the server but not requested")
    if profile.prefill_delayer_max_passes is not None:
        required["prefill_delayer_max_delay_passes"] = profile.prefill_delayer_max_passes
    if profile.prefill_delayer_queue_ratio is not None:
        required["prefill_delayer_queue_min_ratio"] = profile.prefill_delayer_queue_ratio
    if profile.prefill_delayer_max_delay_ms is not None:
        required["prefill_delayer_max_delay_ms"] = profile.prefill_delayer_max_delay_ms
    if profile.mtp:
        required.update(speculative_algorithm="EAGLE", speculative_num_steps=profile.mtp_steps,
                        speculative_eagle_topk=1, speculative_num_draft_tokens=profile.verify_tokens,
                        speculative_draft_kv_cache_dtype="fp8_e4m3", speculative_attention_mode="prefill",
                        disable_overlap_schedule=False, speculative_adaptive=False,
                        enable_multi_layer_eagle=False, linear_replayssm_cache_len=16,
                        speculative_use_rejection_sampling=profile.mtp_rejection_sampling,
                        speculative_moe_runner_backend="triton")
    for name, expected in required.items():
        if info.get(name) != expected:
            raise RuntimeError(f"HiSparse server profile mismatch: {name}={info.get(name)!r}, expected {expected!r}")
    if not profile.mtp and info.get("speculative_algorithm") not in (None, "NONE"):
        raise RuntimeError("non-MTP arm unexpectedly enabled speculative decoding")
    if json.loads(info["hisparse_config"]) != profile.hisparse_config():
        raise RuntimeError("HiSparse config differs from requested per-DP/session budgets")
    states = info["internal_states"]
    if len(states) != profile.attn_dp:
        raise RuntimeError(f"HiSparse needs {profile.attn_dp} allocated attention-DP pools, got {len(states)}")
    workers = []
    for dp, state in enumerate(states):
        worker = {"index": dp, "pool": int(state["memory_usage"]["token_capacity"]),
                  "running": int(state["effective_max_running_requests_per_dp"]),
                  "mamba": int(state["max_mamba_cache_size"]), "memory": state["memory_usage"]}
        if worker["pool"] != profile.logical_tokens or worker["running"] != profile.per_dp:
            raise RuntimeError(f"HiSparse allocated per-DP capacity mismatch: {worker}")
        if worker["mamba"] < profile.per_dp + profile.snapshot_staging_slots:
            raise RuntimeError(f"HiSparse Mamba pool cannot cover live plus snapshot staging: {worker}")
        workers.append(worker)
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, help="audited source; otherwise use model/verified_identity.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path,
                        default=REPO / "patches/sglang-0.5.20/experimental-hisparse/SOURCE_SHA256")
    parser.add_argument("--running", type=int, default=8, help="global across both DP groups")
    parser.add_argument("--logical-tokens", type=int, default=1200000, help="per DP, includes parked sessions")
    parser.add_argument("--host-reserve-gib", type=float, required=True, help="one whole-node reserve")
    parser.add_argument("--hot-blocks", type=int, default=512)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--snapshot-interval", type=int, default=64)
    parser.add_argument("--snapshot-staging-slots", type=int, default=4)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--prefill-max-requests", type=int,
                        help="opt-in prefill batch request limit; use 1 for the initial C64 memory profile")
    parser.add_argument("--mem-fraction", type=float, default=0.90)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--sparse-graphs", action="store_true",
                        help="opt-in fewer CUDA graph shapes with runtime null-row padding; GPU validation required")
    parser.add_argument("--index-fp8", action="store_true", help="resident index FP8 capacity option")
    parser.add_argument("--mtp", action="store_true",
                        help="opt-in NEXTN + ReplaySSM-spec; requires guarded source and B>=512*(steps+1)")
    parser.add_argument("--mtp-steps", type=int, choices=range(1, 8), default=3,
                        help="explicit NEXTN width; default3, steps1 requires MTP and pinned h38 source")
    parser.add_argument("--draft-fp8-direct", action="store_true",
                        help="opt-in h22 split direct FP8 draft attention; requires MTP, SM89 A/B unverified")
    parser.add_argument("--fold-in-verify", action="store_true",
                        help="opt-in h20 ReplaySSM fold in next verify; requires MTP, SM89 A/B unverified")
    parser.add_argument("--gdn-state-fmt", choices=("i10h", "i12h"),
                        help="h24 packed GDN state on --fold-in-verify (ARC_GDN_STATE_FMT + _DISTRIBUTED=1)")
    parser.add_argument("--mtp-rejection-sampling", action="store_true",
                        help="opt-in native MTP sampled proposals and rejection sampling; requires MTP")
    parser.add_argument("--native-mtp-host-kv", action="store_true",
                        help="opt-in h35 full-history mapped-host draft FP8 K/V; GPU index retained, requires MTP")
    parser.add_argument("--cpu-affinity", action="store_true",
                        help="opt-in reviewed64-core/128-thread per-rank CPU affinity; topology/readback gates required")
    parser.add_argument("--no-debug-checks", action="store_false", dest="debug_checks",
                        help="opt-in diagnostic host-readback removal; keeps distributed lifecycle protocol checks")
    parser.add_argument("--port", type=int, default=30021)
    parser.add_argument("--deterministic", action="store_true",
                        help="quality A/B only: the three run-to-run nondeterminism switches (slower)")
    parser.add_argument("--swap-in-block-size", type=int,
                        help="HiSparse swap-in threads per CTA (default 960); lower it when a hot size fails to launch")
    parser.add_argument("--attn-dp", type=int, choices=(1, 2), default=2,
                        help="1 = pure TP4 attention (KV heads replicated); needs a source with the ARC_HISPARSE_TP4 gate")
    parser.add_argument("--custom-all-reduce", action="store_true",
                        help="opt in only after native peer and custom graph replay data gates pass")
    parser.add_argument("--spec-phase-packed", action="store_true",
                        help="opt-in h36 exact phase-local agreement packets; requires MTP and GPU oracle proof")
    parser.add_argument("--dflash", type=Path, help="MTP5-DFlash export dir: replaces the native MTP draft "
                        "(needs --mtp --mtp-rejection-sampling; source with experimental-dflash/l4)")
    parser.add_argument("--dflash-keep", type=Path,
                        help="per-layer kept expert ids for a MoE drafter (default g4/results/keep_freq.json); "
                        "a dense export (config ffn_type dense_shared) takes none")
    parser.add_argument("--python")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dist-timeout", type=int,
                        help="explicit collective timeout in seconds; use 120 for the initial diagnostic arm")
    parser.add_argument("--dry-run", action="store_true", help="asset and CPU profile checks only")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    dflash_dense = bool(args.dflash) and json.loads((args.dflash / "config.json").read_text()).get(
        "ffn_type", "moe") == "dense_shared"
    if dflash_dense and args.dflash_keep is not None:
        raise SystemExit(f"{args.dflash} is a dense DFlash export: it has no experts to keep (drop --dflash-keep)")
    dflash_keep = None if not args.dflash or dflash_dense else str(
        (args.dflash_keep or REPO / "work/qwen38fn_dflash/g4/results/keep_freq.json").resolve())
    profile = Profile(running=args.running, logical_tokens=args.logical_tokens,
                      host_reserve_gib=args.host_reserve_gib, hot_blocks=args.hot_blocks,
                      context=args.context, snapshot_interval=args.snapshot_interval,
                      snapshot_staging_slots=args.snapshot_staging_slots, prefill_chunk=args.prefill_chunk,
                      mem_fraction=args.mem_fraction, eager=args.eager, index_fp8=args.index_fp8,
                      dist_timeout=args.dist_timeout, prefill_max_requests=args.prefill_max_requests,
                      mtp=args.mtp, debug_checks=args.debug_checks, sparse_graphs=args.sparse_graphs,
                      draft_fp8_direct=args.draft_fp8_direct, fold_in_verify=args.fold_in_verify,
                      gdn_state_fmt=args.gdn_state_fmt, mtp_rejection_sampling=args.mtp_rejection_sampling,
                      native_mtp_host_kv=args.native_mtp_host_kv,
                      cpu_affinity=args.cpu_affinity,
                      custom_all_reduce=args.custom_all_reduce, spec_phase_packed=args.spec_phase_packed,
                      mtp_steps=args.mtp_steps, attn_dp=args.attn_dp,
                      swap_in_block_size=args.swap_in_block_size, deterministic=args.deterministic,
                      dflash_drafter=str(args.dflash.resolve()) if args.dflash else None,
                      dflash_keep=dflash_keep, dflash_dense=dflash_dense)
    python, env, runtime = runtime_env(args.runtime.resolve(), "hisparse", args.python)
    # The startup-only h30 cache-advice hook imports the carried arcforge helper.
    # runtime_env deliberately rebuilds PYTHONPATH, so add this audited source
    # explicitly instead of relying on the launching notebook's ambient path.
    env["PYTHONPATH"] = str(REPO / "src") + ":" + env["PYTHONPATH"]
    # Experiment-only: ARC_PYTHONPATH_PREPEND puts import hooks first (e.g. the hostcomm sitecustomize of
    # work/qwen38fn_l4/kaggle/hc_inject, as Kaggle v15 does); recorded in launch.json via PYTHONPATH.
    if os.environ.get("ARC_PYTHONPATH_PREPEND"):
        env["PYTHONPATH"] = os.environ["ARC_PYTHONPATH_PREPEND"] + ":" + env["PYTHONPATH"]
    if env.get("ARC_HISPARSE_STARTUP_CACHE_DROP") == "1":
        subprocess.run([python, "-c", "import arcforge.checkpoint_prefetch"],
                       env=env, check=True, timeout=30)
    source = Path(runtime["arms"]["hisparse"]["source"])
    manifest = args.source_manifest.resolve()
    manifest_sha = verify_source_manifest(source, manifest)
    verify_native_host_support(source, profile)
    verify_phase_packed_support(source, profile)
    verify_mtp_steps_support(source, profile)
    verify_rs_relay_support(source, profile)
    verify_dflash_support(source, profile)
    model = args.model.resolve()
    if args.notebook:
        metadata = notebook_metadata(args.notebook)
    else:
        identity = json.loads((model / "verified_identity.json").read_text())
        metadata = {"MANIFEST": {"target_sizes": identity["target_sizes"], "target_sha256": identity["target_sha256"]}}
        if metadata["MANIFEST"]["target_sha256"]["config.json"] not in MODEL_IDENTITIES.values():
            raise RuntimeError("verified model identity is none of the audited checkpoints")
    model_identity = verify_model(model, metadata)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise RuntimeError("run directory is not empty; use a new run tag")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Keep compile caches between eager/graph runs, isolated from the frozen baseline.
    cache = args.runtime.resolve() / "cache-sm89-hisparse"
    cache.mkdir(exist_ok=True)
    env = profile_env(env, manifest, profile, cache)
    shape_checks = verify_h25_shape(python, env, source, profile)
    command = server_args(python, model, args.port, profile)
    # Record the explicit profile only; inherited application credentials never enter artifacts.
    env_record = recorded_env(env, manifest, profile, cache)
    launch = {"args": command, "environment": env_record, "runtime": runtime, "model": model_identity,
              "port": args.port,
              "hisparse_config": profile.hisparse_config(), "source_manifest_sha256": manifest_sha,
              "h25_shape_checks": shape_checks, "status": "dry-run" if args.dry_run else "starting",
              "requested_workload": {"modality": "text+vision", "image_inputs": True},
              "vision_http_validation": "pending",
              "distributed_timeout_seconds": profile.dist_timeout,
              "prefill_max_requests": profile.prefill_max_requests,
              "mtp_requested": profile.mtp, "mtp_generation_validation": "pending" if profile.mtp else "not-requested",
              "diagnostic_host_checks": profile.debug_checks,
              "sparse_graphs": profile.sparse_graphs,
              "graph_batches_requested": [] if profile.eager else profile.graph_batches,
              "draft_fp8_direct": profile.draft_fp8_direct,
              "fold_in_verify": profile.fold_in_verify,
              "gdn_state_fmt": profile.gdn_state_fmt,
              "custom_all_reduce_requested": profile.custom_all_reduce,
              "mtp_rejection_sampling": profile.mtp_rejection_sampling,
              "native_mtp_host_kv": profile.native_mtp_host_kv,
              "dflash_drafter": profile.dflash_drafter, "dflash_keep": profile.dflash_keep,
              "dflash_dense": profile.dflash_dense,
              "cpu_affinity_requested": profile.cpu_affinity,
              "admission_verified": False, "allocated_pools_verified": False}
    write_json(args.run_dir / "launch.json", launch)
    try:
        config_preflight(python, env, command, args.run_dir)
    except BaseException as exc:
        launch.update(status="preflight-failed", error=str(exc))
        write_json(args.run_dir / "launch.json", launch)
        raise
    launch.update(config_preflight_verified=True)
    write_json(args.run_dir / "launch.json", launch)
    if args.dry_run:
        print(json.dumps(launch, indent=2))
        return
    require_free_port(args.port)
    hardware = gpu_probe(python, env, args.run_dir)
    affinity_preflight = None
    if profile.cpu_affinity:
        from runpod_cpu_affinity import collect, validate
        affinity_preflight = validate(collect(hardware["topology"]))
        write_json(args.run_dir / "cpu_affinity_preflight.json", affinity_preflight)
    log_path = args.run_dir / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (args.run_dir / "server.pid").write_text(f"{server.pid}\n")
    launch.update(pid=server.pid, pgid=server.pid)
    write_json(args.run_dir / "launch.json", launch)
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.timeout:
            log = log_path.read_text(errors="replace")
            if not launch["admission_verified"] and log.count(STARTUP_MARKER) == 4:
                receipts = validate_receipts(log, profile)
                write_json(args.run_dir / "admission_receipts.json", {"receipts": receipts, "admission_verified": True})
                launch.update(admission_verified=True)
                write_json(args.run_dir / "launch.json", launch)
            if server.poll() is not None:
                raise RuntimeError(f"owned HiSparse server exited: {server.returncode}; see server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/get_server_info", timeout=5) as response:
                    info = json.load(response)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                time.sleep(5)
                continue
            workers = validate_server_info(info, profile)
            log = log_path.read_text(errors="replace")
            trees = re.findall(r"ARC_HISPARSE_QSA_TREE=([0-9a-f]{64})", log)
            if not trees or any(tree != manifest_sha for tree in trees):
                raise RuntimeError("HiSparse startup source receipts missing or inconsistent")
            receipts = validate_receipts(log, profile)
            write_json(args.run_dir / "admission_receipts.json", {"receipts": receipts, "admission_verified": True})
            host_pools = validate_host_pool_receipts(log, profile, receipts)
            write_json(args.run_dir / "native_draft_pool_receipts.json", {"requested": profile.native_mtp_host_kv,
                                                                         "receipts": host_pools})
            write_json(args.run_dir / "deployment_start.json", {"server_info": info, "workers": workers,
                                                               "args": command, "admission_receipts": receipts,
                                                               "native_draft_host_pools": host_pools,
                                                               "text_only_mode": {key: info.get(key) for key in
                                                                    ("language_model_only", "language_only", "encoder_only")}})
            if affinity_preflight is not None:
                from runpod_cpu_affinity import readback
                affinity = readback(log, server.pid, server.pid, affinity_preflight["recommended_rank_cpus"])
                write_json(args.run_dir / "cpu_affinity_readback.json", affinity)
                launch.update(cpu_affinity_verified=True)
            launch.update(status="ready", startup_seconds=time.monotonic() - started, workers=workers,
                          language_model_only=info.get("language_model_only"),
                          native_draft_host_pools=host_pools,
                          admission_verified=True, allocated_pools_verified=True)
            write_json(args.run_dir / "launch.json", launch)
            print(f"HISPARSE_READY pid={server.pid} port={args.port} run={args.run_dir}", flush=True)
            if args.wait:
                server.wait()
            return
        raise TimeoutError("HiSparse startup timeout; see server.log")
    except BaseException as exc:
        stop_owned(server)
        launch.update(status="failed", error=str(exc), server_returncode=server.returncode)
        write_json(args.run_dir / "launch.json", launch)
        raise


if __name__ == "__main__":
    main()
