"""Opt-in Qwen4-Exp packed INT3 plugin for the pinned SGLang 0.5.20 G4 stack.

CPU audit: python int3_deploy_runtime.py check --model MODEL
Serve: python int3_deploy_runtime.py launch --model-path MODEL [server arguments]
The launcher installs the plugin in spawned interpreters too. It does not start
CUDA during a CPU audit. Main experts use SGLang's native HummingMoEMethod;
the original ModelOpt mixed loader still owns all other modules.
A weight sync (RL update_weights_from_disk) after the first complete load may send the dense tensors alone or
all expert tensors again, never a part of them (log line ARC_INT3_DELTA_LOADED).
ARC_INT3_ACT=fp8 runs the routed experts W3A8 (dynamic per-token e4m3
activations); the default bf16 is W3A16. The checkpoint is the same.
ARC_INT3_TUNING=<table.json> (from int3_humming_tune.py) replaces Humming's
INDEXED schedules for the expert banks; the table must match this Humming
version, activation dtype, layer shapes and default buckets exactly.
ARC_INT3_RUNNER_REUSE=0 turns off the per-layer Humming runner reuse (default 1).
ARC_HOST_POOL_PRIVATE=1 puts the pinned host pools on private anonymous pages
instead of SGLang's shared mapping, which compaction keeps trying to migrate.
ARC_PINNED_HOST_REGISTER=1 does the same for PyTorch's pin_memory=True blocks
(malloc + cudaHostRegister instead of cudaHostAlloc's shared mapping).
ARC_INT3_WEIGHT_PATCH=<dir> serves the checkpoint with a weight patch's tensors in place of its own (WeightPatch;
log line ARC_INT3_WEIGHT_PATCH_LOADED).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import re
import runpy
import struct
import sys
from pathlib import Path

_INSTALLED = False
_EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(w13_qweight|w13_scales|w2_qweight|w2_scales)$")
_SUFFIX = {"w13_qweight": "w13_weight", "w13_scales": "w13_weight_scale",
           "w2_qweight": "w2_weight", "w2_scales": "w2_weight_scale"}
# int8 activations would make Humming fold the signed BF16 group scales into
# integer scales, a different weight numerics path; only fp8 is offered.
_ACTIVATIONS = {"bf16": None, "fp8": "float8e4m3"}


def activation_mode() -> str:
    mode = os.environ.get("ARC_INT3_ACT", "bf16")
    if mode not in _ACTIVATIONS:
        raise ValueError(f"ARC_INT3_ACT must be one of {sorted(_ACTIVATIONS)}, got {mode!r}")
    return mode


def runtime_prefix(name: str) -> str:
    """Use Qwen4's own HF-to-runtime names, including its fused dense linears."""
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model."):]
    elif name.startswith("language_model.model."):
        name = "model." + name[len("language_model.model."):]
    name = name.replace(".self_attn.", ".")
    for projection in ("q_proj", "k_proj", "v_proj"):
        if name.endswith("." + projection):
            name = name[: -len(projection)] + "qkv_proj"
    for projection in ("in_proj_qkv", "in_proj_z"):
        if name.endswith("." + projection):
            name = name[: -len(projection)] + "in_proj_qkvz"
    return name


def validate_contract(config: dict) -> dict:
    q = config["quantization_config"]
    if q["quant_method"] != "modelopt_mixed" or q["quant_algo"] != "MIXED_PRECISION":
        raise ValueError("packed INT3 requires the ModelOpt mixed nonexpert carrier")
    meta = q["arc_int3"]
    expected = {"schema_version": 1, "bits": 3, "group_size": 128,
                "num_experts": 320, "scale_dtype": "bfloat16",
                "layout": "humming_uint3_g128_nozp", "w13_rows": "gate_then_up"}
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"arc_int3.{key}: expected {value!r}, got {meta[key]!r}")
    text = config["text_config"]
    if text["num_hidden_layers"] != 48 or text["num_experts"] != 320:
        raise ValueError("this deployment supports only Qwen4-Exp 48-layer e320")
    if text["num_experts_per_tok"] != 10:
        raise ValueError("expected top-10 routing")
    if not text["norm_topk_prob"]:
        raise ValueError("decoupled routing is not implemented by this packed loader")
    if config["architectures"] != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("expected Qwen4ExpForConditionalGeneration")
    experts = meta["main_expert_modules"]
    expected_experts = [f"model.language_model.layers.{i}.mlp.experts" for i in range(48)]
    if sorted(experts) != sorted(expected_experts) or len(experts) != 48:
        raise ValueError("main_expert_modules must name each of the 48 layers once")
    for field in ("dense_fp8_modules", "dense_bf16_modules"):
        names = meta[field]
        if len(names) != len(set(names)) or any(".mlp.experts" in n or "mtp" in n for n in names):
            raise ValueError(f"invalid {field}")
    fp8 = {runtime_prefix(n) for n in meta["dense_fp8_modules"]}
    bf16 = {runtime_prefix(n) for n in meta["dense_bf16_modules"]}
    if fp8 & bf16:
        raise ValueError("fused dense modules cannot mix FP8 and BF16 shards")
    return meta


def audit_model(model: Path) -> dict:
    """Inspect safetensors headers only; no weights are moved onto a GPU."""
    config = json.loads((model / "config.json").read_text())
    meta = validate_contract(config)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    hidden = config["text_config"]["hidden_size"]
    intermediate = config["text_config"]["moe_intermediate_size"]
    expected = {}
    for prefix in meta["main_expert_modules"]:
        expected.update({prefix + ".w13_qweight": ([320, 2 * intermediate, hidden * 3 // 32], "I32"),
                         prefix + ".w13_scales": ([320, 2 * intermediate, hidden // 128], "BF16"),
                         prefix + ".w2_qweight": ([320, hidden, intermediate * 3 // 32], "I32"),
                         prefix + ".w2_scales": ([320, hidden, intermediate // 128], "BF16")})
    if hidden % 128 or intermediate % 128:
        raise ValueError("INT3 group and packing dimensions must be divisible by 128")
    routed = {k for k in index if k.startswith("model.language_model.layers.") and ".mlp.experts." in k}
    if routed != set(expected):
        raise ValueError(f"unexpected expert keys: missing={sorted(set(expected)-routed)[:4]}, extra={sorted(routed-set(expected))[:4]}")
    by_file = {}
    for name, (shape, dtype) in expected.items():
        by_file.setdefault(index[name], []).append((name, shape, dtype))
    headers = {}
    for filename in set(index.values()):
        path = model / filename
        if Path(filename).name != filename:
            raise ValueError(f"unsafe shard filename {filename!r}")
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header: {filename}")
            size = struct.unpack("<Q", raw)[0]
            if size > 16 * 1024 * 1024:
                raise ValueError(f"oversized safetensors header: {filename}")
            header = json.loads(f.read(size))
        payload_size = path.stat().st_size - 8 - size
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= payload_size:
                raise ValueError(f"truncated tensor payload {filename}:{name}")
        headers[filename] = header
    for name, filename in index.items():
        if name not in headers[filename]:
            raise ValueError(f"index references absent tensor {filename}:{name}")
    for name, (shape, dtype) in expected.items():
        tensor = headers[index[name]][name]
        if tensor["shape"] != shape or tensor["dtype"] != dtype:
            raise ValueError(f"{name}: expected {dtype} {shape}, got {tensor['dtype']} {tensor['shape']}")
        start, end = tensor["data_offsets"]
        if end - start != math.prod(shape) * (4 if dtype == "I32" else 2):
            raise ValueError(f"packed tensor payload size disagrees with header: {name}")
    for name in meta["dense_fp8_modules"]:
        if name + ".weight" not in index or name + ".weight_scale_inv" not in index:
            raise ValueError(f"missing FP8 weight/scale pair: {name}")
        weight = headers[index[name + ".weight"]][name + ".weight"]
        scale = headers[index[name + ".weight_scale_inv"]][name + ".weight_scale_inv"]
        if weight["dtype"] != "F8_E4M3" or scale["dtype"] != "F32":
            raise ValueError(f"invalid FP8 weight/scale dtype: {name}")
        if scale["shape"] != [(n + 127) // 128 for n in weight["shape"]]:
            raise ValueError(f"invalid FP8 block scale shape: {name}")
    for name in meta["dense_bf16_modules"]:
        if name + ".weight" not in index or name + ".weight_scale_inv" in index:
            raise ValueError(f"invalid BF16 dense pair: {name}")
        if headers[index[name + ".weight"]][name + ".weight"]["dtype"] != "BF16":
            raise ValueError(f"invalid BF16 dense dtype: {name}")
    return {"status": "cpu-header-pass", "experts": 320, "layers": 48,
            "expert_tensors": len(expected), "expert_shards": len(by_file),
            "dense_fp8_modules": len(meta["dense_fp8_modules"]),
            "dense_bf16_modules": len(meta["dense_bf16_modules"]),
            "gpu_validated": False}


def expert_coverage(loaded: set, required: set, already_loaded: bool) -> bool:
    """Whether one load_weights call carried the packed experts (True) or only dense tensors (False).

    The first call on a model instance must carry all of them. Later calls (RL weight sync) carry all of them or
    none; a part of the experts is a broken sync and raises instead of leaving the banks half old, half new.
    """
    if loaded == required:
        return True
    if not already_loaded or loaded:
        raise ValueError(f"missing packed expert tensors: {sorted(required-loaded)[:8]}")
    return False


WEIGHT_PATCH_SCHEMA = "arcforge.kaggle.weight_patch.v1"


class WeightPatch:
    """ARC_INT3_WEIGHT_PATCH=<dir> (work/qwen38fn_rl/rl_weight_patch.py): tensors that replace the checkpoint's own
    under the same name as the loader reads them, so an RL version is served off the unchanged package without a
    rewritten shard on disk. Every patch tensor must arrive exactly once, with the checkpoint's dtype and shape, and
    hash to the sha256 patch.json records; a missing one, a duplicate or a second load raises."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "patch.json").read_text())
        if self.manifest["schema"] != WEIGHT_PATCH_SCHEMA:
            raise ValueError(f"{self.root}/patch.json is {self.manifest['schema']!r}, not {WEIGHT_PATCH_SCHEMA}")
        self.pending = dict(self.manifest["tensors"])
        self.done = set()
        self.handles = {}

    def replace(self, name, tensor):
        import hashlib
        import torch
        from safetensors import safe_open
        entry = self.pending.pop(name)
        if entry["file"] not in self.handles:
            self.handles[entry["file"]] = safe_open(str(self.root / entry["file"]), "pt")
        new = self.handles[entry["file"]].get_tensor(name)
        if new.dtype != tensor.dtype or tuple(new.shape) != tuple(tensor.shape):
            raise ValueError(f"weight patch {name}: {new.dtype} {tuple(new.shape)}, the checkpoint has "
                             f"{tensor.dtype} {tuple(tensor.shape)}")
        digest = hashlib.sha256(new.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"weight patch {name} is {digest}, patch.json says {entry['sha256']}")
        self.done.add(name)
        return new.to(tensor.device)

    def substitute(self, weights):
        for name, tensor in weights:
            if name in self.done:
                raise ValueError(f"weight patch: {name} arrived twice")
            yield name, (self.replace(name, tensor) if name in self.pending else tensor)

    def finish(self) -> str:
        if self.pending:
            raise ValueError(f"weight patch: {len(self.pending)} tensors never arrived: {sorted(self.pending)[:4]}")
        self.handles.clear()
        return (f"ARC_INT3_WEIGHT_PATCH_LOADED tensors={len(self.done)} "
                f"deploy_manifest={self.manifest['deploy_manifest_sha256'][:16]}")


def tuned_schedules(table: dict, sublayer: str, defaults: list, gemm_type: str) -> list:
    """The table's schedules for one expert sublayer, after checking they were tuned against these defaults."""
    if gemm_type != "indexed":
        raise ValueError(f"ARC_INT3_TUNING holds INDEXED schedules; SGLang asked for {gemm_type}")
    entry = table["sublayers"][sublayer]
    if json.loads(json.dumps(defaults)) != entry["default"]:
        raise ValueError(f"Humming's default {sublayer} schedules differ from the tuning table's; re-tune")
    if [b[:2] for b in entry["tuned"]] != [b[:2] for b in entry["default"]]:
        raise ValueError(f"tuning table changes the {sublayer} bucket boundaries")
    return [tuple(bucket) for bucket in entry["tuned"]]


def check_shared_block_m(table: dict) -> None:
    """SGLang aligns the routed tokens to w13's block M and runs w2 on that alignment (HummingRunnerCore.
    _prepare_indexed_gemm_kwargs takes moe_block_size from the w13 schedule): a table whose w2 schedule has another
    block M at some row count would run w2 on a misaligned token list there. int3_humming_tune tunes the two
    sublayers apart, so the table must say they agree at every row count (its bucket edges cover every interval)."""
    w13, w2 = (table["sublayers"][s]["tuned"] for s in ("w13", "w2"))

    def block_m(buckets, rows):
        return next(cfg["block_shape"][0] for lo, hi, cfg in buckets if lo < rows <= hi)

    for rows in sorted({hi for _, hi, _ in w13 + w2} | {lo + 1 for lo, _, _ in w13 + w2}):
        if block_m(w13, rows) != block_m(w2, rows):
            raise ValueError(f"tuning table runs w13 at block M {block_m(w13, rows)} and w2 at {block_m(w2, rows)} "
                             f"for {rows} rows; SGLang aligns both to w13's")


def install_tuning(table: dict, humming_version: str, act: str) -> None:
    """Route Humming's schedule lookup for the INT3 expert banks through a measured table."""
    from humming.config import GemmType
    from humming.layer import HummingMethod

    if (table["humming"], table["act"], table["gemm_type"]) != (humming_version, act, "indexed"):
        raise ValueError(f"tuning table is for humming {table['humming']} act={table['act']} {table['gemm_type']}; "
                         f"running humming {humming_version} act={act}")
    if set(table["sublayers"]) != {"w13", "w2"}:
        raise ValueError("tuning table must cover exactly w13 and w2")
    check_shared_block_m(table)
    original = HummingMethod.get_default_tuning_configs.__func__
    reported = set()

    def get_default_tuning_configs(cls, layer, use_f16_accum=False, use_batch_invariant=False,
                                   use_m_major_input_scale=False, gemm_type=GemmType.DENSE, sublayer_name=""):
        defaults = original(cls, layer, use_f16_accum=use_f16_accum, use_batch_invariant=use_batch_invariant,
                            use_m_major_input_scale=use_m_major_input_scale, gemm_type=gemm_type,
                            sublayer_name=sublayer_name)
        if use_f16_accum or use_batch_invariant or use_m_major_input_scale:
            raise ValueError("ARC_INT3_TUNING was measured without f16 accum / batch invariance / m-major scales")
        tuned = tuned_schedules(table, sublayer_name, defaults, GemmType(gemm_type).value)
        if sublayer_name not in reported:
            reported.add(sublayer_name)
            changed = sum(json.loads(json.dumps(a)) != json.loads(json.dumps(b)) for a, b in zip(defaults, tuned))
            print(f"ARC_INT3_TUNING {sublayer_name}: {changed} of {len(tuned)} buckets replaced", flush=True)
        return tuned

    HummingMethod.get_default_tuning_configs = classmethod(get_default_tuning_configs)


def install_runner_reuse() -> None:
    """One HummingRunnerCore per expert layer on the standard-dispatch path.

    SGLang's ("none", "humming") fused func builds a new HummingRunnerCore on every call, so the core's
    schedule cache never hits: every eager (extend) MoE call recomputes Humming's heuristics for w13 and w2
    and json-encodes them (py-spy, G4 W3A8 C110, minute 22). CUDA-graph decode only runs it at capture.
    """
    from sglang.srt.layers.moe.moe_runner.base import FusedOpPool
    from sglang.srt.layers.moe.moe_runner import humming as runner_humming

    key = ("none", "humming")
    if FusedOpPool._fused_funcs.get(key) is not runner_humming.fused_experts_none_to_humming:
        raise RuntimeError("SGLang's none->humming fused func changed; revisit the runner-reuse patch")
    gemm_type = runner_humming.get_standard_humming_moe_gemm_type()

    def fused_experts_none_to_humming(dispatch_output, quant_info, runner_config):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        layer = quant_info.layer
        core = layer.__dict__.get("_arc_humming_core")
        if core is None:
            core = runner_humming.HummingRunnerCore(runner_config)
            layer.__dict__["_arc_humming_core"] = core
        elif core.config is not runner_config:
            raise RuntimeError("one expert layer reached Humming with two runner configs")
        topk_output = dispatch_output.topk_output
        runner_input = runner_humming.HummingRunnerInput(
            hidden_states=dispatch_output.hidden_states, topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids, gemm_type=gemm_type)
        return StandardCombineInput(hidden_states=core.run(runner_input, quant_info, {}).hidden_states)

    FusedOpPool._fused_funcs[key] = fused_experts_none_to_humming


_MADV_POPULATE_WRITE = 23  # Linux 5.14+; the stdlib mmap module may not name it


def private_host_buffer(dims: tuple, dtype) -> "torch.Tensor":
    """dims x dtype of private anonymous memory: every page faulted in, the range not inherited by fork()."""
    import mmap
    import torch

    count = math.prod(dims)
    alloc_bytes = math.ceil(count * torch.empty([], dtype=dtype).element_size() / mmap.PAGESIZE) * mmap.PAGESIZE
    mm = mmap.mmap(-1, alloc_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    mm.madvise(mmap.MADV_DONTFORK)
    mm.madvise(getattr(mmap, "MADV_POPULATE_WRITE", _MADV_POPULATE_WRITE))
    return torch.frombuffer(mm, dtype=dtype, count=count).reshape(dims)


def install_private_host_pools() -> None:
    """Pinned host pools (QSA HiSparse host KV, GDN buffers, PLE table) on private anonymous pages.

    SGLang maps every cudaHostRegister'd host buffer MAP_SHARED | MAP_ANONYMOUS, i.e. shmem (~125 GiB on G4).
    Compaction cannot see a pin on a shmem page until it has unmapped it, so each attempt (kcompactd's proactive
    compaction, direct compaction under the agents' memory pressure) sends a TLB-shootdown IPI to every CPU
    running a scheduler thread, fails on the pin and maps the page back: G4 C110 showed ~15k failed migrations/s
    and the extend forwards 1.5-2.7x slower in those minutes. A pinned private anonymous page fails compaction's
    pinned-anon check before it is isolated, so it is never unmapped. MADV_DONTFORK keeps fork() from copying
    the pinned range (the reason SGLang chose MAP_SHARED).
    """
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.pool_host import common
    from sglang.srt.mem_cache.storage.mmap import mmap_allocator

    if common.alloc_mmap is not mmap_allocator.alloc_mmap:
        raise RuntimeError("SGLang's host allocator no longer calls mmap_allocator.alloc_mmap; revisit ARC_HOST_POOL_PRIVATE")
    if (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip():
        raise RuntimeError("ARC_HOST_POOL_PRIVATE does not cover SGLANG_HUGEPAGE_SIZE hugepage pools")
    common.alloc_mmap = private_host_buffer


def pinned_host_register_conf(conf: str) -> str:
    """PYTORCH_CUDA_ALLOC_CONF with PyTorch's pinned host blocks allocated by malloc + cudaHostRegister.

    cudaHostAlloc (PyTorch's default for pin_memory=True) maps MAP_SHARED /dev/zero pages: shmem, which compaction
    unmaps before it finds the pin, like the host pools ARC_HOST_POOL_PRIVATE moves (~1 GB on G4 C110 after one run).
    """
    if "pinned_use_cuda_host_register" in conf or "pinned_num_register_threads" in conf:
        raise ValueError(f"PYTORCH_CUDA_ALLOC_CONF already sets the pinned allocator: {conf!r}")
    return ",".join(filter(None, [conf, "pinned_use_cuda_host_register:True", "pinned_num_register_threads:8"]))


def install() -> None:
    """Install only with explicit process opt-in; leave other model configs alone."""
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("ARC_INT3_DEPLOY") != "1":
        raise RuntimeError("set ARC_INT3_DEPLOY=1 through the launch entrypoint")
    import torch
    from humming import dtypes
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.humming import HummingLayerQuantizationConfig, HummingMoEMethod
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    humming_version = importlib.metadata.version("humming-kernels")
    if humming_version not in ("0.1.12", "0.1.16", "0.1.18"):
        raise RuntimeError(f"unreviewed humming-kernels version {humming_version}")
    if humming_version == "0.1.18":
        # 0.1.18's SM100 default schedules give w13 and w2 different block M at some row counts (8192-token prefill:
        # 120 / 112); a runner that aligns w2 on w13's token list reads it misaligned there, silently.
        from sglang.srt.layers.moe.moe_runner import humming as runner_humming
        if vars(runner_humming).get("INDEXED_W2_OWN_ALIGNMENT") is not True:
            raise RuntimeError("humming 0.1.18 needs a fork tree whose Humming runner aligns w2 to its own block M "
                               "(sglang-arc-2 c61e351b7 or later)")
    from sglang import __version__ as sglang_version
    if sglang_version != "0.5.20":
        raise RuntimeError(f"unreviewed SGLang version {sglang_version}")
    for function, fields in ((HummingMoEMethod.create_weights, {"num_experts", "hidden_size", "intermediate_size_per_partition"}),
                             (Qwen4ExpForConditionalGeneration.load_weights, {"weights"})):
        if not fields <= set(inspect.signature(function).parameters):
            raise RuntimeError(f"serving API changed: {function.__qualname__}")
    original_from = ModelOptMixedPrecisionConfig.from_config.__func__
    original_method = ModelOptMixedPrecisionConfig.get_quant_method
    original_load = Qwen4ExpForConditionalGeneration.load_weights
    act = activation_mode()
    a_name = _ACTIVATIONS[act]
    input_schema = None if a_name is None else HummingInputSchema(a_dtype=a_name, input_scale_group_size=0)
    expected_a = dtypes.bfloat16 if a_name is None else dtypes.DataType.from_str(a_name)

    tuning_path = os.environ.get("ARC_INT3_TUNING")
    if tuning_path:
        install_tuning(json.loads(Path(tuning_path).read_text()), humming_version, act)
    reuse = os.environ.get("ARC_INT3_RUNNER_REUSE", "1")
    if reuse not in ("0", "1"):
        raise ValueError(f"ARC_INT3_RUNNER_REUSE must be 0 or 1, got {reuse!r}")
    if reuse == "1":
        install_runner_reuse()
    print(f"ARC_INT3_RUNNER_REUSE={reuse}", flush=True)
    private = os.environ.get("ARC_HOST_POOL_PRIVATE", "0")
    if private not in ("0", "1"):
        raise ValueError(f"ARC_HOST_POOL_PRIVATE must be 0 or 1, got {private!r}")
    if private == "1":
        install_private_host_pools()
    print(f"ARC_HOST_POOL_PRIVATE={private}", flush=True)
    patch_dir = os.environ.get("ARC_INT3_WEIGHT_PATCH") or None
    if patch_dir is not None and not (Path(patch_dir) / "patch.json").is_file():
        raise FileNotFoundError(f"ARC_INT3_WEIGHT_PATCH={patch_dir} has no patch.json")
    print(f"ARC_INT3_WEIGHT_PATCH={patch_dir}", flush=True)

    class Int3MoEMethod(HummingMoEMethod):
        _arc_int3_direct = True

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            for sublayer in ("w13", "w2"):
                got = layer.humming_metas[sublayer].a_dtype
                if got != expected_a:
                    raise RuntimeError(f"ARC_INT3_ACT={act} but Humming {sublayer} runs a_dtype={got}")

    def from_config(cls, config):
        result = original_from(cls, config)
        if "arc_int3" in config:
            # Model-level geometry is checked by the launcher; each spawned worker
            # also validates the quantizer before allocating expert buffers.
            meta = config["arc_int3"]
            if (meta["bits"], meta["group_size"], meta["num_experts"], meta["scale_dtype"],
                meta["layout"], meta["w13_rows"], meta["schema_version"]) != (
                    3, 128, 320, "bfloat16", "humming_uint3_g128_nozp", "gate_then_up", 1):
                raise ValueError("unsupported packed INT3 contract")
            result._arc_int3 = meta
        return result

    def get_quant_method(self, layer, prefix):
        if not hasattr(self, "_arc_int3"):
            return original_method(self, layer, prefix)
        meta = self._arc_int3
        normalized = runtime_prefix(prefix)
        experts = {runtime_prefix(n) for n in meta["main_expert_modules"]}
        if normalized in experts:
            if not isinstance(layer, FusedMoE):
                raise TypeError(f"INT3 expert prefix is not FusedMoE: {prefix}")
            if layer.moe_ep_size != 1 or layer.moe_tp_size != 1 or layer.num_fused_shared_experts != 0:
                raise ValueError("packed e320 G4 loader requires TP1/EP1 and unfused shared experts")
            if layer._num_local_routed != 320 or layer.top_k != 10:
                raise ValueError("unexpected expert count/top-k")
            schema = HummingWeightSchema(b_dtype="uint3", weight_scale_group_size=128,
                                         has_zero_point=False)
            return Int3MoEMethod(HummingLayerQuantizationConfig(schema, force_input_schema=input_schema))
        fp8 = {runtime_prefix(n) for n in meta["dense_fp8_modules"]}
        bf16 = {runtime_prefix(n) for n in meta["dense_bf16_modules"]}
        if normalized in fp8 | bf16:
            if not isinstance(layer, LinearBase):
                raise TypeError(f"dense prefix is not LinearBase: {prefix}")
            if normalized in bf16:
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True,
                                            activation_scheme="dynamic", weight_block_size=[128, 128]))
        return original_method(self, layer, prefix)

    def load_weights(self, weights):
        modules = {name: module for name, module in self.named_modules()
                   if isinstance(module, FusedMoE) and getattr(module.quant_method, "_arc_int3_direct", False)}
        if not modules:
            return original_load(self, weights)
        if len(modules) != 48:
            raise ValueError(f"packed INT3 expects 48 expert banks, found {len(modules)}")
        loaded = set()
        passed = [0]  # tensors handed to the original loader, i.e. everything that is not a packed expert tensor
        required = {name + "." + suffix for name in modules for suffix in _SUFFIX.values()}
        patch = None
        if patch_dir is not None:
            if "_arc_int3_full_loaded" in self.__dict__:
                raise RuntimeError("ARC_INT3_WEIGHT_PATCH covers the first load only; the patch would overwrite "
                                   "a later weight sync")
            patch = WeightPatch(patch_dir)
            weights = patch.substitute(weights)

        def nonexpert_weights():
            for name, tensor in weights:
                match = _EXPERT.fullmatch(name)
                if match:
                    runtime_name = "model.layers." + match[1] + ".mlp.experts"
                    if runtime_name not in modules:
                        raise KeyError(f"packed expert has no instantiated module: {name}")
                    suffix = _SUFFIX[match[2]]
                    param = getattr(modules[runtime_name], suffix)
                    key = runtime_name + "." + suffix
                    if key in loaded or tensor.shape != param.shape or tensor.dtype != param.dtype:
                        raise ValueError(f"duplicate or incompatible packed tensor {name}: {tensor.dtype} {tuple(tensor.shape)}, expected {param.dtype} {tuple(param.shape)}")
                    with torch.no_grad():
                        param.copy_(tensor)
                    loaded.add(key)
                    continue
                if name.startswith("model.language_model.layers.") and ".mlp.experts." in name:
                    raise ValueError(f"old/unrecognized routed expert tensor in INT3 model: {name}")
                passed[0] += 1
                yield name, tensor

        result = original_load(self, nonexpert_weights())
        # The first load must carry every expert tensor. After that, an in-place weight update (RL
        # update_weights_from_disk) may carry only the dense projections, never a part of the experts.
        carries_experts = expert_coverage(loaded, required, "_arc_int3_full_loaded" in self.__dict__)
        if not isinstance(result, set):
            raise TypeError("Qwen4 load_weights no longer returns a loaded-parameter set")
        if not carries_experts:
            print(f"ARC_INT3_DELTA_LOADED tensors={passed[0]}", flush=True)
            return result
        if patch is not None:
            print(patch.finish(), flush=True)
        self.__dict__["_arc_int3_full_loaded"] = True
        print(f"ARC_INT3_DIRECT_LOADED layers=48 tensors={len(loaded)} humming={humming_version} act={act}", flush=True)
        return result | loaded

    ModelOptMixedPrecisionConfig.from_config = classmethod(from_config)
    ModelOptMixedPrecisionConfig.get_quant_method = get_quant_method
    Qwen4ExpForConditionalGeneration.load_weights = load_weights
    _INSTALLED = True


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "launch":
        args = sys.argv[2:]
        if "--model-path" not in args:
            raise SystemExit("launch requires --model-path")
        model = Path(args[args.index("--model-path") + 1])
        print(json.dumps(audit_model(model)), flush=True)
        # Require explicit Humming target backend; the DFlash dense worker remains
        # its own FP8 implementation and does not consume this MoE backend.
        flag = "--moe-runner-backend"
        if flag not in args or args[args.index(flag) + 1] != "humming":
            raise SystemExit("packed INT3 requires --moe-runner-backend humming")
        here = Path(__file__).resolve().parent
        pinned = os.environ.get("ARC_PINNED_HOST_REGISTER", "0")
        if pinned not in ("0", "1"):
            raise ValueError(f"ARC_PINNED_HOST_REGISTER must be 0 or 1, got {pinned!r}")
        if pinned == "1":
            if "PYTORCH_ALLOC_CONF" in os.environ:
                raise ValueError("PYTORCH_ALLOC_CONF is set; ARC_PINNED_HOST_REGISTER only edits PYTORCH_CUDA_ALLOC_CONF")
            # before anything imports torch; the scheduler process inherits it at spawn
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = pinned_host_register_conf(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        print(f"ARC_PINNED_HOST_REGISTER={pinned} PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
        os.environ["ARC_INT3_DEPLOY"] = "1"
        os.environ["PYTHONPATH"] = os.pathsep.join([str(here / "int3_deploy_patch"), str(here), os.environ.get("PYTHONPATH", "")])
        install()
        sys.argv = ["sglang.launch_server", *args]
        runpy.run_module("sglang.launch_server", run_name="__main__")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--model", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(audit_model(options.model), indent=2))


if __name__ == "__main__":
    main()
