"""Startup checks for the accelerated training stack.

Everything here guards against a *silent* wrong-but-running configuration: the
run completes, the loss curve looks plausible, and the number you take away is
answering a different question than the one you asked. Each check therefore
raises rather than warns.

Three of them exist because we were bitten:

* ``assert_liger_moe_patch`` -- ``patches/liger-kernel-0.8.1`` widens liger's MoE
  autotune space to BLOCK_N=256 / BLOCK_K=64, which is where Blackwell tcgen05
  wants to sit (+3.8% end to end, ``docs/research/qwen35_training_speed_opt.md``
  §3), fixes nondeterministic dX, and prunes eight dX configs that cannot compile
  under the 255-register limit. uv installs by hardlink, so any reinstall reverts
  it with no trace -- exactly the failure mode ``patches/README.md`` warns about.

* ``assert_gdn_backend`` -- fla dispatches ``chunk_gated_delta_rule`` through a
  priority-ordered registry. FlashQLA sits at priority 3 and is selected only if
  the ``flash_qla`` package imports; otherwise the call falls through to the
  Triton implementation. Both are correct, but they are not the same speed and
  not the same numerics, so which one ran is part of a run's identity. The
  registry is silent about this by design (it logs at INFO once, to a logger
  nobody configures), so we ask it explicitly.

* ``assert_shards_match_tokenizer`` -- the SFT shards store token *ids*, not
  text. Pointing a trainer at a model whose vocabulary differs from the one the
  shards were rendered with produces a run that trains happily on garbage.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path


def describe_gdn_backend() -> tuple[str, list[str]]:
    """Return (active backend name, one line per registered backend).

    "active" is what the registry would pick for a call it can handle; a
    backend can still be rejected per-call by its verifier (dtype, head dim,
    unsupported kwargs), in which case dispatch continues down the list and
    ultimately reaches fla's own Triton implementation. ``triton`` is therefore
    the name for "no registered backend took the call".
    """
    from fla.ops.backends import BackendRegistry
    from fla.ops.gated_delta_rule.backends import gdr_registry

    BackendRegistry.ensure_initialized("gated_delta_rule")

    lines = []
    for be in gdr_registry._get_sorted_backends():
        lines.append(
            f"{be.backend_type} (priority {be.priority}, package {be.package_name}): "
            f"available={be.is_available()} enabled={be.is_enabled()}"
        )

    active = gdr_registry.get_active()
    return (active.backend_type if active is not None else "triton"), lines


def assert_gdn_backend(expected: str) -> str:
    """Fail unless the gated-delta-net backend fla will use is `expected`.

    `expected` is one of the registered backend_type strings (e.g. "flash_qla")
    or "triton" for fla's built-in implementation.
    """
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1" and expected != "triton":
        raise RuntimeError(
            f"--gdn-backend {expected} requested but FLA_DISABLE_BACKEND_DISPATCH=1 "
            "bypasses dispatch entirely, so the Triton path always runs."
        )

    active, lines = describe_gdn_backend()
    if active != expected:
        detail = "\n  ".join(lines)
        raise RuntimeError(
            f"--gdn-backend {expected} requested but fla would dispatch to {active!r}.\n"
            f"  {detail}\n"
            f"  Set --gdn-backend {active} to accept what is installed, or install/enable "
            f"{expected} (FlashQLA is gated on the flash_qla package and FLA_FLASH_QLA)."
        )
    if active == "flash_qla":
        assert_flash_qla_stack()
    return active


# The only apache-tvm-ffi version FlashQLA and FA4 can both live on. Upstream
# flash-qla hard-pins 0.1.9, whose intersection with flash-attn-4's
# >=0.1.12,<0.2 is empty; patches/flash-qla-0.1.2 re-pins it onto tilelang
# 0.1.13 (window >=0.1.11,<0.1.13), leaving exactly 0.1.12.
_FLASH_QLA_STACK = {"apache-tvm-ffi": "0.1.12", "tilelang": "0.1.13"}


def _call_on_t(statement: ast.stmt, name: str) -> ast.Call | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "T"
        and func.attr == name
    ):
        return call
    return None


def _statement_lists(tree: ast.AST):
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and all(isinstance(item, ast.stmt) for item in value):
                yield value


def _ordered_m_store_count(
    tree: ast.AST,
    *,
    fragment: str,
    shared: str,
    sync_start: int,
    destination: str,
) -> int:
    count = 0
    for statements in _statement_lists(tree):
        for offset in range(max(0, len(statements) - 3)):
            first, fence, sync, store = statements[offset : offset + 4]
            first_call = _call_on_t(first, "copy")
            fence_call = _call_on_t(fence, "fence_proxy_async")
            sync_call = _call_on_t(sync, "sync_threads")
            store_call = _call_on_t(store, "copy")
            if not all((first_call, fence_call, sync_call, store_call)):
                continue
            if [ast.unparse(arg) for arg in first_call.args] != [fragment, shared]:
                continue
            if fence_call.args:
                continue
            if [ast.literal_eval(arg) for arg in sync_call.args] != [sync_start, 128]:
                continue
            if [ast.unparse(arg) for arg in store_call.args] != [shared, destination]:
                continue
            count += 1
    return count


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"function {name!r} missing")


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise ValueError(f"{class_name}.{method_name} missing")


def _argument_defaults(function: ast.FunctionDef) -> dict[str, object]:
    args = function.args.args
    defaults = function.args.defaults
    names = [arg.arg for arg in args[-len(defaults) :]]
    return {name: ast.literal_eval(value) for name, value in zip(names, defaults)}


def _named_call(function: ast.FunctionDef, dotted_name: str) -> ast.Call:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted_name:
            return node
    raise ValueError(f"call {dotted_name!r} missing from {function.name}")


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise ValueError(f"keyword {name!r} missing from call")


def _calls_on_t(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "T"
        and node.func.attr == name
    ]


def _sm100_sync_errors(label: str, tree: ast.AST) -> list[str]:
    errors = []
    for call in _calls_on_t(tree, "sync_threads"):
        try:
            barrier_id, count = (ast.literal_eval(arg) for arg in call.args)
        except (TypeError, ValueError):
            errors.append(f"{label} has a non-literal T.sync_threads call")
            continue
        if not isinstance(barrier_id, int) or not 1 <= barrier_id <= 15:
            errors.append(f"{label} uses invalid PTX barrier id {barrier_id!r}")
        if not isinstance(count, int) or count <= 0 or count % 32:
            errors.append(f"{label} uses invalid barrier participant count {count!r}")
    return errors


def _has_true_pass_config(tree: ast.AST, key_name: str) -> bool:
    expected = f"tilelang.PassConfigKey.{key_name}"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if key is None:
                continue
            if ast.unparse(key) == expected:
                try:
                    return ast.literal_eval(value) is True
                except (TypeError, ValueError):
                    return False
    return False


def flash_qla_source_patch_errors(package_root: Path) -> list[str]:
    """Return missing live-source invariants without importing CUDA/TileLang."""
    errors = []
    prepare_path = package_root / "ops/gated_delta_rule/chunk/blackwell/prepare_h.py"
    cp_fwd_path = package_root / "ops/gated_delta_rule/chunk/blackwell/cp_fwd.py"
    fused_fwd_path = package_root / "ops/gated_delta_rule/chunk/blackwell/fused_fwd.py"
    chunk_path = package_root / "ops/gated_delta_rule/chunk/__init__.py"
    try:
        prepare_tree = ast.parse(prepare_path.read_text(encoding="utf-8"))
        cp_fwd_tree = ast.parse(cp_fwd_path.read_text(encoding="utf-8"))
        fused_fwd_tree = ast.parse(fused_fwd_path.read_text(encoding="utf-8"))
        chunk_tree = ast.parse(chunk_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return [f"cannot inspect patched FlashQLA source: {exc}"]

    stores = (
        ("right", "m_fragment_R", "m_shared_R", 14, "mt[bb, bh, 0:DK, DK // 2:]"),
        ("left", "m_fragment_L", "m_shared_L", 15, "mt[bb, bh, 0:DK, :DK // 2]"),
    )
    for label, fragment, shared, sync_start, destination in stores:
        count = _ordered_m_store_count(
            prepare_tree,
            fragment=fragment,
            shared=shared,
            sync_start=sync_start,
            destination=destination,
        )
        if count != 2:
            errors.append(
                f"SM100 final-M {label} store needs 2 ordered copy→fence→writer-sync→TMA paths, found {count}"
            )

    for label, tree in (("prepare_h", prepare_tree), ("fused_fwd", fused_fwd_tree)):
        errors.extend(_sm100_sync_errors(label, tree))
        if not _has_true_pass_config(tree, "TL_DISABLE_THREAD_STORAGE_SYNC"):
            errors.append(f"{label} must retain TL_DISABLE_THREAD_STORAGE_SYNC=True")

    try:
        correct_h0 = _function(cp_fwd_tree, "tilelang_correct_h0")
        h_copies = [
            call
            for call in _calls_on_t(correct_h0, "copy")
            if len(call.args) >= 2
            and ast.unparse(call.args[0]).startswith("ht_buffer[")
            and ast.unparse(call.args[1]) == "h_shared"
        ]
        if len(h_copies) != 2:
            errors.append(f"SM100 correction needs 2 ht_buffer→h_shared copies, found {len(h_copies)}")
        else:
            for call in h_copies:
                if ast.unparse(_keyword(call, "disable_tma")) != ("buffer_dtype == torch.float32"):
                    errors.append("SM100 FP32 correction h copy does not disable the racy TMA path")
                    break

        warmup = _function(cp_fwd_tree, "tilelang_get_warmup_chunks")
        initializes_fallback = any(
            isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "fallback_mask[bb, i_h]" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
            for node in ast.walk(warmup)
        )
        if not initializes_fallback:
            errors.append("SM100 warmup terminal fallback_mask is not initialized false")
    except (ValueError, TypeError) as exc:
        errors.append(f"SM100 AutoCP kernel hardening missing: {exc}")

    try:
        public = _function(chunk_tree, "chunk_gated_delta_rule")
        forward = _method(chunk_tree, "ChunkGatedDeltaRuleFunction", "forward")
        backward = _method(chunk_tree, "ChunkGatedDeltaRuleFunction", "backward")
        public_defaults = _argument_defaults(public)
        forward_defaults = _argument_defaults(forward)
        public_expected = {
            "auto_cp": True,
            "enable_fwd_cp_cache": True,
            "auto_cp_backward": None,
        }
        forward_expected = {**public_expected, "auto_cp_backward": False}
        if {name: public_defaults.get(name) for name in public_expected} != public_expected:
            errors.append("public AutoCP defaults are not forward=True/env-controlled backward")
        if {name: forward_defaults.get(name) for name in forward_expected} != forward_expected:
            errors.append("autograd AutoCP defaults are not forward=True/backward=False")

        env_default = [
            node
            for node in chunk_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_AUTO_CP_BACKWARD_DEFAULT" for target in node.targets
            )
        ]
        if len(env_default) != 1 or ast.unparse(env_default[0].value) != (
            "os.environ.get('FLASH_QLA_AUTO_CP_BACKWARD', '0') == '1'"
        ):
            errors.append("public backward AutoCP environment default is not safe/off")
        env_fallback = any(
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "auto_cp_backward is None"
            and any(
                isinstance(item, ast.Assign) and ast.unparse(item.value) == "_AUTO_CP_BACKWARD_DEFAULT"
                for item in node.body
            )
            for node in ast.walk(public)
        )
        if not env_fallback:
            errors.append("public wrapper does not resolve the backward AutoCP env toggle")

        apply_call = _named_call(public, "ChunkGatedDeltaRuleFunction.apply")
        if [ast.unparse(arg) for arg in apply_call.args[-3:]] != [
            "auto_cp",
            "enable_fwd_cp_cache",
            "auto_cp_backward",
        ]:
            errors.append("public wrapper does not pass auto_cp_backward to autograd")

        fwd_call = _named_call(forward, "chunk_gated_delta_rule_fwd")
        if ast.unparse(_keyword(fwd_call, "auto_cp")) != "auto_cp":
            errors.append("forward AutoCP is not controlled by auto_cp")
        cache_expr = ast.unparse(_keyword(fwd_call, "enable_fwd_cp_cache"))
        if cache_expr != "auto_cp and auto_cp_backward and enable_fwd_cp_cache":
            errors.append("forward CP cache is not gated by backward AutoCP")

        assignments = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "ctx.auto_cp_backward" for target in node.targets)
        ]
        effective_bwd_cp = "auto_cp and auto_cp_backward"
        if len(assignments) != 1 or ast.unparse(assignments[0].value) != effective_bwd_cp:
            errors.append("autograd context does not gate auto_cp_backward with master auto_cp")

        bwd_call = _named_call(backward, "chunk_gated_delta_rule_bwd")
        if ast.unparse(_keyword(bwd_call, "auto_cp")) != "ctx.auto_cp_backward":
            errors.append("low-level backward does not consume auto_cp_backward")
        if ast.unparse(_keyword(bwd_call, "cp_cache")) != "cp_cache":
            errors.append("low-level backward does not receive the gated CP cache")

        returns = [node for node in ast.walk(backward) if isinstance(node, ast.Return)]
        input_count = len(forward.args.args) - 1
        if (
            len(returns) != 1
            or not isinstance(returns[0].value, ast.Tuple)
            or len(returns[0].value.elts) != input_count
        ):
            errors.append("autograd backward gradient count does not match forward inputs")
    except (ValueError, TypeError) as exc:
        errors.append(f"safe backward AutoCP API patch missing: {exc}")
    return errors


def _flash_qla_package_root() -> Path:
    spec = importlib.util.find_spec("flash_qla")
    if spec is None:
        raise RuntimeError("FlashQLA is active but its package source cannot be located")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    if spec.origin:
        return Path(spec.origin).parent
    raise RuntimeError("FlashQLA is active but its package source has no filesystem path")


def assert_flash_qla_stack() -> None:
    """Fail unless the tvm-ffi / tilelang pair FlashQLA and FA4 both tolerate is installed.

    Installing upstream `flash-qla` from PyPI drags apache-tvm-ffi down to
    0.1.9, and the resulting failure is both remote from its cause and easy to
    misread: FA4 imports fine, then the first kernel call dies inside
    nvidia-cutlass-dsl with `make_kwargs_wrapper() got an unexpected keyword
    argument 'map_dataclass_to_tuple'` (tvm_ffi_provider.py:642 -- the
    parameter arrived in tvm-ffi 0.1.10). Name the cause here instead.
    """
    import importlib.metadata as md

    wrong = []
    for pkg, want in _FLASH_QLA_STACK.items():
        try:
            got = md.version(pkg)
        except md.PackageNotFoundError:
            got = "<not installed>"
        if got != want:
            wrong.append(f"{pkg}: want {want}, have {got}")
    if wrong:
        raise RuntimeError(
            "FlashQLA is active but its validated stack is not installed: " + "; ".join(wrong) + ".\n"
            "  FA4 needs apache-tvm-ffi>=0.1.12; upstream flash-qla pins ==0.1.9. "
            "Reinstall the re-pinned fork:\n"
            "    uv pip install 'apache-tvm-ffi==0.1.12' 'tilelang==0.1.13'\n"
            "    uv pip install --no-deps -e reference/FlashQLA   "
            "(see patches/flash-qla-0.1.2/)"
        )

    source_errors = flash_qla_source_patch_errors(_flash_qla_package_root())
    if source_errors:
        raise RuntimeError(
            "FlashQLA is active but its validated SM100 source fixes are not live: " + "; ".join(source_errors) + ".\n"
            "  Rebuild the c18a486 fork with patches/flash-qla-0.1.2/0001 and 0002, then:\n"
            "    uv pip install --no-deps -e reference/FlashQLA"
        )


def assert_liger_moe_patch() -> None:
    """Fail unless patches/liger-kernel-0.8.1 is live in this interpreter.

    Checks the live generic/dW/dX autotuner effects rather than patch files, so a
    reinstall that silently restores the stock kernel is caught even if every
    patch file is still sitting in patches/.
    """
    from liger_kernel.ops.fused_moe_kernels import (
        _AUTOTUNE_DISABLED,
        _get_dW_autotune_configs,
        _get_dx_autotune_configs,
        _get_gemm_autotune_configs,
        _moe_bwd_dX_expanded_kernel,
    )

    if _AUTOTUNE_DISABLED:
        raise RuntimeError(
            "liger fused-MoE autotune is disabled (LIGER_FUSED_MOE_AUTOTUNE=0). "
            "The BN=256 patch only takes effect through autotune; unset it."
        )

    gemm_bn = {c.kwargs["BLOCK_N"] for c in _get_gemm_autotune_configs()}
    dw_bn = {c.kwargs["BLOCK_N"] for c in _get_dW_autotune_configs()}
    dw_bk = {c.kwargs["BLOCK_K"] for c in _get_dW_autotune_configs()}
    dx_configs = _moe_bwd_dX_expanded_kernel.configs
    dx_helper_configs = _get_dx_autotune_configs()

    missing = []
    if 256 not in gemm_bn:
        missing.append(f"GEMM BLOCK_N=256 (have {sorted(gemm_bn)})")
    if 256 not in dw_bn:
        missing.append(f"dW BLOCK_N=256 (have {sorted(dw_bn)})")
    if 64 not in dw_bk:
        missing.append(f"dW BLOCK_K=64 (have {sorted(dw_bk)})")
    if any(c.kwargs["BLOCK_N"] == 256 and c.num_warps == 4 for c in dx_configs):
        missing.append("dX still contains uncompilable BLOCK_N=256/num_warps=4")
    if not any(c.kwargs["BLOCK_N"] == 256 and c.num_warps == 8 for c in dx_configs):
        missing.append("dX lost the valid BLOCK_N=256/num_warps=8 candidates")

    def config_key(config):
        return (
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.num_stages,
        )

    if {config_key(c) for c in dx_configs} != {config_key(c) for c in dx_helper_configs}:
        missing.append("dX autotuner does not use _get_dx_autotune_configs()")
    if missing:
        raise RuntimeError(
            "liger MoE autotune patch is NOT applied: " + "; ".join(missing) + ".\n"
            "  Run: bash tools/apply_patches.sh liger-kernel-0.8.1\n"
            "  (uv installs by hardlink, so a reinstall or `uv cache prune` reverts it "
            "silently -- see patches/README.md.)"
        )


def assert_shards_match_tokenizer(model_path: str, max_id: int) -> None:
    """Fail unless `model_path`'s vocabulary can represent the shards' token ids.

    `max_id` is the largest id observed in the shards. Cheap and one-sided: it
    catches a shrunk or reordered vocabulary, which is the way this goes wrong
    when shards rendered for one model get pointed at another.
    """
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    vocab = tc["vocab_size"]
    if max_id >= vocab:
        raise RuntimeError(
            f"shards contain token id {max_id} but {model_path} has vocab_size={vocab}. "
            "These shards were rendered for a different tokenizer -- re-render with "
            "render_qwen.py against this model before training."
        )
