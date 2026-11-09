"""The marker gate for the sglang 0.5.20 patch stack, as data.

A patch stack that is silently one patch short is the failure this guards: the
server starts, serves, and is wrong in a way no log line mentions. Every entry
is a string that must appear in a file of the installed source -- either because
one of patches/sglang-0.5.20/*.patch put it there, or, for the four patches
upstream absorbed, because upstream's own version of the fix has to be present.

The same dict is embedded in work/qwen38fn_eval/build_sglang_0520.sh, which is
the B200 line's build gate. This module is a second reader of the same list, not
a second list: test_sglang_0520_markers.py parses the dict back out of that
script and fails if the two have drifted. It lives here rather than there so
that the Kaggle packager can import it without sourcing a build script.

Keys are paths relative to the sglang checkout root. Merge markers for one file
into one tuple -- a repeated dict key silently discards the earlier entry.
"""
from __future__ import annotations

from pathlib import Path

MARKERS: dict[str, tuple[str, ...]] = {
    "python/sglang/srt/layers/attention/qsa/mqa.py": (
        "groups_per_cta = 1 if rows <= 8 else 2 if rows <= 24 else 4",   # 0001 (was 0006)
        "_grouped_qsa_effective_min_rows",                               # 0002 (was 0007)
        "_tilelang_qsa_mqa_grouped_verify_kernel",
        "_qsa_verify_group_size",                                        # 0025
        "if use_grouped_verify and (grouped_size * query_heads) % 8 == 0:",  # 0026
        "ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM",                             # 0027
        "_GROUPED_QSA_MIN_ROWS_SCALE = {2: 2.0, 4: 1.0, 8: 2.0}"),       # 0028
    "python/sglang/srt/layers/attention/qsa/qsa_indexer.py": ("_qsa_grouped_queries_per_request",),
    "python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py": (
        "_fused_mamba_state_scatter_with_mask_strided_kernel",),         # 0003 (was 0008)
    "python/sglang/kernels/ops/speculative/topk1.py": ("draft_topk1_postprocess_distributed",),  # 0004
    "python/sglang/kernels/ops/elementwise/fast_topk.py": ("fast_topk_zero_starts",),            # 0005
    # 0005 passes row_starts=nullptr; v0.5.20 deleted the kernel's null check and
    # the patch puts it back. Without it every decode step is an illegal address.
    "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh": ("params.row_starts == nullptr",),
    "python/sglang/srt/models/qwen4_exp.py": ("_qwen4_mnnvl_hc_precompute_gate",),               # 0006
    "python/sglang/srt/layers/flashinfer_comm_fusion.py": ("can_use_qwen4_mnnvl_hc",),
    "python/sglang/kernels/ops/communication/qwen4_mnnvl_hc.py": ("compile_qwen4_mnnvl_hc",),
    "python/sglang/kernels/ops/elementwise/hc_gate.py": ("compile_hc_gate",),
    "python/sglang/srt/speculative/eagle_worker_v2.py": ("_bind_native_qwen_mtp_before_pool",),  # 0007
    # --- 0008-0014: the NVFP4 target-KV arm. Dormant unless ARC_QSA_NVFP4 is set,
    #     but the markers still have to be there: this is the arm Kaggle serves.
    "python/sglang/srt/layers/attention/flashinfer_backend.py": (
        "ARC_FULL_NVFP4_PREFILL=fa2_packed", "arc_nvfp4_paged_forward"),                  # 0008/0010
    "python/sglang/srt/model_executor/pool_configurator.py": (
        "Native packed prefill allocates no pool-sized dequant workspace",),              # 0009
    "python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py": (
        "_arc_native_fp4_prefill",),                                                      # 0011
    "python/sglang/srt/layers/attention/qsa/sparse_attn.py": (
        "qwen_sparse_nvfp4_expand_triton",                                                # 0012
        "DST_V_GROUP",                                                                    # 0014
        "_compact_kv_nvfp4_four_plane",                                                   # 0024
        # upstream absorbed the old 0004/0015: assert upstream's behaviour is present
        "ZERO_FILL", "keys = keys.to(q_values.dtype)", "slots.to(tl.int64)"),
    # 0013: the field moved to arg_groups/fields/spec.py in v0.5.20
    "python/sglang/srt/arg_groups/fields/spec.py": ('"bfloat16", "nvfp4"',),
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py": (
        "_trtllm_sparse_nvfp4_sm100",                                                     # 0014
        "qwen_sparse_nvfp4_gather4_triton",                                               # 0024
        "zero_fill_cols=stride"),                                                         # upstream
    "python/sglang/srt/speculative/spec_utils.py": (
        "_commit_ple_side_states_after_replayssm(",),                                     # 0016
    "python/sglang/srt/entrypoints/openai/serving_responses.py": (
        "ARC_0017_MERGE_ASSISTANT_TURN",                                                  # 0017
        "ARC_0018_TOOL_OUTPUT_MEDIA"),                                                    # 0018
    # --- upstream absorbed the old 0013/0016 ---
    "python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py": (
        "out.hidden_states[:raw_bs].clone()",),
    "python/sglang/srt/models/qwen4_exp_mtp.py": (
        "hc_hidden_states = hc_hidden_states[select_index]",),
}
# SM120 only. The decode-kernel knob raises on a non-SM120 card if it is set, so
# the B200 build gate does not carry it and this is checked separately.
SM120_MARKERS: dict[str, tuple[str, ...]] = {
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py": (
        "ARC_QSA_SM120_DECODE",),                                                         # 0015
}


def missing(root: Path, markers: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Every marker that is not where it should be, named so the patch is obvious."""
    absent = []
    for relative, needles in (markers or MARKERS).items():
        path = root / relative
        try:
            text = path.read_text()
        except OSError as error:
            absent.append(f"{relative}: unreadable ({type(error).__name__}: {error})")
            continue
        absent += [f"{relative}: missing {needle!r}" for needle in needles if needle not in text]
    return absent
