# sglang-0.5.20: Qwen3.8-Flash-Next main line (B200)

> **Frozen since 2026-10-04: sglang changes now live in the private fork [`ycchen-tw/sglang-arc-2`](https://github.com/ycchen-tw/sglang-arc-2)** (one change per commit).
> The deployment target is set by [`deploy/sglang.lock.json`](../../deploy/sglang.lock.json); tools are in [`work/sglang_fork/`](../../work/sglang_fork/).
> Every sglang patch here has a corresponding branch in the fork (`deploy/*`, `exp/*`, `legacy/*`); only the old QSA offload line
> (0.5.20 `experimental-qsa`, pr36497 0044–0059) was deliberately not carried over. **Do not add patches to `patches/sglang-*` anymore.**

Base tag `v0.5.20` (`94602c9`, see `BASE`). Upstream already supports `qwen4_exp`/QSA natively, so this directory holds only
the remainder, currently 32 patches: 0001–0007 are the FP8 production path, 0008–0014 the NVFP4 target-KV arm (for comparison experiments), 0015 an SM120 decode-kernel knob, 0016 the commit of PLE side state under ReplaySSM (conditional correctness; required whenever `--enable-linear-replayssm-spec` is on), 0017 makes the Responses API merge one turn's assistant items into one block (required for codex / arc-code), 0018 makes image/video returned by tools actually reach the model (required for `view_image` / `view_video`), 0019–0023 the QSA + NVFP4 path for sm_89 (Ada / L4), 0024–0028 QSA kernel speedups for spec verification (measured on B200, all five bitwise identical; 0027 off by default), 0029 fixes scale passing in the NVFP4 KV decode graph, 0030–0031 small speedups for the sm_120 c55 spec step, 0032 opt-in load-time block-FP8 (see the end).
Applying and the marker gate: `work/qwen38fn_eval/build_sglang_0520.sh`;
server start-up: `work/qwen38fn_eval/serve_qwen38fn_0520.sh` (FP8, port 30032) / `serve_qwen38fn_nvfp4_0520.sh` (NVFP4 KV, port 30033);
SM120 (RTX PRO 6000 / Colab G4): `build_sglang_0520_sm120.sh` + `serve_sm120_0520.sh`, see `docs/research/sglang_0520_sm120_port.md`.
After changing any patch, regenerate `TREE_DIGEST` (`work/qsa_hisparse/g4_runtime.py tree-write`); the G4 runtime assertion uses it to check the whole sglang package.

Forward-ported from `sglang-qwen4-pr36497/` (PR #36497 @ `73a2552`, frozen);
upgrade analysis and per-patch verdicts in `docs/research/sglang_0520_upgrade.md`.

| patch | original no. | purpose | default | changes made in the forward port |
|---|---|---|---|---|
| 0001 qsa-mqa-auto-gpc | 0006 | QSA index kernel picks groups/CTA automatically from rows | `SGLANG_QSA_MQA_GROUPS_PER_CTA=0` (=auto) | none |
| 0002 qsa-grouped-verify | 0007 | target-verify shares a K tile across 4 rows/request | **ON** (`SGLANG_QSA_GROUPED_VERIFY=1`) | none; upstream made `num_tokens_per_req` resolvable per phase, **not verified at runtime** |
| 0003 gdn-track-strided | 0008 | strided track kernel for the Mamba state scatter + tests | always-on | none |
| 0004 qwen4-mtp-distributed-topk1 | 0009 | greedy MTP all-gathers only each rank's (max, id) | **ON** | adopted upstream's `[:raw_bs]` + `.clone()` structure, fields added back |
| 0005 qsa-trim-launches | 0011 | fewer QSA decode launches; fast_topk zero-starts | always-on | **restored the `row_starts == nullptr` guard** — upstream deleted it, and applying as-is would null-deref |
| 0006 qwen4-mnnvl-hc | 0012 | custom kernel for TP4/B200 attention all-reduce + HC epilogue | **ON** (`SGLANG_QWEN4_MNNVL_HC=1`) | imports merged; flashinfer 0.6.17→0.6.18, **internal structs not statically verified** |
| 0007 qwen-mtp-early-head-alias | 0014 | native MTP's draft KV counted in the cell size; early alias | always-on | upstream rewrote `_resolve_eagle_aux_hidden_state`; added a `return` that keeps the target's SWA layer count out of the draft field; tests switched to `get_spec()` |

| 0008 full-nvfp4-packed-prefill | 0023 | FA2 prefill consumes packed FP4 + `kv_cache_sf` directly; the pool budget no longer allocates a dequant workspace | `ARC_FULL_NVFP4=0` (the QSA arm doesn't use it) | restored `import os` in three files; `kv_cache_configurator` follows upstream in using `hf_config`. **On SM100 this prefill scale path has no gate at all** (see the upgrade doc) |
| 0009 full-nvfp4-budget | 0024 | budget fix for the same | same | none |
| 0010 full-nvfp4-forward-adapter | 0025 | NVFP4 adapter for flashinfer paged forward | same | none |
| 0011 qsa-nvfp4-kv | 0029 | under QSA the target KV is stored as packed FP4; decode goes through trtllm-gen | on only with `ARC_QSA_NVFP4=1` | removed the descale/direct-decode dependency on 0004/0005; gather switched to upstream's `zero_fill_cols=`; the Hybrid default `k_scale=1.0` overriding the GPU scale is fixed by 0029 |
| 0012 qsa-nvfp4-fused-expand | 0030 | the FP4→FP8 expansion of prefix chunks fused into one Triton kernel | `ARC_QSA_NVFP4_PREFILL=fp8` | removed `scale_kwargs` (the upstream chunk kernel has no such parameter and casts unconditionally anyway) |
| 0013 speculative-draft-kv-nvfp4 | 0034 | `--speculative-draft-kv-cache-dtype` accepts `nvfp4` | — | **file changed to `arg_groups/fields/spec.py`** (upstream moved it; still no nvfp4) |
| 0014 qsa-nvfp4-sm100 | 0035 | FP4 KV for B200 trtllm-gen: HND, E4m3 signature, value scale grouped by 4 tokens | with 0011 | re-stitched the dst formula onto upstream's `_compact_kv` skeleton (`ZERO_FILL`/int64 offset kept). The `DST_V_GROUP` layout was checked byte by byte with `test_dst_v_group_layout.py` (B200 ALL OK) |
| 0015 qsa-sm120-decode-knob | 0022a | A/B knob for the QSA decode kernel on SM120 `ARC_QSA_SM120_DECODE=xqa\|fa2` | `xqa` (= upstream behavior); setting it on a non-SM120 machine raises | shrunk to 14 lines: upstream already defaults to trtllm-gen/XQA on SM120, so only the `fa2` arm is kept. **`fa2` needs FA2's sm_120 wheel** — without FA2 the 0.5.20 fallback drops to FA4 cute, whose MLIR graph build always fails on sm_120 (measured on G4); the venv only has flash-attn-4 |
| 0016 ple-side-state-commit-after-replayssm | 0018 | GDN + `--enable-linear-replayssm-spec`: after verify, commit the PLE short-conv / n-gram intermediate back to the main slot | always-on (takes effect only in the ReplaySSM ring branch) | upstream made `replayssm_spec_fold` KDA-only; GDN goes through the ring branch in `spec_utils.py` and ends before `return`, skipping `update_mamba_state_after_mtp_verify` (upstream's own `_update_ple_state_after_mtp_verify` is only called there). Re-cut to call that upstream function instead of building the pairs itself; the PLE pool exists only in the static pool (unified raises outright), the indices are already physical, and a virtual-id pool raises. **No gate means no fix**: `serve_qwen38fn_0520.sh` PYCHK16 / both SM120 scripts / `build_sglang_0520.sh` markers all assert it. Upstream PR candidate (hits any model with PLE side state + this flag). Regression test `work/qwen38fn_eval/test_ple_side_state_commit_0016.py` (AST structure + a state-level check with a real pool and real kernel; fails before the fix); **an end-to-end greedy fingerprint cannot see this defect**: rerunning at temperature 0 already diverges by 0.05–0.08 |
| 0017 responses-merge-assistant-turn | — (new) | in the Responses history, the reasoning / message / function_call items of one assistant turn are merged into one chat message, **not split by `phase`** | always-on | upstream 0.5.20 added "merge only when `phase` is equal" to `_merge_consecutive_assistant_messages`; the items codex sends back have different phases (reasoning none, message=commentary, function_call), so one turn was rendered as three assistant blocks, two of them empty `<think>\n\n</think>\n\n` followed by `<\|im_end\|>`, and Qwen3.8 imitates that shape: after ~100 reasoning tokens it emits `\n\n` and ends the turn without a tool call. arc-code, 25 games: 8 judged as giving up within 90 seconds; 6-game comparison 4.56 vs 12.24 on the old line; sessions 25 vs 6. `responses_load_probe.py`, 16 parallel × 4 turns: 0.5.20 4–10/60, old line 0/64, reasoning item stripped 0–1/64. **Must run before deploying**: `responses_load_probe.py PORT MODEL --conc 16 --turns 4 --filler-kb 100`, threshold ≤1/64 (needs a live server, so it can't be a start-up gate). First upstream PR candidate |
| 0018 responses-tool-output-media | — | when a Responses `function_call_output` carries image/video, the media stay in their original order in the `tool` message's own content list (the Qwen template renders them inside `<tool_response>`); a turn with media gets no implicit output budget, and the scheduler clamps it by the expanded length | always-on | **upstream lacks this feature** (`main` is equally broken). `_flatten_tool_output` takes only `text`, so media returned by tools are silently dropped here — the only symptom is that `prompt_tokens` excludes the attachments and the model guesses from scratch, while the tool call itself "succeeds". **This is why `view_image` was always broken**, not just video. The first version split the media into an immediately following `user` message: the model saw two consecutive user turns, results of parallel calls got separated, and the interleaved image/text order was lost; on 2026-09-24 it went back to keeping them in the tool message (the chat side's `normalize_tool_content` already keeps tool lists that carry media). An unrecognized part type now returns 400 instead of being silently dropped. Also fixes a second defect: `default_max_tokens` was computed from the placeholder length before expansion, so once media entered the prompt the allowance was over-issued; the first version's `context_len // 2` only works while the input is under half the window — Kaggle v5 `dc22x5` was rejected viewing an image at 76,811 input (106,749 > 106,496), codex retried as-is 5 times, and then that game ended. Regression: `work/qwen38fn_eval/test_responses_tool_media_0018.py`; A/B with PC Qwen3.5-4B hitting `/v1/responses` directly: see journal 2026-09-24 |

## Absorbed upstream, no longer carried locally (the build/serve gates now assert the upstream behavior exists)

| original patch | upstream equivalent | asserted marker |
|---|---|---|
| 0004 fp8kv-qsa-upcast | `qsa/sparse_attn.py` chunk-prefill `keys.to(q.dtype)`, pack kernel dequant-on-gather, **also fixes an int64 offset overflow** | `keys = keys.to(q_values.dtype)`, `slots.to(tl.int64)` |
| 0015 qsa-fp8-trt-page-tail | `ZERO_FILL` + page-aligned store mask (clears the whole row stride, a superset of `ZERO_PAGE_TAIL`); comes with `test_qsa_strided_zero_fill.py` | `ZERO_FILL`, backend `zero_fill_cols=stride` |
| 0013 eagle-draft-extend-selected-rows | upstream PR #35546 merged verbatim, no env gate; **adds a `.clone()`** | `out.hidden_states[:raw_bs].clone()` |
| 0016 eagle-draft-extend-hidden-anchor | root cause fixed in the `qwen4_exp_mtp.py` model wrapper | `hc_hidden_states = hc_hidden_states[select_index]` |

Upstream's TRT paged-decode scratch therefore went from FP8 to BF16 (1.03→2.06 MiB per row, ~256 rows ≈ 0.5 GB more);
the +3% throughput in `qwen38fn_fp8kv_page_tail_fix.md` was measured with the FP8 scratch and **must be re-measured**.

## Not carried (pull them from the frozen directory if needed; read the upgrade doc's verdicts first)

Research instruments 0001/0003/0021/0002/0022b/0028, experiment switches 0005/0010/0017, conditional correctness 0019/0020 (0018 was re-cut as 0016),
XQA chain-verify 0026/0027, 0031 (debug), 0032 (upstream already dispatches `FP8_BLOCK_SCALES` natively),
0033 (needed by the NVIDIA NVFP4 **weight** checkpoint arm; not fixed upstream), the six mtp3 patches.

## Status

- 0029 is equivalent to frozen-line 0043; checked with zero fuzz against the 0.5.20 source tree, **not yet verified on a 0.5.20 GPU runtime**. For the old runtime's 12L/TP2 graph capture and batch-2 verification, see the frozen line's README.
- 2026-09-19: forward port done, `py_compile` and all markers pass; verified at runtime on B200 GPUs 4–7 (`fp8kv_repro.py` 9 cells, 0; pool +2.3%;
  MNNVL probe bitwise; MTP accept 2.93–2.99; see upgrade doc §5.0).
- 2026-09-19: SM120 (Colab G4) verified 0001–0015: fp8 KV and nvfp4 KV arms smoke / fp8kv 9 cells, 0 / MTP accept 2.66–2.77;
  nvfp4 pool 3,913,984 vs 3,809,600 on the old line with the same flags. **Server start-up must set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**,
  otherwise the pool is 19% smaller (allocator fragmentation is counted as used by `mem_get_info`). See `docs/research/sglang_0520_sm120_port.md`.

## QSA speedups for spec verification: 0024–0028 (2026-09-22, measured on B200)

Measurements, evidence and rejected directions are in
`docs/research/qwen38fn_qsa_verify_kernels_20260922.md`.
All five **change only speed, not output**; the regression tests compare bit for bit, so there is no numeric arm for A/B.

| patch | purpose | default |
|---|---|---|
| 0024 qsa-nvfp4-fused-gather | the NVFP4 selected-KV gather goes from two same-shape launches (one for the payload, one for the block scale) to one four-plane launch; the head axis is folded into the feature axis | `ARC_QSA_NVFP4_GATHER=fused` (`split` is the kill switch) |
| 0025 qsa-verify-subgroup-indexer | when the verify window width has no table entry (qpr=8 for 8-step MTP), use the widest supported group that divides it (4) instead of falling all the way back to the per-row kernel | always-on (with `SGLANG_QSA_GROUPED_VERIFY`) |
| 0026 qsa-indexer-skip-head-pad | the grouped indexer no longer zero-pads the 4 index heads to 8 — the grouped MMA N is `qpr × heads`, already a multiple of 8 | always-on (only on the grouped path) |
| 0027 qsa-indexer-occupancy-knob | second argument of the grouped indexer's `__launch_bounds__`: trades registers for blocks/SM | **`ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM=0` (off)**; 10 is worth 1.09–1.12× at B≥80 |
| 0028 qsa-verify-group-eight | raises the group cap from 4 to 8 (by a row-count threshold), and writes down the real constraint on which widths can be grouped | always-on; width 8/16 at rows≥128 1.10–1.30× |

**0024**: `_forward_trtllm_sparse_nvfp4` used to call gather twice, both re-running the same addressing chain
logical position → request table → slot; the scale pass moves only 1/8 of the bytes
but takes as long as the payload because its feature width is only 16. B200, ctx 32,100, graph replay median:

| shape | two passes | fused | |
|---|---:|---:|---:|
| B=30 S=4 | 97.8 µs | 46.5 µs | 2.10× |
| B=110 S=8 | 646.6 µs | 273.8 µs | 2.36× |

The full QSA verify path including attention gains 1.53× (B=110 S=1) to 1.79× (B=110 S=8).
**This one already hits the memory roofline**: just writing the 1.07 GB scratch takes 157 µs (6.8 TB/s; B200 HBM is about 8 TB/s),
so there is no second 2× left in this kernel; going faster means not materializing it.
The `ARC_QSA_GATHER4_BLOCK_TOPK` / `_WARPS` defaults 64/8 come from a sweep.

**0025**: `_GROUPED_QSA_MIN_ROWS_SCALE` only has qpr 2 and 4. The 8-step MTP verify window is 8 rows;
with no entry it falls all the way back to the per-row kernel, so each of a request's 8 rows re-reads its whole compressed-K.
The grouped kernel's real contract is narrower than "the whole window" — it only requires "consecutive qpr rows of the same request, with increasing length",
and splitting an 8-row window into two groups of 4 satisfies that fully; the kernel itself was already parameterized by group size, **so this is a pure dispatch change**.
B=110, ctx 32,100: per-row 329 µs → groups of 4 233 µs. qpr=3 still returns None (the three-warp lowering for N=24
measured slower than the baseline, and 3 has no supported divisor greater than 1).

**0026**: `tilelang_qsa_mqa_decode` unconditionally zero-pads the 4 index heads to 8, on the grounds that
"MMA needs N to be a multiple of 8" — true for the **per-row** kernel (where N is `heads`),
but the grouped kernel's N is `qpr × heads`, which at the supported group sizes (qpr 2→8, qpr 4→16) is already
a multiple of 8. On the grouped path that pad therefore only buys "one `torch.cat` per call" and
"a GEMM with half of its columns computing zeros". Measured on B200 (logits bit-identical):

| shape | padded to 8 | unpadded | |
|---|---:|---:|---:|
| B=30, ctx 32,100 | 32.4 µs | 20.1 µs | 1.61× |
| B=110, ctx 32,100 | 132.1 µs | 85.2 µs | 1.55× |
| B=110, ctx 96,000 | 356.2 µs | 231.3 µs | 1.54× |

0025 + 0026 combined, against today's production path (**production page-table width**, see below):

| shape (B=110) | today | 0025+0026 | |
|---|---:|---:|---:|
| S=1, any ctx | — | — | 1.00× (untouched) |
| S=2 ctx 32,100 | 95.3 µs | 72.7 µs | 1.31× |
| S=4 ctx 8,100 | 52.4 µs | 36.0 µs | 1.45× |
| S=4 ctx 32,100 | 148.3 µs | 97.3 µs | 1.52× |
| S=4 ctx 96,000 | 367.4 µs | 234.8 µs | 1.57× |
| S=8 ctx 8,100 | 189.2 µs | 66.7 µs | **2.84×** |
| S=8 ctx 32,100 | 426.7 µs | 174.8 µs | **2.44×** |
| S=8 ctx 96,000 | 1018.2 µs | 424.8 µs | **2.40×** |

**Always measure this kernel at the production page-table width.** The page table the server hands out is pre-allocated for
the **configured** context (262,144 → 4096 compressed pages), and the grid is sized from it;
shrinking the table to just enough (as my first version did) leaves the kernel with almost no empty CTAs, while in production 88% are empty at ctx 32k
and 97% at ctx 8k. The first version therefore **both** overestimated the S=4 gain (reported 1.93×, actually 1.52×)
and led to a wrong decision — see below.

**0027, rejected by myself**: at the wrong (shrunk-to-fit) width, a sweep found "one group per CTA + 8 warps +
no Q loads for idle CTAs" worth 1.20×, and it was committed. Re-swept at production width, **it is a
regression at short context** (B=110 ctx 32,100: 0026's 97.2 µs became 120.0 µs), because gpc=1 makes a grid that is already 88%
empty another 4× larger. At production width the best configuration varies with context (8.1k wants gpc2, 32.1k gpc4,
96k gpc1) and is worth only 1.02–1.14×; there is no robust rule to write, so it was reverted entirely.

**0027 (a knob that is off by default, not a default value)**: `ncu` at B=110 / ctx 96,000 diagnoses
**not bandwidth saturation** (Max Bandwidth 43.1%, vs 69.9% for the gather on the same card), but
**71.6% of warp stalls waiting on the L1TEX scoreboard, with theoretical occupancy capped at 50% by registers**
(block limit registers 8, SM limit 32). `T.annotate_min_blocks_per_sm(10)`
pushes it to 62.5% (achieved 61.2%), with **zero spilling** and Max Bandwidth 47.6%.

| shape (B=110) | off | set to 10 | |
|---|---:|---:|---:|
| S=4 ctx 32,100 | 97.2 µs | 89.0 µs | 1.09× |
| S=4 ctx 96,000 | 234.7 µs | 210.0 µs | 1.12× |
| S=8 ctx 32,100 | 174.3 µs | 154.7 µs | 1.13× |
| S=8 ctx 96,000 | 424.9 µs | 378.0 µs | 1.12× |

**Why it isn't the default**: the gain only materializes with enough groups, and the crossover is **non-monotonic** — at ctx 32,100,
B=48 is 1.09×, B=64 drops to 0.97×, B=80 is 1.20× again, and B=8/16 is 0.90×. Writing a rule from three batch points is
exactly how the previous 0027 went wrong, so this time it is only a knob with the measurements attached, and the deploying side decides
from its actual batch. The differences among 10/12/16/20/24 are small (not knife-edge tuning).

**0028, and the end-to-end conclusion for 8-step MTP**: after actually starting an 8-step server and measuring
(it needs the experimental `0018-qsa-speculative-ring` to pass the `draft_tokens > compress_ratio`
gate; reverted bit for bit after measuring), it turned out that **0025/0026 have no effect at all on 8-step MTP (measured 1.00×)**.

The reason is my own mistake: **the verify window of k-step MTP is k+1 rows**. 8 steps is **9** rows, and
`_qsa_verify_group_size(9)` returns None — 9 is divisible by neither 2 nor 4. All my earlier microbenchmarks with S=8
actually correspond to **7** steps.

Dropping the comments and asking the kernel directly, the constraint is hard: the MMA lowering requires `N = group × heads`
to be divisible by 8 **and** `N/8` to divide the warp count, so group 3 (N=12 raw / 24 padded) and
group 9 (36 / 72) **don't even compile**. **group must be a power of 2 ⇒ odd widths cannot be grouped.**

| steps | window width | usable group | real server's indexer (B=32, clen≈11.3k, per call) |
|---:|---:|---|---:|
| 7 | 8 | **8** | **65.0 µs** |
| 8 | 9 | none (falls back to per-row) | 158.3 µs (156.0 after context correction) |

**Only 12.5% more verify rows, but the indexer costs 2.40×. When running MTP with QSA, choose an odd number of steps.**

0028 itself was found along this path: the group cap was stuck at 4, but MMA allows up to 8 (N=32, n_warp=4,
m_warp=1). Widths 8/16 now use group 8, and each request's compressed-K is read once instead of twice:

| shape (width 8) | group 4 | group 8 | |
|---|---:|---:|---:|
| B=30 ctx 32,100 | 40.0 µs | 34.1 µs | 1.17× |
| B=64 ctx 8,100 | 41.2 µs | 32.4 µs | 1.27× |
| B=110 ctx 8,100 | 66.8 µs | 51.2 µs | 1.30× |
| B=110 ctx 32,100 | 174.7 µs | 149.4 µs | 1.17× |

At small batch (rows<128) group 8 is 1–2 µs slower, so it uses the crossover-threshold mechanism the file already has
(`8: 2.0` added to `_GROUPED_QSA_MIN_ROWS_SCALE`) and falls back to group 4 when there are too few rows —
instead of inventing a curve of its own. Against the per-row kernel, width 8 is now 2.4–3.4×.

Regression tests (each needs one GPU):
`work/qwen38fn_eval/test_qsa_nvfp4_gather4_0024.py` (the four destination buffers compared bit for bit with the two-pass version,
for both the sm_100 paged scale layout and sm_120 token-major, tiles swept up to 128 — **only 128 crosses a destination page**;
at small tiles a page-index broadcast along the wrong axis still gives the right answer),
`work/qwen38fn_eval/test_qsa_verify_subgroup_0025.py` (shared by 0025 and 0026: group-size table, **asserts that the grouped path is actually taken** (the default min-rows=64 makes small shapes take the per-row path in both arms, so the comparison would be vacuous), **asserts that the grouped kernel is really built with heads=4** (0026 could silently fall back to "correct but still padding", which a logits comparison can't see), **asserts that every supported width (2/4/6/8/16) builds**, prints the 0027 flag state (**run the same test again with `ARC_QSA_INDEXER_MIN_BLOCKS_PER_SM=10`**; it is a compiler hint and must not change a single bit), and per-row vs grouped logits identical element by element).

## sm_89 / Ada line: 0019–0023 (migrated from the frozen line on 2026-09-21)

Kaggle L4×4 and the local 4090 are both sm_89. On this line, QSA decode had no usable kernel on non-SM100 machines
(trtllm-gen/XQA is SM100+, and the FA4 cute fallback doesn't build), and neither did NVFP4 KV quantize
(`nvfp4_kv_quantize` is SM100/120 PTX; flashinfer `fp4_quantize` returns
`Invalid backend: 89` on Ada). These five fill the gap.

| patch | purpose | default |
|---|---|---|
| 0019 qsa-direct-decode-sm89 | Triton direct decode (with SM89 tiles chosen by KV width, FP8 K/V descale) | only with `SGLANG_QSA_DIRECT_DECODE=1` |
| 0020 qsa-fp4-fused-decode-sm89 | QSA decode reads packed FP4 directly; prefill reuses this line's existing `qwen_sparse_nvfp4_expand_triton` | `ARC_QSA_FP4_FUSED=1` (on by default) |
| 0021 nvfp4-triton-quantize-sm89 | Triton NVFP4 quantizer + relaxed nvfp4 arch gate in `kv_cache_hook` | with 0019 |
| 0022 ple-host-and-loader-sm89 | the PLE loader frees its temporary map; the n-gram table is built directly on CPU (a transient of 11.9 GiB at TP4 / 23.8 GiB at TP2) | effective with `--ple-offload-embedding` |
| 0023 nvfp4-nibble-decode-int-shift | `tl.exp2` in the shared E2M1 decoder replaced by an integer shift | always-on |

**0023 is not only for sm_89**: that helper is also used by this line's 0012 prefill expand, so
FP4 prefill benefits too. Semantics unchanged (both forms are exact on the E2M1 ladder); measured on the decode kernel
212 → 158 µs.

### What this line had already solved, found during the migration

The frozen line had seven patches (`sglang-qwen4-pr36497/0036–0042`); five remain after the migration:

| frozen line | status on this line |
|---|---|
| 0037 SM89 tile, 0038 FP8 descale | merged into 0019 (three patches on the frozen line for historical reasons) |
| the **prefill** half of 0039 | this line's 0012 already has `qwen_sparse_nvfp4_expand_triton`, wired up; **not needed** |
| the decode half of 0039 | = 0020, using upstream's `_nvfp4_decode_nibble` instead of carrying a second copy |
| **0042 dequant workspace budget** | this line's 0009 already skips that allocation under `ARC_QSA_NVFP4=1`; **not migrated at all** — just set the flag |

Redo it with `work/qwen38fn_l4/port_to_0520.py`
(asserts every anchor). On this line the nvfp4 arch gate in `server_args.py` moved to
`arg_groups/kv_cache_hook.py`, and the architecture check changed from `is_sm100_supported()` to
`get_platform().is_sm100`.

### Verification status

- All five round-tripped against `0.5.20 + 0001–0017`: apply forward, apply in reverse back to pristine, apply forward again;
  six files bit-identical; all pass `ast.parse`.
- **Decode kernel verified on GPU**: kernels extracted from the ported tree pass the correctness tests, with relative error
  **0.00249–0.00336** against a float32 reference, bit-identical to the frozen-line version.
- ⚠️ **No end-to-end yet**: it needs a 0.5.20 sm_89 venv (there is none locally). 0020's quantizer and
  0021's PLE changes have only syntax- and anchor-level guarantees and have never run on a real load path.

## sm_120 c55 spec step: 0030–0031 (2026-09-29, measured on G4)

Measurements and A/B in `docs/research/qsa_hisparse_opt1_stage1_g4_20260929.md`.

| patch | purpose | default | numerics |
|---|---|---|---|
| 0030 gdn-replayssm-commit-live-rows | the ReplaySSM compact commit loads only the ring rows that have to be folded or tracked (≤ accept), instead of every program reading all 16 rows; the launcher splits out `launch_gdn_replayssm_compact_commit` | on (`ARC_GDN_COMMIT_LIVE_ROWS=0` is the kill switch) | bit-identical (the skipped rows had weight 0 anyway) |
| 0031 moe-triton-config-mtp-draft-rtx-pro-6000 | tuned config for the MTP draft layer's triton fused MoE (E=320, N=640, fp8_w8a8, [128,128]) on RTX PRO 6000; previously it logged `Using default MoE kernel config` | used whenever the file is present (pointing `SGLANG_MOE_CONFIG_DIR` elsewhere returns to the default) | touches only the draft; the output distribution is unchanged under rejection sampling |

## Load-time block-FP8: 0032 (2026-09-29, measured on G4)

Measurements and quality checks in `docs/research/qsa_hisparse_0032_online_block_fp8_g4_20260929.md`.

| patch | purpose | default | numerics |
|---|---|---|---|
| 0032 online-block-fp8-dense | after loading, converts the specified BF16 linear layers to 128×128 block FP8 and swaps in sglang's `Fp8LinearMethod` (sm_120 uses CUTLASS; no new kernel). `ARC_ONLINE_FP8` takes a subset of `gdn_in,gdn_out,qsa_qkv,qsa_o,lm_head` or `all`; TP≠1, overlap load, non-CUTLASS dispatch and names that match no layer are all rejected outright | unset = untouched (reverse-apply comparison byte-identical) | changes numerics: `lm_head` passes the noise ruler; the 4 large projections, −0.093 nats per token, fail |
