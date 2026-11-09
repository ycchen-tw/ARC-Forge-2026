"""Build a private, offline ARC2 L4x4 h28 notebook locally; never upload or submit.

The mounted source asset and runtime_setup.install contract come from the supplied
manifest. Capacity settings are recorded unchanged; real cgroup admission may fail.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import textwrap

MODEL = "threerabbits/qwen38fn-nvfp4-reap320-arccode/Transformers/default/1"
# l4opt 10-10: the packed INT3 GPTQ deployment (gEAN320, the G4 W3A8 model; experts in Humming uint3 g128, dense
# FP8 / BF16) for humming_moe "int3packed". The asset manifest names the model; check_manifest takes either.
MODEL_INT3 = "threerabbits/qwen38fn-gean320-int3-gptq/Transformers/default/1"
# 10-15: version 2 = the same package with the Joint QAD v2 run1 step-147 values (same files, shapes and config.json;
# docs/research/qwen38fn_gean320_qad_gap_and_gates_20261015.md). The asset's model pin decides which one a build serves.
MODEL_INT3_QAD = "threerabbits/qwen38fn-gean320-int3-gptq/Transformers/default/2"
# 10-24: version 3 = the student-trajectory QAD, reverse KL32_tail step 1000 (qadmix-R-v1000; same files, sizes and
# config.json as v2, only deploy_manifest.json / qad_manifest.json and shard values differ;
# docs/research/qwen38fn_qad_student_traj_20261023.md). Once v3 exists Kaggle drops v1 / v2 pins at push time.
MODEL_INT3_QAD3 = "threerabbits/qwen38fn-gean320-int3-gptq/Transformers/default/3"
MODELS_INT3 = (MODEL_INT3, MODEL_INT3_QAD, MODEL_INT3_QAD3)
MODELS = (MODEL, *MODELS_INT3)
RUNTIME = "threerabbits/qwen38fn-sglang-0520-sm120-cp312"
RUNTIME_SHA = "cbf427cb3677313cf060ec4e4c4e53d2cf7a9dbf06ac884ce31e93447345c1b3"
ZSTD = "threerabbits/zstd-x86-64-kaggle"
SOURCE_DATASET = "threerabbits/qwen38fn-l4-hisparse-h28"
SOURCE_DIGEST = "b9cbd753acb18f0a40802b198f7d64fd92a48d5bc2117c2918bcff2b9b8ebb22"
SOURCE_MANIFEST_SHA = "a80d8d481679e25c17c4f9c3089c9d77812ef5a676eaab371541df2187de7f84"
FAST_SOURCE_DIGEST = "df5847fd1fb73f62c09e5432c7a263d85ea97832594a0836880dd3f02e252b28"
FAST_SOURCE_MANIFEST_SHA = "3e71105a9ebcb21dee2c030c65b5b24a94c8aeca825d2b381bc0290c8251c710"
PROVENANCE_SHA = "812fd475d9056e54c107272d93960c8aa2770c8e4945b476846811e30b908680"
COMPETITION = "arc-prize-2026-arc-agi-2"
LAUNCHER = "work/qwen38fn_l4/probes/runpod_hisparse.py"
KNOWN_BAD_RS_CANDIDATES = {"r7-r4", "r8-r2", "r9"}
CANDIDATE_IDS = ("r7-r4", "r8-r2", "r9", "r9-r2", "arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")
NOMTP_CANDIDATES = ("r9-r2", "arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")  # target-only control (v15 r9-r2; v16's fallback on the fork)
# v16: the sglang-arc-2 fork trees that carry the shared MTP5-DFlash worker (speculative/arc_dflash): arc/main dfd2fd8f
# (MoE drafter) and exp/dense-dflash 86932128 (= arc/main + the dense-routed drafter, ffn_type dense_shared)
DFLASH_CANDIDATES = ("arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")
DENSE_DFLASH_CANDIDATES = ("arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")
# fork exp/planb-serve-l4 c8500d9fe and later: the plan B drafter, whose export config says residual "plain" (one stream,
# pre-norm), context "inject" (committed tokens' K/V from their input fusion, no trunk rows) and ffn_type "dense" (one
# MLP of dense_intermediate_size). Its DFlash spec pins exactly these three fields as "architecture"; a spec without it
# is the hyper-connection dense_shared export. Other trees' dense loader takes dense_shared only.
PLAIN_DFLASH_ARCHITECTURE = {"residual": "plain", "context": "inject", "ffn_type": "dense"}
PLAIN_DFLASH_TREES = ("a8fc06fce469b6c58494effc1b59e1357162e505be15f886143ac60c2bdbd806",  # c8500d9fe
                      "6ce650f1fe19603e69c4015f2849d9a8038ea8ae23efc8b0eae0c568a0c6a72f",  # 60997af63a
                      "8aa4af5f6a3e55ba21c7836f6b29ea0a90b933b2498b73327d43e9fcc133e281")  # all-merge-1020 a5c67c41cd
# fork 60997af63a (exp/planb-opt's serving opt-ins on the L4 tree): Profile dflash_merge / dflash_inject_kv /
# dflash_fuse_once / dflash_kv_dtype / dflash_ring_fp8_dot = ARC_DFLASH_MERGE / _INJECT_KV / _FUSE_ONCE / _KV_DTYPE /
# _RING_FP8_DOT, each the fork's default unless set (field -> default). The plain drafter only.
PLAIN_DFLASH_OPT_TREES = ("6ce650f1fe19603e69c4015f2849d9a8038ea8ae23efc8b0eae0c568a0c6a72f",
                          "8aa4af5f6a3e55ba21c7836f6b29ea0a90b933b2498b73327d43e9fcc133e281")  # all-merge-1020 a5c67c41cd
PLAIN_DFLASH_OPTS = {"dflash_merge": False, "dflash_inject_kv": "layers", "dflash_fuse_once": False,
                     "dflash_kv_dtype": "bf16", "dflash_ring_fp8_dot": False}
ONLINE_FP8_LAYERS = ("gdn_in", "gdn_out", "qsa_qkv", "qsa_o")  # arc_hostcomm_hook ARC_HC_ONLINE_FP8; lm_head stays BF16
HARNESSES = ("frozen-arc2", "codex")
CODEX_DATASET = "threerabbits/arcforge-codex-0154-p8"
CODEX_SERVED_MODEL = "qwen38fn"  # the G4 hs55 codex notebook's served name
# 10-31: the competition closes 2026-11-02 23:59 UTC and a submission queues for hours before its notebook starts, so
# the notebook reads the clock: the codex deadline is capped so that the whole notebook ends by END_BY_UTC.
END_BY_UTC = "2026-11-02T23:30:00+00:00"
END_BY_MARGIN_SECONDS = 1200  # the harness grace kill (900 s) + the steps after it (90 min SaveRun 10-31: under 2 min)
END_BY_BOOT_SECONDS = 1500  # notebook start to harness start: 17.5 min (x36-base 12 h, 10-31), 19.3 min (x34 12 h, 10-29)
END_BY_MIN_HARNESS_SECONDS = 600  # less harness time than this before the deadline: the placeholder submission instead
# humming-kernels 0.1.16, the PyPI wheel unchanged (the Runpod 10-06 INT3 arm ran these exact files)
HUMMING_WHEEL = {"dataset": "threerabbits/humming-kernels-0116",
                 "file": "humming_kernels-0.1.16-cp310-abi3-manylinux_2_28_x86_64.whl",
                 "sha256": "683212692f5e1339961e4322aff1ba116523e2681b2aafeaf7c081bc4e1f5627"}
SPEC_TRACE_MODES = ("compare", "host", "pair", "hostinit", "pair_prefill", "pair_decode", "dump")
HC_DEBUG_MODES = ("streams", "no_attn", "zero_freed")
DFLASH_COMPILE_MODES = ("0", "det")  # arc-main/arc-dense compile only on "1" (their default): "0" turns it off there
DFLASH_DET_CANDIDATES = ("arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")  # fork 135068d5a's "det" (inductor, no per-process autotune)
DRAFT_VOCAB_CANDIDATES = ("arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")  # fork exp/l4-dense-det-vocab: SubsetPairHead (ARC_DFLASH_DRAFT_VOCAB)
# fork exp/l4-tp4-attn: pure TP4 attention (attn_dp 1) under ARC_HISPARSE_TP4=1; unset, the arc-dense-vocab paths
TP4_ATTENTION_CANDIDATES = ("arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")
SHARDED_VERIFY_CANDIDATES = ("arc-dense-vocab-tp4-shv",)  # fork exp/l4-sharded-verify (ARC_SHARDED_VERIFY)
# fork exp/l4-sharded-verify from 53192d1a: ARC_PLE_INT5 (INT5 host PLE table) and the unpadded PLE extend that a prefill
# forward of more than one request needs
PLE_INT5_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
# fork exp/l4-sharded-verify from addbbf3d (+ efbcbcc4's int64 index fix): G4's extend without host syncs (main d2e4567b)
EXTEND_NOSYNC_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
# fork exp/l4-sharded-verify from 33991165: DFlash exact prefill / session park / async refresh (main 56d78f62, 06fd8d9c)
DFLASH_PARK_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
# fork 8e3f880f and later: ARC_DFLASH_SHARED_GRAPH_POOL_GROUP (drafter graphs in the target pool on an attention-TP group)
SHARED_GRAPH_POOL_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
QSA_SM89_CANDIDATES = ("arc-dense-vocab-tp4-shv",)  # fork 21bd6674 and later: ARC_QSA_SM89_DECODE / _STAGES
B200OPT_CANDIDATES = ("arc-dense-vocab-tp4-shv",)  # fork 1888206438 and later: RING_OUT_T / BATCHED_COMMIT / FUSED_PROPOSE / HC gate
H24_TP_CANDIDATES = ("arc-dense-vocab-tp4-shv",)  # fork 2f31c0cb and later: h24 packed GDN state on TP > 1 (gdn_state_fmt)
TARGET_VOCAB_CANDIDATES = ("arc-dense-vocab-tp4-shv",)  # fork d94aef29 and later: ARC_TARGET_VOCAB (target_vocab)
# fork exp/l4-sharded-verify from l4opt34: main's multimodal host-RAM fixes (ARC_MM_RELEASE_AFTER_PREFILL, ARC_MM_LAZY_MEDIA,
# the retract media refill) and the drafter's ARC_DFLASH_W4A8 / selector codebook pruning / sliced attention clones
MM_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
W4A8_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
W4A8_SPEC = re.compile(r"(int4|uint4z):(32|64|128)(?::(all|mlp))?")
HEAPDIAG_CANDIDATES = ("arc-dense-vocab-tp4-shv",)  # fork 3b48fa36 and later (l4opt37): ARC_HEAPDIAG (heap_diag)
# Source trees with srt/distributed/arc_shm_gather.py (fork all-merge 8d4dea09 = tree 07b13538, payload l4opt43):
# ARC_SHM_GATHER (shm_gather). The registered arc-dense-vocab-tp4-shv pin (c8c83127) does not have it.
SHM_GATHER_TREES = ("07b1353858f8291830f30951871b81f34bab67991f3a112419fc1468b39577b7",
                    "4b6cb81b24fba28fd0da4a217384be41e4f7970a791532fa7dc09a0e64510e6b",  # + exp/l4-qsa-index-shard 5c196331
                    "a8fc06fce469b6c58494effc1b59e1357162e505be15f886143ac60c2bdbd806",  # + exp/planb-serve-l4 c8500d9fe
                    "6ce650f1fe19603e69c4015f2849d9a8038ea8ae23efc8b0eae0c568a0c6a72f",  # + plan B opt-ins 60997af63a
                    "8aa4af5f6a3e55ba21c7836f6b29ea0a90b933b2498b73327d43e9fcc133e281")  # all-merge-1020 a5c67c41cd
# fork exp/l4-qsa-index-shard (from 8d4dea09): ARC_QSA_INDEX_SHARD=layer (qsa_index_shard). The run also checks the
# installed tree for the switch's read points before it sets the env (QSA_INDEX_SHARD_MARKERS in MANAGED).
QSA_INDEX_SHARD_CANDIDATES = ("arc-dense-vocab-tp4-shv",)
# Qwen3.8-FN tokenizer: padded vocabulary and the added tokens (<|endoftext|> .. <|audio_pad|>, chat / tool / think
# markers and both EOS ids); a target_vocab list must keep every one of them.
TARGET_VOCAB_SIZE = 248320
TOKENIZER_ADDED_IDS = range(248044, 248077)
# arc_hostcomm_hook ARC_HC_HUMMING_MOE: int3:128 requantizes the NVFP4 experts at load (v16 K8 profile); int3packed
# loads MODEL_INT3's packed GPTQ experts as stored
HUMMING_MOE_SPECS = ("int3:128", "int3packed")
HUMMING_MOE_TILES = ("global", "local")  # ARC_HC_HUMMING_MOE_TILE
HUMMING_MOE_ACTS = ("bf16", "int8", "fp8")  # ARC_HC_HUMMING_MOE_ACT (int8 with int3:128, fp8 with int3packed)
HUMMING_DENSE_SPECS = ("int4:128",)
HC_AG_KINDS = ("ag_fp8v", "ag_fp8q")  # hostcomm all-gather variants besides the default ag_bf16 (ARC_HC_AG_KIND)
HC_AR2_KINDS = ("ar1_fp8v",)
HC_SUMLEN_KINDS = ("bf16", "fp8")  # ARC_HC_SUMLEN 1 / fp8  # attention-pair all-reduce variants besides the default ar1_bf16 (ARC_HC_AR2_KIND)


@dataclass(frozen=True)
class Profile:
    running: int = 64
    logical_tokens: int = 3250048
    mem_fraction: float = 0.95
    host_reserve_gib: float = 8.0
    validation_limit: int = 8  # Original notebook Config.limit_manual.
    capacity_phase: bool = False
    saverun_minutes: float | None = None
    rerun_minutes: float | None = None
    live_logging: bool = True
    validation_repeat: int = 1  # SaveRun only: copies of the public roster, to load-test like the rerun.
    candidate_id: str | None = None
    mtp: bool = False
    hot_blocks: int = 512
    prefill_chunk: int = 1024
    sparse_graphs: bool = False
    draft_fp8_direct: bool = False
    fold_in_verify: bool = False
    mtp_rejection_sampling: bool = False
    native_mtp_host_kv: bool = False
    cpu_affinity: bool = False
    debug_checks: bool = True
    spec_phase_packed: bool = False
    mtp_steps: int = 3
    nomtp_control: bool = False
    hostcomm: bool = False  # v15: arc_hostcomm_hook via sitecustomize (Vast 10-03), r9-r2 NoMTP only
    vision_offload: bool = False  # ARC2 never sends images; needs the hostcomm hook to install
    # v15 harness knobs. None / False = the v13 command line (original notebook Config: ctx_cap 50000).
    harness_ctx_cap: int | None = None
    deadline_rate_adaptive: bool = False
    time_aware_budget: bool = False
    # v16 (sglang-arc-2 arc/main): MTP5-DFlash drafter with verify width K (None = no DFlash), server context,
    # HiSparse swap-in CTA width, hostcomm-hook extras, dp-lm-head, the codex harness and the startup gates.
    dflash_k: int | None = None
    context_length: int = 131072
    swap_in_block_size: int | None = None
    online_fp8: str | None = None  # comma list of ONLINE_FP8_LAYERS (hostcomm hook ARC_HC_ONLINE_FP8)
    hc_ag_anydtype: bool = False   # hostcomm all-gather of int64/int32/fp32 (PLE ngram ids) instead of NCCL
    hc_slot_mb: int = 21
    dp_lm_head: bool = False       # --enable-dp-lm-head
    # SGLANG_FORCE_COARSE_WAR_BARRIER=1: the scheduler's next-batch writes wait for the whole forward. The fine barrier
    # falls back to PRE_REPLAY when a backend has no in-graph marker (decode_cuda_graph_runner._resolve_shared_read_ends),
    # releasing table writes while QSA/HiSparse still read them inside the verify graph (v16 K8 illegal address).
    coarse_war_barrier: bool = False
    # ARC_THINK_BUDGET=<N> (sglang-arc-2 speculative/arc_think_budget.py, in the EAGLE chain verify the DFlash worker
    # runs): a response's thinking block is closed after N output tokens and the model goes on to its tool call, so one
    # response cannot push the history past what codex's compaction request still fits in (local loop 10-25..10-29:
    # single responses up to 191k; 227 of 5,382 compaction requests did not fit 153.6k, after one ~99k response).
    think_budget: int | None = None
    # ARC_V16_MARLIN_BLOCK_REMAP=1 (kaggle/v16_patch/sitecustomize.py): fused_marlin_moe runs block 64 where its rule
    # picks 48 (M in [230, 346): K8 C32 verify, M=256), the suspect of the v16 K8 illegal address.
    marlin_block_remap: bool = False
    # ARC_HC_HUMMING_MOE (arc_hostcomm_hook): the target's routed experts are requantized from NVFP4 at load and run on
    # Humming, so no marlin in the target MoE (the drafter's stays marlin). Runpod 10-06 K8 C32 with int3:128 ran 4 h
    # clean where marlin (with or without the block remap) crashed every ~62 min; Kaggle v5 (remap) crashed at 97 min.
    humming_moe: str | None = None
    # l4opt 10-08: ARC_HC_HUMMING_MOE_TILE (row count the Humming tuning heuristic sees: "global" = M*top_k as before,
    # "local" = the rows of this rank's experts; global over-sizes block_m 4x at C64 verify) and ARC_HC_HUMMING_MOE_ACT
    # ("int8": W3A8, per-token INT8 activations quantized inside Humming).
    humming_moe_tile: str = "global"
    humming_moe_act: str = "bf16"
    # l4opt (Vast 10-03 stack, not yet on Kaggle): ARC_HC_HUMMING_DENSE=<spec> over ARC_HC_HUMMING_LAYERS (weight-only
    # W4A16 for the chosen dense linears, disjoint from online_fp8), ARC_HC_MIX_W8=1 (hyper-connection mixers INT8
    # weight-only, Triton kernel in probes/skinny_gemm, frees ~0.6 GiB/card) and ARC_EMBED_OFFLOAD=1 (embed_tokens
    # from one host-mapped table, frees the attention-TP shard).
    humming_dense: str | None = None
    humming_layers: str | None = None
    hc_mix_w8: bool = False
    embed_offload: bool = False
    # l4opt: ARC_HC_MOE_RS=1 (MoE combine reduces only this DP's rows on host-mapped memory instead of AR4 + dp_scatter)
    # and ARC_HC_AG_KIND (dp_gather all-gather payload: ag_bf16 default, ag_fp8v = FP8 with a scale per 128 values).
    # The L4 DFlash K4 C32 verify (p2b, 10-08) spends 38.8 ms of 108.7 in these collectives vs 10.6 ms in a NoMTP step.
    hc_moe_rs: bool = False
    hc_ag_kind: str | None = None
    # l4opt DP-aware hostcomm (bitwise equal to the defaults): hc_ag_dp = ARC_HC_AG_DP=1 (MoE dispatch gather reads only
    # the other DP's shards; hc_ag_dp_check also verifies the pair replication it relies on), hc_moe_rs_owner =
    # ARC_HC_MOE_RS=owner (combine reads S instead of 1.5S per rank). hc_numa_node = ARC_HC_NUMA_NODE (first-touch the
    # hostcomm regions on that NUMA node; multi-socket hosts such as Runpod EPYC, Kaggle is single socket).
    # moe_stats = ARC_MOE_STATS=1 (distinct experts per decode MoE call in the log, measurement only).
    hc_ag_dp: bool = False
    hc_ag_dp_check: bool = False
    # 10-13 (docs/research/kaggle_l4x4_link_hostcomm_20261012.md): hc_pipe_ar2 / hc_pipe_tp4 = ARC_HC_PIPE_STAGES2 / 4
    # (1-8: that group's ar1_fp8v, ag_fp8q / ag_dp_fp8q and owner_fp8 combine run writer and reader blocks, the reads
    # of a step overlapping the writes of the next on the host link; bitwise identical; Kaggle B=188 pays only for the
    # pair AR2), hc_ar2_blocks = ARC_HC_AR2_BLOCKS (the pair all-reduce's blocks / chunks, hook default 32). hc_ag_dp
    # with hc_ag_kind ag_fp8q = the FP8 gather that quantizes the sibling's shard locally, in MAX_LEN decode / verify
    # forwards only. hc_ag_dp_check_every = ARC_HC_AG_DP_CHECK_EVERY (check one in N of those gathers, so one run
    # measures the saving and checks replication).
    hc_pipe_ar2: int = 0
    hc_pipe_tp4: int = 0
    hc_ar2_blocks: int | None = None
    hc_ag_dp_check_every: int = 1
    # 10-13 review (docs/research/kaggle_l4x4_link_hostcomm_20261012.md, hc_variants_test / l4_hcbench --exact):
    # hc_fold_epoch = ARC_HC_FOLD_EPOCH (FP8 collectives advance the epoch in-kernel, no bump kernel), hc_ar2_p2p =
    # ARC_HC_AR2_P2P (the pair ar1_fp8v over CUDA peer access, bitwise; not with hc_pipe_ar2), hc_rs_stage2 =
    # ARC_HC_RS_STAGE2 ("fp8": the owner_fp8 combine's pair exchange in FP8, lossy), hc_rs_sib_wait =
    # ARC_HC_RS_SIBWAIT (that exchange waits for the sibling only, bitwise), hc_ag_blocks = ARC_HC_AG_BLOCKS (dispatch
    # gather chunks, hook default 16), gpu_monitor = ARC_GPU_MONITOR (rank 0 samples SM clock / power / clock-event
    # reasons of every GPU, NVML PCIe TX / RX throughput and host PSI / cgroup throttling, logged per ARC_FWD_TIMING
    # window; needs fwd_timing), hisparse_miss_stats = ARC_HISPARSE_MISS_STATS (fork exp/l4-shv-review c8c83127 and
    # later: per-layer swap-in counters, read by the hook per window on ranks 0 / 2; needs fwd_timing).
    hc_fold_epoch: bool = False
    hc_ar2_p2p: bool = False
    hc_rs_stage2: str = "bf16"
    hc_rs_sib_wait: bool = False
    hc_ag_blocks: int | None = None
    gpu_monitor: bool = False
    hisparse_miss_stats: bool = False
    hc_moe_rs_owner: bool = False
    hc_numa_node: int | None = None
    moe_stats: bool = False
    # ARC_MROPE_DECODE_PIN=1: decode M-RoPE deltas through a pinned non-blocking copy (the stock torch.tensor(list,
    # device=cuda) syncs the stream every decode step of a batch with an image request; 52 ms/run_batch, NoMTP C64).
    mrope_decode_pin: bool = False
    # ARC_FWD_TIMING=1: CUDA events around every ModelRunner.forward, read back without a host sync; each rank logs
    # GPU ms per (runner, forward mode), the stream time between forwards and prefill forwards by size every 60 s.
    fwd_timing: bool = False
    # ARC_SPEC_AGREE_DEVICE=1: the distributed HiSparse spec guards (verify input, accept, snapshot claim) compare the
    # pair's device tensors on the GPU (hostcomm gather + device assert) instead of .cpu() + gloo inside run_batch,
    # which made the scheduler wait for the draft and the verify (100 + 11 ms per run_batch, K4 C32 Runpod 10-08).
    # "ab" alternates device / original guards every 500 verify cycles in the same server and logs per-mode cycle
    # medians (ARC_SPEC_AGREE_AB): an A/B under the same host-communication speed.
    spec_agree_device: bool | str = False
    # l4opt topology (10-09): attn_dp 2 = TP4 + attention DP2 x TP2 + EP4 (the deployment); 1 = pure TP4 (one
    # scheduler, attention TP4 with the 2 KV heads replicated, EP4 MoE with an all-reduce instead of the DP gather /
    # combine), launcher --attn-dp 1 and ARC_HISPARSE_TP4=1 on a TP4_ATTENTION_CANDIDATES tree. logical_tokens and
    # running stay per attention DP / global, so attn_dp 1 needs twice the logical tokens for the same per-request room.
    attn_dp: int = 2
    # ARC_HC_LOCAL_SLOTS=1: each rank first-touches the hostcomm pages it writes on its GPU's NUMA node (Runpod dual
    # socket: GPU1-3 otherwise write across the socket at 1.2-2.9 GB/s; hc_buffer_probe MoE combine 2.1-3.3x faster)
    hc_local_slots: bool = False
    # l4opt FP8 communication (10-10, docs/plans/qwen38fn_l4_fp8_comm_20261010.md), e4m3 + one fp32 scale per 128
    # values, hidden-width tensors only: hc_ar2_kind ar1_fp8v = ARC_HC_AR2_KIND (attention-pair all-reduce: out_proj,
    # embedding and the drafter's), hc_moe_rs_fp8 = ARC_HC_MOE_RS=owner_fp8 (the quarter-owner MoE combine sends the
    # owners' partials in FP8, pair exchange bf16), hc_ag_kind ag_fp8q = ag_fp8v whose payload the int3packed W3A8
    # Humming MoE takes as gate_up input (group 128) instead of requantizing per token.
    hc_ar2_kind: str | None = None
    hc_moe_rs_fp8: bool = False
    # ARC_HC_SUMLEN (10-11): prefill / extend forwards (SUM_LEN layout) gather with ag_sum (each pair rank publishes
    # half of its DP's rows, reads only the other DP's; bitwise) and combine with rs_sum_owner instead of the
    # zero-filled two-shot AR4s (det-g4: 2 x 48 x ~3.1 ms in a 2k-token extend of ~830 ms). "bf16" = bitwise the AR4
    # path, "fp8" = the combine's partials to the owners in FP8 (as hc_moe_rs_fp8). Needs hc_moe_rs_owner.
    hc_sumlen: str | None = None
    # ARC_HC_MOE_OVERLAP (10-21, probes/l4opt/overlap/moe_combine_overlap.py): "<w1>+<w2>+..." (multiples of 128 summing
    # to the hidden size 2560, e.g. "512+1024+1024"): the Humming down weights split along N at load; decode / verify
    # MoE layers run each chunk's rs_owner combine on a second stream under the next chunk's GEMM (L4 C112 48-layer
    # chain: -3.1 ms). Bitwise the unsplit path (probes/hostcomm/moe_overlap_test.py). Needs hc_moe_rs_owner and the
    # int3packed Humming MoE, without hc_rs_stage2 fp8 / hc_rs_sib_wait / hc_pipe_tp4 (not verified; the hook raises).
    hc_moe_overlap: str | None = None
    # ARC_HC_MIX_VERIFY (10-11): GatedResidual mixes of lo < rows <= hi ("<lo>:<hi>", e.g. "16:256") on
    # probes/skinny_gemm/hc_mix_rows.py (deterministic split-K down, gate-mean fused into the up GEMM, tiles per row
    # bucket from hc_mix_rows_l4.json) instead of the five-kernel torch.compile chain (det-g4 verify, 187 rows / DP:
    # ~11 ms of a 119 ms step). BF16 weights, fp32 accumulation: not bitwise against the chain.
    hc_mix_verify: str | None = None
    # ARC_LMHEAD_TRITON (10-11): the target lm_head GEMM over lo < rows <= hi ("<lo>:<hi>", e.g. "64:256") on
    # probes/skinny_gemm/lm_head_gemm.py instead of torch.matmul (L4, 192 rows x 124,160 x 2560: 2,765 vs 3,016 us).
    # fp32 accumulation like cuBLAS, another kernel: not bitwise.
    lmhead_triton: str | None = None
    # l4opt sharded verify (10-10, fork exp/l4-sharded-verify): ARC_SHARDED_VERIFY=1 keeps the target-verify logits
    # vocab-sharded on the attention-TP group and samples the chain from per-row shard summaries the pair exchanges
    # (~12 KB per step instead of the 24-32 MB logits gather; same distribution as the chain RS kernel). Greedy, top-k /
    # top-p, penalty and logprob batches gather as before. Needs a SHARDED_VERIFY_CANDIDATES tree, DFlash and the LM head
    # vocab-sharded over attention TP (dp_lm_head, or attn_dp 1).
    sharded_verify: bool = False
    # l4opt 10-11: ARC_PLE_INT5=1 stores the host-pinned PLE table as symmetric INT5 (group 16, e4m3 scale; 110 B per
    # 160-wide row instead of 160 B FP8): 47.7 -> 32.8 GiB host over the four ranks, room for codex sandboxes.
    ple_int5: bool = False
    # 10-24: ARC_PLE_NVFP4=1 (fork arc_ple_nvfp4, same plumbing as ple_int5): NVFP4 rows, 90 B per 160-wide row instead of
    # INT5's 110 B -- 8.20 -> 6.71 GiB host per rank, ~6 GiB over the four. Lossy: row relative RMS 9.5 % (INT5 4.2 %).
    # Excludes ple_int5.
    ple_nvfp4: bool = False
    # requests per prefill forward (--prefill-max-requests; 1 since the 09-30 bring-up). > 1 needs the unpadded PLE
    # extend of PLE_INT5_CANDIDATES (the padded path grew with requests x longest row).
    prefill_max_requests: int = 1
    # --enable-prefill-delayer (attention DP2): a prefill on one DP waits up to prefill_delayer_max_passes scheduler
    # passes for the other DP so both prefill in one forward instead of one running IDLE (10-11 s3 C80: prefill 8.8 % of
    # the wall, the other DP's IDLE 8.3 %); queue ratio / max delay ms set the optional queue trigger (unset = off).
    prefill_delayer: bool = False
    prefill_delayer_max_passes: int | None = None
    prefill_delayer_queue_ratio: float | None = None
    prefill_delayer_max_delay_ms: float | None = None
    # l4opt 10-11: ARC_HOST_POOL_PRIVATE=1 (v16_patch) puts the pinned host pools (QSA host KV, PLE table; ~19 GiB per
    # scheduler) on private anonymous pages, and PyTorch's pin_memory blocks go malloc + cudaHostRegister. The Runpod L4
    # host had kcompactd failing ~7.5M migrations/min on that shmem and NUMA balancing scanning the schedulers; G4 10-10
    # (same fix, int3_deploy_runtime.py): extend forwards 1.3-1.9x slower in such minutes, three runs flat after it.
    host_private: bool = False
    # l4opt 10-11: G4's sync-free extend (fork main d2e4567b): ARC_GDN_HOST_CHUNK_INDICES=1 (the GDN chunk prefill plan
    # from the host extend lengths, pinned non-blocking copies) and ARC_MM_NO_SYNC=1 (a new image's pixels through pinned
    # memory, the vision encoder's max_seqlen from the host cu_seqlens). G4 c110: no stream sync left in the target
    # extend, -2.2 ms per extend batch; L4's eager extend is host-launch bound (~250 ms fixed per forward), so the head
    # start is the point. Neither logs; the recorded server env carries them.
    extend_nosync: bool = False
    # l4opt 10-11: G4's DFlash prefill (fork main 56d78f62 / 06fd8d9c): ARC_DFLASH_EXACT_PREFILL=1 (a prompt's drafter
    # window is exactly one refresh over its last layers x (window + 3) = 5,135 positions), ARC_DFLASH_SESSION_PARK=1 (a
    # finished session request keeps its window; the session's next request adopts it when the ids match and appends
    # only what lies past it -- the codex proxy sends session_id on every /responses) and ARC_DFLASH_REFRESH_ASYNC=1 (no
    # host read of device state in the refresh). G4 c110: 97.4% of parks adopted, refresh GPU -17%, CPU -10 ms per
    # extend batch. Without it every turn's window restarts from the turn's own chunk (ARC_DFLASH_PREFILL_TAIL 2048).
    dflash_session_park: bool = False
    # ARC_DFLASH_SHARED_GRAPH_POOL_GROUP=1 (fork 8e3f880f, opt-in): the drafter's CUDA graphs capture into the target's
    # graph pool on the target's capture stream on each attention-TP pair, as G4 does on one rank (f1f7836e: 0.44 -> 0.08
    # GiB at c110). L4 C96: draft graphs 0.20 GiB per card in their own pool.
    dflash_shared_graph_pool: bool = False
    # B200 decode-cycle opt-ins (fork exp/b200-perf-extend, 10-09; on exp/l4-sharded-verify since 10-11):
    # dflash_step_opt = ARC_DFLASH_RING_OUT_T=1 (ring attention writes [B, n, Hq, D], bitwise), ARC_DFLASH_BATCHED_COMMIT=1
    # (24 commit kernels instead of ~90, bitwise) and ARC_DFLASH_FUSED_PROPOSE=1 (the selector chain as one Triton
    # kernel; identically distributed draws). PC 4090 drafter step at bs 48: 5.88 -> 5.49 ms. ARC_DFLASH_RING_TILE=auto
    # is left out: its 128x64 tile needs 148 KB of shared memory (sm_89 has 101 KB). hc_gate_precompute =
    # ARC_HC_GATE_PRECOMPUTE=1: verify / decode HC combines over > 32 rows compute the gate on a side stream next to the
    # attention / MoE block and only apply it after the all-reduce (bitwise; B200 verify -0.46 ms).
    dflash_step_opt: bool = False
    hc_gate_precompute: bool = False
    # QSA HiSparse sm_89 direct decode (fork 21bd6674): ARC_QSA_SM89_DECODE = E2M1 decode (0 arithmetic as shipped,
    # 1 fp32 bit construction, 3 bf16 bit construction + bf16 scale multiply; all the same bits) and
    # ARC_QSA_SM89_STAGES (2 as shipped). 4090 at the L4 verify shape: decode 3 -31 %, stages 3 another -4 %.
    qsa_sm89_decode: int | None = None
    qsa_sm89_stages: int | None = None
    # server env TORCH_BLAS_PREFER_CUBLASLT=1 (10-11): aten bf16 GEMMs through cuBLASLt's heuristics. Legacy cuBLAS on
    # L4 picks the 1-stage s1688gemm for the HC mixer's up projection at 192 rows (verify, 48 requests per DP): HC
    # chain 88.7 us vs 79.2 at 188 rows; cuBLASLt 77.2 (hctest_hck.sh). Other kernels, so not bitwise.
    blas_lt: bool = False
    # h24 packed GDN state (10-11): ARC_GDN_STATE_FMT=i10h / i12h (+ ARC_GDN_STATE_FMT_DISTRIBUTED=1, the fork's gate for
    # distributed HiSparse) on the h20 fold-in-verify. The state is stored at 65.6 % (i10h) / 78.1 % (i12h) of the fp16
    # bytes: L4 ReplaySSM verify 387.5 -> 306.5 us per layer (i10h), mamba SSM 27.4 -> ~18 MiB per slot. Needs fork
    # 2f31c0cb (TP > 1; H24_TP_CANDIDATES).
    gdn_state_fmt: str | None = None
    # GDN snapshot host buffers per rank (launcher snapshot_host_buffers, payloads from l4opt34): None keeps the launcher's
    # 2 x (R/DP + 16), which is the fork's anchor sizing with anchors off; the fork's no-anchor default R/DP + 16 is
    # one buffer per session plus slack (C96: 128 -> 64 buffers, 29.6 MB each per rank, ~7.6 GB of pinned host)
    snapshot_host_buffers: int | None = None
    # h23 resume anchors (launcher session_anchor -> hisparse_config qsa_session_anchor, payloads from l4opt34): a parked
    # session record also keeps its resume point, so a request agreeing only up to the previous turn (a retry, a dropped
    # last generation) truncates to it instead of a full re-prefill; anchors use the second half of the default
    # 2 x (R/DP + 16) buffers, and a buffer shortage drops the LRU record's anchor before the record
    session_anchor: bool = False
    # l4opt 10-12, main's multimodal host-RAM fixes (MM_CANDIDATES): ARC_MM_RELEASE_AFTER_PREFILL=1 drops a request's
    # pixel features once its prefill completes (codex resends every image each turn: G4 c110 scheduler host RAM 9.6 ->
    # 1.3 GiB/h); ARC_MM_LAZY_MEDIA=1 sends media seen before without pixels (the frontend keeps their expanded token
    # blocks; one tokenizer worker). A retracted request whose media carry no feature waits for a refill. The ARC2
    # harness attaches every train / test grid PNG to the first message and lets the model view_image its renders.
    mm_release_after_prefill: bool = False
    mm_lazy_media: bool = False
    # l4opt 10-12: ARC_DFLASH_W4A8=<int4|uint4z>:<group>[:<all|mlp>] (W4A8_CANDIDATES): the drafter's block-FP8 linears as
    # Humming INT4 weights (RTN from the FP8 values) x per-token FP8 activations, half the weight bytes at the FP8 GEMM's
    # speed on sm_89; only the proposals change, so it is judged by accept and speed. Needs the Humming wheel.
    dflash_w4a8: str | None = None
    # 10-20 plan B serving opt-ins (fork 60997af63a, PLAIN_DFLASH_OPT_TREES; the plain drafter only; defaults = the
    # fork's, PLAIN_DFLASH_OPTS, and a default stays out of CONFIG): dflash_merge = ARC_DFLASH_MERGE=1 (per layer q|k|v,
    # gate|up and the injected k|v as one linear each), dflash_inject_kv "gemm" = ARC_DFLASH_INJECT_KV=gemm (every
    # layer's injected K/V from one BF16 GEMM), dflash_fuse_once = ARC_DFLASH_FUSE_ONCE=1 (the injected step's input
    # fusions in one pass), dflash_kv_dtype "fp8_e4m3" = ARC_DFLASH_KV_DTYPE (ring pool K/V as e4m3, no scale),
    # dflash_ring_fp8_dot = ARC_DFLASH_RING_FP8_DOT=1 (e4m3 x e4m3 ring attention; needs the e4m3 pool). Each changes
    # only the proposals' rounding (the 10-20 pod pair test: the attention-TP pair bitwise in agreement).
    dflash_merge: bool = False
    dflash_inject_kv: str = "layers"
    dflash_fuse_once: bool = False
    dflash_kv_dtype: str = "bf16"
    dflash_ring_fp8_dot: bool = False
    # l4opt 10-12: ARC_HEAPDIAG=<seconds> (HEAPDIAG_CANDIDATES), a diagnostic: every scheduler logs its glibc heap
    # counters (the [heap] Rss, mallinfo2 in use / free, malloc_info per arena) every 60 s, and at init done and
    # <seconds> later a deep dump (CPU torch storages by mapping, big Python objects) followed by malloc_trim(0) and the
    # counters after it. Each deep dump holds the GIL for seconds; the first runs before any request.
    heap_diag: int | None = None
    # l4opt 10-16: ARC_SHM_GATHER=1 (a SHM_GATHER_TREES source): the scheduler's per-cycle CPU collectives -- the
    # attention-TP consensus of distributed HiSparse spec (agree / ready_count) and the DP-attention MLP sync -- go
    # through a POSIX shared-memory buffer instead of Gloo (Runpod 10-04 C48: 0.36 / 0.7 ms -> ~30 us per exchange,
    # cycle 154.0 -> 144.1 ms). Same payloads in the same lockstep; a sequence mismatch or a stalled peer raises.
    shm_gather: bool = False
    # l4opt 10-16: the notebook samples the cgroup's memory.stat / memory.events every 10 s from just before the server
    # starts (server/host_memory.jsonl): what the startup host admission counts as used (page cache vs anon) and how
    # host memory moves through the gates and the harness. Read-only; no server change.
    host_memory_trace: bool = False
    # l4opt 10-16 host RAM: malloc_trim_seconds = hook ARC_MALLOC_TRIM_SECONDS (every scheduler gives glibc's free heap
    # back every N s; 10-12 ARC_HEAPDIAG: ~1.0 GiB per scheduler after ~30 min of serving; needs a payload whose
    # arc_hostcomm_hook.py is this checkout's); admission_reclaimable_file = fork ARC_ADMISSION_RECLAIMABLE_FILE=1 (the
    # startup host admission counts inactive page cache as available, as on Vast: Kaggle counted ~20-28 GB of page
    # cache as used); inductor_compile_threads = TORCHINDUCTOR_COMPILE_THREADS (1: no inductor compile worker pool,
    # ~1.4 GiB of host in the 10-12 accounting).
    malloc_trim_seconds: int | None = None
    admission_reclaimable_file: bool = False
    inductor_compile_threads: int | None = None
    # 10-16: ARC_QSA_INDEX_SHARD=layer (QSA_INDEX_SHARD_CANDIDATES): each attention-TP rank keeps the QSA index of half
    # the 12 QSA layers (i % 2) and hands its top-k to the pair; selection and outputs are bitwise the unsharded ones.
    # Index 396 -> 198 B/token per rank (C128 L 5,701,632: 2.10 -> 1.05 GiB), which hisparse_plan gives to hot blocks;
    # costs one int16 attention-TP all-gather per QSA layer and forward (docs/research/qsa_index_layer_shard_l4_20261016.md).
    # "layer-check" (needs deterministic): no saving, every rank keeps and scores every layer and checks each received
    # top-k against its own on the device; a differing row stops the server. The validation run of the receiving path.
    qsa_index_shard: str | None = None
    # launcher deterministic (ARC_QSA_TOPK_EXACT=1, ARC_HC_MIX_DETERMINISTIC=1, SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0):
    # every DP replica's two attention-TP ranks compute the drafter and the replicated target ops each on their own,
    # so any run-to-run nondeterminism lets them draft different tokens. Kaggle v6 (int3:128) died in the batch gate on
    # "attention-TP lifecycle mismatch at spec-verify-input" (DP1); the Runpod run of the same profile never hit it.
    deterministic: bool = False
    # server env ARC_DFLASH_COMPILE: "0" runs the drafter eager (no inductor). fork 135068d5a (B200, not in arc-main or
    # arc-dense): inductor autotunes each Triton kernel per process on first launch, so the two ranks of an attention-TP
    # group can run different configs and draft different tokens; COMPILE=0 passed there. Kaggle ladder step 1 (dense
    # K4, 10-07) died on the same lifecycle mismatch with neither marlin nor Humming in the drafter.
    dflash_compile: str | None = None
    # server env ARC_DFLASH_DRAFT_VOCAB: repo-relative JSON {"ids": [...]} the drafter's head scores instead of the
    # whole vocabulary (fork exp/l4-draft-vocab SubsetPairHead on the L4 dp-lm-head pairs; proposals only, the target
    # verifies against its full distribution). e1 trace: the full draft head is ~2.9 ms of each draft step at C48.
    draft_vocab: str | None = None
    # server env ARC_TARGET_VOCAB: repo-relative JSON {"ids": [...]} the target lm_head keeps (fork target_vocab.py:
    # every other id gets probability 0; sharded verify runs in the compact id space). Any list works: build one with
    # probes/l4opt/target_vocab_build.py; it has to keep the added tokens, contain draft_vocab and split into 64-row
    # slices per attention-TP rank. 98,304 ids: 636 -> 252 MB of head per L4 rank.
    target_vocab: str | None = None
    # debug: v16_patch ARC_V16_SPEC_TRACE compares the DFlash drafter's inputs (target GR, anchors) and proposals across
    # each attention-TP pair at every prefill / draft / decode-extend call and prints the first mismatches.
    spec_trace: str | None = None  # None, "compare", "host" or "pair" (see v16_patch _patch_spec_trace)
    hc_debug: str | None = None  # None, "streams" (log HostComm launch streams), "no_attn" (no HostComm in the pairs),
    # "zero_freed" (zero the Marlin experts Humming frees; needs humming_moe)
    harness: str = "frozen-arc2"
    gates_only: bool = False       # stop after the startup gates and the compiled-cache export (cache seeding runs)
    min_free_gib: float = 0.25     # per-card free device memory after startup and after the image gate
    vision_peak_images: int = 20
    vision_peak_concurrency: int = 4
    long_prefill_lengths: str = "8192,65536"  # comma list (a str keeps the CONFIG JSON round trip exact)
    seed_cache_from: str | None = None  # gates-only cache-seeding runs: start from this local compiled cache

    def __post_init__(self):
        if self.seed_cache_from is not None and (not self.gates_only or not self.seed_cache_from.startswith("/")):
            raise ValueError("seeding the compiled cache from a local path is for gates-only runs (absolute path)")
        if self.harness not in HARNESSES:
            raise ValueError(f"harness must be one of {HARNESSES}")
        if self.candidate_id is not None and self.candidate_id not in CANDIDATE_IDS:
            raise ValueError(f"unknown candidate {self.candidate_id!r}")
        if self.dflash_k is not None:
            if type(self.dflash_k) is not int or not 2 <= self.dflash_k <= 8:
                raise ValueError("DFlash verify width K must be an integer in 2..8")
            if self.candidate_id not in DFLASH_CANDIDATES:
                raise ValueError("DFlash needs a sglang-arc-2 candidate carrying speculative/arc_dflash (arc-main)")
            # fold_in_verify (h20) is a target-side GDN option: DFlash keeps SGLang's verify and commit
            if self.gdn_state_fmt is not None and (self.gdn_state_fmt not in ("i10h", "i12h") or not self.fold_in_verify
                                                   or self.candidate_id not in H24_TP_CANDIDATES):
                raise ValueError(f"gdn_state_fmt {self.gdn_state_fmt!r}: i10h or i12h, on fold_in_verify, with a "
                                 f"candidate carrying the TP > 1 packed state ({H24_TP_CANDIDATES})")
            if self.snapshot_host_buffers is not None and (
                    type(self.snapshot_host_buffers) is not int
                    or self.snapshot_host_buffers < self.running // self.attn_dp + 16):
                raise ValueError(f"snapshot_host_buffers {self.snapshot_host_buffers!r}: an integer >= R/DP + 16 "
                                 f"({self.running // self.attn_dp + 16}, the fork's no-anchor default)")
            if type(self.session_anchor) is not bool:
                raise ValueError(f"session_anchor {self.session_anchor!r}: a bool")
            if self.nomtp_control or any((self.mtp, self.draft_fp8_direct,
                                          self.mtp_rejection_sampling, self.native_mtp_host_kv,
                                          self.spec_phase_packed)) or self.mtp_steps != 3:
                raise ValueError("DFlash replaces the native MTP draft: no --mtp/--nomtp-control or native-draft options")
            # (A strict > 512 x K was tried after v16 v2: HOT 5120 also hit the K8 illegal address, so the verify
            # union is not the cause; see marlin_block_remap.)
            if self.hot_blocks < 512 * self.dflash_k or self.hot_blocks % 16:
                raise ValueError(f"DFlash K={self.dflash_k} needs hot blocks >= {512 * self.dflash_k} (verify union) and align16")
        if self.hostcomm and not (self.nomtp_control or self.dflash_k is not None):
            raise ValueError("hostcomm hook was validated on the repaired r9-r2 NoMTP control (Vast 10-03) and the "
                             "arc-main DFlash profile (Runpod 10-04) only")
        if self.online_fp8 is not None:
            names = self.online_fp8.split(",")
            if not self.hostcomm or not names or len(set(names)) != len(names) or not set(names) <= set(ONLINE_FP8_LAYERS):
                raise ValueError(f"online FP8 is a hostcomm-hook option over {ONLINE_FP8_LAYERS}")
        if self.marlin_block_remap and not (self.hostcomm and self.dflash_k is not None):
            raise ValueError("marlin_block_remap is chained from the hostcomm sitecustomize (hc_inject) on the DFlash profile")
        if self.humming_moe is not None and (self.humming_moe not in HUMMING_MOE_SPECS
                                             or not (self.hostcomm and (self.dflash_k is not None or self.nomtp_control))):
            raise ValueError(f"humming MoE is a hostcomm-hook option (DFlash or NoMTP profile), one of {HUMMING_MOE_SPECS}")
        if self.humming_moe_tile not in HUMMING_MOE_TILES or self.humming_moe_act not in HUMMING_MOE_ACTS:
            raise ValueError(f"humming_moe_tile one of {HUMMING_MOE_TILES}, humming_moe_act one of {HUMMING_MOE_ACTS}")
        if self.humming_moe is None and (self.humming_moe_tile, self.humming_moe_act) != ("global", "bf16"):
            raise ValueError("humming_moe_tile / humming_moe_act set the Humming MoE: they need humming_moe")
        if self.humming_moe == "int3packed":
            # MODEL_INT3's dense linears are FP8 / BF16 in the checkpoint: the hook's dense rewrites expect BF16
            if self.humming_moe_act not in ("bf16", "fp8"):
                raise ValueError("int3packed runs bf16 or fp8 activations (int8 would refold its signed BF16 scales)")
            if self.humming_dense is not None or self.online_fp8 is not None:
                raise ValueError("int3packed serves the checkpoint's own FP8 / BF16 dense: no humming_dense / online_fp8")
        elif self.humming_moe_act == "fp8":
            raise ValueError("humming_moe_act fp8 is the packed INT3 checkpoint's W3A8 (humming_moe int3packed)")
        if (self.humming_dense is None) != (self.humming_layers is None):
            raise ValueError("humming_dense and humming_layers go together")
        if self.humming_dense is not None:
            layers = self.humming_layers.split(",")
            if (self.humming_dense not in HUMMING_DENSE_SPECS or not self.hostcomm or not layers
                    or len(set(layers)) != len(layers) or not set(layers) <= set(ONLINE_FP8_LAYERS)):
                raise ValueError(f"humming dense is a hostcomm-hook option, one of {HUMMING_DENSE_SPECS} over {ONLINE_FP8_LAYERS}")
            if self.online_fp8 is not None and set(layers) & set(self.online_fp8.split(",")):
                raise ValueError("a dense layer is either online FP8 or Humming, not both")
        if self.hc_ag_kind is not None and self.hc_ag_kind not in HC_AG_KINDS:
            raise ValueError(f"hc_ag_kind must be one of {HC_AG_KINDS}")
        if self.hc_ag_kind == "ag_fp8q" and (self.humming_moe != "int3packed" or self.humming_moe_act != "fp8"
                                             or not (self.hc_moe_rs or self.hc_moe_rs_owner)):
            raise ValueError("hc_ag_kind ag_fp8q feeds the int3packed W3A8 Humming MoE (humming_moe_act fp8) under "
                             "the hc_moe_rs* layer hook")
        if self.hc_ar2_kind is not None and (self.hc_ar2_kind not in HC_AR2_KINDS or self.attn_dp != 2):
            raise ValueError(f"hc_ar2_kind is the attention-pair all-reduce (attn_dp 2), one of {HC_AR2_KINDS}")
        if self.hc_moe_rs_fp8 and not self.hc_moe_rs_owner:
            raise ValueError("hc_moe_rs_fp8 is the FP8 variant of the quarter-owner combine: needs hc_moe_rs_owner")
        if self.hc_sumlen is not None and (self.hc_sumlen not in HC_SUMLEN_KINDS or not self.hc_moe_rs_owner
                                           or self.attn_dp != 2):
            raise ValueError(f"hc_sumlen is one of {HC_SUMLEN_KINDS} inside the quarter-owner MoE hook "
                             "(hc_moe_rs_owner, attn_dp 2)")
        if self.hc_moe_overlap is not None:
            parts = self.hc_moe_overlap.split("+") if isinstance(self.hc_moe_overlap, str) else []
            if (len(parts) < 2 or not all(p.isdigit() for p in parts) or any(int(p) <= 0 or int(p) % 128 for p in parts)
                    or sum(int(p) for p in parts) != 2560):
                raise ValueError(f"hc_moe_overlap {self.hc_moe_overlap!r}: '<w1>+<w2>+...', at least two positive "
                                 "multiples of 128 summing to the hidden size 2560")
            if (not self.hc_moe_rs_owner or self.humming_moe != "int3packed" or self.hc_rs_stage2 != "bf16"
                    or self.hc_rs_sib_wait or self.hc_pipe_tp4):
                raise ValueError("hc_moe_overlap overlaps the int3packed Humming down GEMM with the quarter-owner combine: "
                                 "needs hc_moe_rs_owner and humming_moe int3packed, without hc_rs_stage2 fp8 / "
                                 "hc_rs_sib_wait / hc_pipe_tp4")
        if self.hc_mix_verify is not None:
            lo_hi = self.hc_mix_verify.split(":")
            if (len(lo_hi) != 2 or not all(v.isdigit() for v in lo_hi) or not 0 <= int(lo_hi[0]) < int(lo_hi[1])
                    or self.hc_mix_w8):
                raise ValueError(f"hc_mix_verify is '<lo>:<hi>' with 0 <= lo < hi and excludes hc_mix_w8, "
                                 f"got {self.hc_mix_verify!r}")
        if self.lmhead_triton is not None:
            lo_hi = self.lmhead_triton.split(":")
            if len(lo_hi) != 2 or not all(v.isdigit() for v in lo_hi) or not 0 <= int(lo_hi[0]) < int(lo_hi[1]):
                raise ValueError(f"lmhead_triton is '<lo>:<hi>' with 0 <= lo < hi, got {self.lmhead_triton!r}")
        if self.sharded_verify and (self.candidate_id not in SHARDED_VERIFY_CANDIDATES or self.dflash_k is None
                                    or not (self.dp_lm_head or self.attn_dp == 1)):
            raise ValueError("sharded_verify needs a SHARDED_VERIFY_CANDIDATES tree, DFlash, and the LM head vocab-sharded "
                             "over attention TP (dp_lm_head, or attn_dp 1)")
        if self.ple_int5 and self.candidate_id not in PLE_INT5_CANDIDATES:
            raise ValueError("ple_int5 needs a PLE_INT5_CANDIDATES tree (fork ARC_PLE_INT5)")
        if self.ple_nvfp4 and (self.ple_int5 or self.candidate_id not in PLE_INT5_CANDIDATES):
            raise ValueError("ple_nvfp4 excludes ple_int5 and needs a PLE_INT5_CANDIDATES tree (fork ARC_PLE_NVFP4)")
        if self.prefill_max_requests < 1 or (self.prefill_max_requests > 1 and self.candidate_id not in PLE_INT5_CANDIDATES):
            raise ValueError("prefill_max_requests is >= 1, and > 1 needs the unpadded PLE extend (PLE_INT5_CANDIDATES)")
        if self.prefill_delayer and (self.attn_dp != 2 or self.candidate_id not in PLE_INT5_CANDIDATES):
            raise ValueError("prefill_delayer pairs the two attention-DP schedulers' prefills: attn_dp 2, PLE_INT5_CANDIDATES")
        if (self.qsa_sm89_decode is not None or self.qsa_sm89_stages is not None) and (
                self.candidate_id not in QSA_SM89_CANDIDATES or self.qsa_sm89_decode not in (None, 0, 1, 3)
                or self.qsa_sm89_stages not in (None, 2, 3, 4)):
            raise ValueError(f"qsa_sm89_decode (0 / 1 / 3) and qsa_sm89_stages (2-4) need a QSA_SM89_CANDIDATES tree "
                             f"{QSA_SM89_CANDIDATES}")
        if (self.dflash_step_opt or self.hc_gate_precompute) and self.candidate_id not in B200OPT_CANDIDATES:
            raise ValueError(f"dflash_step_opt / hc_gate_precompute need a B200OPT_CANDIDATES tree {B200OPT_CANDIDATES}")
        if self.dflash_step_opt and self.dflash_k is None:
            raise ValueError("dflash_step_opt is a DFlash drafter option")
        if self.dflash_session_park and (self.candidate_id not in DFLASH_PARK_CANDIDATES or self.dflash_k is None):
            raise ValueError("dflash_session_park needs DFlash on a DFLASH_PARK_CANDIDATES tree (fork 56d78f62 / 06fd8d9c)")
        if self.dflash_shared_graph_pool and (self.candidate_id not in SHARED_GRAPH_POOL_CANDIDATES or self.dflash_k is None
                                              or not self.sparse_graphs):
            raise ValueError("dflash_shared_graph_pool needs DFlash with target CUDA graphs on a SHARED_GRAPH_POOL_CANDIDATES "
                             "tree (fork 8e3f880f)")
        if (self.mm_release_after_prefill or self.mm_lazy_media) and self.candidate_id not in MM_CANDIDATES:
            raise ValueError(f"mm_release_after_prefill / mm_lazy_media need an MM_CANDIDATES tree {MM_CANDIDATES}")
        if self.dflash_w4a8 is not None and (
                not isinstance(self.dflash_w4a8, str) or W4A8_SPEC.fullmatch(self.dflash_w4a8) is None
                or self.candidate_id not in W4A8_CANDIDATES or self.dflash_k is None
                or (self.humming_moe is None and self.humming_dense is None)):
            raise ValueError(f"dflash_w4a8 {self.dflash_w4a8!r}: <int4|uint4z>:<32|64|128>[:<all|mlp>], DFlash on a "
                             f"W4A8_CANDIDATES tree with the Humming wheel (humming_moe / humming_dense)")
        for name, allowed in (("dflash_merge", (False, True)), ("dflash_inject_kv", ("layers", "gemm")),
                              ("dflash_fuse_once", (False, True)), ("dflash_kv_dtype", ("bf16", "fp8_e4m3")),
                              ("dflash_ring_fp8_dot", (False, True))):
            value = getattr(self, name)
            if type(value) is not type(allowed[0]) or value not in allowed:
                raise ValueError(f"{name} {value!r}: one of {allowed}")
        plain_opts = sorted(name for name, default in PLAIN_DFLASH_OPTS.items() if getattr(self, name) != default)
        if plain_opts and (self.dflash_k is None or self.candidate_id != "arc-dense-vocab-tp4-shv"):
            raise ValueError(f"{plain_opts}: plan B drafter serving opt-ins, for DFlash on arc-dense-vocab-tp4-shv pinned to "
                             f"a PLAIN_DFLASH_OPT_TREES source with the plain drafter's spec (build checks both)")
        if self.dflash_ring_fp8_dot and self.dflash_kv_dtype != "fp8_e4m3":
            raise ValueError("dflash_ring_fp8_dot is the e4m3 ring pool's attention: it needs dflash_kv_dtype fp8_e4m3 "
                             "(fork ring_attention.attend raises on a BF16 pool)")
        if self.heap_diag is not None and (isinstance(self.heap_diag, bool) or not isinstance(self.heap_diag, int)
                                           or self.heap_diag <= 0 or self.candidate_id not in HEAPDIAG_CANDIDATES):
            raise ValueError(f"heap_diag {self.heap_diag!r}: seconds from init done to the second deep dump (a positive "
                             f"int) on a HEAPDIAG_CANDIDATES tree {HEAPDIAG_CANDIDATES}")
        if type(self.shm_gather) is not bool or (self.shm_gather and (self.candidate_id != "arc-dense-vocab-tp4-shv"
                                                                       or self.attn_dp != 2)):
            raise ValueError(f"shm_gather {self.shm_gather!r}: a bool, true only on arc-dense-vocab-tp4-shv pinned to a "
                             f"SHM_GATHER_TREES source with attn_dp 2 (its receipt follows the DP-pair batch gate)")
        if type(self.host_memory_trace) is not bool:
            raise ValueError(f"host_memory_trace must be a bool, got {self.host_memory_trace!r}")
        if self.malloc_trim_seconds is not None and (type(self.malloc_trim_seconds) is not int
                                                     or self.malloc_trim_seconds < 60 or not self.hostcomm):
            raise ValueError(f"malloc_trim_seconds {self.malloc_trim_seconds!r}: an int >= 60, a hostcomm hook option")
        if type(self.admission_reclaimable_file) is not bool or (self.admission_reclaimable_file
                                                                  and self.candidate_id not in DFLASH_CANDIDATES):
            raise ValueError(f"admission_reclaimable_file {self.admission_reclaimable_file!r}: a bool, on a fork tree")
        if self.inductor_compile_threads is not None and (type(self.inductor_compile_threads) is not int
                                                          or self.inductor_compile_threads < 1):
            raise ValueError(f"inductor_compile_threads {self.inductor_compile_threads!r}: an int >= 1")
        # the fork raises at startup otherwise: attention TP exactly 2 (attn_dp 2 of TP4), no native-MTP QSA draft
        # pool, block indices below 0xFFFF on the 16-bit hand-off (context + draft window <= 4 x 65534); layer-check
        # compares two ranks' top-k, equal only with deterministic top-k
        if self.qsa_index_shard is not None and (
                self.qsa_index_shard not in ("layer", "layer-check")
                or self.candidate_id not in QSA_INDEX_SHARD_CANDIDATES
                or self.attn_dp != 2 or self.mtp or self.dflash_k is None or self.context_length + 16 > 4 * 65534
                or (self.qsa_index_shard == "layer-check" and not self.deterministic)):
            raise ValueError(f"qsa_index_shard {self.qsa_index_shard!r}: 'layer' (or 'layer-check' with deterministic) "
                             f"on a QSA_INDEX_SHARD_CANDIDATES tree {QSA_INDEX_SHARD_CANDIDATES} with attn_dp 2, "
                             f"DFlash (no native MTP) and context <= 262120")
        if self.extend_nosync and self.candidate_id not in EXTEND_NOSYNC_CANDIDATES:
            raise ValueError("extend_nosync needs an EXTEND_NOSYNC_CANDIDATES tree (fork d2e4567b)")
        if self.host_private and not (self.hostcomm and (self.humming_moe or self.humming_dense or self.marlin_block_remap)):
            raise ValueError("host_private is a v16_patch switch: needs hostcomm with Humming or the marlin remap")
        if not self.prefill_delayer and (self.prefill_delayer_max_passes is not None
                                         or self.prefill_delayer_queue_ratio is not None
                                         or self.prefill_delayer_max_delay_ms is not None):
            raise ValueError("prefill_delayer_* options without prefill_delayer")
        if (self.hc_mix_w8 or self.embed_offload or self.hc_moe_rs or self.hc_ag_kind or self.hc_ag_dp
                or self.hc_ag_dp_check or self.hc_moe_rs_owner or self.hc_numa_node is not None
                or self.hc_ar2_kind or self.hc_moe_rs_fp8 or self.hc_sumlen or self.hc_moe_overlap or self.hc_mix_verify
                or self.lmhead_triton
                or self.moe_stats or self.mrope_decode_pin or self.spec_agree_device or self.fwd_timing
                or self.hc_pipe_ar2 or self.hc_pipe_tp4 or self.hc_ar2_blocks is not None or self.hc_fold_epoch
                or self.hc_ar2_p2p or self.hc_rs_stage2 != "bf16" or self.hc_rs_sib_wait
                or self.hc_ag_blocks is not None or self.gpu_monitor or self.hisparse_miss_stats) and not self.hostcomm:
            raise ValueError("hc_mix_w8 / embed_offload / hc_* / moe_stats are hostcomm-hook options")
        if self.spec_agree_device not in (False, True, "ab"):
            raise ValueError(f"spec_agree_device must be false, true or 'ab', got {self.spec_agree_device!r}")
        if self.spec_agree_device and self.dflash_k is None:
            raise ValueError("spec_agree_device replaces the distributed spec guards; it needs a DFlash profile")
        if self.hc_moe_rs and self.hc_moe_rs_owner:
            raise ValueError("hc_moe_rs (rows) and hc_moe_rs_owner are two modes of ARC_HC_MOE_RS; pick one")
        if self.hc_ag_dp and self.hc_ag_kind not in (None, "ag_fp8q"):
            raise ValueError("hc_ag_dp gathers bf16 (no hc_ag_kind) or FP8 with the payload (hc_ag_kind ag_fp8q)")
        for name in ("hc_pipe_ar2", "hc_pipe_tp4"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 8:
                raise ValueError(f"{name} must be an int 0..8, got {getattr(self, name)!r}")
        if (self.hc_pipe_ar2 or self.hc_ar2_blocks is not None) and self.attn_dp != 2:
            raise ValueError("hc_pipe_ar2 / hc_ar2_blocks are the attention-pair all-reduce's (attn_dp 2)")
        if self.hc_ar2_blocks is not None and (type(self.hc_ar2_blocks) is not int or not 1 <= self.hc_ar2_blocks <= 128):
            raise ValueError(f"hc_ar2_blocks must be an int 1..128, got {self.hc_ar2_blocks!r}")
        if self.hc_ag_blocks is not None and (type(self.hc_ag_blocks) is not int or not 1 <= self.hc_ag_blocks <= 128):
            raise ValueError(f"hc_ag_blocks must be an int 1..128, got {self.hc_ag_blocks!r}")
        if self.hc_ar2_p2p and (self.attn_dp != 2 or self.hc_ar2_kind != "ar1_fp8v" or self.hc_pipe_ar2):
            raise ValueError("hc_ar2_p2p is the attention pair's ar1_fp8v over peer access: attn_dp 2, hc_ar2_kind "
                             "ar1_fp8v, no hc_pipe_ar2")
        if self.hc_rs_stage2 not in ("bf16", "fp8"):
            raise ValueError(f"hc_rs_stage2 must be bf16 or fp8, got {self.hc_rs_stage2!r}")
        if (self.hc_rs_stage2 != "bf16" or self.hc_rs_sib_wait) and not (self.hc_moe_rs_owner and self.hc_moe_rs_fp8):
            raise ValueError("hc_rs_stage2 / hc_rs_sib_wait are options of the owner_fp8 combine (hc_moe_rs_owner + "
                             "hc_moe_rs_fp8)")
        for name in ("hc_fold_epoch", "hc_ar2_p2p", "hc_rs_sib_wait", "gpu_monitor", "hisparse_miss_stats"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if (self.gpu_monitor or self.hisparse_miss_stats) and not self.fwd_timing:
            raise ValueError("gpu_monitor / hisparse_miss_stats log per ARC_FWD_TIMING window: need fwd_timing")
        if (type(self.hc_ag_dp_check_every) is not int or self.hc_ag_dp_check_every < 1
                or (self.hc_ag_dp_check_every != 1 and not self.hc_ag_dp_check)):
            raise ValueError(f"hc_ag_dp_check_every is an int >= 1 and != 1 only with hc_ag_dp_check, "
                             f"got {self.hc_ag_dp_check_every!r}")
        if self.hc_ag_dp_check and not self.hc_ag_dp:
            raise ValueError("hc_ag_dp_check checks hc_ag_dp")
        if self.hc_numa_node is not None and (type(self.hc_numa_node) is not int or self.hc_numa_node < 0):
            raise ValueError("hc_numa_node must be a NUMA node index")
        if self.hc_local_slots and (not self.hostcomm or self.hc_numa_node is not None):
            raise ValueError("hc_local_slots is a hostcomm placement; it excludes hc_numa_node")
        if self.attn_dp not in (1, 2):
            raise ValueError(f"attn_dp must be 2 (attention DP2 x TP2) or 1 (pure TP4), got {self.attn_dp!r}")
        if self.attn_dp == 1:
            if self.candidate_id not in TP4_ATTENTION_CANDIDATES:
                raise ValueError(f"attn_dp 1 needs a tree with the ARC_HISPARSE_TP4 gates: {TP4_ATTENTION_CANDIDATES}")
            if self.dp_lm_head or self.hc_moe_rs or self.hc_moe_rs_owner or self.hc_ag_dp or self.hc_ag_dp_check:
                raise ValueError("dp_lm_head / hc_moe_rs* / hc_ag_dp* need attention DP (attn_dp 2)")
            if self.mtp:
                raise ValueError("attn_dp 1 runs DFlash or NoMTP; the native MTP draft was not ported to pure TP4")
        if self.dflash_compile is not None and (self.dflash_compile not in DFLASH_COMPILE_MODES or self.dflash_k is None):
            raise ValueError(f"dflash_compile is a DFlash option, one of {DFLASH_COMPILE_MODES}")
        if self.dflash_compile == "det" and self.candidate_id not in DFLASH_DET_CANDIDATES:
            raise ValueError(f"ARC_DFLASH_COMPILE=det exists only in {DFLASH_DET_CANDIDATES} (the others read it as off)")
        if self.draft_vocab is not None:
            if self.dflash_k is None or not (self.dp_lm_head or self.attn_dp == 1):
                raise ValueError("draft_vocab is a DFlash option for the dp-lm-head pairs or the pure-TP4 group "
                                 "(SubsetPairHead)")
            if self.candidate_id not in DRAFT_VOCAB_CANDIDATES:
                raise ValueError(f"draft_vocab needs a tree with SubsetPairHead: {DRAFT_VOCAB_CANDIDATES}")
            ids = json.loads((REPO / self.draft_vocab).read_text())["ids"]
            if not ids or len(set(ids)) != len(ids) or not all(type(i) is int and i >= 0 for i in ids):
                raise ValueError(f"draft_vocab {self.draft_vocab}: ids must be unique non-negative integers")
        if self.target_vocab is not None:
            if self.candidate_id not in TARGET_VOCAB_CANDIDATES or self.draft_vocab is None or not self.sharded_verify:
                raise ValueError(f"target_vocab needs {TARGET_VOCAB_CANDIDATES}, draft_vocab and sharded_verify")
            ids = json.loads((REPO / self.target_vocab).read_text())["ids"]
            if (not ids or len(set(ids)) != len(ids)
                    or not all(type(i) is int and 0 <= i < TARGET_VOCAB_SIZE for i in ids)):
                raise ValueError(f"target_vocab {self.target_vocab}: ids must be unique integers in [0, {TARGET_VOCAB_SIZE})")
            attn_tp = 4 // self.attn_dp
            if len(ids) % (64 * attn_tp):
                raise ValueError(f"target_vocab {self.target_vocab}: {len(ids)} ids do not split into {attn_tp} "
                                 "attention-TP ranks x a multiple of 64 rows")
            kept = set(ids)
            missing = [i for i in TOKENIZER_ADDED_IDS if i not in kept]
            if missing:
                raise ValueError(f"target_vocab {self.target_vocab}: added tokens {missing} are not kept")
            off = set(json.loads((REPO / self.draft_vocab).read_text())["ids"]) - kept
            if off:
                raise ValueError(f"target_vocab {self.target_vocab}: {len(off)} draft_vocab ids are not kept")
        if self.spec_trace is not None and self.spec_trace not in SPEC_TRACE_MODES:
            raise ValueError(f"spec_trace must be one of {SPEC_TRACE_MODES}")
        if self.spec_trace and not (self.dflash_k is not None and (self.marlin_block_remap or self.humming_moe)):
            raise ValueError("spec_trace lives in v16_patch, which the DFlash profile loads with the remap or Humming")
        if self.hc_debug is not None and (self.hc_debug not in HC_DEBUG_MODES
                                          or not (self.dflash_k is not None and (self.marlin_block_remap or self.humming_moe))):
            raise ValueError(f"hc_debug is a v16_patch option on the DFlash profile, one of {HC_DEBUG_MODES}")
        if self.hc_debug == "zero_freed" and self.humming_moe is None:
            raise ValueError("hc_debug zero_freed zeroes the Marlin experts the Humming MoE frees: needs humming_moe")
        if self.hc_ag_anydtype and not self.hostcomm:
            raise ValueError("hc_ag_anydtype is a hostcomm-hook option")
        if type(self.hc_slot_mb) is not int or self.hc_slot_mb <= 0:
            raise ValueError("hostcomm slot MiB must be a positive integer")
        if self.vision_offload and not self.hostcomm:
            raise ValueError("vision offload is installed by the hostcomm hook")
        if self.harness == "codex":
            if self.vision_offload:
                raise ValueError("the codex harness sends board images in its first message; vision must stay on GPU")
            if self.candidate_id not in DFLASH_CANDIDATES:
                raise ValueError("the codex harness is wired for the sglang-arc-2 arc-main candidate")
        if self.dp_lm_head and self.candidate_id not in DFLASH_CANDIDATES:
            raise ValueError("--enable-dp-lm-head was measured only on sglang-arc-2 arc-main")
        if self.context_length < 131072 or self.context_length % 64 or self.context_length > self.logical_tokens:
            raise ValueError("context length must be >=131072, page64 aligned and within per-DP logical tokens")
        if self.think_budget is not None and (type(self.think_budget) is not int or self.dflash_k is None
                                              or not 0 < self.think_budget < self.context_length):
            raise ValueError("think budget must be a positive integer below the context length, on DFlash (it acts in "
                             "the EAGLE verify)")
        if self.think_budget is not None and self.sharded_verify and not (
                self.hostcomm and (self.humming_moe or self.humming_dense or self.marlin_block_remap)):
            # 10-31: on vocab-shard verify logits the fork's apply indexes out of bounds; v16_patch forces on the
            # gathered rows instead (_patch_think_budget)
            raise ValueError("think budget on sharded verify rides on the v16_patch sitecustomize: needs hostcomm with "
                             "Humming or the marlin remap")
        if self.swap_in_block_size is not None and (type(self.swap_in_block_size) is not int or self.swap_in_block_size <= 0):
            raise ValueError("swap-in block size must be a positive integer")
        if not math.isfinite(self.min_free_gib) or self.min_free_gib < 0:
            raise ValueError("min free GiB must be finite and nonnegative")
        if self.vision_peak_images <= 0 or self.vision_peak_concurrency <= 0 or not self.long_prefill_lengths or any(
                not n.isdigit() or not 0 < int(n) < self.context_length for n in self.long_prefill_lengths.split(",")):
            raise ValueError("startup gate sizes must be positive (long prefills below the context length)")
        if self.gates_only and self.capacity_phase:
            raise ValueError("gates-only runs no capacity phase")
        if self.harness_ctx_cap is not None and (type(self.harness_ctx_cap) is not int or self.harness_ctx_cap <= 0):
            raise ValueError("harness ctx cap must be a positive integer")
        if self.candidate_id in KNOWN_BAD_RS_CANDIDATES:
            raise ValueError("MTP candidate has known draft-probability relay defect; use r9-r2+h39")
        if type(self.mtp_steps) is not int or self.mtp_steps not in (1, 3):
            raise ValueError("MTP steps must be explicitly 1 or 3")
        if self.mtp_steps == 1 and self.candidate_id != "r9-r2":
            raise ValueError("NEXTN1 requires explicit r9-r2+h39 candidate")
        if self.dflash_k is not None:
            pass  # checked above
        elif self.nomtp_control:
            if self.candidate_id not in NOMTP_CANDIDATES:
                raise ValueError("NoMTP control requires explicit repaired r9-r2 (or arc-main) candidate")
            if any((self.mtp, self.draft_fp8_direct, self.fold_in_verify, self.mtp_rejection_sampling,
                    self.native_mtp_host_kv, self.spec_phase_packed)) or self.mtp_steps != 3:
                raise ValueError("NoMTP control excludes MTP, draft, fold, RS, native host, packed and nondefault MTP steps")
            if self.hot_blocks < 512 or self.hot_blocks % 16:
                raise ValueError("NoMTP hot blocks must cover top512 and align16")
        elif self.candidate_id is not None:
            if self.candidate_id != "r9-r2" or not self.mtp or not self.mtp_rejection_sampling:
                raise ValueError("candidate requires explicit MTP and rejection sampling for original ARC2 sampling")
            minimum_hot = 512 * (self.mtp_steps + 1)
            if self.hot_blocks < minimum_hot or self.hot_blocks % 16:
                raise ValueError(f"MTP hot blocks must cover verify{self.mtp_steps + 1} union >={minimum_hot} and align16")
        elif self.mtp or any((self.sparse_graphs, self.draft_fp8_direct, self.fold_in_verify,
                             self.mtp_rejection_sampling, self.native_mtp_host_kv, self.cpu_affinity,
                             self.spec_phase_packed, not self.debug_checks)) or self.hot_blocks != 512 or self.prefill_chunk != 1024:
            raise ValueError("experimental profile requires explicit candidate ID")
        if self.candidate_id in DFLASH_CANDIDATES and self.dflash_k is None and not self.nomtp_control:
            raise ValueError("a sglang-arc-2 candidate runs either DFlash (--dflash-k) or the NoMTP control")
        if self.candidate_id is not None and (self.prefill_chunk <= 0 or self.prefill_chunk % 128):
            raise ValueError("candidate prefill chunk must align128")
        if self.spec_phase_packed and self.candidate_id != "r9-r2":
            raise ValueError("phase-packed agreement requires explicit r9-r2+h39 candidate")
        if self.running <= 0 or self.running % 2:
            raise ValueError("running is a positive even global DP2 request limit")
        if self.logical_tokens < 131072 or self.logical_tokens % 64:
            raise ValueError("logical tokens are per DP, >=131072 and page64 aligned")
        if not math.isfinite(self.mem_fraction) or not 0 < self.mem_fraction < 1:
            raise ValueError("mem fraction must be finite and between0and1")
        if not math.isfinite(self.host_reserve_gib) or self.host_reserve_gib < 0:
            raise ValueError("whole-node host reserve must be finite and nonnegative")
        if self.validation_limit <= 0:
            raise ValueError("SaveRun validation limit must be positive")
        if self.validation_repeat < 1:
            raise ValueError("SaveRun validation repeat must be at least1")
        if self.capacity_phase and (self.running != 64 or self.logical_tokens != 3250048):
            raise ValueError("canonical long capacity phase requires global C64 and per-DP L3250048")
        for minutes in (self.saverun_minutes, self.rerun_minutes):
            if minutes is not None and (not math.isfinite(minutes) or minutes <= 0 or minutes > 720):
                raise ValueError("explicit notebook time budget must be in(0,720]minutes")


def relative(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path == PurePosixPath(".") or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe asset path: {value!r}")


def digest(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("manifest SHA256/digest must be64 lowercase hex characters")


def check_manifest(manifest: dict) -> None:
    if manifest["schema_version"] != 1:
        raise ValueError("asset manifest schema_version must be1")
    expected = {"runtime_dataset": RUNTIME, "zstd_dataset": ZSTD}
    for key, value in expected.items():
        if manifest[key] != value:
            raise ValueError(f"asset identity differs: {key}")
    if manifest["model_source"] not in MODELS:
        raise ValueError(f"asset model {manifest['model_source']!r} is none of {MODELS}")
    if not re.fullmatch(r"threerabbits/[a-z0-9-]+", manifest["source_dataset"]):
        raise ValueError("source dataset must be an explicit owned private asset ref")
    setup, payload, source, runtime = (manifest[key] for key in ("runtime_setup", "payload", "source", "runtime_archive"))
    for entry, filename in ((setup, "file"), (payload, "archive"), (source, "archive"), (runtime, "file")):
        relative(entry[filename]); digest(entry["sha256"])
    if setup["file"] != "runtime_setup.py" or payload["launcher"] != LAUNCHER:
        raise ValueError("asset must contain the agreed installer and h28 launcher")
    digest(payload["launcher_sha256"])
    relative(source["manifest"]); digest(source["manifest_sha256"]); digest(source["digest"])
    allowed_sources = {(SOURCE_DIGEST, 4901, SOURCE_MANIFEST_SHA),
                       (FAST_SOURCE_DIGEST, 4901, FAST_SOURCE_MANIFEST_SHA)}
    candidate = None
    if "candidate" in manifest:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_candidate_setup", Path(__file__).with_name("runtime_setup.py"))
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)
        candidate = setup.candidate_identity(manifest)
    if candidate is None and (source["digest"], source["files"], source["manifest_sha256"]) not in allowed_sources:
        raise ValueError("source must pin h28 or its startup-only h30 candidate")
    if candidate is None and not source["archive"].endswith(source["digest"][:12] + ".tar.zst"):
        raise ValueError("source archive filename must pin its whole-tree digest")
    for entry in (payload, source, runtime):
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError("asset byte counts must be positive integers")
    if type(source["files"]) is not int or source["files"] <= 0:
        raise ValueError("source tree file count must be positive")
    if runtime["sha256"] != RUNTIME_SHA or runtime["bytes"] != 3496285060:
        raise ValueError("not the pinned0520v2 runtime archive")
    digest(runtime["provenance_sha256"])
    if runtime["provenance_sha256"] != PROVENANCE_SHA:
        raise ValueError("not the pinned0520v2 runtime provenance")
    if not isinstance(manifest["model_identity"], dict) or not manifest["model_identity"]:
        raise ValueError("exact model identity must be supplied")
    if "live_status" in manifest:
        relative(manifest["live_status"]["file"]); digest(manifest["live_status"]["sha256"])
        if manifest["live_status"]["file"] != "work/qwen38fn_l4/kaggle/live_status.py":
            raise ValueError("live status observer must be the unit-owned payload file")
    if "compiled_cache" in manifest:
        cache = manifest["compiled_cache"]
        if not re.fullmatch(r"threerabbits/[a-z0-9-]+", cache["dataset"]):
            raise ValueError("compiled cache must be an explicit owned private asset")
        digest(cache["manifest_sha256"])
        helper = manifest["compiled_cache_helper"]
        relative(helper["file"])
        digest(helper["sha256"])
        if helper["file"] != "work/qwen38fn_l4/kaggle/compiled_cache.py":
            raise ValueError("compiled cache helper must be the unit-owned payload file")


# A new kernel gets Kaggle's latest image (Python 3.13 since late 09); the runtime needs 3.12.13 (v15 v1 died in
# runtime_setup). This is the image the h28 v13 kernel stays pinned to.
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


# ---------------------------------------------------------------------------------------------------------------
# v16 build-time HiSparse capacity check: sglang-arc-2 srt/mem_cache/qsa_hisparse_capacity.py plan_qsa_hisparse with
# the L4 attention-TP2 constants read off the server logs ("QSA HiSparse sizing: ... KV 3456 B/token, index 396
# B/token, landing zone 37767168 B" at C=131072; device rows = prefill + R*(B+16)*4 gives 4-token blocks).
L4_KV_BYTES_PER_TOKEN = 3456          # 12 QSA layers, NVFP4 K/V, one KV head per rank
L4_INDEX_BYTES_PER_TOKEN = 396        # FP8 index (ARC_QSA_INDEX_FP8=1)
L4_INDEX_BYTES_PER_TOKEN_SHARDED = 198  # qsa_index_shard layer: 6 of the 12 layers' index per rank (logged as index 198)
QSA_BLOCK_TOKENS, QSA_PAGE_TOKENS, QSA_LAYERS, QSA_TOPK, QSA_MAX_HOT = 4, 64, 12, 512, 32752
SM89_SMEM_OPTIN = 101376
#: KV budget V per rank as the server logged it ("QSA HiSparse sizing: budget X GiB", two decimals, so the lower
#: edge is used). V does not depend on logical tokens or context (those are spent inside it); it depends on what
#: is resident before the pool: running (mamba slots), memory fraction, drafter, hostcomm slots, FP8 dense, lm_head.
MEASURED_BUDGETS = {
    # (running, mem_fraction, dflash_k, online_fp8, hostcomm slot MiB or None, dp_lm_head, vision_offload)
    (32, 0.97, 4, None, None, False, False): (1.38, "Vast Utah 10-03 dfl-k4c32 server.log (arc-forge l01 tree)"),
    (48, 0.97, 4, "gdn_in,gdn_out,qsa_qkv,qsa_o", 32, True, False):
        (2.36, "Runpod 10-04 dfl-dplm48 server.log (fork 5bdc95de, l4-dflash-opt)"),
    (48, 0.97, 4, "gdn_in,gdn_out,qsa_qkv,qsa_o", 21, True, False):
        (2.36, "Runpod 10-04 dfl-v16c48 server.log (arc/main dfd2fd8f, L 2100032: 3568 fit, 0.57 GiB after graphs)"),
    # K=8 (hot 4096): C32 L 1500032 -> 7920 fit, 0.76 GiB after all graphs, image/long/smoke gates pass;
    # C40 L 1800000 -> 5200 fit but verify graph 0.61 + draft graph 0.22 GiB leave 0.03 GiB: the image gate OOMed it.
    (32, 0.97, 8, "gdn_in,gdn_out,qsa_qkv,qsa_o", 21, True, False):
        (2.67, "Runpod 10-04 dfl-v16k8c32 server.log (arc/main dfd2fd8f, K8 hot 4096, L 1500032: 0.76 GiB after graphs)"),
    # K4 C32 HOT 4096 L 1500032 (Runpod 10-05 k4h4096): 8208 fit, 1.21 GiB after graphs, 0.60-0.64 GiB free under
    # full codex load (16/DP) for 59 min, 0 CUDA errors, decode 543 tok/s (K8 arms 290-310 and crashing).
    (32, 0.97, 4, "gdn_in,gdn_out,qsa_qkv,qsa_o", 21, True, False):
        (2.73, "Runpod 10-05 k4h4096 server.log (arc/main dfd2fd8f, K4 hot 4096, L 1500032: 8208 fit)"),
    (40, 0.97, 8, "gdn_in,gdn_out,qsa_qkv,qsa_o", 21, True, False):
        (2.49, "Runpod 10-04 dfl-v16k8c40 server.log (arc/main dfd2fd8f, K8; 0.03 GiB after graphs, image OOM, not deployable)"),
    # C56 starts (2592 fit at L 2100032) but leaves 0.15-0.17 GiB after graphs: the 20-board image gate OOMed it.
    (56, 0.97, 4, "gdn_in,gdn_out,qsa_qkv,qsa_o", 21, True, False):
        (2.20, "Runpod 10-04 dfl-v16c56 server.log (arc/main dfd2fd8f; image request OOM, not deployable)"),
}


def swap_in_smem_bytes(hot_blocks: int, top_k: int = QSA_TOPK) -> int:
    chunks = -(-hot_blocks // 32)
    return 4 * (top_k + 2 * (chunks + 1) + 2 * top_k + 2 + chunks) + 2 * (hot_blocks + 2 * top_k)


def index_bytes_per_token(profile: "Profile") -> int:
    """QSA index bytes per logical token on one rank: halved under qsa_index_shard layer (layer-check keeps it all)."""
    return L4_INDEX_BYTES_PER_TOKEN_SHARDED if profile.qsa_index_shard == "layer" else L4_INDEX_BYTES_PER_TOKEN


def hisparse_plan(*, running: int, logical_tokens: int, context: int, prefill_chunk: int, budget_bytes: int,
                  attn_dp: int = 2, index_bytes: int = L4_INDEX_BYTES_PER_TOKEN) -> dict:
    """Per-rank plan of one attention-DP pool (R = running / attn_dp, chunk = global prefill chunk / attn_dp; pure TP4
    replicates the KV heads, so each rank holds the same per-token bytes for all R requests). index_bytes: the QSA
    index bytes per logical token this rank holds (index_bytes_per_token: halved under qsa_index_shard)."""
    r, c, kv, bt = running // attn_dp, context, L4_KV_BYTES_PER_TOKEN, QSA_BLOCK_TOKENS
    page_blocks = QSA_PAGE_TOKENS // bt
    ctx_blocks = -(-c // bt)
    round_up = lambda x, m: -(-x // m) * m  # noqa: E731
    prefill = round_up(c + prefill_chunk // attn_dp, QSA_PAGE_TOKENS)
    landing = kv // QSA_LAYERS * (round_up(c, QSA_PAGE_TOKENS) + QSA_PAGE_TOKENS)
    fixed = (logical_tokens * index_bytes + prefill * kv + logical_tokens // bt * 8
             + r * (ctx_blocks + page_blocks) * 8 + landing)
    per_slot = r * (bt * kv + QSA_LAYERS * (4 + 4 + 2))
    if budget_bytes <= fixed:
        raise ValueError(f"HiSparse budget {budget_bytes} B does not cover index + prefill area + maps ({fixed} B) "
                         f"for R={r}/DP, L={logical_tokens}, C={c}")
    fit = min(((budget_bytes - fixed) // per_slot - page_blocks) // 16 * 16, QSA_MAX_HOT)
    return {"per_dp_running": r, "prefill_tokens": prefill, "landing_zone_bytes": landing, "fixed_bytes": fixed,
            "per_slot_bytes": per_slot, "hot_blocks_fit": fit, "budget_bytes": budget_bytes,
            "index_bytes_per_token": index_bytes}


def capacity_check(profile: "Profile", budget_gib: float | None) -> dict:
    """Refuse a v16 build whose hot blocks cannot cover the DFlash verify union or do not fit the measured budget."""
    key = (profile.running, profile.mem_fraction, profile.dflash_k, profile.online_fp8,
           profile.hc_slot_mb if profile.hostcomm else None, profile.dp_lm_head, profile.vision_offload)
    if budget_gib is None:
        if key not in MEASURED_BUDGETS or profile.attn_dp != 2:
            raise ValueError(f"no measured HiSparse budget for {key}; start this shape once and pass the logged "
                             "'QSA HiSparse sizing: budget' value as --hisparse-budget-gib")
        budget_gib, source = MEASURED_BUDGETS[key]
    else:
        source = "--hisparse-budget-gib"
    budget = int((budget_gib - 0.005) * (1 << 30))
    plan = hisparse_plan(running=profile.running, logical_tokens=profile.logical_tokens,
                         context=profile.context_length, prefill_chunk=profile.prefill_chunk, budget_bytes=budget,
                         attn_dp=profile.attn_dp, index_bytes=index_bytes_per_token(profile))
    need = QSA_TOPK * (profile.dflash_k or 1)
    if profile.hot_blocks < need:
        raise ValueError(f"hot blocks {profile.hot_blocks} below the verify union {need} (512 x K)")
    if profile.hot_blocks > plan["hot_blocks_fit"]:
        raise ValueError(f"hot blocks {profile.hot_blocks} do not fit: {plan['hot_blocks_fit']} fit in the "
                         f"{budget_gib} GiB budget ({source}) at R={profile.running}, L={profile.logical_tokens}, "
                         f"C={profile.context_length}")
    if swap_in_smem_bytes(profile.hot_blocks) > SM89_SMEM_OPTIN:
        raise ValueError(f"hot blocks {profile.hot_blocks} exceed the sm_89 swap-in shared-memory limit")
    return {**plan, "budget_gib": budget_gib, "budget_source": source, "hot_blocks": profile.hot_blocks,
            "verify_union_blocks": need}


def load_dflash_spec(path: Path) -> dict:
    """A MoE drafter pins its keep set and per-layer expert count; a dense one (config ffn_type dense_shared, exp/dense-dflash)
    has no experts to keep and pins its FFN width instead ("dense": the export's dense_intermediate_size). The plain
    drafter (fork c8500d9fe) is a dense spec that also pins "architecture", exactly PLAIN_DFLASH_ARCHITECTURE."""
    spec = json.loads(path.read_text())
    moe = {"dataset", "export_dir", "keep", "experts_kept", "files_sha256", "source"}
    dense = {"dataset", "export_dir", "dense", "files_sha256", "source"}
    plain = dense | {"architecture"}
    if set(spec) not in (moe, dense, plain):
        raise ValueError(f"{path}: DFlash spec needs exactly {sorted(moe)} (MoE), {sorted(dense)} (dense_shared) or "
                         f"{sorted(plain)} (plain)")
    if "architecture" in spec and spec["architecture"] != PLAIN_DFLASH_ARCHITECTURE:
        raise ValueError(f"{path}: architecture {spec['architecture']!r}; the plain drafter is exactly "
                         f"{PLAIN_DFLASH_ARCHITECTURE}")
    if "dense" in spec and (type(spec["dense"]) is not int or spec["dense"] <= 0):
        raise ValueError(f"{path}: dense is the export's dense_intermediate_size (a positive integer)")
    if not re.fullmatch(r"threerabbits/[a-z0-9-]+", spec["dataset"]):
        raise ValueError("DFlash drafter must be an owned private dataset")
    for name in ("config.json", "initialization_manifest.json") + (() if "dense" in spec else (spec["keep"],)):
        if name not in spec["files_sha256"]:
            raise ValueError(f"DFlash spec does not pin {name}")
    for name, value in spec["files_sha256"].items():
        relative(name)
        digest(value)
    if spec["export_dir"]:
        relative(spec["export_dir"])
    return spec


#: What the codex harness needs at run time (work/arc_code/arc2/run_arc2.py and the arc3 arm it reuses); globbed
#: here, then every listed required file must exist. kaggle_run.py and its config are the agreed entry point
#: (10-12: kaggle_run.py runs run_arc2_seq -- s0 + s2 + refiner -- on top of run_arc2.solve).
CODEX_HARNESS_GLOBS = ("work/arc_code/arc2/*.py", "work/arc_code/arc2/*.sh", "work/arc_code/arc2/configs/*.json",
                       "work/arc_code/arc2/priors/*.json",   # x39: the estimator's priors (run_arc2_seq loads them from HERE)
                       "work/arc_code/arc2/prompts/**/*.md", "work/arc_code/arc2/*_TOOLS.md", "work/arc_code/arms/*.py", "work/arc_code/driver_files/**/*",
                       "src/arcforge/arc_code/*.py")
CODEX_HARNESS_FILES = ("work/arc_code/arc2/kaggle_run.py", "work/arc_code/arc2/run_arc2.py",
                       "work/arc_code/arc2/run_arc2_seq.py", "work/arc_code/arc2/image_check.py",
                       "work/arc_code/arc2/harvest_sandbox.py", "work/arc_code/arc2/sandbox_runner.py",
                       "work/arc_code/arc2/prompts/v3refine/brief.md", "work/arc_code/arc2/model_proxy.py", "work/arc_code/arc2/arc2_render.py",
                       "work/arc_code/arc2/check.py", "work/arc_code/arc2/arc.py", "work/arc_code/arc2/fence_arc2.sh",
                       "work/arc_code/arc2/netns_inner.sh", "work/arc_code/posix_fence.py", "work/arc_code/fence_posix.sh",
                       "work/arc_code/jail.sh", "work/arc_code/prompts.py", "work/arc_code/arms/__init__.py",
                       "work/arc_code/arms/codex.py", "src/arcforge/__init__.py", "src/arcforge/arc_code/bridge.py")
V16_TOOL_FILES = ("work/qwen38fn_l4/kaggle/v16_gates.py", "work/qwen38fn_l4/kaggle/v16_patch/sitecustomize.py")


def harness_bundle(root: Path, codex: bool, codex_config: str | None) -> tuple[bytes, dict, dict]:
    """Deterministic tar.gz of the carried files (mtime/owner zeroed, gzip mtime 0) and their sha256 roster."""
    import gzip
    import io
    import subprocess
    import tarfile
    names = set(V16_TOOL_FILES)
    if codex:
        names |= set(CODEX_HARNESS_FILES) | {codex_config}
        for pattern in CODEX_HARNESS_GLOBS:
            names |= {p.relative_to(root).as_posix() for p in root.glob(pattern)
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    missing = sorted(name for name in names if not (root / name).is_file())
    if missing:
        raise ValueError(f"carried files missing under {root}: {missing}")
    names = sorted(names)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in names:
                data = (root / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid, info.uname, info.gname = len(data), 0, 0, 0, "", ""
                info.mode = 0o755 if (root / name).stat().st_mode & 0o111 else 0o644
                archive.addfile(info, io.BytesIO(data))
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    git = {"head": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty": subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *names],
                                   capture_output=True, text=True).stdout.strip().splitlines()}
    return buffer.getvalue(), files, git


def kernel_metadata(manifest: dict, kernel_id: str, title: str, extra_datasets: tuple = (), extra_models: tuple = ()) -> dict:
    if not re.fullmatch(r"threerabbits/[a-z0-9-]+", kernel_id):
        raise ValueError("kernel id must be an explicit owned slug")
    if kernel_id == "threerabbits/arc2-qwen38fn-reap320-l4x4-debug":
        raise ValueError("use a new private kernel id; preserve the historical debug notebook")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(title) < 5 or kernel_id.split("/")[1] != slug:
        raise ValueError("title must resolve to the requested kernel slug")
    datasets = [RUNTIME, ZSTD, manifest["source_dataset"]]
    if "compiled_cache" in manifest:
        datasets.append(manifest["compiled_cache"]["dataset"])
    datasets += [ref for ref in extra_datasets if ref not in datasets]
    return {"id": kernel_id, "title": title, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaL4", "competition_sources": [COMPETITION],
            "dataset_sources": datasets, "kernel_sources": [],
            "model_sources": [manifest["model_source"], *extra_models], "docker_image": DOCKER_IMAGE}


COMMON = r'''
import hashlib, importlib.util, json, os, pathlib, platform, shutil, subprocess, sys, threading, time, traceback
NOTEBOOK_STARTED = time.monotonic()
NOTEBOOK_STARTED_UNIX = time.time()
# 10-12 codex2 v1: RUNNING 2.5 h with not one log line (the stream API returned 500 throughout); a flushed first line
# separates "the notebook never ran" from "it ran and went quiet" on the next attempt.
print("h28: notebook started " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(NOTEBOOK_STARTED_UNIX)), flush=True)

WORK = pathlib.Path("/tmp/qwen38fn_l4")
OUT = pathlib.Path("/kaggle/working/qwen38fn_l4")
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def source_asset(ref):
    owner, slug = ref.split("/")
    candidates = [pathlib.Path("/kaggle/input/datasets") / owner / slug,
                  pathlib.Path("/kaggle/input") / slug]
    found = {p.resolve() for p in candidates if p.is_dir()}
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one mounted source asset {ref}, got {found}")
    return found.pop()

def competition_file(name):
    candidates = [pathlib.Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2") / name,
                  pathlib.Path("/kaggle/input/arc-prize-2026-arc-agi-2") / name]
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) != 1:
        raise RuntimeError(f"expected exact ARC2 input {name}, got {found}")
    return found.pop()

write_json(OUT / "requested_profile.json", CONFIG)
write_json(OUT / "asset_manifest.json", ASSET_MANIFEST)
'''

# v16 (arc-main): the carried bundle (codex harness + v16 gates), the drafter read-ahead and the submission guarantee.
# Kept outside WORK: runtime_setup.install insists on a fresh WORK directory.
V16_COMMON = r'''
import base64, io, tarfile
V16_DIR = pathlib.Path("/tmp/qwen38fn_l4_v16")
V16_DIR.mkdir(parents=True, exist_ok=True)
BUNDLE_ROOT = V16_DIR / "bundle"

def unpack_bundle():
    data = base64.b64decode(BUNDLE_B64)
    if hashlib.sha256(data).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError("carried bundle differs from its build receipt")
    BUNDLE_ROOT.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(BUNDLE_ROOT, filter="data")
    found = {p.relative_to(BUNDLE_ROOT).as_posix(): sha256(p) for p in BUNDLE_ROOT.rglob("*") if p.is_file()}
    if found != BUNDLE_FILES:
        raise RuntimeError("carried bundle file roster/hash differs from the build receipt")
    write_json(OUT / "bundle_receipt.json", {"sha256": BUNDLE_SHA256, "files": BUNDLE_FILES, "git": BUNDLE_GIT})

unpack_bundle()
SERVER_LOG_SUMMARY = load_module(BUNDLE_ROOT / "work/qwen38fn_l4/kaggle/v16_gates.py", "arcforge_v16_gates").ServerLogSummary

class DrafterRead(threading.Thread):
    """sha256 of every pinned drafter file, read while the runtime installs (it also warms the page cache)."""
    def __init__(self, root, names):
        super().__init__(daemon=True)
        self.root, self.names, self.result, self.error = root, names, None, None
        self.started_at = time.monotonic()
    def run(self):
        try:
            self.result = {name: sha256(self.root / name) for name in self.names}
            self.seconds = time.monotonic() - self.started_at
        except BaseException as exc:
            self.error = exc
    def join(self):
        super().join()
        if self.error is not None:
            raise RuntimeError(f"DFlash drafter read failed: {self.error}") from self.error
        print(f"h28: DFlash drafter files hashed in {self.seconds:.0f}s", flush=True)
        return self.result

if CONFIG.get("dflash_asset"):
    DFLASH_ROOT = source_asset(CONFIG["dflash_asset"]["dataset"])
    DFLASH_PREFETCH = DrafterRead(DFLASH_ROOT / CONFIG["dflash_asset"]["export_dir"], sorted(CONFIG["dflash_asset"]["files_sha256"]))
    DFLASH_PREFETCH.start()

def repeated_roster(roster, repeat):
    """SaveRun load test: `repeat` copies of a {task_id: value} roster, the copies under distinct IDs (task-rN) so the
    harness runs and scores each independently (the rerun roster is 240 tasks, the public evaluation 120)."""
    return {**roster, **{f"{task_id}-r{copy}": json.loads(json.dumps(value))
                         for copy in range(1, repeat) for task_id, value in roster.items()}}

def codex_deadline(config, rerun, started):
    """The codex harness deadline: the profile's minutes after the notebook started, capped so that the notebook ends by
    end_by_unix (end_by_margin_seconds holds the harness's grace kill and the steps after it). A notebook that starts
    at or after end_by_unix is no queued submission (Kaggle re-running it after the close): no cap."""
    stamp = lambda unix: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(unix))  # noqa: E731
    planned = started + config["rerun_minutes" if rerun else "saverun_minutes"] * 60
    cap = config["end_by_unix"] - config["end_by_margin_seconds"]
    applies = started < config["end_by_unix"]
    deadline = min(planned, cap) if applies else planned
    return deadline, {"started": stamp(started), "planned_deadline": stamp(planned), "end_by": stamp(config["end_by_unix"]),
                      "cap": stamp(cap), "deadline": stamp(deadline), "applies": applies, "capped": deadline < planned,
                      "minutes": round((deadline - started) / 60, 1)}

def end_by_gate(deadline, needed_seconds, stage):
    """Too little time before the (capped) deadline: raise, so the guard writes the placeholder submission (a rerun is
    scored, not an error) instead of booting a server or starting a harness that cannot finish."""
    left = deadline - time.time()
    if left < needed_seconds:
        raise TimeoutError(f"end-by cutoff at {stage}: {left / 60:.1f} min left before the harness deadline "
                           f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(deadline))}, "
                           f"{needed_seconds / 60:.0f} min needed")

def selected_challenges(rerun, config):
    if rerun:
        return json.loads(competition_file("arc-agi_test_challenges.json").read_text())
    challenges = json.loads(competition_file("arc-agi_evaluation_challenges.json").read_text())
    selected = dict(list(repeated_roster(challenges, config["validation_repeat"]).items())[:config["validation_limit"]])
    if len(selected) != config["validation_limit"]:
        raise RuntimeError(f"SaveRun roster holds {len(selected)} tasks (validation_repeat "
                           f"{config['validation_repeat']}), validation_limit is {config['validation_limit']}")
    return selected

def ensure_submission(challenges, reason):
    """/kaggle/working/submission.json must exist and cover every task: keep the harness's valid attempts, fill every
    missing or malformed task with a placeholder, and say so loudly."""
    path = OUT.parent / "submission.json"
    existing, problem = {}, None
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, dict):
                problem, existing = "submission.json is not an object", {}
        except json.JSONDecodeError as exc:
            problem = f"submission.json is not JSON: {exc}"
    else:
        problem = "submission.json missing"
    def valid(task_id, value):
        return (isinstance(value, list) and len(value) == len(challenges[task_id]["test"]) and all(
            isinstance(row, dict) and all(isinstance(row.get(k), list) and row[k] and all(isinstance(r, list) and r for r in row[k])
                                          for k in ("attempt_1", "attempt_2")) for row in value))
    filled = [task_id for task_id in challenges if not valid(task_id, existing.get(task_id))]
    final = {task_id: existing[task_id] if task_id not in filled else
             [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in challenges[task_id]["test"]] for task_id in challenges}
    if filled or problem:
        print("!" * 100 + f"\nSUBMISSION FALLBACK ({reason}): {problem or ''} placeholder for {len(filled)} of "
              f"{len(challenges)} tasks: {filled[:20]}\n" + "!" * 100, flush=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(final))
    temporary.replace(path)
    receipt = {"reason": reason, "problem": problem, "tasks": len(challenges), "placeholder_tasks": filled,
               "harness_tasks_kept": len(challenges) - len(filled), "sha256": sha256(path)}
    write_json(OUT / "submission_receipt.json", receipt)
    return receipt
'''

V16_GUARD_HEAD = r'''
V16_RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in ("1", "true")
try:
'''

V16_GUARD_TAIL = r'''
except BaseException as V16_ERROR:
    traceback.print_exc()
    print("!" * 100 + f"\nNOTEBOOK FAILED: {type(V16_ERROR).__name__}: {V16_ERROR}\n" + "!" * 100, flush=True)
    V16_SUBMISSION = ensure_submission(selected_challenges(V16_RERUN, CONFIG),
                                       f"notebook failed: {type(V16_ERROR).__name__}: {V16_ERROR}")
    if not V16_RERUN:
        raise
    # A competition rerun that raises is scored as an error; the guaranteed submission above is scored instead.
    print("competition rerun: the failure is logged above; not re-raising so the submission is scored", flush=True)
'''

# The codex notebook's first step: a queued submission that starts too close to the end-by cutoff writes the
# placeholder submission (V16_GUARD_TAIL) instead of booting a server it cannot use.
V16_END_BY = r'''
V16_DEADLINE, V16_END_BY_RECEIPT = codex_deadline(CONFIG, V16_RERUN, NOTEBOOK_STARTED_UNIX)
write_json(OUT / "end_by.json", V16_END_BY_RECEIPT)
print("h28: end-by " + json.dumps(V16_END_BY_RECEIPT), flush=True)
if V16_END_BY_RECEIPT["capped"] or not V16_END_BY_RECEIPT["applies"]:
    V16_EB = V16_END_BY_RECEIPT
    print("!" * 100 + "\nEND-BY: " + (f"harness deadline capped to {V16_EB['deadline']} ({V16_EB['minutes']} min after "
          f"the notebook started; planned {V16_EB['planned_deadline']}) so the notebook ends by {V16_EB['end_by']}"
          if V16_EB["applies"] else f"notebook started after the end-by time {V16_EB['end_by']}: no cap, the planned "
          f"deadline {V16_EB['planned_deadline']}") + "\n" + "!" * 100, flush=True)
end_by_gate(V16_DEADLINE, CONFIG["end_by_boot_seconds"] + CONFIG["end_by_min_harness_seconds"], "notebook start")
'''

HARDWARE = r'''
def read_cpu_file(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError as exc:
        return {"unavailable": str(exc)}

def hardware_command(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"hardware command failed: {command}: {result.stderr}")
    return result.stdout

hardware = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "libc": platform.libc_ver(), "cpu_count": os.cpu_count(),
            "cgroup": {name: read_cpu_file("/sys/fs/cgroup/" + name)
                       for name in ("memory.max", "memory.current", "cpu.max")},
            "memory_stat": read_cpu_file("/sys/fs/cgroup/memory.stat"),
            "proc_meminfo": read_cpu_file("/proc/meminfo"),
            "nvidia_smi": hardware_command(["nvidia-smi"]),
            "topology": hardware_command(["nvidia-smi", "topo", "-m"]),
            "gpu_names": hardware_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines(),
            "runtime_environment": {name: os.environ[name] for name in
                ("KAGGLE_IS_COMPETITION_RERUN", "CUDA_VISIBLE_DEVICES") if name in os.environ}}
write_json(OUT / "hardware.json", hardware)
if len(hardware["gpu_names"]) != 4 or any("L4" not in name for name in hardware["gpu_names"]):
    raise RuntimeError("this notebook requires exactly fourL4 GPUs; see hardware.json")
print(json.dumps({"gpus": hardware["gpu_names"], "cgroup": hardware["cgroup"],
                  "requested_profile": CONFIG}), flush=True)
'''

INSTALL = r'''
asset = source_asset(ASSET_MANIFEST["source_dataset"])
setup = asset / ASSET_MANIFEST["runtime_setup"]["file"]
if sha256(setup) != ASSET_MANIFEST["runtime_setup"]["sha256"]:
    raise RuntimeError("mounted runtime_setup.py differs from the pinned installer")
runtime_setup = load_module(setup, "arcforge_h28_runtime_setup")
prefetch_spec = ASSET_MANIFEST["checkpoint_prefetch"]
prefetch_file = asset / prefetch_spec["file"]
if sha256(prefetch_file) != prefetch_spec["sha256"]:
    raise RuntimeError("checkpoint reader differs from its pinned ARC3 implementation")
checkpoint_reader = load_module(prefetch_file, "arcforge_checkpoint_prefetch")
mounted_model = runtime_setup.model_mount(ASSET_MANIFEST["model_source"])
model_identity = ASSET_MANIFEST["model_identity"]
for name, digest in model_identity["target_sha256"].items():
    runtime_setup.verify_file(mounted_model / name, digest, model_identity["target_sizes"][name])
print("h28: starting original ARC3 checkpoint prefetch while runtime installs", flush=True)
# reads (and so the page cache) on the NUMA node with the most free memory; a two-socket host's ranks need their own
# node free for HostComm slots and host pools (Runpod 10-11: 80 GiB of checkpoint cache on node 1)
prefetch_handle = checkpoint_reader.start_prefetch(mounted_model, model_identity, place_on_roomiest_node=True)
try:
    INSTALLED = runtime_setup.install(ASSET_MANIFEST, work_dir=WORK, output_dir=OUT)
finally:
    prefetch_receipt = checkpoint_reader.join_prefetch(prefetch_handle)
    write_json(OUT / "checkpoint_prefetch.json", prefetch_receipt)
required = {"python", "env", "runtime", "model", "payload_repo", "source_manifest"}
if not isinstance(INSTALLED, dict) or not required <= set(INSTALLED):
    raise RuntimeError("offline installer did not return the agreed runtime contract")
if not isinstance(INSTALLED["env"], dict):
    raise RuntimeError("offline installer env must be a dictionary")
payload = pathlib.Path(INSTALLED["payload_repo"])
launcher_file = payload / ASSET_MANIFEST["payload"]["launcher"]
if sha256(launcher_file) != ASSET_MANIFEST["payload"]["launcher_sha256"]:
    raise RuntimeError("installed launcher differs from the frozen h28NoMTP candidate")
if sha256(INSTALLED["source_manifest"]) != ASSET_MANIFEST["source"]["manifest_sha256"]:
    raise RuntimeError("installed SOURCE_SHA256 differs from the frozen h28 manifest")
write_json(OUT / "installed_paths.json", {name: str(INSTALLED[name]) for name in required - {"env"}})
'''

MANAGED = r'''
def memory_receipt():
    group = pathlib.Path("/sys/fs/cgroup")
    stats = dict(line.split() for line in (group / "memory.stat").read_text().splitlines())
    return {"current_bytes": int((group / "memory.current").read_text()),
            **{name + "_bytes": int(stats[name]) for name in ("anon", "file", "unevictable")}}

def cgroup_shmem_bytes():
    stats = dict(line.split() for line in pathlib.Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
    return int(stats["shmem"])

def memory_events():
    """cgroup v2 memory.events (oom / oom_kill counters) and shmem: v16 v2 lost a TP rank silently."""
    group = pathlib.Path("/sys/fs/cgroup")
    events = dict(line.split() for line in (group / "memory.events").read_text().splitlines())
    stats = dict(line.split() for line in (group / "memory.stat").read_text().splitlines())
    return {"current_gib": round(int((group / "memory.current").read_text()) / 2**30, 2),
            "anon_gib": round(int(stats["anon"]) / 2**30, 2), "file_gib": round(int(stats["file"]) / 2**30, 2),
            "shmem_gib": round(int(stats["shmem"]) / 2**30, 2), **{name: int(events[name]) for name in events}}

def start_host_memory_trace(path, every_s=10.0):
    """Append {t, current, events, stat} for /sys/fs/cgroup every every_s seconds (daemon thread). A failed read
    writes one {"t", "error"} record and stops the thread, so a gap in the trace is visible as such."""
    group = pathlib.Path("/sys/fs/cgroup")
    def sample():
        with open(path, "a") as out:
            while True:
                try:
                    record = {"t": round(time.time(), 1), "current": int((group / "memory.current").read_text()),
                              "events": dict((k, int(v)) for k, v in (line.split() for line in
                                             (group / "memory.events").read_text().splitlines())),
                              "stat": dict((k, int(v)) for k, v in (line.split() for line in
                                           (group / "memory.stat").read_text().splitlines()))}
                except Exception as exc:
                    out.write(json.dumps({"t": round(time.time(), 1), "error": repr(exc)}) + "\n")
                    out.flush()
                    raise
                out.write(json.dumps(record) + "\n")
                out.flush()
                time.sleep(every_s)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return thread

def top_rss(limit):
    out = subprocess.run(["ps", "-eo", "pid,rss,comm", "--sort=-rss"], capture_output=True, text=True,
                         timeout=60, check=True).stdout.splitlines()[1:limit + 1]
    return [f"{row.split()[2]}:{row.split()[0]}:{int(row.split()[1]) // 1024}MiB" for row in out]



def run_command(command, env, log, timeout):
    with pathlib.Path(log).open("w") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"owned phase failed({result.returncode}); see {log}")

def arc2_progress(output, selected, started):
    receipts = list((output / "hisparse-transport").glob("*/call-*.receipt.json"))
    progress = {"phase": "original-arc2", "elapsed_seconds": round(time.monotonic() - started),
        "task_records": len(list((output / "tasks").glob("*.json"))), "selected_tasks": len(selected),
        "completed_model_calls": 0, "native_generated_tokens": 0, "failed_model_calls": 0,
        "receipts_being_written": 0}
    for path in receipts:
        try:
            receipt = json.loads(path.read_text())
        except json.JSONDecodeError:
            progress["receipts_being_written"] += 1
            continue
        if "error" in receipt:
            progress["failed_model_calls"] += 1
        elif "native_output_id_count" in receipt:
            progress["completed_model_calls"] += 1
            progress["native_generated_tokens"] += receipt["native_output_id_count"]
    server_log = output.parent / "server/server.log"
    with server_log.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        tail = stream.read().decode(errors="replace")
    progress["server_log_age_seconds"] = round(time.time() - server_log.stat().st_mtime, 1)
    metrics = {}
    for line in tail.splitlines():
        if "Prefill batch" in line or "Decode batch" in line:
            dp = re.search(r"(?:\bDP([01]) )?TP\d+ EP\d+\]", line)  # pure TP4 (attn_dp 1) logs no DP: DP 0
            if dp:
                metrics[dp.group(1) or "0"] = line
    progress["latest_scheduler_lines"] = metrics
    # v15h-B (10-12): this heartbeat had no GPU or host readings, so its Save & Run showed neither the VRAM floor nor
    # how close the node came to the cgroup limit; the codex heartbeat carries both.
    progress["gpu_free_gib"] = [row["free_gib"] for row in gpu_free_rows()]
    progress["host_memory"] = memory_events()
    return progress

def run_original_arc2(command, env, log, output, selected, monitor=None, timeout=None):
    stopped = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not stopped.is_set():
            print("h28: " + json.dumps(arc2_progress(output, selected, started)), flush=True)
            if monitor is not None:
                snapshot = monitor["observer"].snapshot()
                with (monitor["output"] / "snapshots.jsonl").open("a") as stream:
                    stream.write(json.dumps(snapshot) + "\n")
                    stream.flush()
                print("ARC2_LIVE=" + json.dumps(monitor["module"].concise(snapshot)), flush=True)
            stopped.wait(30)
    progress = threading.Thread(target=heartbeat, daemon=True)
    progress.start()
    try:
        run_command(command, env, log, timeout)
    finally:
        stopped.set()
        progress.join(timeout=1)

def run_capacity_logged(command, env, log, fast):
    stopped = threading.Event()
    started = time.monotonic()
    def heartbeat():
        while not stopped.wait(30):
            requests = fast / "case/requests"
            print("h28: " + json.dumps({"phase": "capacity-c64-100k", "elapsed_seconds": round(time.monotonic() - started),
                "cold_started": len(list(requests.glob("cold-*/request.json"))),
                "cold_finished": len(list(requests.glob("cold-*/result.json"))),
                "continuation_started": len(list(requests.glob("continuation-*/request.json"))),
                "continuation_finished": len(list(requests.glob("continuation-*/result.json")))}), flush=True)
    progress = threading.Thread(target=heartbeat, daemon=True)
    progress.start()
    try:
        run_command(command, env, log, None)
    finally:
        stopped.set()
        progress.join(timeout=1)

def phase_result(directory, expected="passed"):
    result = json.loads((pathlib.Path(directory) / "result.json").read_text())
    if result["status"] != expected:
        raise RuntimeError(f"phase result failed: {directory}")
    return result

def decode_graph_receipts(logs, running_per_dp):
    rows = []
    for dp in (0, 1):
        found = []
        for line in logs.splitlines():
            if not re.search(r"\bDP" + str(dp) + r"\b", line) or "Decode batch" not in line:
                continue
            count = re.search(r"#running-req:\s*(\d+)", line)
            if count and "cuda graph: True" in line and int(count.group(1)) == running_per_dp:
                found.append(line)
        if not found:
            raise RuntimeError(f"no actual graph decode receipt at requested batch forDP{dp}")
        rows.append({"dp_rank": dp, "running_requests": running_per_dp, "graph": True, "raw_line": found[0]})
    return rows

def original_arc2_command(python, model, payload, config, manifest, base_url, output, rerun):
    record = manifest["original_arc2"]
    root = pathlib.Path(payload) / record["root"]
    if sha256(root / "ORIGINAL_MANIFEST.json") != record["manifest_sha256"]:
        raise RuntimeError("original ARC2 source manifest changed")
    original = record["config"]
    output.mkdir(parents=True, exist_ok=True)
    source = competition_file("arc-agi_test_challenges.json" if rerun else "arc-agi_evaluation_challenges.json")
    challenges = json.loads(source.read_text())
    if not rerun:
        solutions_file = competition_file("arc-agi_evaluation_solutions.json")
        solutions = json.loads(solutions_file.read_text())
        for task_id, task in challenges.items():
            if len(task["test"]) != len(solutions[task_id]):
                raise RuntimeError("public evaluation solution count differs")
            for test, answer in zip(task["test"], solutions[task_id], strict=True):
                test["output"] = answer  # Original scorer reads this; prompts use only test input.
        repeat = config["validation_repeat"]
        if repeat > 1:
            # Load test only: copies get distinct IDs so the harness runs and scores each independently.
            challenges = {**challenges, **{f"{task_id}-r{copy}": json.loads(json.dumps(task))
                          for copy in range(1, repeat) for task_id, task in challenges.items()}}
        source = output / "public_evaluation_scored.json"
        write_json(source, challenges)
    selected = list(challenges) if rerun else list(challenges)[:config["validation_limit"]]
    if not rerun and len(selected) != config["validation_limit"]:
        raise RuntimeError("SaveRun does not contain all requested validation tasks")
    override = config.get("rerun_minutes" if rerun else "saverun_minutes")
    total_minutes = original["total_budget_min"] if override is None else override
    elapsed_minutes = (time.monotonic() - NOTEBOOK_STARTED) / 60
    remaining_minutes = total_minutes - elapsed_minutes - original["reserve_min"]
    if remaining_minutes <= 0:
        raise RuntimeError("original notebook wall budget is exhausted")
    ctx_cap = original["ctx_cap"]
    if config["harness_ctx_cap"] is not None:
        if config["harness_ctx_cap"] < ctx_cap:
            raise RuntimeError(f"harness ctx cap {config['harness_ctx_cap']} is below the original notebook's {ctx_cap}")
        ctx_cap = config["harness_ctx_cap"]
    command = [python, str(root / "hisparse_bridge.py"), "--hisparse-sessions",
               "--task-source", "kaggle", "--challenges", str(source),
               "--submission-out", str(OUT.parent / "submission.json"), "--out", str(output),
               "--base-url", base_url + "/v1", "--model", "tir", "--tokenizer", str(model), "--backend", "openai",
               "--compact", "--max-hops", str(original["max_hops"]), "--pass-gate-hop", str(original["max_hops"]),
               "--max-turns", str(original["max_turns"]), "--n-explorers", str(original["explorers"]), "--base-explorers", "2",
               "--reasoning-effort", original["effort"], "--ctx-cap", str(ctx_cap),
               "--report-reserve", str(original["report_reserve"]), "--context-margin", str(original["context_margin"]),
               "--compact-gen-budget", str(original["gen_budget"]), "--refine-gen-budget", str(original["refine_budget"]),
               "--task-budget", str(original["task_budget"]), "--lineage-budget", str(original["lineage_budget"]),
               "--min-session-budget", str(original["min_session_budget"]), "--handoff-budget", str(original["handoff_budget"]),
               "--gen-budget-hard", "--skip-refine-when-agree",
               "--max-live", str(config["running"]), "--task-live", str(max(1, config["running"] * 3 // 4)),
               "--dp-size", str(config.get("attn_dp", 2)), "--deadline-min", str(remaining_minutes),
               "--deadline-tok-per-s", str(original["deadline_tok_per_s"]),
               "--max-error-rate", str(original["max_error_rate"]), "--error-grace", "4", "--sandbox-python", python]
    if config["deadline_rate_adaptive"]:
        command.append("--deadline-rate-adaptive")
    if config["time_aware_budget"]:
        command.append("--time-aware-budget")
    if not rerun:
        command += ["--limit", str(config["validation_limit"])]
    write_json(output / "original_invocation.json", {"args": command, "selected_tasks": selected,
        "original_config": original, "original_notebook_sha256": record["notebook_sha256"],
        "user_run_overrides": {"validation_limit": config["validation_limit"], "notebook_total_minutes": total_minutes,
            "elapsed_minutes_before_harness": elapsed_minutes, "cleanup_reserve_minutes_from_original": original["reserve_min"],
            "harness_remaining_minutes": remaining_minutes, "harness_ctx_cap": ctx_cap,
            "deadline_rate_adaptive": config["deadline_rate_adaptive"], "time_aware_budget": config["time_aware_budget"],
            "scope": "whole notebook including install/startup/cleanup"},
        "serving_concurrency_source": "user requested conc" + str(config["running"]) + "; solver parameters from original notebook",
        "validation_repeat": config["validation_repeat"],
        "invocation_template": "work/kaggle_submit/mtp97c/make_notebook.py:714", "rerun": rerun})
    return command, selected

def original_arc2_result(output, selected, rerun):
    records = {path.stem: json.loads(path.read_text()) for path in (output / "tasks").glob("*.json")}
    if not selected or len(selected) != len(set(selected)) or not set(records).issubset(selected):
        raise RuntimeError("original ARC2 task records are outside the unique requested roster")
    completed = [task_id for task_id in selected if task_id in records]
    missing = [task_id for task_id in selected if task_id not in records]
    result = {"harness": "unchanged original ARC2 agentic TIR", "selected_tasks": selected,
              "requested_tasks": len(selected), "completed_tasks": len(completed),
              "completed_task_ids": completed, "missing_task_ids": missing,
              "missing_or_unstarted_tasks": len(missing),
              "full_requested_roster_completed": not missing, "rerun": rerun,
              "quality_scope": "hidden test; no ground truth" if rerun else "public completed subset of requested validation tasks"}
    for task_id, record in records.items():
        n_test, rows = record["n_test"], record["results"]
        if record["task_id"] != task_id or type(n_test) is not int or n_test <= 0 or not isinstance(rows, list) or len(rows) != n_test:
            raise RuntimeError("invalid original ARC2 task/test result record")
        indices = [row["test_idx"] for row in rows]
        if any(type(index) is not int for index in indices) or set(indices) != set(range(n_test)):
            raise RuntimeError("incomplete or duplicate original ARC2 test results")
        if any(not isinstance(row["attempts"], list) for row in rows):
            raise RuntimeError("invalid original ARC2 attempt result record")
        if rerun:
            if any(row["correct"] is not None for row in rows):
                raise RuntimeError("hidden-test records unexpectedly contain ground-truth scores")
        elif any(type(row["correct"]) is not bool for row in rows):
            raise RuntimeError("missing original public ground-truth scoring records")
    result["scored_tasks"] = 0 if rerun else len(completed)
    result["score_denominator_tasks"] = result["scored_tasks"]
    if not rerun:
        if not completed:
            raise RuntimeError("no completed original public ground-truth scoring records")
        result["partial_pass_at_2_percent"] = sum(sum(row["correct"] for row in record["results"]) /
            record["n_test"] for record in records.values()) / len(completed) * 100
        result["strict_pass_at_2_percent"] = sum(all(row["correct"] for row in record["results"])
            for record in records.values()) / len(completed) * 100
    write_json(output / "baseline_result.json", result)
    return result

def compiled_identity(python, runtime, env, manifest, resolved, gpu):
    hisparse = json.loads(resolved["hisparse_config"])
    launch = {"tp_size": resolved["tp_size"], "ep_size": resolved["ep_size"],
        "dp_size": resolved["dp_size"], "enable_dp_attention": resolved["enable_dp_attention"],
        "running": resolved["max_running_requests"], "logical_tokens": hisparse["qsa_logical_tokens"],
        "context_length": resolved["context_length"], "page_size": resolved["page_size"],
        "hot_blocks": hisparse["qsa_hot_blocks"], "index_dtype": "fp8" if env["ARC_QSA_INDEX_FP8"] == "1" else "bfloat16",
        "kv_dtype": resolved["kv_cache_dtype"], "ssm_dtype": resolved["mamba_ssm_dtype"],
        "graph": not resolved["disable_cuda_graph"] and not resolved["disable_decode_cuda_graph"],
        "cuda_graph_bs_decode": resolved["cuda_graph_bs_decode"],
        "cuda_graph_config": resolved["cuda_graph_config"], "speculative_algorithm": resolved["speculative_algorithm"],
        "mem_fraction_static": resolved["mem_fraction_static"], "hisparse_config": hisparse,
        "chunked_prefill_size": resolved["chunked_prefill_size"], "prefill_max_requests": resolved["prefill_max_requests"],
        "dtype": resolved["dtype"], "nccl_p2p_disable": env["NCCL_P2P_DISABLE"],
        "disable_custom_all_reduce": resolved["disable_custom_all_reduce"]}
    if "candidate" in manifest:
        launch["candidate_features"] = {name: env[name] for name in (
            "ARC_NATIVE_MTP_KV_HOST", "ARC_QSA_FP8_DIRECT", "ARC_GDN_FOLD_IN_VERIFY",
            "ARC_HISPARSE_QSA_DEBUG", "SGLANG_SET_CPU_AFFINITY")}
        launch["rejection_sampling"] = resolved["speculative_use_rejection_sampling"]
        launch["replayssm_spec_cache_len"] = resolved["linear_replayssm_cache_len"]
        if manifest["candidate"]["id"] in ("r8-r2", "r9", "r9-r2", "arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"):
            launch["candidate_features"]["ARC_HISPARSE_SPEC_PHASE_PACKED"] = env["ARC_HISPARSE_SPEC_PHASE_PACKED"]
        if manifest["candidate"]["id"] in ("r9-r2", "arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"):
            launch["candidate_features"]["ARC_HISPARSE_MTP_STEPS"] = env["ARC_HISPARSE_MTP_STEPS"]
            launch["speculative_num_steps"] = resolved["speculative_num_steps"]
            launch["speculative_num_draft_tokens"] = resolved["speculative_num_draft_tokens"]
        if manifest["candidate"]["id"] in ("arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"):
            # v16: the drafter and the hostcomm-hook switches change which kernels exist (keep set by content)
            launch["dflash"] = {"enabled": "ARC_DFLASH_DRAFTER" in env,
                                "keep_sha256": sha256(env["ARC_DFLASH_KEEP"]) if "ARC_DFLASH_KEEP" in env else None}
            launch["hostcomm"] = {name: env.get(name) for name in ("ARC_HOSTCOMM", "ARC_HC_AR", "ARC_HC_AG",
                "ARC_HC_SKIP_RS2", "ARC_HC_SLOT_MB", "ARC_HC_ONLINE_FP8", "ARC_HC_AG_ANYDTYPE", "ARC_VISION_OFFLOAD",
                "ARC_HC_AG_DP", "ARC_HC_AG_DP_CHECK", "ARC_HC_MOE_RS", "ARC_HC_NUMA_NODE", "ARC_MOE_STATS",
                "ARC_MOE_STATS_DIR", "ARC_MROPE_DECODE_PIN", "ARC_SPEC_AGREE_DEVICE", "ARC_FWD_TIMING",
                "ARC_HC_HUMMING_MOE", "ARC_HC_HUMMING_MOE_TILE", "ARC_HC_HUMMING_MOE_ACT")}
            if env.get("ARC_HC_MOE_OVERLAP"):  # chunk-width Humming kernels; absent otherwise (keeps older identities)
                launch["hostcomm"]["ARC_HC_MOE_OVERLAP"] = env["ARC_HC_MOE_OVERLAP"]
            launch["enable_dp_lm_head"] = resolved["enable_dp_lm_head"]
            launch["swap_in_block_size"] = hisparse.get("swap_in_block_size")
    # distribution(name) resolves along sys.path like import does. distributions() also lists the Kaggle image's own
    # /usr/local/lib/python3.12/dist-packages torch/triton (v16 v1: "cache identity mismatch: versions" although
    # the imported torch was the runtime's 2.13.0), and a dict built from it keeps whichever copy comes last.
    version_code = "import importlib.metadata as m,json; out={}\nfor n in ('torch','triton','flashinfer-python','sgl-kernel','sglang-kernel','tilelang'):\n    try: d=m.distribution(n)\n    except m.PackageNotFoundError: continue\n    out[d.metadata['Name']]=d.version\nprint(json.dumps(out))"
    versions = json.loads(subprocess.check_output([python, "-c", version_code], env=env, text=True))
    nvcc = subprocess.check_output([env["CUDA_HOME"] + "/bin/nvcc", "--version"], env=env, text=True)
    # The driver and its compat dir differ between the machine that seeds a cache (Runpod) and Kaggle; compiled
    # Triton/JIT entries are keyed by source, options and sm_89, so they are recorded in the export, not matched.
    gpu_identity = {name: gpu[name] for name in ("gpus", "capabilities", "torch", "cuda", "sglang")}
    gpu_identity["python"] = gpu["python"].split()[0]  # the version; the build string differs between images
    return {"source": manifest["source"], "runtime_archive": manifest["runtime_archive"],
        "versions": versions, "gpu": gpu_identity, "nvcc": nvcc, "launch_profile": launch,
        "runtime_path": str(runtime), "cache_paths": {name: env[name] for name in
            ("SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE")}}

def restore_compiled_cache(python, runtime, env, manifest, gpu, payload):
    if "compiled_cache" not in manifest:
        return {"status": "cold-start", "speedup_measured": False}
    record = manifest["compiled_cache"]
    asset = source_asset(record["dataset"])
    if sha256(asset / "manifest.json") != record["manifest_sha256"]:
        raise RuntimeError("mounted SM89 cache manifest differs from the pinned asset")
    helper_record = manifest["compiled_cache_helper"]
    helper = pathlib.Path(payload) / helper_record["file"]
    if sha256(helper) != helper_record["sha256"]:
        raise RuntimeError("compiled-cache restore helper differs from the pinned payload")
    module = load_module(helper, "arcforge_sm89_cache_restore")
    resolved = json.loads((OUT / "server/config_preflight.json").read_text())["resolved"]
    expected = compiled_identity(python, runtime, env, manifest, resolved, gpu)
    return module.restore(asset, expected, {name: pathlib.Path(path) for name, path in expected["cache_paths"].items()})

def export_compiled_cache(python, runtime, env, manifest, config, gpu, stage):
    import tarfile
    paths = {name: pathlib.Path(env[name]) for name in ("SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE")}
    destination = OUT / ("compiled-sm89-" + stage)
    destination.mkdir(exist_ok=True)
    # .noextract: this directory is uploaded as the next run's cache dataset, and Kaggle unpacks *.tar.gz uploads
    archive = destination / "compiled-kernels.tar.gz.noextract"
    files = []
    with tarfile.open(archive, "w:gz") as stream:
        for name, directory in paths.items():
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(directory).as_posix()
                stream.add(path, arcname=name + "/" + relative, recursive=False)
                files.append({"cache": name, "file": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    preflight = json.loads((OUT / "server/config_preflight.json").read_text())
    identity = compiled_identity(python, runtime, env, manifest, preflight["resolved"], gpu)
    write_json(destination / "manifest.json", {**identity,
        "archive": archive.name, "archive_sha256": sha256(archive), "files": files,
        "recorded_only": {name: gpu.get(name) for name in ("driver", "cuda_compat_dir")},
        "scope": "actual SM89 compiled files; CUDA graphs must be recaptured; no next-start reuse claim yet"})
    return {"manifest": str(destination / "manifest.json"), "files": len(files), "bytes": archive.stat().st_size}

def launch_profile(launcher, config, dflash=None):
    options = dict(mem_fraction=config["mem_fraction"], index_fp8=True,
                   prefill_chunk=config.get("prefill_chunk", 1024), prefill_max_requests=config["prefill_max_requests"],
                   dist_timeout=120)
    if config.get("prefill_delayer", False):
        options.update(prefill_delayer=True, prefill_delayer_max_passes=config["prefill_delayer_max_passes"],
                       prefill_delayer_queue_ratio=config["prefill_delayer_queue_ratio"],
                       prefill_delayer_max_delay_ms=config["prefill_delayer_max_delay_ms"])
    if config.get("attn_dp", 2) != 2:  # the launcher's Profile gained attn_dp with the Vast 10-03 TP4 arm
        options["attn_dp"] = config["attn_dp"]
    if "context_length" in config:
        options["context"] = config["context_length"]
    if config.get("swap_in_block_size") is not None:
        options["swap_in_block_size"] = config["swap_in_block_size"]
    if config.get("dflash_k") is not None:
        # serve_l4.sh's validated shape: NEXTN K-1 / K, top-k 1, rejection sampling, the drafter replacing the MTP draft
        if dflash is None or config.get("candidate_id") not in ("arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"):
            raise RuntimeError("DFlash needs the staged drafter export and a sglang-arc-2 candidate")
        options.update({name: config[name] for name in ("hot_blocks", "sparse_graphs", "cpu_affinity", "debug_checks")})
        options.update(mtp=True, mtp_steps=config["dflash_k"] - 1, mtp_rejection_sampling=True, dist_timeout=300,
                       dflash_drafter=dflash["drafter"], dflash_keep=dflash["keep"], deterministic=config["deterministic"],
                       fold_in_verify=config["fold_in_verify"])
        if config.get("gdn_state_fmt") is not None:  # only payloads from l4opt30 on know the launcher field
            options["gdn_state_fmt"] = config["gdn_state_fmt"]
        if config.get("snapshot_host_buffers") is not None:  # launcher field from payload l4opt34 on
            options["snapshot_host_buffers"] = config["snapshot_host_buffers"]
        if config.get("session_anchor"):  # launcher field from payload l4opt34 on
            options["session_anchor"] = True
        if dflash["dense"] is not None:  # only the arc-dense payload's launcher knows the flag (arc-main's predates it)
            options["dflash_dense"] = True
        return launcher.Profile(config["running"], config["logical_tokens"], config["host_reserve_gib"], **options)
    if config.get("nomtp_control", False):
        if config.get("candidate_id") not in ("r9-r2", "arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv") or any(config.get(name, False) for name in (
            "mtp", "draft_fp8_direct", "fold_in_verify", "mtp_rejection_sampling",
            "native_mtp_host_kv", "spec_phase_packed")) or config.get("mtp_steps", 3) != 3:
            raise RuntimeError("NoMTP control requires repaired r9-r2 and excludes speculative options")
        options.update({name: config[name] for name in ("hot_blocks", "sparse_graphs", "cpu_affinity", "debug_checks")})
    elif config.get("candidate_id") == "r9-r2" and not config.get("mtp", False):
        raise RuntimeError("repaired candidate target-only mode requires explicit NoMTP control")
    if config.get("mtp", False):
        options.update({name: config[name] for name in ("mtp", "hot_blocks", "sparse_graphs",
            "draft_fp8_direct", "fold_in_verify", "mtp_rejection_sampling", "native_mtp_host_kv",
            "cpu_affinity", "debug_checks")})
        if config["candidate_id"] in ("r8-r2", "r9", "r9-r2"):
            options["spec_phase_packed"] = config["spec_phase_packed"]
        if config["candidate_id"] == "r9-r2":
            options["mtp_steps"] = config["mtp_steps"]
    return launcher.Profile(config["running"], config["logical_tokens"], config["host_reserve_gib"], **options)

def hostcomm_env(python, env, payload, runtime, config):
    """Build libhostcomm.so with the runtime's own nvcc and return the env for the hook (Vast 10-03 best stack)."""
    hook_dir = payload / "work/qwen38fn_l4/probes/hostcomm"
    inject_dir = payload / "work/qwen38fn_l4/kaggle/hc_inject"
    for path in (hook_dir / "arc_hostcomm_hook.py", hook_dir / "hostcomm.py", hook_dir / "hostcomm.cu",
                 inject_dir / "sitecustomize.py"):
        if not path.is_file():
            raise RuntimeError(f"hostcomm payload file missing: {path}")
    build_dir = runtime / "hostcomm-build"
    started = time.monotonic()
    subprocess.run([python, "-c", "import os, sys, pathlib; sys.path.insert(0, sys.argv[1]); import hostcomm; "
                    "print(hostcomm.build(os.environ['CUDA_HOME'], pathlib.Path(sys.argv[2])))",
                    str(hook_dir), str(build_dir)], env={**env, "ARC_HC_TIMEOUT_S": "600"}, check=True)
    lib = build_dir / "libhostcomm.so"
    if not lib.is_file():
        raise RuntimeError("libhostcomm.so was not produced")
    hc_env = {"ARC_HOSTCOMM": "1", "ARC_HC_AR": "1", "ARC_HC_AG": "1", "ARC_HC_SKIP_RS2": "1",
              "ARC_HC_SLOT_MB": str(config.get("hc_slot_mb", 21)), "ARC_QSA_DECODE": "1", "ARC_HC_LIB": str(lib),
              "PYTHONPATH": str(inject_dir) + ":" + str(hook_dir) + ":" + env["PYTHONPATH"]}
    humming = bool(config.get("humming_moe") or config.get("humming_dense"))
    if config.get("marlin_block_remap", False) or humming:
        # after hc_inject: its sitecustomize chains to the next one on the path, which is v16_patch's
        patch_dir = BUNDLE_ROOT / "work/qwen38fn_l4/kaggle/v16_patch"
        if not (patch_dir / "sitecustomize.py").is_file():
            raise RuntimeError(f"v16 patch missing from the bundle: {patch_dir}")
        hc_env["PYTHONPATH"] = str(inject_dir) + ":" + str(patch_dir) + ":" + str(hook_dir) + ":" + env["PYTHONPATH"]
        hc_env["ARC_V16_HC_INJECT_DIR"] = str(inject_dir)
        if config["spec_trace"]:
            hc_env["ARC_V16_SPEC_TRACE"] = ("1" if config["spec_trace"] == "compare"
                                                            else config["spec_trace"])
        if config["hc_debug"]:
            hc_env[{"streams": "ARC_V16_HC_STREAMS", "no_attn": "ARC_V16_HC_NO_ATTN",
                     "zero_freed": "ARC_V16_ZERO_FREED"}[config["hc_debug"]]] = "1"
        if config.get("host_private", False):
            hc_env["ARC_HOST_POOL_PRIVATE"] = "1"
            # the runtime env carries PYTORCH_ALLOC_CONF=expandable_segments:True, which torch reads ahead of
            # PYTORCH_CUDA_ALLOC_CONF: the pinned options go after it, and both variables get the same value
            conf = env.get("PYTORCH_ALLOC_CONF", "")
            if env.get("PYTORCH_CUDA_ALLOC_CONF") not in (None, conf) or "pinned_" in conf:
                raise RuntimeError(f"host_private sets the pinned allocator itself; env has PYTORCH_ALLOC_CONF={conf!r}, "
                                   f"PYTORCH_CUDA_ALLOC_CONF={env.get('PYTORCH_CUDA_ALLOC_CONF')!r}")
            hc_env["PYTORCH_ALLOC_CONF"] = hc_env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(filter(None, [
                conf, "pinned_use_cuda_host_register:True", "pinned_num_register_threads:8"]))
    if config.get("marlin_block_remap", False):
        hc_env["ARC_V16_MARLIN_BLOCK_REMAP"] = "1"
    if humming:
        hc_env.update(humming_env(runtime, config))
    if config.get("humming_dense"):
        hc_env["ARC_HC_HUMMING_DENSE"] = config["humming_dense"]
        hc_env["ARC_HC_HUMMING_LAYERS"] = config["humming_layers"]
    if config.get("hc_mix_w8", False):
        skinny = payload / "work/qwen38fn_l4/probes/skinny_gemm"
        if not (skinny / "w8a16.py").is_file():
            raise RuntimeError(f"hc_mix_w8 needs the skinny_gemm payload: {skinny}")
        hc_env["ARC_HC_MIX_W8"] = "1"
        hc_env["ARC_SKINNY_DIR"] = str(skinny)
    if config.get("embed_offload", False):
        hc_env["ARC_EMBED_OFFLOAD"] = "1"
    if config.get("hc_moe_rs", False):
        hc_env["ARC_HC_MOE_RS"] = "1"
    if config.get("hc_moe_rs_owner", False):
        hc_env["ARC_HC_MOE_RS"] = "owner_fp8" if config.get("hc_moe_rs_fp8", False) else "owner"
    if config.get("hc_sumlen"):
        hc_env["ARC_HC_SUMLEN"] = {"bf16": "1", "fp8": "fp8"}[config["hc_sumlen"]]
    if config.get("hc_moe_overlap"):
        hc_env["ARC_HC_MOE_OVERLAP"] = config["hc_moe_overlap"]
    if config.get("hc_mix_verify"):
        skinny = payload / "work/qwen38fn_l4/probes/skinny_gemm"
        for name in ("hc_mix_rows.py", "hc_mix_rows_l4.json"):
            if not (skinny / name).is_file():
                raise RuntimeError(f"hc_mix_verify needs {name} in the skinny_gemm payload: {skinny}")
        hc_env["ARC_HC_MIX_VERIFY"] = config["hc_mix_verify"]
        hc_env["ARC_SKINNY_DIR"] = str(skinny)
    if config.get("lmhead_triton"):
        skinny = payload / "work/qwen38fn_l4/probes/skinny_gemm"
        if not (skinny / "lm_head_gemm.py").is_file():
            raise RuntimeError(f"lmhead_triton needs lm_head_gemm.py in the skinny_gemm payload: {skinny}")
        hc_env["ARC_LMHEAD_TRITON"] = config["lmhead_triton"]
        hc_env["ARC_SKINNY_DIR"] = str(skinny)
    if config.get("hc_ag_dp", False):
        hc_env["ARC_HC_AG_DP"] = "1"
    if config.get("hc_ag_dp_check", False):
        hc_env["ARC_HC_AG_DP_CHECK"] = "1"
    if config.get("hc_ag_dp_check_every", 1) != 1:
        hc_env["ARC_HC_AG_DP_CHECK_EVERY"] = str(config["hc_ag_dp_check_every"])
    if config.get("hc_pipe_ar2", 0):
        hc_env["ARC_HC_PIPE_STAGES2"] = str(config["hc_pipe_ar2"])
    if config.get("hc_pipe_tp4", 0):
        hc_env["ARC_HC_PIPE_STAGES4"] = str(config["hc_pipe_tp4"])
    if config.get("hc_ar2_blocks") is not None:
        hc_env["ARC_HC_AR2_BLOCKS"] = str(config["hc_ar2_blocks"])
    if config.get("hc_ag_blocks") is not None:
        hc_env["ARC_HC_AG_BLOCKS"] = str(config["hc_ag_blocks"])
    if config.get("hc_fold_epoch", False):
        hc_env["ARC_HC_FOLD_EPOCH"] = "1"
    if config.get("hc_ar2_p2p", False):
        hc_env["ARC_HC_AR2_P2P"] = "1"
    if config.get("hc_rs_stage2", "bf16") != "bf16":
        hc_env["ARC_HC_RS_STAGE2"] = config["hc_rs_stage2"]
    if config.get("hc_rs_sib_wait", False):
        hc_env["ARC_HC_RS_SIBWAIT"] = "1"
    if config.get("malloc_trim_seconds") is not None:
        hc_env["ARC_MALLOC_TRIM_SECONDS"] = str(config["malloc_trim_seconds"])
    if config.get("gpu_monitor", False):
        hc_env["ARC_GPU_MONITOR"] = "1"
    if config.get("hisparse_miss_stats", False):
        hc_env["ARC_HISPARSE_MISS_STATS"] = "1"  # read by the fork's coordinator and by the hook
    if config.get("hc_numa_node") is not None:
        hc_env["ARC_HC_NUMA_NODE"] = str(config["hc_numa_node"])
    if config.get("hc_local_slots", False):
        hc_env["ARC_HC_LOCAL_SLOTS"] = "1"
    if config.get("moe_stats", False):
        hc_env["ARC_MOE_STATS"] = "1"
        hc_env["ARC_MOE_STATS_DIR"] = str(OUT / "server/moe_routing")  # topk_ids ring snapshots for replay
    if config.get("mrope_decode_pin", False):
        hc_env["ARC_MROPE_DECODE_PIN"] = "1"
    if config.get("fwd_timing", False):
        hc_env["ARC_FWD_TIMING"] = "1"
    if config.get("spec_agree_device", False):
        hc_env["ARC_SPEC_AGREE_DEVICE"] = "ab" if config["spec_agree_device"] == "ab" else "1"
    if config.get("hc_ag_kind"):
        hc_env["ARC_HC_AG_KIND"] = config["hc_ag_kind"]
    if config.get("hc_ar2_kind"):
        hc_env["ARC_HC_AR2_KIND"] = config["hc_ar2_kind"]
    if config.get("vision_offload", False):
        hc_env["ARC_VISION_OFFLOAD"] = "1"
    if config.get("online_fp8"):
        hc_env["ARC_HC_ONLINE_FP8"] = config["online_fp8"]  # block-FP8 dense layers (lm_head stays BF16)
    if config.get("hc_ag_anydtype", False):
        hc_env["ARC_HC_AG_ANYDTYPE"] = "1"
    print(f"h28: built libhostcomm.so in {time.monotonic() - started:.1f}s", flush=True)
    return {"env": hc_env, "lib_sha256": sha256(lib), "build_seconds": time.monotonic() - started, "timeout_s": 600}

def humming_env(runtime, config):
    """Unpack the pinned Humming wheel and point Humming at the runtime's pip CUDA 13 (v16_patch)."""
    import zipfile
    wheel = source_asset(HUMMING_WHEEL["dataset"]) / HUMMING_WHEEL["file"]
    if sha256(wheel) != HUMMING_WHEEL["sha256"]:
        raise RuntimeError(f"humming wheel hash mismatch: {wheel}")
    root = runtime / "humming"
    if root.exists():
        raise RuntimeError(f"{root} already exists")
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(root)
    # zip keeps no execute bits; Humming runs _native/x86_64/nvrtc_compile as a child process
    native = root / "humming/_native/x86_64"
    for path in native.iterdir():
        path.chmod(0o755)
    cuda = runtime / "package/lib/python3.12/site-packages/nvidia/cu13"
    if not (cuda / "lib").is_dir() or not (cuda / "include").is_dir():
        raise RuntimeError(f"runtime pip CUDA 13 missing: {cuda}")
    moe = ({"ARC_HC_HUMMING_MOE": config["humming_moe"], "ARC_HC_HUMMING_MOE_TILE": config["humming_moe_tile"],
            "ARC_HC_HUMMING_MOE_ACT": config["humming_moe_act"]} if config.get("humming_moe") else {})
    return {**moe, "ARC_HUMMING_DIR": str(root),
            "ARC_V16_HUMMING_CUDA_ROOT": str(cuda)}

def shm_gather_receipts(server_pgid, enabled):
    """After the batch gate's decode, read the four scheduler processes of this server (setproctitle
    sglang::scheduler*, in its process group; a forked child keeps the title, so a titled process whose parent is
    titled too is not one). Their small unlinked POSIX segments with multiprocessing's default name (psm_<8 hex>) are
    arc_shm_gather's buffers -- SGLang names its own segments sgl_shm_*, HostComm arc_hc_* / arc_embed_*. On: every
    scheduler maps at least one (the attention-TP pair's spec consensus runs every verify; the MLP sync adds one for
    the group of four when it goes over the CPU group); off: none. (/proc/<pid>/environ cannot tell: setproctitle
    reuses the environment block for the title; the launch env is in run_result builder_env.)"""
    mapping = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) \S+ \S+ \S+ \d+\s+/dev/shm/(psm_[0-9a-f]{8}) \(deleted\)$")
    titled = {}
    for proc in pathlib.Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if not (proc / "cmdline").read_bytes().startswith(b"sglang::scheduler") or os.getpgid(int(proc.name)) != server_pgid:
                continue
            ppid = int(re.search(r"^PPid:\s+(\d+)$", (proc / "status").read_text(), re.M)[1])
            maps = (proc / "maps").read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited between the listing and the read: not a scheduler of the running server (counted below)
        segments = sorted({m[3] for m in map(mapping.match, maps) if m and int(m[2], 16) - int(m[1], 16) <= 65536})
        titled[int(proc.name)] = {"pid": int(proc.name), "ppid": ppid, "segments": segments}
    rows = [row for row in titled.values() if row["ppid"] not in titled]
    if len(rows) != 4:
        raise RuntimeError(f"expected four sglang::scheduler processes in process group {server_pgid}, found {rows}")
    for row in rows:
        if (not row["segments"]) if enabled else row["segments"]:
            raise RuntimeError(f"shm_gather={enabled}: scheduler {row['pid']} maps psm_ segments {row['segments']}")
    return {"enabled": enabled, "schedulers": rows, "forked_children": len(titled) - len(rows)}

def hostcomm_receipts(logs, config):
    """Fail loudly unless every TP rank installed the hook and joined the TP4 hostcomm group."""
    installed = [line for line in logs.splitlines() if "ARC_HOSTCOMM_INSTALLED" in line]
    tp4 = [line for line in logs.splitlines() if "ARC_HOSTCOMM_INIT" in line and "ranks=[0, 1, 2, 3]" in line]
    if len(tp4) != 4:
        raise RuntimeError(f"expected four TP4 hostcomm group receipts, found {len(tp4)}")
    # Scheduler subprocesses install the hook before SGLang configures their logger, so ARC_HOSTCOMM_INSTALLED
    # usually appears once (main process); the four TP4 init receipts above only exist when the hook is live.
    if not installed:
        raise RuntimeError("hostcomm hook install receipt missing")
    receipts = {"installed": installed, "tp4_init": tp4}
    if config.get("hc_local_slots", False):
        # every HostComm (TP4, and the attention-TP pairs under attn_dp 2) logs where this rank's written pages landed
        inits = re.findall(r"ARC_HOSTCOMM_INIT group=(\S+) ranks=(\[[0-9, ]+\]) slot=\d+ cfg=\{.*?\} placement=(.*)$",
                           logs, re.M)
        want = 4 if config.get("attn_dp", 2) == 1 else 8
        placed = [(group, ranks, json.loads(raw)) for group, ranks, raw in inits]
        def on_node(pl):  # gpu_node -1: a one-node host (sysfs reports no node), any single node is local
            pages = pl["pages_per_node"]
            if pl["gpu_node"] < 0:
                return len(pages) == 1 and sum(pages.values()) == pl["own_pages"]
            return pages == {f"N{pl['gpu_node']}": pl["own_pages"]}
        bad = [row for row in placed if row[2] is None or not on_node(row[2])]
        if len(placed) != want or bad:
            raise RuntimeError(f"hc_local_slots: expected {want} HostComm inits with every written page on the rank's "
                               f"GPU node, found {len(placed)}, off-node or missing {bad[:2]}")
        receipts["local_slots"] = [{"group": g, "ranks": r, **pl} for g, r, pl in placed]
    if config.get("marlin_block_remap", False):
        # every scheduler hooks both modules and, by the end of graph capture (K8 C32 verify is M=256), remapped 48
        remap = {}
        for kind in ("patched sglang.srt.layers.moe.fused_moe_triton pid", "patched sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe pid",
                     "align M=\\d+ 48->64 pid"):
            remap[kind] = sorted(set(re.findall(r"ARC_V16_MARLIN_BLOCK_REMAP " + kind + r"=(\d+)", logs)))
            if len(remap[kind]) < 4:
                raise RuntimeError(f"expected ARC_V16_MARLIN_BLOCK_REMAP '{kind}' from four schedulers, found pids {remap[kind]}")
        receipts["marlin_block_remap"] = remap
    if config.get("humming_moe"):
        moe = re.findall(r"\] ARC_HC_HUMMING_MOE (\S+) tile=(\S+) act=(\S+) layers=(\d+) cuda_bytes_freed=(-?\d+)", logs)
        want = (config["humming_moe"], config["humming_moe_tile"], config["humming_moe_act"])
        packed = config["humming_moe"] == "int3packed"  # placeholders -> INT3 banks: device memory grows
        if (len(moe) != 4 or len({layers for *_, layers, _ in moe}) != 1
                or any((spec, tile, act) != want or int(layers) <= 0 or (int(freed) < 0) != packed
                       for spec, tile, act, layers, freed in moe)):
            raise RuntimeError(f"expected Humming MoE {want} on four ranks, same layer count, found {moe}")
        if packed:
            loaded = re.findall(r"\] ARC_HC_INT3_PACKED experts=(\d+) fp8_linears=(\d+) bf16_linears=(\d+)", logs)
            if len(loaded) != 4 or len(set(loaded)) != 1 or loaded[0][0] != "48":
                raise RuntimeError(f"expected ARC_HC_INT3_PACKED experts=48 with one dense count on four ranks, found {loaded}")
            receipts["int3_packed"] = {"experts": 48, "fp8_linears": int(loaded[0][1]), "bf16_linears": int(loaded[0][2])}
        cuda = sorted(set(re.findall(r"ARC_V16_HUMMING_CUDA root=\S+ pid=(\d+)", logs)))
        if len(cuda) < 4:
            raise RuntimeError(f"expected ARC_V16_HUMMING_CUDA from four schedulers, found pids {cuda}")
        receipts["humming_moe"] = {"ranks": [{"spec": spec, "tile": tile, "act": act, "layers": int(layers),
                                              "cuda_bytes_freed": int(freed)} for spec, tile, act, layers, freed in moe],
                                   "cuda_root_pids": cuda}
    if config.get("vision_offload", False):
        offload = [line for line in logs.splitlines() if "ARC_VISION_OFFLOAD payload_bytes=" in line]
        if len(offload) != 4:
            raise RuntimeError(f"expected vision offload on four ranks, found {len(offload)}")
        receipts["vision_offload"] = offload
    if config.get("online_fp8"):
        fp8 = re.findall(r"\] ARC_HC_ONLINE_FP8 (\S+) cuda_bytes_freed=(\d+)", logs)
        if len(fp8) != 4 or any(names != config["online_fp8"] or int(freed) <= 0 for names, freed in fp8):
            raise RuntimeError(f"expected online FP8 {config['online_fp8']} on four ranks with freed bytes, found {fp8}")
        receipts["online_fp8"] = [{"layers": names, "cuda_bytes_freed": int(freed)} for names, freed in fp8]
    for key, pattern in (("humming_dense", r"\] ARC_HC_HUMMING_DENSE (\S+) (\{.*?\}) cuda_bytes_freed=(\d+)"),
                         ("hc_mix_w8", r"\] ARC_HC_MIX_W8 mixers=(\d+) cuda_bytes_freed=(\d+)"),
                         ("embed_offload", r"\] ARC_EMBED_OFFLOAD module=\S+ tp=\d+ table_bytes=(\d+) cuda_bytes_freed=(\d+)")):
        if config.get(key):
            found = re.findall(pattern, logs)
            if len(found) != 4 or any(int(row[-1]) <= 0 for row in found):
                raise RuntimeError(f"expected {key} on four ranks with freed bytes, found {found}")
            if key == "humming_dense" and any(row[0] != config["humming_dense"] for row in found):
                raise RuntimeError(f"humming dense spec mismatch: {found}")
            receipts[key] = [list(row) for row in found]
    if config.get("hc_ag_dp") or config.get("hc_moe_rs_owner") or config.get("hc_numa_node") is not None:
        installed = re.findall(r"\] ARC_HOSTCOMM_DP_AWARE rank=\d ag_dp=(\w+) ag_dp_check=(\w+) moe_rs=(\S+) numa_node=(\S+)", logs)
        want = (str(bool(config.get("hc_ag_dp"))), str(bool(config.get("hc_ag_dp_check"))),
                ("owner_fp8" if config.get("hc_moe_rs_fp8") else "owner") if config.get("hc_moe_rs_owner")
                else ("1" if config.get("hc_moe_rs") else "0"),
                str(config["hc_numa_node"]) if config.get("hc_numa_node") is not None else "None")
        if len(installed) != 4 or any(tuple(row) != want for row in installed):
            raise RuntimeError(f"expected ARC_HOSTCOMM_DP_AWARE {want} on four ranks, found {installed}")
        receipts["hostcomm_dp_aware"] = [list(row) for row in installed]
    if config.get("hc_sumlen"):
        want_sumlen = {"bf16": "1", "fp8": "fp8"}[config["hc_sumlen"]]
        moe_rs = re.findall(r"\] ARC_HOSTCOMM_MOE_RS mode=\S+ sumlen=(\S+) sumlen_blocks=(\d+) installed", logs)
        if len(moe_rs) != 4 or any(row[0] != want_sumlen for row in moe_rs):
            raise RuntimeError(f"expected ARC_HOSTCOMM_MOE_RS sumlen={want_sumlen} on four ranks, found {moe_rs}")
        receipts["hostcomm_sumlen"] = [list(row) for row in moe_rs]
    if config.get("hc_mix_verify"):
        lo, hi = config["hc_mix_verify"].split(":")
        mixv = re.findall(r"\] ARC_HC_MIX_VERIFY rows \((\d+), (\d+)\] buckets=(\[[0-9, ]*\])", logs)
        if len(mixv) != 4 or any(row[:2] != (lo, hi) for row in mixv):
            raise RuntimeError(f"expected ARC_HC_MIX_VERIFY rows ({lo}, {hi}] on four ranks, found {mixv}")
        receipts["hc_mix_verify"] = [list(row) for row in mixv]
    if config.get("lmhead_triton"):
        lo, hi = config["lmhead_triton"].split(":")
        lmh = re.findall(r"\] ARC_LMHEAD_TRITON rows \((\d+), (\d+)\] cfg=(\{[^}]*\})", logs)
        if len(lmh) != 4 or any(row[:2] != (lo, hi) for row in lmh):
            raise RuntimeError(f"expected ARC_LMHEAD_TRITON rows ({lo}, {hi}] on four ranks, found {lmh}")
        receipts["lmhead_triton"] = [list(row) for row in lmh]
    if config.get("hc_ag_kind") or config.get("hc_ar2_kind"):
        # every HostComm logs the payload kinds its group runs: TP4 (MoE dispatch gather) and the attention pairs
        inits = re.findall(r"ARC_HOSTCOMM_INIT group=(\S+) ranks=(\[[0-9, ]+\]) slot=\d+ cfg=(\{.*?\}) placement=", logs)
        ag, ar2 = config.get("hc_ag_kind") or "ag_bf16", config.get("hc_ar2_kind") or "ar1_bf16"
        cfgs = [(ranks, json.loads(raw)) for _, ranks, raw in inits]
        tp4 = [c for r, c in cfgs if r == "[0, 1, 2, 3]"]
        pairs = [c for r, c in cfgs if r in ("[0, 1]", "[2, 3]")]
        if (len(tp4) != 4 or len(pairs) != 4 or any(c["ag_kind"] != ag for c in tp4 + pairs)
                or any(c["ar_kind"] != ar2 for c in pairs)):
            raise RuntimeError(f"expected 4 TP4 + 4 pair HostComms with ag_kind {ag} and pair ar_kind {ar2}, found {cfgs}")
        receipts["hostcomm_kinds"] = {"ag_kind": ag, "ar2_kind": ar2}
    if config.get("hc_pipe_ar2", 0) or config.get("hc_pipe_tp4", 0) or config.get("hc_ar2_blocks") is not None:
        inits = re.findall(r"ARC_HOSTCOMM_INIT group=(\S+) ranks=(\[[0-9, ]+\]) slot=\d+ cfg=(\{.*?\}) placement=", logs)
        cfgs = [(ranks, json.loads(raw)) for _, ranks, raw in inits]
        tp4 = [c for r, c in cfgs if r == "[0, 1, 2, 3]"]
        pairs = [c for r, c in cfgs if r in ("[0, 1]", "[2, 3]")]
        ar2_blocks = 32 if config.get("hc_ar2_blocks") is None else config["hc_ar2_blocks"]
        if (len(tp4) != 4 or len(pairs) != 4 or any(c.get("pipe_stages") != config.get("hc_pipe_tp4", 0) for c in tp4)
                or any(c.get("pipe_stages") != config.get("hc_pipe_ar2", 0) or c["ar_blocks"] != ar2_blocks
                       for c in pairs)):
            raise RuntimeError(f"expected 4 TP4 HostComms with pipe_stages {config.get('hc_pipe_tp4', 0)} and 4 pair "
                               f"HostComms with pipe_stages {config.get('hc_pipe_ar2', 0)}, ar_blocks {ar2_blocks}; "
                               f"found {cfgs}")
        receipts["hc_pipe"] = {"tp4": [c["pipe_stages"] for c in tp4], "pairs": [c["pipe_stages"] for c in pairs],
                               "pair_ar_blocks": [c["ar_blocks"] for c in pairs]}
    review = {k: config.get(k) for k in ("hc_fold_epoch", "hc_ar2_p2p", "hc_rs_stage2", "hc_rs_sib_wait", "hc_ag_blocks")}
    if (review["hc_fold_epoch"] or review["hc_ar2_p2p"] or review["hc_rs_stage2"] not in (None, "bf16")
            or review["hc_rs_sib_wait"] or review["hc_ag_blocks"] is not None):
        inits = re.findall(r"ARC_HOSTCOMM_INIT group=(\S+) ranks=(\[[0-9, ]+\]) slot=\d+ cfg=(\{.*?\}) placement=", logs)
        cfgs = [(ranks, json.loads(raw)) for _, ranks, raw in inits]
        pairs = [c for r, c in cfgs if r in ("[0, 1]", "[2, 3]")]
        tp4 = [c for r, c in cfgs if r == "[0, 1, 2, 3]"]
        ag_blocks = 16 if review["hc_ag_blocks"] is None else review["hc_ag_blocks"]
        if (len(tp4) != 4 or len(pairs) != 4
                or any(c.get("fold_epoch") is not bool(review["hc_fold_epoch"]) for c in tp4 + pairs)
                or any(c.get("ar2_p2p") is not bool(review["hc_ar2_p2p"]) for c in pairs)
                or any(c["ag_blocks"] != ag_blocks for c in tp4)):
            raise RuntimeError(f"expected 4 TP4 + 4 pair HostComms with fold_epoch {review['hc_fold_epoch']}, pair "
                               f"ar2_p2p {review['hc_ar2_p2p']}, TP4 ag_blocks {ag_blocks}; found {cfgs}")
        # One line from the launcher process (the schedulers install before SGLang sets up their logger, see above;
        # codex4 v1 died on expecting four); every process reads the same environment.
        installed = re.findall(r"\] ARC_HOSTCOMM_INSTALLED .* rs_stage2=(\S+) rs_sibwait=(\S+) ag_blocks=(\S+)", logs)
        want = (review["hc_rs_stage2"] or "bf16", "1" if review["hc_rs_sib_wait"] else "0", str(ag_blocks))
        if not installed or any(tuple(row) != want for row in installed):
            raise RuntimeError(f"expected ARC_HOSTCOMM_INSTALLED rs_stage2 / rs_sibwait / ag_blocks {want}, "
                               f"found {installed}")
        receipts["hc_review"] = {"pairs": pairs, "tp4_fold": [c.get("fold_epoch") for c in tp4],
                                 "installed": [list(r) for r in installed]}
    if config.get("gpu_monitor"):
        if not re.search(r"\] ARC_GPU_MONITOR installed every_s=\S+ pcie_every_s=\S+ gpus=4", logs):
            raise RuntimeError("expected ARC_GPU_MONITOR installed with PCIe sampling of 4 GPUs (global rank 0)")
        receipts["gpu_monitor"] = True
    if config.get("hisparse_miss_stats"):
        readers = sorted(re.findall(r"\] ARC_HISPARSE_MISS reader installed rank=(\d)", logs))
        if readers != ["0", "2"]:
            raise RuntimeError(f"expected ARC_HISPARSE_MISS readers on ranks 0 and 2, found {readers}")
        receipts["hisparse_miss_stats"] = readers
    if config.get("hc_ag_dp") and config.get("hc_ag_kind") == "ag_fp8q":
        first = re.findall(r"\] ARC_HC_AG_DP_FP8Q first_capture rank=(\d) rows=(\d+) mode=(\S+) pipe_stages=(\d+) "
                           r"check=(\w+)", logs)
        if sorted(r for r, *_ in first) != ["0", "1", "2", "3"]:
            raise RuntimeError(f"expected ARC_HC_AG_DP_FP8Q in a captured graph on ranks 0-3, found {first}")
        receipts["ag_dp_fp8q"] = [list(row) for row in first]
    if config.get("hc_ag_kind") == "ag_fp8q":
        installed = re.findall(r"\] ARC_HC_FP8Q installed rank=(\d) up_input_group=128 rows_cap=(\d+) prewarm_hits=(\d+)", logs)
        hits = re.findall(r"\] ARC_HC_FP8Q captured_hit rank=(\d) rows=(\d+)", logs)
        if (sorted(r for r, _, _ in installed) != ["0", "1", "2", "3"] or any(int(h) <= 0 for _, _, h in installed)
                or sorted({r for r, _ in hits}) != ["0", "1", "2", "3"]):
            raise RuntimeError(f"expected ARC_HC_FP8Q installed (prewarmed) and a captured payload hit on ranks 0-3, "
                               f"found {installed} / {hits}")
        receipts["fp8q"] = {"installed": installed, "captured_hits": hits}
    if config.get("hc_moe_overlap"):
        installed = re.findall(r"\] ARC_HC_MOE_OVERLAP installed rank=(\d) widths=(\S+) layers=(\d+) selfcheck=ok", logs)
        first = re.findall(r"\] ARC_HC_MOE_OVERLAP first_capture rank=(\d) rows=(\d+) widths=(\S+) combine=(\S+)", logs)
        if (sorted(r for r, _, _ in installed) != ["0", "1", "2", "3"]
                or any(w != config["hc_moe_overlap"] for _, w, _ in installed)
                or sorted({r for r, *_ in first}) != ["0", "1", "2", "3"]):
            raise RuntimeError(f"expected ARC_HC_MOE_OVERLAP installed with widths {config['hc_moe_overlap']} (self-check "
                               f"ok) and in a captured graph on ranks 0-3, found {installed} / {first}")
        receipts["moe_overlap"] = {"installed": [list(r) for r in installed], "first_capture": [list(r) for r in first]}
    if config.get("mrope_decode_pin"):
        found = re.findall(r"\] ARC_MROPE_DECODE_PIN installed on (\S+)", logs)
        if len(found) != 4:
            raise RuntimeError(f"expected ARC_MROPE_DECODE_PIN on four ranks, found {found}")
        receipts["mrope_decode_pin"] = found
    if config.get("fwd_timing"):
        found = re.findall(r"\] ARC_FWD_TIMING installed rank=(\d+)", logs)
        if sorted(found) != ["0", "1", "2", "3"]:
            raise RuntimeError(f"expected ARC_FWD_TIMING on ranks 0-3, found {found}")
        receipts["fwd_timing"] = sorted(found)
    if config.get("spec_agree_device"):
        want = "ab" if config["spec_agree_device"] == "ab" else "1"
        found = re.findall(r"\] ARC_SPEC_AGREE_DEVICE installed rank=(\d+) attn_ranks=(\[[0-9, ]+\]) mode=(\S+)", logs)
        if sorted(int(r) for r, _, _ in found) != [0, 1, 2, 3] or any(m != want for _, _, m in found):
            raise RuntimeError(f"expected ARC_SPEC_AGREE_DEVICE mode={want} on four ranks, found {found}")
        receipts["spec_agree_device"] = found
    return receipts

def startup_memory_receipts(logs, config):
    """Per-rank free device memory after each CUDA-graph capture, as SGLang logs it (its "GB" here is GiB)."""
    rows = {}
    for line in logs.splitlines():
        rank = re.search(r"(?:\bDP(\d) )?TP(\d) EP\d\]", line)  # pure TP4 (attn_dp 1) logs no DP: DP 0
        if rank is None:
            continue
        row = rows.setdefault("TP" + rank.group(2), {"dp": int(rank.group(1) or 0)})
        graph = re.search(r"Capture (.+?) CUDA graph end\..*avail mem=([0-9.]+) GB", line)
        if graph:
            row["after_" + graph.group(1).replace(" ", "_") + "_graph_gib"] = float(graph.group(2))
        draft = re.search(r"DFlash .*draft CUDA graphs for batch sizes .* in [0-9.]+ s, (-?[0-9.]+) GiB", line)
        if draft:
            row["dflash_draft_graph_gib"] = float(draft.group(1))
        pool = re.search(r"Memory pool end\. avail mem=([0-9.]+) GB", line)
        if pool:
            row["after_memory_pool_gib"] = float(pool.group(1))
        final = re.search(r"available_gpu_mem=([0-9.]+) GB", line)
        if final:
            row["available_gpu_mem_gib"] = float(final.group(1))
        sizing = re.search(r"QSA HiSparse sizing: budget ([0-9.]+) GiB, .*B=(\d+) blocks \((\d+) fit", line)
        if sizing:
            row.update(hisparse_budget_gib=float(sizing.group(1)), hot_blocks=int(sizing.group(2)),
                       hot_blocks_fit=int(sizing.group(3)))
        snapshots = re.search(r"QSA HiSparse GDN snapshots: .*?(\d+) host buffers x (\d+) B.*resume anchors (on|off)", line)
        if snapshots:
            row.update(snapshot_host_buffers=int(snapshots.group(1)), snapshot_buffer_bytes=int(snapshots.group(2)),
                       session_anchor=snapshots.group(3) == "on")
    if sorted(rows) != ["TP0", "TP1", "TP2", "TP3"]:
        raise RuntimeError(f"startup memory lines for four ranks expected, got {sorted(rows)}")
    want = config.get("snapshot_host_buffers")
    if want is not None and any(row.get("snapshot_host_buffers") != want for row in rows.values()):
        raise RuntimeError(f"snapshot_host_buffers {want}: ranks logged "
                           f"{ {name: row.get('snapshot_host_buffers') for name, row in rows.items()} }")
    anchors = {name: row.get("session_anchor") for name, row in rows.items()}
    if config.get("session_anchor") and any(value is not True for value in anchors.values()):
        raise RuntimeError(f"session_anchor set: every rank must log resume anchors on, got {anchors}")
    if not config.get("session_anchor") and any(value is True for value in anchors.values()):
        raise RuntimeError(f"resume anchors on without session_anchor: {anchors}")
    for name, row in rows.items():
        graphs = [key for key in row if key.startswith("after_") and key.endswith("_graph_gib")]
        if not graphs:
            raise RuntimeError(f"{name}: no CUDA-graph capture memory line in server.log")
        if config.get("dflash_k") is not None and "dflash_draft_graph_gib" not in row:
            raise RuntimeError(f"{name}: no DFlash draft CUDA graph line in server.log")
        # The DFlash draft graphs are captured after the target's and log only their own size. On fork main's
        # refresh entry (ARC_DFLASH_EXACT_PREFILL, not the injected-context drafter) warm_refresh runs before the
        # drafter measures its free memory and the capture releases its cached blocks, so the logged size goes negative
        # (all-merge-1020 on L4: -0.62 GiB; nvidia-smi after startup is the gate): never count it as freed memory.
        row["after_all_graphs_gib_estimate"] = round(min(row[key] for key in graphs) - max(row.get("dflash_draft_graph_gib", 0.0), 0.0), 3)
    return rows

def mtp_probe_args(config):
    return ["--mtp-steps", str(config["mtp_steps"])] if config["candidate_id"] == "r9-r2" else []

def mtp_counter_kwargs(config):
    return {"mtp_steps": config["mtp_steps"]} if config["candidate_id"] == "r9-r2" else {}

def set_server_flag(command, flag, value):
    if command.count(flag) != 1:
        raise RuntimeError(f"launcher argv must carry {flag} exactly once")
    index = command.index(flag)
    return command[:index + 1] + [value] + command[index + 2:]

def gpu_free_rows():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    rows = [dict(zip(("gpu", "free_mib", "used_mib"), (int(x) for x in line.split(",")))) for line in out.strip().splitlines()]
    for row in rows:
        row["free_gib"] = round(row["free_mib"] / 1024, 3)
    if len(rows) != 4:
        raise RuntimeError(f"nvidia-smi listed {len(rows)} GPUs")
    return rows

def stage_dflash(config):
    """The pinned drafter export, every file hashed (DFLASH_PREFETCH: read while the runtime installs, which also
    warms the page cache), linked into a local directory: the worker stats ARC_DFLASH_DRAFTER every decode cycle
    (fork dfd2fd8f), and a local directory keeps that off the dataset mount."""
    spec = config["dflash_asset"]
    hashes = DFLASH_PREFETCH.join()
    if hashes != spec["files_sha256"]:
        bad = sorted(name for name in spec["files_sha256"] if hashes.get(name) != spec["files_sha256"][name])
        raise RuntimeError(f"mounted DFlash drafter differs from the pinned export: {bad}")
    local = WORK / "dflash-drafter"
    local.mkdir()
    for name in spec["files_sha256"]:
        (local / name).symlink_to(DFLASH_ROOT / spec["export_dir"] / name)
    if "dense" in spec:  # dense-routed export (exp/dense-dflash): no keep set, the launcher passes none
        exported = json.loads((local / "config.json").read_text())
        # The fork's dense loader serves two drafters, told apart by three fields of the export's config, all checked:
        # the hyper-connection dense_shared export (its config predates residual / context, which the fork's Config
        # reads as hc / recompute when unset) and, when the spec pins "architecture", the plain residual with injected
        # context and one dense MLP (fork c8500d9fe).
        if "architecture" in spec:
            architecture = {"residual": "plain", "context": "inject", "ffn_type": "dense"}
            if spec["architecture"] != architecture:
                raise RuntimeError(f"DFlash spec architecture {spec['architecture']} is not the plain drafter {architecture}")
            fields = {name: exported.get(name) for name in architecture}
        else:
            architecture = {"residual": "hc", "context": "recompute", "ffn_type": "dense_shared"}
            fields = {"residual": exported.get("residual", "hc"), "context": exported.get("context", "recompute"),
                      "ffn_type": exported.get("ffn_type")}
        if fields != architecture or exported.get("dense_intermediate_size") != spec["dense"]:
            raise RuntimeError(f"DFlash spec pins a dense {spec['dense']} drafter {architecture}, the export's config says "
                               f"{fields} {exported.get('dense_intermediate_size')}")
        keep_path, experts_kept, layers = None, None, exported["num_layers"]
    else:
        keep = json.loads((local / spec["keep"]).read_text())
        if sorted(keep, key=int) != [str(i) for i in range(len(keep))] or any(len(v) != spec["experts_kept"] for v in keep.values()):
            raise RuntimeError("DFlash keep set does not match the pinned per-layer expert count")
        keep_path, experts_kept = str(local / spec["keep"]), spec["experts_kept"]
        architecture, layers = None, None
    receipt = {"drafter": str(local), "keep": keep_path, "dataset": spec["dataset"], "files_verified": len(hashes),
               "experts_kept": experts_kept, "dense": spec.get("dense"), "architecture": architecture, "layers": layers,
               "k": config["dflash_k"]}
    write_json(OUT / "dflash_asset.json", receipt)
    return receipt

def dflash_receipts(logs, config, dflash):
    """Each rank built the drafter on its attention-TP pair with the pinned keep set and captured its draft graphs."""
    # fork dfd2fd8f: "DFlash drafter <dir> on attention-TP group [0, 1] (...), marlin experts: ..." (the l01 patch tree
    # logged "DFlash L4 drafter ... on attention-TP pair")
    # dense (exp/dense-dflash): "dense experts: 5 layers, experts kept ['dense 10240', ...] (['fp8 35 linears'] on this rank)"
    built = re.findall(r"(?:\bDP(\d) )?TP(\d) EP\d\] DFlash (?:L4 )?drafter (\S+) on attention-TP (?:group|pair) .*?"
                       r"experts kept \[([0-9, ]+|'dense \d+'(?:, 'dense \d+')*)\].*?weights ([0-9.]+) GiB.*?verify width (\d+)"
                       r"(?:, compile (\w+))?", logs)
    graphs = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash (?:L4 )?draft CUDA graphs for batch sizes (\[[0-9, ]+\])", logs)
    ranks = sorted(int(row[1]) for row in built)
    if ranks != [0, 1, 2, 3] or sorted(int(row[0]) for row in graphs) != [0, 1, 2, 3]:
        raise RuntimeError(f"DFlash drafter/graph receipts for four ranks expected: built {ranks}, graphs {graphs}")
    # fork f1f7836e and later name the pool: "... 0.20 GiB (own graph pool)" / "(target's graph pool)"
    pools = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash (?:L4 )?draft CUDA graphs for batch sizes .*? GiB \((own|target's) graph pool\)",
                       logs)
    if config.get("dflash_shared_graph_pool", False):
        if sorted(row[0] for row in pools) != ["0", "1", "2", "3"] or any(row[1] != "target's" for row in pools):
            raise RuntimeError(f"dflash_shared_graph_pool set but the draft graphs logged pools {pools}")
    elif any(row[1] == "target's" for row in pools):
        raise RuntimeError(f"draft graphs in the target's pool without dflash_shared_graph_pool: {pools}")
    # arc-dense-det logs the drafter's compile mode (fast_drafter.compile_mode); the older trees print none
    want_compile = config.get("dflash_compile") or "1"
    for dp, tp, path, kept, weights, width, compiled in built:
        if (compiled or None) != (want_compile if config.get("candidate_id") in ("arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv") else None):
            raise RuntimeError(f"DFlash drafter compile mode on TP{tp}: logged {compiled!r}, profile {want_compile!r}")
        kept_ok = (all(item.strip() == f"'dense {dflash['dense']}'" for item in kept.split(","))
                   and len(kept.split(",")) == dflash["layers"] if dflash["dense"] is not None
                   else all(item.strip().isdigit() and int(item) == dflash["experts_kept"] for item in kept.split(",")))
        if path != dflash["drafter"] or int(width) != config["dflash_k"] or not kept_ok:
            raise RuntimeError(f"DFlash drafter receipt differs from the pinned profile: TP{tp} {path} kept={kept} K={width}")
    # fork c8500d9fe: what each rank built, read off its modules and pool -- "DFlash drafter architecture: residual plain
    # (PlainDecoder, fc_hidden (2560, 10240), output norm), context inject (pool conv boundaries none, trunk rows per step
    # 3), ffn dense (MLP)"; the dense_shared drafter logs residual hc (Decoder, ..., output hyper_connection_mixer),
    # context recompute (pool conv boundaries (layers, 4, slots, 1, hidden), ...), ffn dense_shared (DenseWithShared).
    # The registered arc-dense-vocab-tp4-shv tree logs it on every rank; the older trees log none.
    arch = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash drafter architecture: residual (\w+) \((\w+), fc_hidden "
                      r"\((\d+), (\d+)\), output (\w+)\), context (\w+) \(pool conv boundaries (none|\([0-9, ]+\)), "
                      r"trunk rows per step (\d+)\), ffn (\w+) \((\w+)\)", logs)
    if config.get("candidate_id") == "arc-dense-vocab-tp4-shv":
        want = dflash["architecture"]  # stage_dflash: the export config's residual / context / ffn_type
        layer_class, output = {"plain": ("PlainDecoder", "norm"), "hc": ("Decoder", "hyper_connection_mixer")}[want["residual"]]
        mlp_class = {"dense": "MLP", "dense_shared": "DenseWithShared"}[want["ffn_type"]]
        if sorted(row[0] for row in arch) != ["0", "1", "2", "3"]:
            raise RuntimeError(f"DFlash drafter architecture lines for four ranks expected, got {arch}")
        for row in arch:
            tp, residual, layer, _, _, out, context, boundaries, trunk, ffn, mlp = row
            injected = context == "inject"
            if ((residual, layer, out, context, ffn, mlp) != (want["residual"], layer_class, output, want["context"],
                                                              want["ffn_type"], mlp_class)
                    or (boundaries == "none") != injected or (injected and int(trunk) != config["dflash_k"] - 1)):
                raise RuntimeError(f"DFlash drafter on TP{tp} built {row[1:]}, the pinned export is {want}")
    elif arch:
        raise RuntimeError(f"DFlash drafter architecture lines from a tree other than arc-dense-vocab-tp4-shv's: {arch}")
    # fork 60997af63a (worker.py, the plain drafter only): "DFlash plain drafter serving: linears as stored | merged (qkv
    # <class>, gate_up <class>, kv <class>), injected K/V per layer | one GEMM (<rows>, <hidden>), pool K/V torch.bfloat16 |
    # torch.float8_e4m3fn, fusion per use | once, FP8 op graph break | custom op". The plan B opt-ins (CONFIG carries one
    # only when set) need it on all four ranks as they set it; with none set it may be absent (c8500d9fe prints none)
    # and, when printed, must show the fork's defaults.
    serving = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash plain drafter serving: linears (as stored|merged \(qkv (\w+), "
                         r"gate_up (\w+), kv (\w+)\)), injected K/V (per layer|one GEMM \((\d+), (\d+)\)), pool K/V "
                         r"(torch\.\w+), fusion (per use|once), FP8 op (graph break|custom op)", logs)
    opts = {name: config[name] for name in ("dflash_merge", "dflash_inject_kv", "dflash_fuse_once", "dflash_kv_dtype",
                                            "dflash_ring_fp8_dot") if name in config}
    if opts or serving:
        if dflash["architecture"] != {"residual": "plain", "context": "inject", "ffn_type": "dense"}:
            raise RuntimeError(f"plain drafter serving lines {serving} / opt-ins {opts} without the plain drafter pinned "
                               f"({dflash['architecture']})")
        scope = (None if config.get("dflash_w4a8") is None
                 else "mlp" if config["dflash_w4a8"].endswith(":mlp") else "all")
        trunk = "HummingW4A8Linear" if scope == "all" else "Fp8BlockLinear"  # the stored block-FP8 linears as served
        ffn_linear = "Fp8BlockLinear" if scope is None else "HummingW4A8Linear"
        gemm = opts.get("dflash_inject_kv", "layers") == "gemm"
        # one GEMM's rows: layers x k|v x this rank's one KV head (2 over the attention-TP pair) x head_dim 256, over
        # hidden 2560 (the plan B export's geometry)
        want = ("merged (qkv %s, gate_up %s, kv %s)" % (trunk, ffn_linear, "NoneType" if gemm else trunk)
                if opts.get("dflash_merge", False) else "as stored",
                "one GEMM (%d, 2560)" % (dflash["layers"] * 2 * 256) if gemm else "per layer",
                "torch.float8_e4m3fn" if opts.get("dflash_kv_dtype", "bf16") == "fp8_e4m3" else "torch.bfloat16",
                "once" if opts.get("dflash_fuse_once", False) else "per use", "graph break")
        if sorted(row[0] for row in serving) != ["0", "1", "2", "3"]:
            raise RuntimeError(f"plain drafter serving lines for four ranks expected (opt-ins {opts}), got {serving}")
        for row in serving:
            if (row[1], row[5], row[8], row[9], row[10]) != want:
                raise RuntimeError(f"plain drafter on TP{row[0]} serves {row[1:]}, opt-ins {opts} want {want}")
    # fork exp/l4-sharded-verify: each rank's target LogitsProcessor logs the install once
    sharded = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] ARC_SHARDED_VERIFY installed: target-verify logits stay sharded over "
                         r"(\d+) attention-TP ranks \(vocab (\d+), (dp-lm-head|pure TP)\)", logs)
    if config.get("sharded_verify"):
        want = ("2", "dp-lm-head") if config.get("attn_dp", 2) == 2 else ("4", "pure TP")
        if sorted(row[0] for row in sharded) != ["0", "1", "2", "3"] or any((n, mode) != want for _, n, _, mode in sharded):
            raise RuntimeError(f"expected ARC_SHARDED_VERIFY installed {want} on ranks 0-3, found {sharded}")
    elif sharded:
        raise RuntimeError(f"ARC_SHARDED_VERIFY installed but the profile has no sharded_verify: {sharded}")
    # v16_patch _patch_think_budget: think budget forced on the vocab shards themselves, once per scheduler
    think_sharded = re.findall(r"ARC_THINK_BUDGET_SHARDED patched (\S+) \(shard-native\) pid=(\d+)", logs)
    if config.get("think_budget") is not None and config.get("sharded_verify"):
        target = "sglang.srt.speculative.arc_think_budget.ThinkBudget.apply"
        pids = {pid for name, pid in think_sharded if name == target}
        if len(pids) < 4:
            raise RuntimeError(f"think budget on sharded verify: {target} patched in {sorted(pids)}, not in all "
                               f"four schedulers")
    elif think_sharded:
        raise RuntimeError(f"ARC_THINK_BUDGET_SHARDED patched without think_budget + sharded_verify: {think_sharded}")
    heads =re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash (?:L4 )?drafter .*?, lm_head (\w+), loaded", logs)
    if config.get("draft_vocab") is not None and sorted(heads) != [(str(r), "SubsetPairHead") for r in range(4)]:
        raise RuntimeError(f"draft_vocab set but the drafters built lm_head {heads} (SubsetPairHead on four ranks expected)")
    # fork d94aef29: the restricted target head logs once per rank (file, kept ids, rows on this rank, bytes freed)
    pruned = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] ARC_TARGET_VOCAB (\S+): (\d+) of (\d+) ids kept, attention-TP "
                        r"rank (\d+)/(\d+) holds (\d+) rows \(was (\d+)\), cuda_bytes_freed=(\d+)", logs)
    if config.get("target_vocab") is not None:
        name = config["target_vocab"].rsplit("/", 1)[-1]
        if (sorted(row[0] for row in pruned) != ["0", "1", "2", "3"] or any(row[1] != name for row in pruned)
                or len({row[2] for row in pruned}) != 1 or any(int(row[6]) * int(row[5]) != int(row[2]) for row in pruned)):
            raise RuntimeError(f"target_vocab {name} set but the target heads logged {pruned}")
    elif pruned:
        raise RuntimeError(f"ARC_TARGET_VOCAB restricted the target head but the profile has no target_vocab: {pruned}")
    # fork 3781e841: "(['fp8 0 linears', 'w4a8 int4:128:all 35 linears (15 batch-invariant)', 'selector rows 98304/16384']
    # on this rank)" -- only the linears replicated across the attention-TP pair (the 15 FFN gate/up/down) are on Humming's
    # batch-invariant tuning, the per-rank attention slices on the default heuristic
    w4a8 = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash (?:L4 )?drafter .*?'w4a8 (\S+) (\d+) linears"
                      r"(?: \((\d+) batch-invariant\))?'", logs)
    if config.get("dflash_w4a8") is not None:
        spec = config["dflash_w4a8"] if config["dflash_w4a8"].count(":") == 2 else config["dflash_w4a8"] + ":all"
        if dflash["layers"] is None:
            raise RuntimeError("dflash_w4a8 is for a dense drafter export (its layer count is from stage_dflash)")
        # per layer q, k, v, o + the FFN's gate, up, down (dense_shared: .mlp.routed., dense: .mlp.) / the FFN's three;
        # the FFN linears are the ones every rank of the attention-TP pair computes whole (batch-invariant). Fork
        # 60997af63a dflash_merge (build_dense_drafter / _rebuild_held): q|k|v, o and -- unless dflash_inject_kv gemm
        # makes the injected K/V one BF16 GEMM -- the injected k|v (rank slices), gate|up and down (replicated);
        # dflash_inject_kv gemm alone keeps the trunk's k / v as separate linears.
        merge = config.get("dflash_merge", False)
        ffn = 2 if merge else 3
        attention = (2 if config.get("dflash_inject_kv", "layers") == "gemm" else 3) if merge else 4
        replicated = ffn * dflash["layers"]
        count = (ffn + attention) * dflash["layers"] if spec.endswith(":all") else replicated
        want = (spec, str(count), str(replicated))
        if sorted(row[0] for row in w4a8) != ["0", "1", "2", "3"] or any(row[1:] != want for row in w4a8):
            raise RuntimeError(f"dflash_w4a8 {spec} set: every drafter must log w4a8 {spec} {count} linears ({replicated} "
                               f"batch-invariant), got {w4a8}")
    elif w4a8:
        raise RuntimeError(f"drafters logged W4A8 linears without dflash_w4a8: {w4a8}")
    rows = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash (?:L4 )?drafter .*?'selector rows (\d+)/(\d+)'", logs)
    if config.get("candidate_id") == "arc-dense-vocab-tp4-shv" and config.get("target_vocab") is not None:  # l4opt34 on
        want = (str(pruned[0][2]), str(config["draft_vocab_size"]))  # the target list's kept ids, the draft vocabulary
        if sorted(row[0] for row in rows) != ["0", "1", "2", "3"] or any(row[1:] != want for row in rows):
            raise RuntimeError(f"selector codebooks pruned to {want} expected on four ranks, logged {rows}")
    return {"built": [{"dp": int(dp or 0), "tp": int(tp), "weights_gib": float(w), "verify_width": int(k)}
                      for dp, tp, _, _, w, k, _ in built], "draft_graph_batches": {tp: bs for tp, bs in graphs},
            "lm_head": sorted(heads), "sharded_verify": [list(row) for row in sharded],
            "target_vocab": [{"tp": int(row[0]), "kept": int(row[2]), "rows": int(row[6]), "freed_bytes": int(row[8])}
                             for row in pruned],
            "compile": sorted({row[6] or "unlogged" for row in built}),
            "w4a8": [list(row) for row in w4a8], "selector_rows": [list(row) for row in rows],
            "architecture": [list(row) for row in arch],
            **({"plain_serving": [list(row) for row in serving]} if opts else {})}

def prefill_ple_receipts(logs, config):
    """ARC_PLE_INT5's / ARC_PLE_NVFP4's per-rank load line (fork arc_ple_int5 / arc_ple_nvfp4) on all four ranks iff
    ple_int5 / ple_nvfp4, and the PrefillDelayer init line on both DP schedulers' attention-TP ranks iff prefill_delayer
    (the launcher checks the server args too)."""
    int5 = []
    for env_name, key in (("ARC_PLE_INT5", "ple_int5"), ("ARC_PLE_NVFP4", "ple_nvfp4")):
        rows = re.findall(rf"(?:\bDP\d )?TP(\d) EP\d\] {env_name}: PLE table (\S+), (\d+) rows, ([\d.]+) GiB host", logs)
        if config.get(key, False):
            if sorted(row[0] for row in rows) != ["0", "1", "2", "3"]:
                raise RuntimeError(f"{key} set but {env_name} loaded on ranks {[row[0] for row in rows]} (0-3 expected)")
        elif rows:
            raise RuntimeError(f"{env_name} loaded but the profile has no {key}: {rows}")
        int5 += rows
    delayer = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] PrefillDelayer initialized with max_delay_passes=(\d+) "
                         r"token_usage_low_watermark=\S+ queue_min_ratio=(\S+) max_delay_ms=(\S+)", logs)
    if config.get("prefill_delayer", False):
        if sorted({row[0] for row in delayer}) != ["0", "1", "2", "3"]:
            raise RuntimeError(f"prefill_delayer set but PrefillDelayer initialized on ranks {delayer}")
        if config["prefill_delayer_max_passes"] is not None and any(int(row[1]) != config["prefill_delayer_max_passes"]
                                                                    for row in delayer):
            raise RuntimeError(f"PrefillDelayer max_delay_passes {delayer}, profile {config['prefill_delayer_max_passes']}")
    elif delayer:
        raise RuntimeError(f"PrefillDelayer initialized but the profile has no prefill_delayer: {delayer}")
    exact = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] DFlash exact prefill: a prompt's window from its last (\d+) positions "
                       r".*? session park (on|off)", logs)
    if config.get("dflash_session_park", False):
        if sorted(row[0] for row in exact) != ["0", "1", "2", "3"] or any(row[2] != "on" for row in exact):
            raise RuntimeError(f"dflash_session_park set but the drafters logged exact prefill / park {exact} (0-3 on expected)")
    elif exact:
        raise RuntimeError(f"DFlash exact prefill logged but the profile has no dflash_session_park: {exact}")
    return {"ple_int5": [{"tp": int(tp), "codec": codec, "rows": int(rows), "host_gib": float(gib)}
                         for tp, codec, rows, gib in int5],
            "dflash_exact_prefill": [list(row) for row in exact],
            "prefill_delayer": [list(row) for row in delayer], "prefill_max_requests": config["prefill_max_requests"]}

# fork exp/l4-qsa-index-shard: files that must carry these lines before run_notebook sets ARC_QSA_INDEX_SHARD
QSA_INDEX_SHARD_MARKERS = {"sglang/srt/layers/attention/qsa/index_shard.py": 'ENV = "ARC_QSA_INDEX_SHARD"',
                           "sglang/srt/mem_cache/kv_cache_configurator.py": "qsa_index_shard=index_shard.build_layer_shard(",
                           "sglang/srt/layers/attention/qsa/qsa_indexer.py": "def _forward_shard_partner("}

def qsa_index_shard_receipts(logs, config):
    """ARC_QSA_INDEX_SHARD=<layer|layer-check> iff qsa_index_shard: all four ranks log their owned QSA layers (attention-TP
    rank r of each DP pair owns QSA layer number i iff i % 2 == r: the pair's sets are disjoint and cover all 12), their
    index pool (layer: 6 of 12 layers, layer-check: 12) and the HiSparse sizing (layer: 198 index B/token, layer-check:
    396); layer-check also logs its running count of received top-k rows checked (a differing row stops the server);
    none of these lines without it."""
    owns = re.findall(r"(?:\bDP(\d) )?TP(\d) EP\d\] ARC_QSA_INDEX_SHARD=(layer|layer-check): attention-TP rank (\d) of 2 "
                      r"owns the index of QSA layers \[([\d, ]*)\] \((\d+) of (\d+);", logs)
    pools = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] ARC_QSA_INDEX_SHARD: QSA index state on this rank ([0-9.]+) GiB "
                       r"for (\d+) of (\d+) layers .*?unsharded it would be ([0-9.]+) GiB", logs)
    sizing = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] QSA HiSparse sizing: ARC_QSA_INDEX_SHARD=(?:layer|layer-check) -- index "
                        r"(\d+) B/token on this rank \(unsharded (\d+)\), ([0-9.]+) GiB at L=(\d+) \(unsharded ([0-9.]+) "
                        r"GiB\)", logs)
    budget = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] QSA HiSparse sizing: budget .*?; KV \d+ B/token, index (\d+) B/token", logs)
    checked = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] ARC_QSA_INDEX_SHARD=layer-check: (\d+) top-k rows of the pair's QSA "
                         r"layers checked against this rank's own, 0 differ", logs)
    mode = config.get("qsa_index_shard")
    if mode is None:
        if owns or pools or sizing or checked:
            raise RuntimeError(f"ARC_QSA_INDEX_SHARD logged without qsa_index_shard: {owns} {pools} {sizing} {checked}")
        return {"qsa_index_shard": None}
    four = ["0", "1", "2", "3"]
    if (sorted(row[1] for row in owns) != four or sorted(row[0] for row in pools) != four
            or sorted(row[0] for row in sizing) != four or sorted(row[0] for row in budget) != four):
        raise RuntimeError(f"qsa_index_shard set: four ranks must log ownership / index pool / sizing, got {owns} "
                           f"{pools} {sizing} {budget}")
    pairs = {}
    for dp, tp, logged, attn_rank, layers, held, total in owns:
        if logged != mode or int(attn_rank) != int(tp) % 2 or (held, total) != ("6", "12"):
            raise RuntimeError(f"TP{tp}: ARC_QSA_INDEX_SHARD={logged} (profile {mode}), attention-TP rank {attn_rank}, "
                               f"owns {held} of {total} QSA layers (6 of 12 expected)")
        pairs.setdefault(dp or "0", []).append({int(x) for x in layers.split(", ")})
    for dp, sets in pairs.items():
        if len(sets) != 2 or sets[0] & sets[1] or len(sets[0] | sets[1]) != 12:
            raise RuntimeError(f"DP{dp}: the pair's owned QSA layers {sets} must be disjoint and cover 12 layers")
    held, index = ("6", "198") if mode == "layer" else ("12", "396")
    if any((row[2], row[3]) != (held, "12") for row in pools):
        raise RuntimeError(f"{mode}: index pools must hold {held} of 12 layers: {pools}")
    if any((row[1], row[2]) != (index, "396") for row in sizing) or any(row[1] != index for row in budget):
        raise RuntimeError(f"{mode}: HiSparse sizing must charge {index} index B/token (unsharded 396): {sizing} {budget}")
    most = {}
    for tp, rows in checked:
        most[tp] = max(most.get(tp, 0), int(rows))
    if mode == "layer-check" and (sorted(most) != four or min(most.values()) == 0):
        raise RuntimeError(f"layer-check: every rank must log received top-k rows checked, got {most}")
    if mode == "layer" and checked:
        raise RuntimeError(f"layer-check counts logged under qsa_index_shard layer: {checked}")
    return {"qsa_index_shard": mode, "checked_rows": {f"TP{tp}": n for tp, n in sorted(most.items())},
            "owned_layers": {f"TP{tp}": sorted(int(x) for x in layers.split(", ")) for _, tp, _, _, layers, _, _ in owns},
            "index_pool_gib": {f"TP{tp}": float(gib) for tp, gib, _, _, _ in pools},
            "index_pool_unsharded_gib": {f"TP{tp}": float(gib) for tp, _, _, _, gib in pools},
            "sizing": [{"tp": int(tp), "index_bytes": int(b), "unsharded_bytes": int(u), "index_gib": float(g),
                        "logical_tokens": int(n), "unsharded_gib": float(ug)} for tp, b, u, g, n, ug in sizing]}

def heapdiag_receipts(logs, config):
    """ARC_HEAPDIAG on all four schedulers iff heap_diag, with the profile's delay, and each one's init deep dump."""
    on = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] ARC_HEAPDIAG on: ticks every 60 s, deep dumps \+ malloc_trim\(0\) at "
                    r"init done and (\d+) s later", logs)
    init = re.findall(r"(?:\bDP\d )?TP(\d) EP\d\] ARC_HEAPDIAG deep init \{", logs)
    if config.get("heap_diag") is not None:
        want = str(config["heap_diag"])
        if sorted(row[0] for row in on) != ["0", "1", "2", "3"] or any(row[1] != want for row in on):
            raise RuntimeError(f"heap_diag {want} set but ARC_HEAPDIAG started on {on} (ranks 0-3, {want} s expected)")
        if sorted(init) != ["0", "1", "2", "3"]:
            raise RuntimeError(f"heap_diag set but the init deep dump was logged on ranks {init} (0-3 expected)")
    elif on or init:
        raise RuntimeError(f"ARC_HEAPDIAG logged without heap_diag: on {on}, init {init}")
    return {"on": [list(row) for row in on], "deep_init_ranks": sorted(init)}

def host_private_receipts(logs, config, shmem_bytes):
    """ARC_HOST_POOL_PRIVATE (v16_patch): with host_private, four processes (the schedulers) allocated GiBs of private
    pinned host pools each and the cgroup's shmem is down to HostComm's /dev/shm slots (s3 C96 without it: 79 GiB,
    ~19 GiB per scheduler); without host_private no private allocation may appear."""
    per_pid = {}
    for size, pid in re.findall(r"ARC_HOST_POOL_PRIVATE alloc (\d+) B pid=(\d+)", logs):
        per_pid[pid] = per_pid.get(pid, 0) + int(size)
    if config.get("host_private", False):
        big = sorted(pid for pid, size in per_pid.items() if size >= 2**30)
        if len(big) != 4:
            raise RuntimeError(f"host_private set but {len(big)} processes hold >= 1 GiB of private host pools: {per_pid}")
        if shmem_bytes > 8 * 2**30:
            raise RuntimeError(f"host_private set but the cgroup still holds {shmem_bytes / 2**30:.1f} GiB of shmem")
    elif per_pid:
        raise RuntimeError(f"ARC_HOST_POOL_PRIVATE allocations without host_private: {per_pid}")
    return {"host_private": config.get("host_private", False), "private_bytes_per_pid": per_pid,
            "cgroup_shmem_bytes": shmem_bytes}

def dflash_batch_counters(jobs):
    """Every batch-gate request went through speculative verify (the drafter is live); acceptance is a sanity record."""
    rows = []
    for job in jobs:
        meta = job["meta_info"]
        if type(meta.get("spec_verify_ct")) is not int or meta["spec_verify_ct"] <= 0:
            raise RuntimeError(f"batch request without a speculative verify step: {json.dumps(meta)[:300]}")
        rows.append({"dp": job["dp"], "verify": meta["spec_verify_ct"], "completion_tokens": meta["completion_tokens"]})
    verify = sum(row["verify"] for row in rows)
    return {"requests": len(rows), "verify_steps": verify,
            "delivered_tokens_per_verify": sum(row["completion_tokens"] for row in rows) / verify}

def seed_compiled_cache(source, cache):
    """Cache-seeding runs only (gates-only builds): start from an existing local compiled cache, so a run at the
    deployment's memory fraction does not have to survive a cold Triton autotune; its export becomes the dataset."""
    if not source.is_dir() or any(cache.iterdir()):
        raise RuntimeError(f"seed cache {source} must exist and {cache} must be empty")
    shutil.copytree(source, cache, dirs_exist_ok=True)
    files = [p for p in cache.rglob("*") if p.is_file()]
    return {"status": "seeded-from-local", "source": str(source), "files": len(files),
            "autotune_entries": sorted(p.relative_to(cache).as_posix() for p in files if p.name.endswith(AUTOTUNE_SUFFIXES))}

AUTOTUNE_SUFFIXES = (".autotune.json", ".best_config")  # Triton autotuner; inductor (vision tower torch.compile)

def new_autotune_entries(cache, restore):
    """Autotune results written after the restore/seed (each one is a benchmark that ran at serving time, with a
    scratch allocation of its own: Triton's do_bench takes 256 MiB)."""
    if restore["status"] == "restored":
        manifest = json.loads((source_asset(ASSET_MANIFEST["compiled_cache"]["dataset"]) / "manifest.json").read_text())
        before = {"triton/" + row["file"] for row in manifest["files"] if row["cache"] == "TRITON_CACHE_DIR"}
    elif restore["status"] == "seeded-from-local":
        before = set(restore["autotune_entries"])
    else:
        before = set()
    found = sorted(p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.name.endswith(AUTOTUNE_SUFFIXES))
    return [name for name in found if name not in before]

def make_agent_venv():
    """The interpreter codex's sandboxed shell runs: the image's Python (numpy, Pillow) behind a venv, readable by
    the fenced per-task uids."""
    venv = WORK / "agent-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)], check=True, timeout=300)
    python = venv / "bin/python"
    subprocess.run([str(python), "-I", "-c", "import numpy, PIL"], check=True, timeout=120)
    subprocess.run(["chmod", "-R", "a+rX,go-w", str(venv)], check=True, timeout=120)
    return python

def codex_arc2_command(python, model, config, base_url, output, rerun):
    """The codex harness entry point (work/arc_code/arc2/kaggle_run.py), as agreed with its implementation."""
    harness = BUNDLE_ROOT / "work/arc_code/arc2/kaggle_run.py"
    harness_config = BUNDLE_ROOT / config["codex_config"]
    tokenizer = pathlib.Path(model) / "tokenizer.json"
    for path in (harness, harness_config, tokenizer):
        if not path.is_file():
            raise RuntimeError(f"codex harness input missing: {path}")
    if output.exists() or output.resolve().is_relative_to("/kaggle"):
        raise RuntimeError(f"codex --out must be a new directory outside /kaggle: {output}")
    deadline_unix, _ = codex_deadline(config, rerun, NOTEBOOK_STARTED_UNIX)
    command = [python, str(harness), "--config", str(harness_config)]
    if rerun:
        command += ["--challenges", str(competition_file("arc-agi_test_challenges.json"))]
        selected = list(json.loads(competition_file("arc-agi_test_challenges.json").read_text()))
    else:
        challenges_file = competition_file("arc-agi_evaluation_challenges.json")
        solutions_file = competition_file("arc-agi_evaluation_solutions.json")
        if config["validation_repeat"] > 1:
            # The harness reads files and hides them by closing /kaggle, so the repeated roster goes under OUT.
            repeated = (OUT / "saverun_challenges_repeated.json", OUT / "saverun_solutions_repeated.json")
            for path, source in zip(repeated, (challenges_file, solutions_file), strict=True):
                write_json(path, repeated_roster(json.loads(source.read_text()), config["validation_repeat"]))
            challenges_file, solutions_file = repeated
        command += ["--challenges", str(challenges_file), "--solutions", str(solutions_file),
                    "--limit", str(config["validation_limit"])]
        selected = list(selected_challenges(False, config))
    command += ["--out", str(output), "--submission-out", str(OUT.parent / "submission.json"),
                "--archive-out", str(OUT.parent / "arc2_run.tar.gz"),
                "--model-base-url", base_url + "/v1", "--model", CODEX_SERVED_MODEL,
                "--dp-size", str(config.get("attn_dp", 2)), "--concurrency", str(config["running"]),
                "--deadline-unix", str(int(deadline_unix)),
                "--codex-dataset", str(source_asset(CODEX_DATASET)), "--agent-python", str(make_agent_venv()),
                "--tokenizer", str(tokenizer)]
    return command, selected, deadline_unix

def server_death(server):
    """None while the server looks alive; a reason once the launcher exited or any GPU was released.
    v16 v2: one TP rank vanished without a traceback, the others hung and /metrics kept answering stale gauges,
    so the harness sat on 32 dead streams. A rank's GPU memory coming back is the signal that does not depend on
    the dead process logging anything."""
    if server.poll() is not None:
        return f"server launcher exited with {server.returncode}"
    released = [row["gpu"] for row in gpu_free_rows() if row["free_gib"] > 10]
    if released:
        return f"GPU memory released on {released} (a server rank exited)"
    return None

def dump_server_death(reason, server_log):
    print("!" * 100 + "\nSERVER DIED: " + reason, flush=True)
    print("h28: memory events " + json.dumps(memory_events()), flush=True)
    print("h28: top rss " + json.dumps(top_rss(25)), flush=True)
    with pathlib.Path(server_log).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 400000))
        tail = stream.read().decode(errors="replace").splitlines()[1:]
    tail = [line for line in tail if "[transformers]" not in line][-300:]
    print("h28: server.log tail (" + str(len(tail)) + " lines)\n" + "\n".join(line[:400] for line in tail), flush=True)
    print("!" * 100, flush=True)

def run_codex_arc2(command, env, log, output, server_log, timeout, server):
    stopped = threading.Event()
    died = []
    started = time.monotonic()
    summary = SERVER_LOG_SUMMARY(server_log)
    offset = [0]
    keys = ("ARC2_KAGGLE", "START", "DONE", "HEARTBEAT", "DEADLINE", "FAILED", "SUMMARY", "Traceback")
    def harness_lines():
        if not pathlib.Path(log).is_file():
            return
        with pathlib.Path(log).open("rb") as stream:
            stream.seek(offset[0])
            data = stream.read()
        cut = data.rfind(b"\n") + 1
        offset[0] += cut
        for line in data[:cut].decode(errors="replace").splitlines():
            if any(key in line for key in keys):
                print("codex: " + line[-600:], flush=True)
    def heartbeat():
        try:
            beat()
        except BaseException as error:
            # This thread is the only thing watching the server; dying quietly would leave the harness on dead streams.
            import traceback
            traceback.print_exc()
            died.append(f"heartbeat thread failed: {error!r}")
            print("!" * 100 + "\nHEARTBEAT THREAD FAILED: " + repr(error) + "\n" + "!" * 100, flush=True)
            harness.terminate()
    def beat():
        while not stopped.wait(60):
            harness_lines()
            snapshot = summary.update()
            results = output / "run" / "results"
            print("h28: " + json.dumps({"phase": "codex-arc2", "elapsed_seconds": round(time.monotonic() - started),
                "task_results": len(list(results.glob("*.json"))) if results.is_dir() else None,
                "accept_len": {dp: row["mean_accept_len"] for dp, row in snapshot["decode"].items()},
                "running": {dp: row["mean_running"] for dp, row in snapshot["decode"].items()},
                "dflash_timing": snapshot["dflash_timing"], "retracted_requests": snapshot["retracted_requests"],
                "errors": snapshot["error_lines"], "gpu_free_gib": [row["free_gib"] for row in gpu_free_rows()],
                "host_memory": memory_events(), "top_rss": top_rss(6)}), flush=True)
            with (OUT / "server-log-summary.jsonl").open("a") as stream:
                stream.write(json.dumps({"t": time.time(), **snapshot}) + "\n")
            reason = server_death(server)
            if reason is not None and not died:
                died.append(reason)
                dump_server_death(reason, server_log)
                # SIGTERM: kaggle_run.py stops opening tasks, kills the running ones, harvests and writes submission.
                harness.terminate()
    with pathlib.Path(log).open("w") as stream:
        harness = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    progress = threading.Thread(target=heartbeat, daemon=True)
    progress.start()
    try:
        try:
            returncode = harness.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            harness.kill()
            harness.wait()
            raise
    finally:
        stopped.set()
        progress.join(timeout=120)
        harness_lines()
    if died:
        raise RuntimeError(f"codex harness stopped because the server died: {died[0]}")
    if returncode:
        raise RuntimeError(f"owned phase failed({returncode}); see {log}")

def codex_arc2_result(output):
    summary = output / "run" / "summary.json"
    results = output / "run" / "results"
    result = {"summary": json.loads(summary.read_text()) if summary.is_file() else None,
              "task_results": len(list(results.glob("*.json"))) if results.is_dir() else 0}
    print("h28: codex harness result " + json.dumps(result)[:2000], flush=True)
    return result

def run_notebook(installed, config, manifest):
    import urllib.error, urllib.request
    if config.get("mtp", False) and config.get("candidate_id") in ("r7-r4", "r8-r2", "r9"):
        raise RuntimeError("MTP candidate has known draft-probability relay defect; use r9-r2+h39")
    payload = pathlib.Path(installed["payload_repo"])
    probes = payload / "work/qwen38fn_l4/probes"
    sys.path[:0] = [str(probes), str(payload / "src")]
    launcher = load_module(payload / manifest["payload"]["launcher"], "arcforge_h28_launcher")
    dflash = stage_dflash(config) if config.get("dflash_k") is not None else None
    spec_decode = config.get("mtp", False) or dflash is not None
    profile = launch_profile(launcher, config, dflash)
    runtime = pathlib.Path(installed["runtime"])
    python, base_env, runtime_receipt = launcher.runtime_env(runtime, "hisparse", str(installed["python"]))
    env = {**base_env, **installed["env"], "NCCL_P2P_DISABLE": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "ARC_HISPARSE_DISTRIBUTED_SPEC": "0",
           "ARC_HISPARSE_STARTUP_CACHE_DROP": "1"}
    if config.get("coarse_war_barrier", False):
        env["SGLANG_FORCE_COARSE_WAR_BARRIER"] = "1"
    if config.get("think_budget") is not None:
        if not spec_decode:
            raise RuntimeError("ARC_THINK_BUDGET acts in the speculative verify only")
        env["ARC_THINK_BUDGET"] = str(config["think_budget"])
    if spec_decode:
        env["ARC_ATTN_TP_CUSTOM_AR"] = "0"
        for name in ("ARC_FLASHINFER_PCIE_VENDOR", "ARC_FLASHINFER_PCIE_REPORT",
                     "ARC_FLASHINFER_PCIE_ATTN_VENDOR", "ARC_FLASHINFER_PCIE_ATTN_REPORT"):
            env.pop(name, None)
    source_manifest = pathlib.Path(installed["source_manifest"])
    source = pathlib.Path(runtime_receipt["arms"]["hisparse"]["source"])
    source_sha = launcher.verify_source_manifest(source, source_manifest)
    model = pathlib.Path(installed["model"])
    identity = json.loads((model / "verified_identity.json").read_text())
    # The asset manifest names the checkpoint (runtime_setup verified its pinned files); the profile must serve its
    # family: the gEAN320 packed INT3 versions for humming_moe int3packed, REAP320 NVFP4 otherwise. verify_model checks
    # the pinned config.json against runpod_runtime.MODEL_IDENTITIES.
    want_model = "/".join(part.lower() if i == 2 else part for i, part in enumerate(manifest["model_source"].split("/")))
    int3_family = want_model.startswith("threerabbits/qwen38fn-gean320-int3-gptq/transformers/default/")
    if int3_family != (config.get("humming_moe") == "int3packed"):
        raise RuntimeError(f"asset model {want_model} does not match humming_moe {config.get('humming_moe')!r}")
    launcher.verify_model(model, {"MANIFEST": {"target_sizes": identity["target_sizes"],
                                               "target_sha256": identity["target_sha256"]}, "IDENTITY": want_model})
    cache = runtime / "cache-sm89-hisparse"
    cache.mkdir(exist_ok=True)
    env = launcher.profile_env(env, source_manifest, profile, cache)
    env["PYTHONPATH"] = str(payload / "src") + ":" + env["PYTHONPATH"]
    if config.get("candidate_id") in ("arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"):
        # SGLang's inductor cache (vision tower torch.compile kernels and their autotune) otherwise lands in
        # ~/.cache/sglang/inductor, outside the four compiled-cache roots: cold on every Kaggle run (Runpod 10-04:
        # 110-190 s before the first image prefill). Inside the Triton root it is exported and restored with it.
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "triton" / "inductor")
    if config.get("dflash_compile") is not None:
        env["ARC_DFLASH_COMPILE"] = config["dflash_compile"]
    if config.get("sharded_verify", False):
        env["ARC_SHARDED_VERIFY"] = "1"
    if config.get("ple_int5", False):
        env["ARC_PLE_INT5"] = "1"
    if config.get("ple_nvfp4", False):
        env["ARC_PLE_NVFP4"] = "1"
    if config.get("extend_nosync", False):
        env["ARC_GDN_HOST_CHUNK_INDICES"] = "1"
        env["ARC_MM_NO_SYNC"] = "1"
    if config.get("dflash_session_park", False):
        env.update({"ARC_DFLASH_EXACT_PREFILL": "1", "ARC_DFLASH_SESSION_PARK": "1", "ARC_DFLASH_REFRESH_ASYNC": "1"})
    if config.get("dflash_shared_graph_pool", False):
        env["ARC_DFLASH_SHARED_GRAPH_POOL_GROUP"] = "1"
    if config.get("mm_release_after_prefill", False):
        env["ARC_MM_RELEASE_AFTER_PREFILL"] = "1"
    if config.get("mm_lazy_media", False):
        env["ARC_MM_LAZY_MEDIA"] = "1"
    if config.get("dflash_w4a8") is not None:
        env["ARC_DFLASH_W4A8"] = config["dflash_w4a8"]
    # plan B drafter serving opt-ins (fork 60997af63a): CONFIG carries one only when it is not the fork's default
    for name, variable in (("dflash_merge", "ARC_DFLASH_MERGE"), ("dflash_inject_kv", "ARC_DFLASH_INJECT_KV"),
                           ("dflash_fuse_once", "ARC_DFLASH_FUSE_ONCE"), ("dflash_kv_dtype", "ARC_DFLASH_KV_DTYPE"),
                           ("dflash_ring_fp8_dot", "ARC_DFLASH_RING_FP8_DOT")):
        if name in config:
            env[variable] = {True: "1", "gemm": "gemm", "fp8_e4m3": "fp8_e4m3"}[config[name]]
    if config.get("shm_gather", False):
        # the module and both call sites must be in the installed tree, or the flag would be read by nothing
        for rel, marker in (("sglang/srt/distributed/arc_shm_gather.py", "def all_gather("),
                            ("sglang/srt/managers/scheduler_components/dp_attn.py", "_arc_shm_gather.enabled()"),
                            ("sglang/srt/mem_cache/qsa_hisparse_distributed.py", "arc_shm_gather.all_gather(own")):
            if marker not in (source / rel).read_text():
                raise RuntimeError(f"shm_gather: {marker!r} is not in the installed {source / rel}")
        env["ARC_SHM_GATHER"] = "1"
    if config.get("admission_reclaimable_file", False):
        rel = "sglang/srt/mem_cache/qsa_hisparse_distributed.py"
        if 'os.environ.get("ARC_ADMISSION_RECLAIMABLE_FILE") == "1"' not in (source / rel).read_text():
            raise RuntimeError(f"admission_reclaimable_file: the installed {source / rel} does not read it")
        env["ARC_ADMISSION_RECLAIMABLE_FILE"] = "1"
    if config.get("inductor_compile_threads") is not None:
        env["TORCHINDUCTOR_COMPILE_THREADS"] = str(config["inductor_compile_threads"])
    if config.get("heap_diag") is not None:
        env["ARC_HEAPDIAG"] = str(config["heap_diag"])
    if config.get("qsa_index_shard") is not None:
        # an installed tree without the switch would ignore ARC_QSA_INDEX_SHARD and silently serve the unsharded index
        # under the sharded hot-block plan: refuse it before the server starts
        for rel, marker in QSA_INDEX_SHARD_MARKERS.items():
            path = source / rel
            if not path.is_file() or marker not in path.read_text():
                raise RuntimeError(f"qsa_index_shard {config['qsa_index_shard']}: installed tree {source} lacks "
                                   f"{marker!r} in {rel} (fork exp/l4-qsa-index-shard)")
        env["ARC_QSA_INDEX_SHARD"] = config["qsa_index_shard"]
    if config.get("dflash_step_opt", False):
        env.update({"ARC_DFLASH_RING_OUT_T": "1", "ARC_DFLASH_BATCHED_COMMIT": "1", "ARC_DFLASH_FUSED_PROPOSE": "1"})
    if config.get("hc_gate_precompute", False):
        env["ARC_HC_GATE_PRECOMPUTE"] = "1"
    if config.get("qsa_sm89_decode") is not None:
        env["ARC_QSA_SM89_DECODE"] = str(config["qsa_sm89_decode"])
    if config.get("qsa_sm89_stages") is not None:
        env["ARC_QSA_SM89_STAGES"] = str(config["qsa_sm89_stages"])
    if config.get("blas_lt", False):
        env["TORCH_BLAS_PREFER_CUBLASLT"] = "1"
    if config.get("draft_vocab") is not None:
        draft_vocab = payload / config["draft_vocab"]
        if not draft_vocab.is_file():
            raise RuntimeError(f"draft vocabulary {draft_vocab} is not in the payload")
        env["ARC_DFLASH_DRAFT_VOCAB"] = str(draft_vocab)
    if config.get("target_vocab") is not None:
        target_vocab = payload / config["target_vocab"]
        if not target_vocab.is_file():
            raise RuntimeError(f"target vocabulary {target_vocab} is not in the payload")
        env["ARC_TARGET_VOCAB"] = str(target_vocab)
    hostcomm = hostcomm_env(python, env, payload, runtime, config) if config.get("hostcomm", False) else None
    if hostcomm is not None:
        env.update(hostcomm["env"])
    if config.get("think_budget") is not None and config.get("sharded_verify") and (
            hostcomm is None or "v16_patch" not in env["PYTHONPATH"]):
        raise RuntimeError("ARC_THINK_BUDGET on ARC_SHARDED_VERIFY needs the v16_patch sitecustomize on the server's "
                           "PYTHONPATH (it forces on the gathered rows)")
    weight_patch = None
    if config.get("model_patch") is not None:
        # the v16_patch sitecustomize (on the server's PYTHONPATH through hostcomm_env) swaps the patch's tensors into
        # the target's weight iterator; the patch must have been made against exactly this package
        if hostcomm is None or "v16_patch" not in env["PYTHONPATH"]:
            raise RuntimeError("model_patch needs the v16_patch sitecustomize on the server's PYTHONPATH")
        patch_dir = runtime_setup.model_mount(config["model_patch"]["model"])
        if sha256(patch_dir / "patch.json") != config["model_patch"]["patch_sha256"]:
            raise RuntimeError(f"{patch_dir}/patch.json is not the pinned {config['model_patch']['patch_sha256']}")
        patch_manifest = json.loads((patch_dir / "patch.json").read_text())
        if patch_manifest["base_manifest_sha256"] != identity["target_sha256"]["deploy_manifest.json"]:
            raise RuntimeError(f"weight patch was made against deploy_manifest {patch_manifest['base_manifest_sha256']}, "
                               f"the served package is {identity['target_sha256']['deploy_manifest.json']}")
        for name, entry in sorted({row["file"]: row for row in patch_manifest["tensors"].values()}.items()):
            if not (patch_dir / name).is_file():
                raise RuntimeError(f"weight patch file missing: {patch_dir / name}")
        weight_patch = {"dir": str(patch_dir), "tensors": len(patch_manifest["tensors"]),
                        "deploy_manifest_sha256": patch_manifest["deploy_manifest_sha256"]}
        env["ARC_L4_WEIGHT_PATCH"] = str(patch_dir)
        env["ARC_L4_WEIGHT_PATCH_MODEL"] = str(model)
        print(f"h28: weight patch {config['model_patch']['model']} ({weight_patch['tensors']} tensors) over {model}",
              flush=True)
    command = launcher.server_args(python, model, config["port"], profile)
    # 'auto' preprocesses images on GPU0 inside the tokenizer process (~220 MB CUDA context);
    # at C80 that OOMed the vision gate (Kaggle v11). The frozen launcher has no hook for it.
    command += ["--image-processor-backend", config["image_processor_backend"]]
    if config.get("vision_offload", False):
        # SGLang's startup warmup for a VLM sends a 32x32 image; with the vision tower on CPU it dies in the
        # tower's attention-TP all-reduce (Kaggle v15 v2). The batch gate is the first (text) request instead.
        command.append("--skip-server-warmup")
    if config.get("harness") == "codex":
        # The G4 hs55 codex notebook's serving contract: its served name, and OpenAI sampling defaults because codex
        # sends no temperature/top_p (the parsers stay qwen3 / qwen3_coder: what G4's "auto" resolves to).
        command = set_server_flag(command, "--served-model-name", CODEX_SERVED_MODEL)
        command += ["--sampling-defaults", "openai"]
    if config.get("dp_lm_head", False):
        command.append("--enable-dp-lm-head")
    if config.get("mtp", False):
        launcher.verify_native_host_support(source, profile)
        if config["candidate_id"] in ("r8-r2", "r9", "r9-r2"):
            launcher.verify_phase_packed_support(source, profile)
        if config["candidate_id"] == "r9-r2":
            launcher.verify_mtp_steps_support(source, profile)
    if dflash is not None:
        launcher.verify_dflash_support(source, profile)
    if spec_decode and "--speculative-use-rejection-sampling" not in command:
        raise RuntimeError("original ARC2 stochastic sampling requires explicit MTP rejection sampling")
    forbidden = ("--disable-cuda-graph", "--language-only", "--language-model-only") if spec_decode else (
        "--speculative-algorithm", "--enable-linear-replayssm-spec")
    if any(flag in command for flag in (*forbidden,
                                        "--disable-cuda-graph", "--language-only", "--language-model-only")):
        raise RuntimeError("frozen h28 launch must retain graph and vision withNoMTP")
    server_dir = OUT / "server"
    server_dir.mkdir()
    evidence = {"status": "starting", "profile": config, "source_manifest_sha256": source_sha,
                "args": command, "environment": launcher.recorded_env(env, source_manifest, profile, cache),
                "phases": [], "cleanup": "pending", "long_context_capacity": "unverified",
                "hostcomm": hostcomm, "dflash": dflash,
                "dflash_compile_env": env.get("ARC_DFLASH_COMPILE"),  # recorded_env (launcher) does not list it
                # builder-set switches recorded_env does not list (the hostcomm env above carries v16_patch's)
                "builder_env": {k: env.get(k) for k in ("ARC_SHARDED_VERIFY", "ARC_PLE_INT5", "ARC_PLE_NVFP4", "ARC_GDN_HOST_CHUNK_INDICES",
                                                        "ARC_MM_NO_SYNC", "ARC_DFLASH_EXACT_PREFILL",
                                                        "ARC_DFLASH_SESSION_PARK", "ARC_DFLASH_REFRESH_ASYNC",
                                                        "ARC_DFLASH_RING_OUT_T", "ARC_DFLASH_BATCHED_COMMIT",
                                                        "ARC_DFLASH_FUSED_PROPOSE", "ARC_HC_GATE_PRECOMPUTE",
                                                        "ARC_QSA_SM89_DECODE", "ARC_QSA_SM89_STAGES",
                                                        "TORCH_BLAS_PREFER_CUBLASLT", "ARC_MM_RELEASE_AFTER_PREFILL",
                                                        "ARC_MM_LAZY_MEDIA", "ARC_DFLASH_W4A8", "ARC_SHM_GATHER",
                                                        "ARC_ADMISSION_RECLAIMABLE_FILE",
                                                        "TORCHINDUCTOR_COMPILE_THREADS", "ARC_QSA_INDEX_SHARD")}
                               | {k: env.get(k) for name, k in (("dflash_merge", "ARC_DFLASH_MERGE"),
                                                               ("dflash_inject_kv", "ARC_DFLASH_INJECT_KV"),
                                                               ("dflash_fuse_once", "ARC_DFLASH_FUSE_ONCE"),
                                                               ("dflash_kv_dtype", "ARC_DFLASH_KV_DTYPE"),
                                                               ("dflash_ring_fp8_dot", "ARC_DFLASH_RING_FP8_DOT"))
                                  if name in config},
                "draft_vocab": None if env.get("ARC_DFLASH_DRAFT_VOCAB") is None else {
                    "path": env["ARC_DFLASH_DRAFT_VOCAB"], "sha256": sha256(env["ARC_DFLASH_DRAFT_VOCAB"])},
                "target_vocab": None if env.get("ARC_TARGET_VOCAB") is None else {
                    "path": env["ARC_TARGET_VOCAB"], "sha256": sha256(env["ARC_TARGET_VOCAB"])}}
    server = None
    started = time.monotonic()
    try:
        print("h28: checking launch configuration and native CUDA", flush=True)
        # Imports can create flashinfer_jit.log; keep restore destinations fresh.
        probe_env = launcher.profile_env(env, source_manifest, profile, runtime / "cache-sm89-preflight")
        if "TORCHINDUCTOR_CACHE_DIR" in env:
            probe_env["TORCHINDUCTOR_CACHE_DIR"] = str(runtime / "cache-sm89-preflight" / "triton" / "inductor")
        launcher.verify_h25_shape(python, probe_env, source, profile)
        launcher.config_preflight(python, probe_env, command, server_dir)
        launcher.require_free_port(config["port"])
        gpu = launcher.gpu_probe(python, probe_env, server_dir)
        affinity_preflight = None
        if config.get("cpu_affinity", False):
            from runpod_cpu_affinity import collect, validate
            affinity_preflight = validate(collect(gpu["topology"]))
            write_json(server_dir / "cpu_affinity_preflight.json", affinity_preflight)
        if config.get("seed_cache_from"):
            evidence["compiled_cache_restore"] = seed_compiled_cache(pathlib.Path(config["seed_cache_from"]), cache)
        else:
            evidence["compiled_cache_restore"] = restore_compiled_cache(
                python, runtime, env, manifest, gpu, payload)
        write_json(server_dir / "compiled_cache_restore.json", evidence["compiled_cache_restore"])
        print("h28: compiled cache " + evidence["compiled_cache_restore"]["status"], flush=True)
        print("h28: rewarming original checkpoint immediately before model load", flush=True)
        write_json(server_dir / "checkpoint_rewarm.json", checkpoint_reader.rewarm(model, identity, place_on_roomiest_node=True))
        if config.get("host_memory_trace", False):
            start_host_memory_trace(server_dir / "host_memory.jsonl")
        with (server_dir / "server.log").open("w") as stream:
            server = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        evidence.update(pid=server.pid, pgid=server.pid)
        write_json(OUT / "run_result.json", evidence)
        base_url = "http://127.0.0.1:" + str(config["port"])
        deadline = time.monotonic() + 1800
        next_progress = 0
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"owned model exited during startup({server.returncode}); see server.log")
            if time.monotonic() >= next_progress:
                print(f"h28: starting model, elapsed {time.monotonic() - started:.0f}s", flush=True)
                with (server_dir / "server.log").open("rb") as progress_log:
                    progress_log.seek(0, 2)
                    progress_log.seek(max(0, progress_log.tell() - 4096))
                    print("\n".join(progress_log.read().decode(errors="replace").splitlines()[-5:]), flush=True)
                next_progress = time.monotonic() + 30
            try:
                with urllib.request.urlopen(base_url + "/get_server_info", timeout=5) as response:
                    info = json.load(response)
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                # A dead server is reported by server.poll() on the next iteration.
                time.sleep(5)
                continue
            logs = (server_dir / "server.log").read_text(errors="replace")
            if "The server is fired up and ready to roll!" not in logs:
                # Uvicorn answers before SGLang's warmup request finishes; its first prefill
                # compiles/autotunes Triton kernels and blocks the scheduler, so a gate that
                # starts now times out on /server_info (Kaggle v12).
                time.sleep(5)
                continue
            workers = launcher.validate_server_info(info, profile)
            trees = re.findall(r"ARC_HISPARSE_QSA_TREE=([0-9a-f]{64})", logs)
            if not trees or any(tree != source_sha for tree in trees):
                raise RuntimeError("model source receipt differs from the pinned SOURCE manifest")
            admission = launcher.validate_receipts(logs, profile)
            if config.get("mtp", False):
                host_pools = launcher.validate_host_pool_receipts(logs, profile, admission)
                write_json(server_dir / "native_draft_pool_receipts.json", {
                    "requested": profile.native_mtp_host_kv, "receipts": host_pools})
            cache_receipts = [json.loads(line.split("ARC_HISPARSE_STARTUP_CACHE_DROP=", 1)[1])
                              for line in logs.splitlines() if line.startswith("ARC_HISPARSE_STARTUP_CACHE_DROP=")]
            if {row["global_rank"] for row in cache_receipts} != {0, 1, 2, 3} or any(
                    row["status"] != "advised" or not row["all_ranks_acknowledged"] for row in cache_receipts):
                raise RuntimeError("missing actual four-rank checkpoint-loaded cache release receipts")
            write_json(server_dir / "checkpoint_cache_release.json", cache_receipts)
            patch_receipts = [line for line in logs.splitlines() if line.startswith("ARC_L4_WEIGHT_PATCH_LOADED ")]
            if weight_patch is None:
                if patch_receipts or "ARC_L4_WEIGHT_PATCH" in logs:
                    raise RuntimeError("weight patch receipts without a configured model_patch")
            else:
                ranks = sorted(int(re.search(r" global_rank=(-?[0-9]+)", line)[1]) for line in patch_receipts)
                want = f"ARC_L4_WEIGHT_PATCH_LOADED tensors={weight_patch['tensors']} "
                if ranks != [0, 1, 2, 3] or any(not line.startswith(want) or
                        f"deploy_manifest={weight_patch['deploy_manifest_sha256']} " not in line for line in patch_receipts):
                    raise RuntimeError(f"weight patch not loaded once on each of the four ranks: {patch_receipts}")
                write_json(server_dir / "weight_patch_receipts.json", {**weight_patch, "receipts": patch_receipts})
                print("h28: " + patch_receipts[0], flush=True)
            write_json(server_dir / "admission_receipts.json", {"receipts": admission, "admission_verified": True})
            if hostcomm is not None:
                write_json(server_dir / "hostcomm_receipts.json", hostcomm_receipts(logs, config))
            if config.get("dp_lm_head", False) and info.get("enable_dp_lm_head") is not True:
                raise RuntimeError(f"--enable-dp-lm-head requested, server reports {info.get('enable_dp_lm_head')!r}")
            if dflash is not None:
                write_json(server_dir / "dflash_receipts.json", dflash_receipts(logs, config, dflash))
            write_json(server_dir / "prefill_ple_receipts.json", prefill_ple_receipts(logs, config))
            write_json(server_dir / "heapdiag_receipts.json", heapdiag_receipts(logs, config))
            write_json(server_dir / "qsa_index_shard_receipts.json", qsa_index_shard_receipts(logs, config))
            write_json(server_dir / "host_private_receipts.json", host_private_receipts(logs, config, cgroup_shmem_bytes()))
            if config.get("harness") == "codex":
                disabled = [line for line in logs.splitlines() if "Responses API (/v1/responses) disabled" in line]
                if disabled or info.get("served_model_name") != CODEX_SERVED_MODEL:
                    raise RuntimeError(f"codex needs /v1/responses served as {CODEX_SERVED_MODEL!r}: "
                                       f"{disabled[:1]} served_model_name={info.get('served_model_name')!r}")
            if affinity_preflight is not None:
                from runpod_cpu_affinity import readback
                affinity = readback(logs, server.pid, server.pid, affinity_preflight["recommended_rank_cpus"])
                write_json(server_dir / "cpu_affinity_readback.json", affinity)
            write_json(server_dir / "deployment_start.json", {"workers": workers, "server_info": info, "args": command})
            evidence.update(status="ready", startup_seconds=time.monotonic() - started,
                            admission_verified=True, allocated_pools_verified=True)
            break
        else:
            raise TimeoutError("owned model startup deadline exceeded")
        print(f"h28: model ready in {evidence['startup_seconds']:.1f}s", flush=True)
        write_json(server_dir / "memory_after_startup.json", memory_receipt())
        v16 = config.get("candidate_id") in ("arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv")
        if v16:
            # Fail before any traffic when a card is already near its limit after every graph capture and warmup.
            logs = (server_dir / "server.log").read_text(errors="replace")
            memory = {"log_per_rank": startup_memory_receipts(logs, config), "nvidia_smi": gpu_free_rows()}
            write_json(server_dir / "gpu_memory_after_startup.json", memory)
            print("h28: free GiB after graphs " + json.dumps({name: row["after_all_graphs_gib_estimate"]
                  for name, row in memory["log_per_rank"].items()}) + " nvidia-smi " +
                  json.dumps([row["free_gib"] for row in memory["nvidia_smi"]]), flush=True)
            low = [row for row in memory["nvidia_smi"] if row["free_mib"] / 1024 < config["min_free_gib"]]
            if low:
                raise RuntimeError(f"GPU free memory after startup below {config['min_free_gib']} GiB: {low}")
            evidence["gpu_memory_after_startup"] = memory
        commands = [("batch", [python, str(probes / "runpod_batch.py"), "--base-url", base_url,
            "--model", str(model), "--output-dir", str(OUT / "batch"), "--concurrency", str(config["running"]),
            "--new-tokens", "128", "--timeout", "300"]),
            ("vision", [python, str(probes / "runpod_vision.py"), "--base-url", base_url,
             "--model", str(model), "--output-dir", str(OUT / "vision"), "--timeout", "300"])]
        if config.get("vision_offload", False):
            # The vision tower now lives on CPU and any image request fails loudly by design.
            commands = [entry for entry in commands if entry[0] != "vision"]
            evidence["phases"].append({"name": "vision", "status": "skipped: vision offload"})
        if config.get("attn_dp", 2) == 1:
            # runpod_batch / runpod_vision route every request to DP 0 or 1 and check both DPs' receipts; pure TP4 has
            # one scheduler. The v16 vision-peak / long-prefill gates below and the harness still run.
            for name, _ in commands:
                evidence["phases"].append({"name": name, "status": "skipped: DP-pair gate on pure TP4 (attn_dp 1)"})
            commands = []
        for name, phase_command in commands:
            phase_started = time.monotonic()
            print(f"h28: running {name} gate", flush=True)
            run_command(phase_command, env, OUT / (name + ".log"), 1200)
            result = phase_result(OUT / name)
            if name == "batch":
                if result["completed_requests"] != config["running"] or result["completed_requests_by_dp"] != {
                        "0": config["running"] // 2, "1": config["running"] // 2}:
                    raise RuntimeError("short batch did not complete the requested both-DP concurrency")
                graphs = decode_graph_receipts((server_dir / "server.log").read_text(errors="replace"), config["running"] // 2)
                write_json(server_dir / "decode_graph_receipts.json", {"receipts": graphs})
                if v16:
                    write_json(server_dir / "shm_gather_receipts.json",
                               shm_gather_receipts(server.pid, config.get("shm_gather", False)))
                if dflash is not None:
                    write_json(OUT / "batch" / "dflash_acceptance.json", dflash_batch_counters(result["jobs"]))
                if config.get("mtp", False):
                    from runpod_mtp import aggregate_counters, request_counters
                    for row in result["jobs"]:
                        row["mtp_counters"] = request_counters(row["meta_info"], **mtp_counter_kwargs(config))
                    write_json(OUT / "batch" / "mtp_acceptance.json", {
                        "all": aggregate_counters(result["jobs"], **mtp_counter_kwargs(config)),
                        "by_dp": {str(dp): aggregate_counters([row for row in result["jobs"] if row["dp"] == dp], **mtp_counter_kwargs(config))
                                  for dp in (0, 1)}})
            evidence["phases"].append({"name": name, "status": "passed"})
            print(f"h28: {name} gate passed in {time.monotonic() - phase_started:.1f}s", flush=True)
            write_json(OUT / "run_result.json", evidence)
        if config.get("mtp", False):
            # Native model/TP2 sampling and retained-history gate; independent of ARC2 budgets.
            sampling_out = OUT / "mtp-sampling-sessions"
            sampling_command = [python, str(probes / "runpod_capacity_mtp.py"), "--mtp",
                "--base-url", base_url, "--model", str(model), "--output-dir", str(sampling_out),
                "--concurrency", str(min(config["running"], 8)), "--input-tokens", "8192",
                "--warm-tokens", "128", "--output-tokens", "512", "--ignore-eos",
                "--temperature", "1", "--top-p", ".95", "--top-k", "-1",
                "--min-active-seconds", ".5", "--poll-interval", ".1"]
            sampling_command += mtp_probe_args(config)
            run_command(sampling_command, env, OUT / "mtp-sampling-sessions.log", 1800)
            sampling_result = phase_result(sampling_out)
            fixture = json.loads((sampling_out / "requests/cold-0000/request.json").read_text())
            prompts = OUT / "mtp-correctness-prompts.json"
            write_json(prompts, {"p9001": fixture["input_ids"]})
            correctness = OUT / "mtp-correctness.json"
            run_command([python, str(probes / "runpod_correctness.py"), "http", "--mtp",
                "--base-url", base_url, "--prompts", str(prompts), "--snapshot-interval", "64",
                "--new-tokens", "128", "--sessions-per-dp", str(min(config["running"] // 2, 2)),
                "--output", str(correctness), *mtp_probe_args(config)], env, OUT / "mtp-correctness.log", 1800)
            result = json.loads(correctness.read_text())
            if result["metadata_lifecycle_verdict"] != "PASSED" or result["cleanup_errors"]:
                raise RuntimeError("MTP snapshot/resume/mismatch/close lifecycle gate failed")
            evidence["phases"].append({"name": "mtp-native-sampling-and-session-lifecycle", "status": "passed"})
            evidence["mtp_sampling"] = sampling_result["mtp_acceptance"]
            write_json(OUT / "run_result.json", evidence)
        if v16:
            # Images like codex's first message (every board as a PNG) and long prefills, now rather than mid-harness:
            # the vision tower's activation peak and any kernel first seen on a long prefill land in a gate.
            served = CODEX_SERVED_MODEL if config.get("harness") == "codex" else "tir"
            gates = BUNDLE_ROOT / "work/qwen38fn_l4/kaggle/v16_gates.py"
            for name, gate_command in (
                    ("vision-peak", [python, str(gates), "vision-peak", "--out", str(OUT / "vision-peak"),
                                     "--base-url", base_url, "--model", served,
                                     "--images", str(config["vision_peak_images"]),
                                     "--concurrency", str(config["vision_peak_concurrency"]),
                                     "--min-free-gib", str(config["min_free_gib"])]),
                    ("long-prefill", [python, str(gates), "long-prefill", "--out", str(OUT / "long-prefill"),
                                      "--base-url", base_url, "--min-free-gib", str(config["min_free_gib"]),
                                      "--lengths", *config["long_prefill_lengths"].split(",")])):
                phase_started = time.monotonic()
                print(f"h28: running {name} gate", flush=True)
                run_command(gate_command, env, OUT / (name + ".log"), 1800)
                result = phase_result(OUT / name)
                print(f"h28: {name} gate passed in {time.monotonic() - phase_started:.1f}s, GPU free GiB "
                      + json.dumps([row["free_gib"] for row in result["gpu_free_after"]]), flush=True)
                evidence["phases"].append({"name": name, "status": "passed",
                                           "gpu_free_after": result["gpu_free_after"]})
                write_json(OUT / "run_result.json", evidence)
            evidence["new_autotune_entries"] = new_autotune_entries(cache, evidence["compiled_cache_restore"])
            print("h28: Triton autotune entries created after the cache restore: "
                  + json.dumps(evidence["new_autotune_entries"]), flush=True)
            evidence["server_log_summary"] = SERVER_LOG_SUMMARY(server_dir / "server.log").update()
            write_json(OUT / "run_result.json", evidence)
        rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in ("1", "true")
        evidence["compiled_cache_before_harness"] = export_compiled_cache(python, runtime, env, manifest, config, gpu, "before-arc2")
        if config.get("gates_only", False):
            evidence["status"] = "gates_passed_no_harness"
            evidence["phases"].append({"name": "harness", "status": "skipped: gates-only build"})
            return evidence
        if config.get("harness") == "codex":
            output = V16_DIR / "arc2run"  # kaggle_run.py: outside /kaggle, which it closes (chmod 700) while it runs
            command, selected, deadline_unix = codex_arc2_command(sys.executable, model, config, base_url, output, rerun)
            write_json(OUT / "submission_invocation.json", {"args": command, "mode": "competition-rerun" if rerun else "saverun",
                                                            "deadline_unix": deadline_unix, "tasks": len(selected)})
            print("h28: running the codex ARC2 harness on " + str(len(selected)) + " tasks, deadline "
                  + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(deadline_unix)), flush=True)
            # A server boot slower than end_by_boot_seconds can still leave too little harness time before the deadline.
            end_by_gate(deadline_unix, config["end_by_min_harness_seconds"], "harness start")
            # The harness owns the deadline; this only kills a harness that overruns it by the grace period.
            hard_remaining = deadline_unix + config["harness_grace_seconds"] - time.time()
            # The harness runs on the image's Python with the notebook's own environment: the server env carries the
            # runtime PYTHONPATH and the hostcomm sitecustomize, neither of which belongs in the agent's world.
            # kaggle_run closes /kaggle (chmod 700) and restores it in a finally that a SIGKILL (the grace timeout
            # below) never reaches: put the mode back here whatever happens.
            kaggle_mode = os.stat("/kaggle").st_mode & 0o7777
            try:
                run_codex_arc2(command, dict(os.environ), OUT / "codex-arc2.log", output, server_dir / "server.log", hard_remaining, server)
            finally:
                if os.stat("/kaggle").st_mode & 0o7777 != kaggle_mode:
                    os.chmod("/kaggle", kaggle_mode)
                    print("h28: restored /kaggle to " + oct(kaggle_mode) + " after the codex harness", flush=True)
            evidence["harness_result"] = codex_arc2_result(output)
            # Autotune at serving time is a per-rank benchmark: the two ranks of an attention-TP pair can pick
            # different configs, and DFlash proposals then diverge (k8cold 10-04: lifecycle mismatch). Record it.
            evidence["new_autotune_entries_after_harness"] = new_autotune_entries(cache, evidence["compiled_cache_restore"])
            print("h28: autotune entries created by the end of the harness: "
                  + json.dumps(evidence["new_autotune_entries_after_harness"]), flush=True)
            evidence["compiled_cache"] = export_compiled_cache(python, runtime, env, manifest, config, gpu, "after-arc2")
            evidence["submission"] = ensure_submission(selected_challenges(rerun, config), "codex harness returned")
            evidence["server_log_summary"] = SERVER_LOG_SUMMARY(server_dir / "server.log").update()
            evidence["phases"].append({"name": "competition-rerun" if rerun else "saverun-validation", "status": "completed"})
            evidence["status"] = "codex_harness_completed"
            return evidence
        if config.get("capacity_phase", False) and not rerun:
            if config.get("mtp", False):
                raise RuntimeError("frozen NoMTP capacity entry cannot be used for the MTP candidate; use its explicit native capacity probe")
            record = manifest["capacity_phase"]
            entry = payload / record["file"]
            if sha256(entry) != record["sha256"]:
                raise RuntimeError("long-capacity entry differs from the pinned payload")
            print("h28: running canonical C64 x100k retained-history capacity phase", flush=True)
            phase_output = OUT / "capacity-c64-100k"
            capacity_command = [python, str(entry), "--base-url", base_url, "--model", str(model),
                "--fast-dir", str(WORK / "capacity-c64-100k"), "--output-dir", str(phase_output)]
            run_capacity_logged(capacity_command, env, OUT / "capacity-c64-100k.log", WORK / "capacity-c64-100k")
            capacity = json.loads((phase_output / "phase-result.json").read_text())
            if capacity["status"] != "passed" or not capacity["final_mirror_completed"]:
                raise RuntimeError("canonical C64 x100k phase failed its native history/cleanup gate")
            evidence["long_context_capacity"] = "passed-c64-100k"
            evidence["capacity"] = capacity
            evidence["phases"].append({"name": "capacity-c64-100k", "status": "passed"})
            write_json(OUT / "run_result.json", evidence)
        original_output = OUT / "original-arc2"
        submission, selected = original_arc2_command(python, model, payload, config, manifest, base_url, original_output, rerun)
        monitor = None
        if config.get("live_logging", False):
            from types import SimpleNamespace
            observer_record = manifest["live_status"]
            observer_file = payload / observer_record["file"]
            if sha256(observer_file) != observer_record["sha256"]:
                raise RuntimeError("live status observer differs from pinned payload")
            observer_module = load_module(observer_file, "arcforge_arc2_live_status")
            monitor_output = OUT / "live-status"
            monitor_output.mkdir()
            observer = observer_module.Observer(SimpleNamespace(base_url=base_url,
                transport_dir=original_output / "hisparse-transport", dp_size=2, interval=30., http_timeout=2.))
            monitor = {"module": observer_module, "observer": observer, "output": monitor_output}
        print("h28: running original ARC2 agentic TIR harness", flush=True)
        write_json(OUT / "submission_invocation.json", {"args": submission, "mode": "competition-rerun" if rerun else "saverun"})
        budget = json.loads((original_output / "original_invocation.json").read_text())["user_run_overrides"]
        print("h28: time budget " + json.dumps(budget), flush=True)
        hard_remaining = NOTEBOOK_STARTED + budget["notebook_total_minutes"] * 60 - time.monotonic()
        if hard_remaining <= 0:
            raise TimeoutError("whole-notebook time budget exhausted before original ARC2 inference")
        run_original_arc2(submission, env, OUT / "original-arc2.log", original_output, selected, monitor, hard_remaining)
        evidence["baseline"] = original_arc2_result(original_output, selected, rerun)
        evidence["compiled_cache"] = export_compiled_cache(python, runtime, env, manifest, config, gpu, "after-arc2")
        final_submission = OUT.parent / "submission.json"
        if not isinstance(json.loads(final_submission.read_text()), dict):
            raise RuntimeError("original harness did not produce the submission object")
        evidence["submission_sha256"] = sha256(final_submission)
        evidence["phases"].append({"name": "competition-rerun" if rerun else "saverun-validation", "status": "completed"})
        evidence["status"] = "deployment_validated_baseline_measured"
        return evidence
    except BaseException as exc:
        evidence.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        cleanup_errors = []
        try:
            if server is not None:
                try:
                    launcher.stop_owned(server)
                    evidence["model_exit_code"] = server.returncode
                except BaseException as exc:
                    cleanup_errors.append(f"model {type(exc).__name__}: {exc}")
            if cleanup_errors:
                raise RuntimeError("; ".join(cleanup_errors))
            evidence["cleanup"] = "owned-model-stopped" if server is not None else "model-not-started"
        except BaseException as exc:
            evidence.update(status="failed", cleanup=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            evidence["elapsed_seconds"] = time.monotonic() - started
            write_json(OUT / "run_result.json", evidence)
            print(json.dumps({key: evidence[key] for key in ("status", "profile", "cleanup", "elapsed_seconds")}), flush=True)

RUN_RESULT = run_notebook(INSTALLED, CONFIG, ASSET_MANIFEST)
'''


V16_FIELDS = ("dflash_k", "context_length", "swap_in_block_size", "online_fp8", "hc_ag_anydtype", "hc_slot_mb",
              "dp_lm_head", "coarse_war_barrier", "think_budget", "marlin_block_remap", "humming_moe", "humming_moe_tile", "humming_moe_act", "humming_dense", "humming_layers",
              "hc_mix_w8", "embed_offload", "hc_moe_rs", "hc_ag_kind", "hc_ag_dp", "hc_ag_dp_check", "hc_pipe_ar2", "hc_pipe_tp4", "hc_ar2_blocks", "hc_ag_dp_check_every", "hc_fold_epoch", "hc_ar2_p2p", "hc_rs_stage2", "hc_rs_sib_wait", "hc_ag_blocks", "gpu_monitor", "hisparse_miss_stats", "hc_moe_rs_owner", "hc_numa_node", "moe_stats", "mrope_decode_pin", "fwd_timing", "spec_agree_device", "deterministic", "dflash_compile", "draft_vocab", "target_vocab", "spec_trace", "hc_debug", "harness", "gates_only", "min_free_gib", "vision_peak_images", "vision_peak_concurrency",
              "long_prefill_lengths", "seed_cache_from", "attn_dp", "hc_local_slots", "hc_ar2_kind", "hc_moe_rs_fp8", "hc_sumlen", "hc_moe_overlap", "hc_mix_verify", "lmhead_triton",
              "sharded_verify", "ple_int5", "ple_nvfp4", "host_private", "extend_nosync", "dflash_session_park", "dflash_shared_graph_pool", "dflash_step_opt", "hc_gate_precompute", "qsa_sm89_decode", "qsa_sm89_stages", "blas_lt", "prefill_max_requests", "prefill_delayer", "prefill_delayer_max_passes",
              "prefill_delayer_queue_ratio", "prefill_delayer_max_delay_ms", "gdn_state_fmt", "snapshot_host_buffers",
              "session_anchor", "mm_release_after_prefill", "mm_lazy_media", "dflash_w4a8", "heap_diag", "shm_gather", "host_memory_trace",
              "malloc_trim_seconds", "admission_reclaimable_file", "inductor_compile_threads", "qsa_index_shard",
              *PLAIN_DFLASH_OPTS)
REPO = Path(__file__).resolve().parents[3]
#: payload hostcomm hooks whose ARC_MALLOC_TRIM code is this checkout's: l4opt47 (035981c2); 3f0147e1 (ARC_HC_MOE_OVERLAP)
#: changed the hook since then without touching the malloc_trim path (git diff 035981c2 3f0147e1 has no malloc / trim line)
MALLOC_TRIM_HOOKS = ("2d83177f5cdd6c6ef5e40a4c15842bb7e55417213b8856494c4c5d0bc26f79e6",)


def build(manifest: dict, profile: Profile, kernel_id: str, title: str, out: Path, *, dflash_spec: Path | None = None,
          codex_config: str | None = None, hisparse_budget_gib: float | None = None, harness_root: Path = REPO,
          compiled_cache: tuple[str, str] | None = None, model_patch: dict | None = None) -> dict:
    if profile.candidate_id in KNOWN_BAD_RS_CANDIDATES:
        raise ValueError("MTP candidate has known draft-probability relay defect; use r9-r2+h39")
    v16 = profile.candidate_id in DFLASH_CANDIDATES
    if not v16 and (dflash_spec or codex_config or hisparse_budget_gib or compiled_cache or any(
            getattr(profile, name) != Profile.__dataclass_fields__[name].default for name in V16_FIELDS)):
        raise ValueError("v16 options need the sglang-arc-2 arc-main candidate")
    if profile.malloc_trim_seconds is not None:
        hook = "work/qwen38fn_l4/probes/hostcomm/arc_hostcomm_hook.py"
        accepted = {hashlib.sha256((REPO / hook).read_bytes()).hexdigest(), *MALLOC_TRIM_HOOKS}
        if manifest["payload"]["files_sha256"].get(hook) not in accepted:
            raise ValueError(f"malloc_trim_seconds: the payload's {hook} is neither this checkout's nor one of "
                             f"MALLOC_TRIM_HOOKS (ARC_MALLOC_TRIM)")
    if profile.shm_gather:
        tree = manifest["candidate"]["expect_tree"] if "candidate" in manifest else None
        if tree not in SHM_GATHER_TREES:
            raise ValueError(f"shm_gather needs a SHM_GATHER_TREES source (fork all-merge 8d4dea09); the asset manifest "
                             f"pins {tree}")
    if compiled_cache is not None:
        dataset, manifest_sha = compiled_cache
        if not re.fullmatch(r"threerabbits/[a-z0-9-]+", dataset):
            raise ValueError("compiled cache must be an explicit owned private dataset")
        digest(manifest_sha)
        manifest = {**manifest, "compiled_cache": {"dataset": dataset, "manifest_sha256": manifest_sha}}
    check_manifest(manifest)
    original_ctx_cap = manifest["original_arc2"]["config"]["ctx_cap"]
    if profile.harness_ctx_cap is not None and profile.harness_ctx_cap < original_ctx_cap:
        raise ValueError(f"harness ctx cap {profile.harness_ctx_cap} is below the original notebook's {original_ctx_cap}")
    if profile.candidate_id != manifest.get("candidate", {}).get("id"):
        raise ValueError("profile and asset candidate IDs differ; no automatic promotion")
    if (profile.humming_moe == "int3packed") != (manifest["model_source"] in MODELS_INT3):
        raise ValueError(f"humming_moe {profile.humming_moe!r} and the asset model {manifest['model_source']!r} disagree "
                         f"(int3packed serves exactly one of {MODELS_INT3})")
    if profile.draft_vocab is not None:  # the notebook reads it from the payload, not from this checkout
        packed = manifest["payload"]["files_sha256"].get(profile.draft_vocab)
        if packed != hashlib.sha256((REPO / profile.draft_vocab).read_bytes()).hexdigest():
            raise ValueError(f"draft_vocab {profile.draft_vocab}: payload has {packed}, this checkout differs or lacks it")
    if profile.target_vocab is not None:  # likewise read from the payload
        packed = manifest["payload"]["files_sha256"].get(profile.target_vocab)
        if packed != hashlib.sha256((REPO / profile.target_vocab).read_bytes()).hexdigest():
            raise ValueError(f"target_vocab {profile.target_vocab}: payload has {packed}, this checkout differs or lacks it")
    if profile.candidate_id is not None and profile.capacity_phase:
        raise ValueError("canonical frozen capacity phase is NoMTP; candidate capacity uses separate probe")
    if profile.live_logging and "live_status" not in manifest:
        raise ValueError("live logging requires the pinned observer in the payload")
    extra_config, extra_datasets, capacity, bundle = {}, [], None, None
    if v16:
        if (profile.dflash_k is None) != (dflash_spec is None):
            raise ValueError("--dflash-k and --dflash-spec go together")
        if (profile.harness == "codex") != (codex_config is not None):
            raise ValueError("the codex harness needs --codex-config (and only it)")
        if profile.live_logging:
            raise ValueError("the v15 live-status observer reads the frozen harness transport; v16 logs its own heartbeat")
        capacity = capacity_check(profile, hisparse_budget_gib)
        if profile.dflash_k is not None:
            spec = load_dflash_spec(dflash_spec)
            if ("dense" in spec) != (profile.candidate_id in DENSE_DFLASH_CANDIDATES):
                raise ValueError(f"a dense DFlash spec goes with {DENSE_DFLASH_CANDIDATES}, a MoE one with arc-main "
                                 f"(candidate {profile.candidate_id}, spec {dflash_spec})")
            if "architecture" in spec and manifest["candidate"]["expect_tree"] not in PLAIN_DFLASH_TREES:
                raise ValueError(f"the plain DFlash drafter ({dflash_spec}) needs a PLAIN_DFLASH_TREES source (fork "
                                 f"c8500d9fe); the asset manifest pins {manifest['candidate']['expect_tree']}")
            plain_opts = sorted(name for name, default in PLAIN_DFLASH_OPTS.items() if getattr(profile, name) != default)
            if plain_opts and ("architecture" not in spec
                               or manifest["candidate"]["expect_tree"] not in PLAIN_DFLASH_OPT_TREES):
                raise ValueError(f"{plain_opts} serve the plain drafter (a spec with architecture; {dflash_spec} has "
                                 f"{spec.get('architecture', 'none: dense_shared')}) on a PLAIN_DFLASH_OPT_TREES source "
                                 f"(fork 60997af63a); the asset manifest pins {manifest['candidate']['expect_tree']}")
            extra_config["dflash_asset"] = spec
            if profile.draft_vocab is not None:  # dflash_receipts: the pruned successor codebook's row count
                extra_config["draft_vocab_size"] = len(json.loads((REPO / profile.draft_vocab).read_text())["ids"])
            extra_datasets.append(spec["dataset"])
        if profile.harness == "codex":
            relative(codex_config)
            extra_config.update(codex_config=codex_config, harness_grace_seconds=900,
                                end_by_unix=int(datetime.fromisoformat(END_BY_UTC).timestamp()),
                                end_by_margin_seconds=END_BY_MARGIN_SECONDS, end_by_boot_seconds=END_BY_BOOT_SECONDS,
                                end_by_min_harness_seconds=END_BY_MIN_HARNESS_SECONDS)
            if END_BY_MARGIN_SECONDS < extra_config["harness_grace_seconds"] + 300:
                raise ValueError("END_BY_MARGIN_SECONDS must hold the harness grace kill and 5 min of steps after it")
            extra_datasets.append(CODEX_DATASET)
            for minutes in (profile.saverun_minutes, profile.rerun_minutes):
                if minutes is None:
                    raise ValueError("the codex harness needs explicit --saverun-minutes and --rerun-minutes (its deadline)")
        if profile.humming_moe is not None or profile.humming_dense is not None:
            extra_datasets.append(HUMMING_WHEEL["dataset"])
        bundle = harness_bundle(harness_root, profile.harness == "codex", codex_config)
    extra_models = []
    if model_patch is not None:
        # 10-29: an RL version as a weight patch (work/qwen38fn_rl/rl_weight_patch.py) over the unchanged mounted package;
        # the v16_patch sitecustomize swaps its tensors into the target's weight iterator (ARC_L4_WEIGHT_PATCH)
        if not v16 or not profile.hostcomm or profile.humming_moe is None:
            raise ValueError("model_patch rides on the v16_patch sitecustomize: v16 + hostcomm + humming_moe")
        if set(model_patch) != {"model", "patch_sha256"} or not re.fullmatch(
                r"threerabbits/[a-z0-9-]+/Transformers/default/[0-9]+", model_patch["model"]) or not re.fullmatch(
                r"[0-9a-f]{64}", model_patch["patch_sha256"]):
            raise ValueError(f"model_patch must be {{model: threerabbits/<slug>/Transformers/default/<N>, "
                             f"patch_sha256: <hex>}}, got {model_patch}")
        if model_patch["model"].rsplit("/", 3)[0] == manifest["model_source"].rsplit("/", 3)[0]:
            raise ValueError("model_patch must be its own Kaggle model, not a version of the package it patches")
        extra_config["model_patch"] = dict(model_patch)
        extra_models.append(model_patch["model"])
    metadata = kernel_metadata(manifest, kernel_id, title, tuple(extra_datasets), tuple(extra_models))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory must be empty; preserve prior notebook versions")
    profile_config = asdict(profile)
    if not v16:
        for name in V16_FIELDS:
            del profile_config[name]  # v15 and earlier notebooks rebuild unchanged
    if profile.candidate_id is None:
        for name in ("candidate_id", "mtp", "hot_blocks", "prefill_chunk", "sparse_graphs", "draft_fp8_direct",
                     "fold_in_verify", "mtp_rejection_sampling", "native_mtp_host_kv", "cpu_affinity", "debug_checks",
                     "spec_phase_packed", "mtp_steps", "nomtp_control", "hostcomm", "vision_offload"):
            del profile_config[name]
    elif profile.candidate_id == "r7-r4":
        del profile_config["spec_phase_packed"]  # Frozen R7 launcher does not accept the new keyword.
    if profile.candidate_id in ("r7-r4", "r8-r2"):
        del profile_config["mtp_steps"]  # Preserve old payload constructor / helper signatures.
    if not profile.nomtp_control:
        profile_config.pop("nomtp_control", None)
    if not profile.hostcomm:
        profile_config.pop("hostcomm", None)
        profile_config.pop("vision_offload", None)
    for name, default in PLAIN_DFLASH_OPTS.items():  # a plan B opt-in at the fork's default stays out of CONFIG
        if v16 and profile_config[name] == default:
            del profile_config[name]
    config = {**profile_config, "logical_tokens_scope": "per-attention-DP", "running_scope": "global",
              "port": 30021, "candidate": "h28NoMTP", "index_dtype": "fp8", "kv_dtype": "nvfp4",
              "ssm_dtype": "float16", "graph": True, "vision": True, "prefill_max_requests": profile.prefill_max_requests,
              "nccl_p2p_disable": "1", "automatic_capacity_downgrade": False, "image_processor_backend": "pil"}
    introduction = (f"# {title}\n\nPrivate ARC2 L4×4, h28NoMTP plus startup-only h30. Runpod h28 graph/vision passed; "
        "this Kaggle run must validate its own hardware and cgroup admission.\n\n"
        f"Requested globalR={profile.running}, per-DP L={profile.logical_tokens}, mem={profile.mem_fraction}, "
        f"whole-node reserve={profile.host_reserve_gib}GiB. Values are never reduced automatically.\n\n"
        "Runpod reference used reserve32GiB; the Kaggle reserve is explicitly set above. "
        "ARC3 checkpoint prefetch and rewarm precede loading; h30 releases owned pages after all four loaders finish. "
        "Admission uses actual cgroup memory, without guessed reclaimable bytes.\n\n"
        f"SaveRun uses the original ARC2 agentic harness on {profile.validation_limit} evaluation tasks; competition rerun processes the full test roster. "
        "Execution writes local evidence and submission output; publishing a competition submission is a separate action.\n")
    if profile.candidate_id is not None:
        mode = "NoMTP control" if profile.nomtp_control else "MTP candidate"
        config["candidate"] = profile.candidate_id + ("NoMTP" if profile.nomtp_control else "MTP")
        gates = ("retained-session lifecycle" if profile.mtp else "target-only readiness")
        sampling = "native stochastic sampling and " if profile.mtp else ""
        introduction = (f"# {title}\n\nPrivate offline {profile.candidate_id} {mode}. Source and payload are hash pinned; "
            "packaging proves CPU identity only. This Kaggle host must pass admission, graph, vision, "
            f"{sampling}{gates} gates before the unchanged original ARC2 harness.\n\n"
            f"Requested global C={profile.running}, per-DP L={profile.logical_tokens}, "
            f"hot blocks={profile.hot_blocks}, memory fraction={profile.mem_fraction}, reserve={profile.host_reserve_gib}GiB. "
            "No automatic capacity reduction; no publishing or competition submission.\n")
    if v16:
        config.update(extra_config, candidate=profile.candidate_id + ("DFlashK" + str(profile.dflash_k) if profile.dflash_k else "NoMTP"),
                      hisparse_capacity_check=capacity)
        spec_text = (f"MTP5-DFlash drafter K={profile.dflash_k} ({extra_config['dflash_asset']['dataset']}, "
                     + (("plain residual, injected context, " if "architecture" in extra_config["dflash_asset"] else "")
                        + f"dense {extra_config['dflash_asset']['dense']} FFN)" if "dense" in extra_config["dflash_asset"]
                        else f"{extra_config['dflash_asset']['experts_kept']} experts/layer)") if profile.dflash_k else "NoMTP control")
        stack = ", ".join(name for name, on in (("hostcomm", profile.hostcomm), (f"FP8 dense {profile.online_fp8}", profile.online_fp8),
                                                 (f"Humming {profile.humming_moe} target MoE (tile {profile.humming_moe_tile}, "
                                                  f"{profile.humming_moe_act} activations)", profile.humming_moe),
                                                 (f"Humming {profile.humming_dense} dense {profile.humming_layers}", profile.humming_dense),
                                                 ("HC mixer W8A16", profile.hc_mix_w8),
                                                 (f"HC mixer hc_mix_rows {profile.hc_mix_verify}", profile.hc_mix_verify),
                                                 (f"lm_head Triton GEMM {profile.lmhead_triton}", profile.lmhead_triton),
                                                 ("embed_tokens on host", profile.embed_offload),
                                                 (f"packed GDN state {profile.gdn_state_fmt}", profile.gdn_state_fmt),
                                                 ("dp-lm-head", profile.dp_lm_head),
                                                 ("plan B drafter " + " ".join(
                                                     f"{name.removeprefix('dflash_')} {getattr(profile, name)}"
                                                     for name, default in PLAIN_DFLASH_OPTS.items()
                                                     if getattr(profile, name) != default),
                                                  any(getattr(profile, name) != default
                                                      for name, default in PLAIN_DFLASH_OPTS.items()))) if on) \
            or "no hostcomm-hook extras"
        introduction = (f"# {title}\n\nPrivate offline ARC2 L4x4 v16: sglang-arc-2 arc/main "
            f"({manifest['candidate_validation']['fork_commit'][:12]}), {spec_text}, {stack}, vision tower on GPU.\n\n"
            f"Requested global C={profile.running}, per-DP L={profile.logical_tokens}, context {profile.context_length}, "
            f"hot blocks={profile.hot_blocks} ({capacity['hot_blocks_fit']} fit the {capacity['budget_gib']} GiB budget), "
            f"memory fraction={profile.mem_fraction}, reserve={profile.host_reserve_gib}GiB. Startup gates: admission, graphs, "
            f"free memory >= {profile.min_free_gib} GiB per card, batch, vision, {profile.vision_peak_images}-board image "
            f"requests, long prefills.\n\n"
            + (f"Harness: codex ({codex_config}); deadline {profile.rerun_minutes:.0f} min after the notebook starts "
               f"(SaveRun {profile.saverun_minutes:.0f} min, first {profile.validation_limit} public evaluation tasks), "
               f"capped at {END_BY_MARGIN_SECONDS // 60} min before {END_BY_UTC} so that the notebook ends by then. "
               if profile.harness == "codex" else "")
            + ("GATES ONLY: no harness (compiled-cache seeding / server validation). " if profile.gates_only else "")
            + "/kaggle/working/submission.json always exists at the end; a failure writes placeholders and logs loudly. "
            "No publishing or competition submission from here.\n")
        data, files, git = bundle
        bundle_cell = ("BUNDLE_B64 = " + repr(base64.b64encode(data).decode()) + "\nBUNDLE_SHA256 = "
                       + repr(hashlib.sha256(data).hexdigest()) + "\nBUNDLE_FILES = " + repr(files)
                       + "\nBUNDLE_GIT = " + repr(git) + "\nCODEX_SERVED_MODEL = " + repr(CODEX_SERVED_MODEL)
                       + "\nCODEX_DATASET = " + repr(CODEX_DATASET) + "\nHUMMING_WHEEL = " + repr(HUMMING_WHEEL) + "\n")
        config_cell = ("import re\nCONFIG = " + repr(config) + "\nASSET_MANIFEST = " + repr(manifest) + "\n"
                       + bundle_cell + COMMON + V16_COMMON)
        body = "\n".join(textwrap.indent(section, "    ") for section in
                         ((V16_END_BY,) if profile.harness == "codex" else ()) + (HARDWARE, INSTALL, MANAGED))
        sections = [("markdown", introduction), ("code", config_cell),
                    ("code", V16_GUARD_HEAD + body + V16_GUARD_TAIL)]
    else:
        config_cell = "import re\nCONFIG = " + repr(config) + "\nASSET_MANIFEST = " + repr(manifest) + "\n" + COMMON
        sections = [("markdown", introduction), ("code", config_cell), ("code", HARDWARE),
                    ("code", INSTALL), ("code", MANAGED)]
    cells = []
    for kind, source in sections:
        if kind == "code":
            compile(source, f"notebook-cell{len(cells)}", "exec")
        cells.append({"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
                      **({"execution_count": None, "outputs": []} if kind == "code" else {})})
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "language": "python",
                "display_name": "Python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"notebook_sha256": hashlib.sha256((out / "notebook.ipynb").read_bytes()).hexdigest(),
               "metadata_sha256": hashlib.sha256((out / "kernel-metadata.json").read_bytes()).hexdigest(),
               "asset_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               "profile": config, "kernel_id": kernel_id, "uploaded": False}
    if profile.candidate_id is not None:
        receipt.update(candidate_id=profile.candidate_id, source_tree=manifest["source"]["digest"],
                       cpu_validation="generated cells compiled; supplied CPU archive validation hash pinned",
                       model_validation="pending on this Kaggle host", promotion="not-validated")
    if v16:
        receipt.update(bundle_sha256=hashlib.sha256(bundle[0]).hexdigest(), bundle_files=bundle[1], bundle_git=bundle[2],
                       hisparse_capacity_check=capacity, kernel_datasets=metadata["dataset_sources"])
    (out / "BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--running", type=int, default=64)
    parser.add_argument("--logical-tokens", type=int, default=3250048, help="per attention-DP capacity")
    parser.add_argument("--mem-fraction", type=float, default=0.95)
    parser.add_argument("--host-reserve-gib", type=float, default=8.0)
    parser.add_argument("--validation-limit", type=int, default=8)
    parser.add_argument("--validation-repeat", type=int, default=1, help="SaveRun only: copies of the public evaluation roster")
    parser.add_argument("--capacity-phase", action="store_true", help="SaveRun only: run canonical C64 x100k before the original ARC2 harness")
    parser.add_argument("--saverun-minutes", type=float, help="explicit whole-notebook SaveRun budget, including startup and cleanup")
    parser.add_argument("--rerun-minutes", type=float, help="explicit whole-notebook competition rerun budget, maximum720")
    parser.add_argument("--candidate-id", choices=CANDIDATE_IDS)
    parser.add_argument("--mtp-steps", type=int, choices=(1, 3), default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mtp", action="store_true")
    mode.add_argument("--nomtp-control", action="store_true", help="explicit repaired r9-r2 target-only experimental profile")
    parser.add_argument("--hot-blocks", type=int, default=512)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    for name in ("sparse-graphs", "draft-fp8-direct", "fold-in-verify", "mtp-rejection-sampling",
                 "native-mtp-host-kv", "cpu-affinity", "spec-phase-packed"):
        parser.add_argument("--" + name, action="store_true")
    parser.add_argument("--no-debug-checks", action="store_true")
    parser.add_argument("--hostcomm", action="store_true", help="v15: host-mapped collectives hook (r9-r2 NoMTP only)")
    parser.add_argument("--vision-offload", action="store_true", help="v15: move the unused vision tower to CPU; skips the vision gate")
    parser.add_argument("--harness-ctx-cap", type=int, help="ARC2 harness --ctx-cap override (>= original notebook's 50000)")
    parser.add_argument("--deadline-rate-adaptive", action="store_true",
                        help="harness: deadline uses min(configured, measured median) per-stream tok/s")
    parser.add_argument("--time-aware-budget", action="store_true",
                        help="harness: shrink each task's budget by remaining time and measured throughput")
    v16 = parser.add_argument_group("v16 (--candidate-id arc-main)")
    v16.add_argument("--dflash-k", type=int, help="MTP5-DFlash verify width K (NEXTN K-1); hot blocks >= 512*K")
    v16.add_argument("--dflash-spec", type=Path, help="drafter dataset pin (e.g. work/qwen38fn_l4/kaggle/dflash-keep128-l4.json)")
    v16.add_argument("--context-length", type=int, default=131072)
    v16.add_argument("--swap-in-block-size", type=int, help="HiSparse swap-in CTA width (serve_l4.sh uses 512)")
    v16.add_argument("--online-fp8", help="hostcomm hook block-FP8 dense layers, comma list of " + ",".join(ONLINE_FP8_LAYERS))
    v16.add_argument("--hc-ag-anydtype", action="store_true", help="hostcomm all-gather for int64/int32/fp32 too")
    v16.add_argument("--hc-slot-mb", type=int, default=21)
    v16.add_argument("--dp-lm-head", action="store_true", help="server flag --enable-dp-lm-head")
    v16.add_argument("--marlin-block-remap", action="store_true",
                     help="v16 sitecustomize: fused_marlin_moe block 48 -> 64 (K8 illegal-address workaround)")
    v16.add_argument("--humming-moe", choices=HUMMING_MOE_SPECS,
                     help="hostcomm hook ARC_HC_HUMMING_MOE: target routed experts on Humming (Kaggle dataset "
                          + HUMMING_WHEEL["dataset"] + ")")
    v16.add_argument("--humming-moe-tile", choices=HUMMING_MOE_TILES, default="global",
                     help="ARC_HC_HUMMING_MOE_TILE: rows the Humming tuning heuristic sees (local = this rank's experts)")
    v16.add_argument("--humming-moe-act", choices=HUMMING_MOE_ACTS, default="bf16",
                     help="ARC_HC_HUMMING_MOE_ACT: int8 = W3A8 (per-token INT8 activations) on int3:128; "
                          "fp8 = W3A8 (per-token e4m3) on int3packed")
    v16.add_argument("--humming-dense", choices=HUMMING_DENSE_SPECS, help="hostcomm hook ARC_HC_HUMMING_DENSE (W4A16 dense)")
    v16.add_argument("--humming-layers", help="ARC_HC_HUMMING_LAYERS: comma subset of " + ",".join(ONLINE_FP8_LAYERS))
    v16.add_argument("--hc-mix-w8", action="store_true", help="hostcomm hook ARC_HC_MIX_W8 (HC mixers INT8 weight-only)")
    v16.add_argument("--embed-offload", action="store_true", help="hostcomm hook ARC_EMBED_OFFLOAD (embed_tokens on host)")
    v16.add_argument("--hc-moe-rs", action="store_true", help="hostcomm hook ARC_HC_MOE_RS (MoE combine to this DP's rows)")
    v16.add_argument("--hc-ag-kind", choices=HC_AG_KINDS, help="hostcomm hook ARC_HC_AG_KIND (dp_gather all-gather payload)")
    v16.add_argument("--hc-ag-dp", action="store_true", help="hostcomm hook ARC_HC_AG_DP (DP-aware MoE dispatch gather)")
    v16.add_argument("--hc-ag-dp-check", action="store_true", help="ARC_HC_AG_DP_CHECK (log pair-replication mismatches)")
    v16.add_argument("--hc-ag-dp-check-every", type=int, default=1,
                     help="ARC_HC_AG_DP_CHECK_EVERY: check one in N ag_dp gathers (with --hc-ag-dp-check)")
    v16.add_argument("--hc-pipe-ar2", type=int, default=0,
                     help="ARC_HC_PIPE_STAGES2 1-8: the attention pairs' pipelined hostcomm kernels (0 = original)")
    v16.add_argument("--hc-pipe-tp4", type=int, default=0,
                     help="ARC_HC_PIPE_STAGES4 1-8: the TP4 group's pipelined hostcomm kernels (0 = original)")
    v16.add_argument("--hc-ar2-blocks", type=int, help="ARC_HC_AR2_BLOCKS: the pair all-reduce's blocks (hook default 32)")
    v16.add_argument("--hc-ag-blocks", type=int, help="ARC_HC_AG_BLOCKS: the dispatch gather's chunks (hook default 16)")
    v16.add_argument("--hc-fold-epoch", action="store_true", help="ARC_HC_FOLD_EPOCH=1 (no bump kernel per collective)")
    v16.add_argument("--hc-ar2-p2p", action="store_true", help="ARC_HC_AR2_P2P=1 (pair ar1_fp8v over CUDA peer access)")
    v16.add_argument("--hc-rs-stage2", choices=("bf16", "fp8"), default="bf16",
                     help="ARC_HC_RS_STAGE2: the owner_fp8 combine's pair exchange (fp8 = lossy)")
    v16.add_argument("--hc-rs-sib-wait", action="store_true", help="ARC_HC_RS_SIBWAIT=1 (combine stage 2 waits sibling)")
    v16.add_argument("--gpu-monitor", action="store_true", help="ARC_GPU_MONITOR=1 (clocks / power / PCIe / PSI per window)")
    v16.add_argument("--hisparse-miss-stats", action="store_true",
                     help="ARC_HISPARSE_MISS_STATS=1 (per-layer swap-in counters per window; fork c8c83127 and later)")
    v16.add_argument("--hc-moe-rs-owner", action="store_true", help="hostcomm hook ARC_HC_MOE_RS=owner (quarter-owner combine)")
    v16.add_argument("--hc-numa-node", type=int, help="ARC_HC_NUMA_NODE: first-touch hostcomm regions on this NUMA node")
    v16.add_argument("--hc-local-slots", action="store_true",
                     help="ARC_HC_LOCAL_SLOTS=1: each rank first-touches the hostcomm pages it writes on its GPU's node")
    v16.add_argument("--hc-ar2-kind", choices=HC_AR2_KINDS,
                     help="hostcomm hook ARC_HC_AR2_KIND (attention-pair all-reduce payload, FP8 for hidden-width)")
    v16.add_argument("--hc-moe-rs-fp8", action="store_true",
                     help="ARC_HC_MOE_RS=owner_fp8 (with --hc-moe-rs-owner: partials to the quarter owners in FP8)")
    v16.add_argument("--hc-sumlen", choices=HC_SUMLEN_KINDS,
                     help="ARC_HC_SUMLEN (with --hc-moe-rs-owner): prefill MoE gather / combine without the zero-filled AR4s")
    v16.add_argument("--hc-moe-overlap", metavar="W1+W2+...",
                     help="ARC_HC_MOE_OVERLAP (with --hc-moe-rs-owner, int3packed): down GEMM chunks of these widths, "
                          "each chunk's combine overlapped with the next chunk's GEMM (e.g. 512+1024+1024)")
    v16.add_argument("--hc-mix-verify", metavar="LO:HI",
                     help="ARC_HC_MIX_VERIFY: HC mixes of LO < rows <= HI on skinny_gemm/hc_mix_rows.py")
    v16.add_argument("--lmhead-triton", metavar="LO:HI",
                     help="ARC_LMHEAD_TRITON: the target lm_head GEMM over LO < rows <= HI on skinny_gemm/lm_head_gemm.py")
    v16.add_argument("--sharded-verify", action="store_true",
                     help="ARC_SHARDED_VERIFY=1: exact verify sampling on vocab-sharded logits (SHARDED_VERIFY_CANDIDATES)")
    v16.add_argument("--ple-int5", action="store_true",
                     help="ARC_PLE_INT5=1: INT5 host PLE table, 32.8 instead of 47.7 GiB host (PLE_INT5_CANDIDATES)")
    v16.add_argument("--ple-nvfp4", action="store_true",
                     help="ARC_PLE_NVFP4=1: NVFP4 host PLE table, 26.8 instead of 32.8 GiB (INT5) host; lossy; excludes --ple-int5")
    v16.add_argument("--dflash-shared-graph-pool", action="store_true",
                     help="ARC_DFLASH_SHARED_GRAPH_POOL_GROUP=1: drafter graphs in the target's graph pool (SHARED_GRAPH_POOL_CANDIDATES)")
    v16.add_argument("--dflash-session-park", action="store_true",
                     help="ARC_DFLASH_EXACT_PREFILL / _SESSION_PARK / _REFRESH_ASYNC=1 (DFLASH_PARK_CANDIDATES)")
    v16.add_argument("--dflash-step-opt", action="store_true",
                     help="ARC_DFLASH_RING_OUT_T / _BATCHED_COMMIT / _FUSED_PROPOSE=1 (B200OPT_CANDIDATES)")
    v16.add_argument("--hc-gate-precompute", action="store_true", help="ARC_HC_GATE_PRECOMPUTE=1 (B200OPT_CANDIDATES)")
    v16.add_argument("--blas-lt", action="store_true", help="server TORCH_BLAS_PREFER_CUBLASLT=1 (aten GEMMs on cuBLASLt)")
    v16.add_argument("--gdn-state-fmt", choices=("i10h", "i12h"),
                     help="h24 packed GDN state on fold_in_verify (ARC_GDN_STATE_FMT + _DISTRIBUTED=1; H24_TP_CANDIDATES)")
    v16.add_argument("--snapshot-host-buffers", type=int,
                     help="GDN snapshot host buffers per rank (>= R/DP + 16; default the launcher's 2 x (R/DP + 16))")
    v16.add_argument("--session-anchor", action="store_true",
                     help="h23 resume anchors (hisparse_config qsa_session_anchor; payloads from l4opt34)")
    v16.add_argument("--mm-release-after-prefill", action="store_true",
                     help="ARC_MM_RELEASE_AFTER_PREFILL=1: drop pixel features at prefill end (MM_CANDIDATES)")
    v16.add_argument("--mm-lazy-media", action="store_true",
                     help="ARC_MM_LAZY_MEDIA=1: media seen before go without pixels (MM_CANDIDATES)")
    v16.add_argument("--dflash-w4a8", help="ARC_DFLASH_W4A8=<int4|uint4z>:<group>[:<all|mlp>] (W4A8_CANDIDATES)")
    v16.add_argument("--dflash-merge", action="store_true", help="ARC_DFLASH_MERGE=1 (plain drafter, PLAIN_DFLASH_OPT_TREES)")
    v16.add_argument("--dflash-inject-kv", choices=("layers", "gemm"), default="layers",
                     help="ARC_DFLASH_INJECT_KV (plain drafter, PLAIN_DFLASH_OPT_TREES; default layers = unset)")
    v16.add_argument("--dflash-fuse-once", action="store_true", help="ARC_DFLASH_FUSE_ONCE=1 (plain drafter)")
    v16.add_argument("--dflash-kv-dtype", choices=("bf16", "fp8_e4m3"), default="bf16",
                     help="ARC_DFLASH_KV_DTYPE (plain drafter; default bf16 = unset)")
    v16.add_argument("--dflash-ring-fp8-dot", action="store_true",
                     help="ARC_DFLASH_RING_FP8_DOT=1 (plain drafter; needs --dflash-kv-dtype fp8_e4m3)")
    v16.add_argument("--shm-gather", action="store_true",
                     help="ARC_SHM_GATHER=1: scheduler CPU collectives over shared memory (SHM_GATHER_TREES source)")
    v16.add_argument("--malloc-trim-seconds", type=int, help="hook ARC_MALLOC_TRIM_SECONDS (>= 60)")
    v16.add_argument("--admission-reclaimable-file", action="store_true",
                     help="fork ARC_ADMISSION_RECLAIMABLE_FILE=1: startup host admission counts inactive page cache")
    v16.add_argument("--inductor-compile-threads", type=int, help="TORCHINDUCTOR_COMPILE_THREADS")
    v16.add_argument("--host-memory-trace", action="store_true",
                     help="notebook samples cgroup memory.stat / memory.events every 10 s into server/host_memory.jsonl")
    v16.add_argument("--qsa-index-shard", choices=("layer", "layer-check"),
                     help="ARC_QSA_INDEX_SHARD=layer: each attention-TP rank keeps half the QSA layers' index and hands "
                          "its top-k to the pair; the capacity check charges 198 index B/token. layer-check (with "
                          "--deterministic): no saving, every received top-k checked against the rank's own "
                          "(QSA_INDEX_SHARD_CANDIDATES)")
    v16.add_argument("--heap-diag", type=int,
                     help="ARC_HEAPDIAG=<seconds>: scheduler glibc heap counters every 60 s, deep dumps + malloc_trim at "
                          "init done and <seconds> later (HEAPDIAG_CANDIDATES)")
    v16.add_argument("--qsa-sm89-decode", type=int, choices=(0, 1, 3),
                     help="ARC_QSA_SM89_DECODE: HiSparse sm_89 direct decode E2M1 path (QSA_SM89_CANDIDATES)")
    v16.add_argument("--qsa-sm89-stages", type=int, choices=(2, 3, 4),
                     help="ARC_QSA_SM89_STAGES: its pipeline stages (QSA_SM89_CANDIDATES)")
    v16.add_argument("--extend-nosync", action="store_true",
                     help="ARC_GDN_HOST_CHUNK_INDICES=1 + ARC_MM_NO_SYNC=1: extend without host syncs (EXTEND_NOSYNC_CANDIDATES)")
    v16.add_argument("--host-private", action="store_true",
                     help="ARC_HOST_POOL_PRIVATE=1 + pinned_use_cuda_host_register: pinned host pools on private pages (v16_patch)")
    v16.add_argument("--prefill-max-requests", type=int, default=1,
                     help="requests per prefill forward; > 1 needs the unpadded PLE extend (PLE_INT5_CANDIDATES)")
    v16.add_argument("--prefill-delayer", action="store_true",
                     help="--enable-prefill-delayer: pair the two DPs' prefills in one forward (attn_dp 2)")
    v16.add_argument("--prefill-delayer-max-passes", type=int, help="--prefill-delayer-max-delay-passes")
    v16.add_argument("--prefill-delayer-queue-ratio", type=float, help="--prefill-delayer-queue-min-ratio (queue trigger)")
    v16.add_argument("--prefill-delayer-max-delay-ms", type=float, help="--prefill-delayer-max-delay-ms (queue trigger cap)")
    v16.add_argument("--attn-dp", type=int, choices=(1, 2), default=2,
                     help="2 = attention DP2 x TP2 (deployed); 1 = pure TP4 attention (TP4_ATTENTION_CANDIDATES)")
    v16.add_argument("--moe-stats", action="store_true", help="ARC_MOE_STATS=1 (distinct experts per decode MoE call)")
    v16.add_argument("--mrope-decode-pin", action="store_true", help="ARC_MROPE_DECODE_PIN=1 (non-syncing decode M-RoPE)")
    v16.add_argument("--fwd-timing", action="store_true", help="ARC_FWD_TIMING=1 (whole-run GPU time per forward mode)")
    v16.add_argument("--spec-agree-device", action="store_true",
                     help="ARC_SPEC_AGREE_DEVICE=1 (distributed spec guards on the GPU, no run_batch host sync)")
    v16.add_argument("--deterministic", action="store_true",
                     help="launcher deterministic: exact QSA top-k, torch HC mix, non-fused MoE finalize (DFlash only)")
    v16.add_argument("--dflash-compile", choices=DFLASH_COMPILE_MODES,
                     help="server env ARC_DFLASH_COMPILE: 0 = drafter eager, no inductor (per-rank autotune, fork 135068d5a)")
    v16.add_argument("--draft-vocab", help="server env ARC_DFLASH_DRAFT_VOCAB: repo-relative draft id list (SubsetPairHead)")
    v16.add_argument("--target-vocab", help="server env ARC_TARGET_VOCAB: repo-relative id list the target lm_head keeps")
    v16.add_argument("--hc-debug", choices=HC_DEBUG_MODES,
                     help="debug: v16_patch logs HostComm launch streams, or keeps HostComm out of the attention pairs")
    v16.add_argument("--spec-trace", choices=SPEC_TRACE_MODES,
                     help="debug: v16_patch compares the drafter's inputs and proposals across each attention-TP pair")
    v16.add_argument("--coarse-war-barrier", action="store_true",
                     help="server env SGLANG_FORCE_COARSE_WAR_BARRIER=1 (schedule stream waits for the whole forward)")
    v16.add_argument("--think-budget", type=int,
                     help="server env ARC_THINK_BUDGET=<N>: close a response's thinking block after N output tokens")
    v16.add_argument("--harness", choices=HARNESSES, default="frozen-arc2")
    v16.add_argument("--codex-config", help="repo-relative codex harness config carried in the notebook")
    v16.add_argument("--harness-root", type=Path, default=REPO, help="checkout the carried harness files come from")
    v16.add_argument("--gates-only", action="store_true", help="stop after the startup gates and compiled-cache export")
    v16.add_argument("--seed-cache-from", help="gates-only: local compiled cache to start from (cache seeding)")
    v16.add_argument("--min-free-gib", type=float, default=0.25)
    v16.add_argument("--vision-peak-images", type=int, default=20)
    v16.add_argument("--vision-peak-concurrency", type=int, default=4)
    v16.add_argument("--long-prefill-lengths", default="8192,65536", help="comma list of prompt lengths")
    v16.add_argument("--hisparse-budget-gib", type=float, help="measured per-rank HiSparse budget for an unlisted shape")
    v16.add_argument("--compiled-cache-dataset", help="owned dataset holding a compiled-sm89 export (manifest.json)")
    v16.add_argument("--compiled-cache-manifest-sha256")
    v16.add_argument("--no-live-logging", action="store_true", help="drop the v15 frozen-harness live-status observer")
    v16.add_argument("--model-patch-model", help="Kaggle model (threerabbits/<slug>/Transformers/default/<N>) holding an "
                     "rl_weight_patch.py patch served over the asset's package")
    v16.add_argument("--model-patch-sha256", help="sha256 of that patch's patch.json")
    parser.add_argument("--preset", type=Path, help="JSON of argument defaults (e.g. work/qwen38fn_l4/kaggle/v16-k4-profile.json); "
                        "flags given on the command line still win")
    known, _ = parser.parse_known_args()
    if known.preset is not None:
        values = {key: value for key, value in json.loads(known.preset.read_text()).items() if not key.startswith("_")}
        unknown = sorted(set(values) - {action.dest for action in parser._actions})
        if unknown:
            parser.error(f"--preset {known.preset} names unknown options {unknown}")
        for key in ("dflash_spec",):
            if key in values:
                values[key] = Path(values[key])
        parser.set_defaults(**values)
    args = parser.parse_args()
    if (args.compiled_cache_dataset is None) != (args.compiled_cache_manifest_sha256 is None):
        parser.error("--compiled-cache-dataset and --compiled-cache-manifest-sha256 go together")
    profile = Profile(args.running, args.logical_tokens, args.mem_fraction, args.host_reserve_gib,
                      args.validation_limit, args.capacity_phase, args.saverun_minutes, args.rerun_minutes,
                      candidate_id=args.candidate_id, mtp=args.mtp, hot_blocks=args.hot_blocks,
                      prefill_chunk=args.prefill_chunk, sparse_graphs=args.sparse_graphs,
                      draft_fp8_direct=args.draft_fp8_direct, fold_in_verify=args.fold_in_verify,
                      mtp_rejection_sampling=args.mtp_rejection_sampling, native_mtp_host_kv=args.native_mtp_host_kv,
                      cpu_affinity=args.cpu_affinity, debug_checks=not args.no_debug_checks,
                      spec_phase_packed=args.spec_phase_packed, mtp_steps=args.mtp_steps,
                      nomtp_control=args.nomtp_control, hostcomm=args.hostcomm, vision_offload=args.vision_offload,
                      validation_repeat=args.validation_repeat, harness_ctx_cap=args.harness_ctx_cap,
                      deadline_rate_adaptive=args.deadline_rate_adaptive, time_aware_budget=args.time_aware_budget,
                      live_logging=not args.no_live_logging, dflash_k=args.dflash_k,
                      context_length=args.context_length, swap_in_block_size=args.swap_in_block_size,
                      online_fp8=args.online_fp8, hc_ag_anydtype=args.hc_ag_anydtype, hc_slot_mb=args.hc_slot_mb,
                      dp_lm_head=args.dp_lm_head, coarse_war_barrier=args.coarse_war_barrier,
                      think_budget=args.think_budget,
                      marlin_block_remap=args.marlin_block_remap, humming_moe=args.humming_moe, humming_moe_tile=args.humming_moe_tile, humming_moe_act=args.humming_moe_act, humming_dense=args.humming_dense, humming_layers=args.humming_layers, hc_mix_w8=args.hc_mix_w8, embed_offload=args.embed_offload, hc_moe_rs=args.hc_moe_rs, hc_ag_kind=args.hc_ag_kind, hc_ag_dp=args.hc_ag_dp, hc_ag_dp_check=args.hc_ag_dp_check, hc_pipe_ar2=args.hc_pipe_ar2, hc_pipe_tp4=args.hc_pipe_tp4, hc_ar2_blocks=args.hc_ar2_blocks, hc_ag_dp_check_every=args.hc_ag_dp_check_every, hc_fold_epoch=args.hc_fold_epoch, hc_ar2_p2p=args.hc_ar2_p2p, hc_rs_stage2=args.hc_rs_stage2, hc_rs_sib_wait=args.hc_rs_sib_wait, hc_ag_blocks=args.hc_ag_blocks, gpu_monitor=args.gpu_monitor, hisparse_miss_stats=args.hisparse_miss_stats, hc_moe_rs_owner=args.hc_moe_rs_owner, hc_numa_node=args.hc_numa_node, hc_local_slots=args.hc_local_slots, hc_ar2_kind=args.hc_ar2_kind, hc_moe_rs_fp8=args.hc_moe_rs_fp8, hc_sumlen=args.hc_sumlen, hc_moe_overlap=args.hc_moe_overlap, hc_mix_verify=args.hc_mix_verify, lmhead_triton=args.lmhead_triton, sharded_verify=args.sharded_verify, ple_int5=args.ple_int5, ple_nvfp4=args.ple_nvfp4, host_private=args.host_private, extend_nosync=args.extend_nosync, dflash_session_park=args.dflash_session_park, dflash_shared_graph_pool=args.dflash_shared_graph_pool, dflash_step_opt=args.dflash_step_opt, hc_gate_precompute=args.hc_gate_precompute, qsa_sm89_decode=args.qsa_sm89_decode, qsa_sm89_stages=args.qsa_sm89_stages, blas_lt=args.blas_lt, gdn_state_fmt=args.gdn_state_fmt, snapshot_host_buffers=args.snapshot_host_buffers, session_anchor=args.session_anchor, mm_release_after_prefill=args.mm_release_after_prefill, mm_lazy_media=args.mm_lazy_media, dflash_w4a8=args.dflash_w4a8, dflash_merge=args.dflash_merge, dflash_inject_kv=args.dflash_inject_kv, dflash_fuse_once=args.dflash_fuse_once, dflash_kv_dtype=args.dflash_kv_dtype, dflash_ring_fp8_dot=args.dflash_ring_fp8_dot, heap_diag=args.heap_diag, qsa_index_shard=args.qsa_index_shard, shm_gather=args.shm_gather, host_memory_trace=args.host_memory_trace, malloc_trim_seconds=args.malloc_trim_seconds,
                      admission_reclaimable_file=args.admission_reclaimable_file,
                      inductor_compile_threads=args.inductor_compile_threads, prefill_max_requests=args.prefill_max_requests, prefill_delayer=args.prefill_delayer, prefill_delayer_max_passes=args.prefill_delayer_max_passes, prefill_delayer_queue_ratio=args.prefill_delayer_queue_ratio, prefill_delayer_max_delay_ms=args.prefill_delayer_max_delay_ms, attn_dp=args.attn_dp, moe_stats=args.moe_stats, mrope_decode_pin=args.mrope_decode_pin, fwd_timing=args.fwd_timing, spec_agree_device=args.spec_agree_device, deterministic=args.deterministic, dflash_compile=args.dflash_compile, draft_vocab=args.draft_vocab, target_vocab=args.target_vocab, spec_trace=args.spec_trace, hc_debug=args.hc_debug, harness=args.harness, gates_only=args.gates_only,
                      min_free_gib=args.min_free_gib, vision_peak_images=args.vision_peak_images,
                      vision_peak_concurrency=args.vision_peak_concurrency,
                      long_prefill_lengths=args.long_prefill_lengths, seed_cache_from=args.seed_cache_from)
    cache = None if args.compiled_cache_dataset is None else (args.compiled_cache_dataset, args.compiled_cache_manifest_sha256)
    if (args.model_patch_model is None) != (args.model_patch_sha256 is None):
        parser.error("--model-patch-model and --model-patch-sha256 go together")
    model_patch = None if args.model_patch_model is None else {"model": args.model_patch_model,
                                                               "patch_sha256": args.model_patch_sha256}
    receipt = build(json.loads(args.asset_manifest.read_text()), profile, args.kernel_id, args.title, args.out_dir,
                    dflash_spec=args.dflash_spec, codex_config=args.codex_config,
                    hisparse_budget_gib=args.hisparse_budget_gib, harness_root=args.harness_root, compiled_cache=cache,
                    model_patch=model_patch)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
