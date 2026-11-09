"""Opt-in sglang integration of host-mapped collectives (hostcomm.cu).

Imported at the end of sglang/srt/layers/dp_attention.py when ARC_HOSTCOMM=1.
Wraps GroupCoordinator.all_reduce / all_gather_into_tensor for the TP4 group and
the attention-TP2 groups, and optionally drops the redundant attention-TP
reduce-scatter in the padded dp_gather (both attention-TP ranks already hold the
same rows after the attention all-reduce).

Env:
  ARC_HC_LIB            path to libhostcomm.so (required)
  ARC_HC_AR=0|1         all-reduce via hostcomm
  ARC_HC_AG=0|1         all_gather_into_tensor via hostcomm
  ARC_HC_SKIP_RS2=0|1   local slice instead of attention-TP reduce-scatter in dp_gather
  ARC_HC_AR4_KIND / ARC_HC_AR4_BLOCKS   (default ar2_bf16 / 8)
  ARC_HC_AR2_KIND / ARC_HC_AR2_BLOCKS   (default ar1_bf16 / 32)
  ARC_HC_AG_KIND  / ARC_HC_AG_BLOCKS    (default ag_bf16 / 16)
                        FP8 kinds (ar1_fp8v, ag_fp8v: e4m3 + one fp32 scale per 128) apply to hidden-width tensors
                        only. ag_fp8q = ag_fp8v that also keeps the gathered e4m3 payload + scales on the device;
                        the int3packed W3A8 Humming MoE then feeds them to gate_up as they travelled (input group
                        128) instead of requantizing the dequantized rows per token (ARC_HC_FP8Q_ROWS, default 1024:
                        payload capacity in rows; a larger gather raises)
  ARC_HC_SLOT_MB        per-rank slot (default 12)
  ARC_HC_AG_DP=0|1      (needs SKIP_RS2) MoE dispatch gather copies the own pair's shards from local rows and reads
                        only the other pair's from host memory; ARC_HC_AG_DP_CHECK=1 also reads the sibling's shard
                        and logs words that differ from the local copy (replication check, measurement only).
                        With ARC_HC_AG_KIND=ag_fp8q the gather is ag_dp_fp8q (the sibling's shard quantized locally,
                        bitwise the ag_fp8q gather while the pair's rows are bitwise equal) and only in MAX_LEN
                        decode / target-verify forwards of a batch without extends; the rest keep ag_fp8q (a prefill's
                        rows differ in the pair: 10-08 r1b counted 9,034 words in the startup prefills, 0 in decode).
                        ARC_HC_AG_DP_CHECK_EVERY=N (default 1): check one such gather in N, decided on the device from
                        a call counter at a position that moves by one every N calls (with N = the gathers per forward
                        the checked layer rotates through all layers under graph replay), so a run can measure the
                        saving and check replication; logged as ARC_HC_AG_DP_CHECK mismatched_words / checked_calls
  ARC_HC_PIPE_STAGES2 / ARC_HC_PIPE_STAGES4 = 0..8  per group size (2 = the attention pairs, 4 = TP4): ar1_fp8v,
                        ag_fp8q / ag_dp_fp8q and rs_owner fp8 of that group run the pipelined kernels (writer and
                        reader blocks, the reads of a step overlap the writes of the next on the host link; bitwise
                        identical, pipe_collectives_test.py); 0 = the original kernels. Kaggle 10-13 (B=188): pays
                        only for the symmetric pair AR2 (178 -> 148 us at 2 stages, 8 chunks); AG4 / RS4 not
  ARC_HC_FOLD_EPOCH=0|1 the FP8 collectives advance the region's epoch in their last block instead of a one-thread
                        bump kernel after every call (one graph node less per collective; hc_variants_test)
  ARC_HC_AR2_P2P=0|1    the attention pair's ar1_fp8v all-reduce over CUDA peer access (PairP2P: each rank writes its
                        payload into the sibling's device memory; bitwise ar1_fp8v) when the rows fit
                        ARC_HC_P2P_SLOT_MB (default 2); larger calls keep hostcomm. Exclusive with ARC_HC_PIPE_STAGES2
  ARC_HC_RS_STAGE2=bf16|fp8  (owner_fp8) the rs_owner pair exchange in FP8 + scales (lossy; the owner keeps the
                        dequantized rows, so the pair stays bitwise equal)
  ARC_HC_RS_SIBWAIT=0|1 (owner_fp8) the rs_owner pair exchange waits for the sibling's flag only
  ARC_HC_MOE_RS=0|1|owner|owner_fp8  MoE combine without AR4 + dp_scatter: 1 = rs_rows (one stage, same bytes as
                        AR4), owner = rs_owner (each rank reduces a quarter of its DP's rows, then pair exchange),
                        owner_fp8 = rs_owner whose partials to a quarter's owner travel in FP8 (pair exchange bf16)
  ARC_HC_SUMLEN=0|1|fp8 (needs ARC_HC_MOE_RS) prefill / extend forwards (dp_attention SUM_LEN layout) without the
                        zero-filled two-shot AR4s: the MoE dispatch gather publishes each pair rank's half of its DP's
                        rows and reads only the other DP's (ag_sum, bitwise), the combine is rs_owner over the SUM_LEN
                        quarters (1 = bf16, bitwise AR4 + dp_scatter; fp8 = partials to the owners in FP8, as
                        owner_fp8). ARC_HC_SUMLEN_BLOCKS (default 16). A forward whose rows exceed the slot keeps the
                        AR4 path (counted as sumlen_over_slot)
  ARC_HC_NUMA_NODE=<n>  first-touch the host region from NUMA node n's CPUs (multi-socket hosts)
  ARC_MROPE_DECODE_PIN=1  decode M-RoPE deltas via a pinned non-blocking copy instead of torch.tensor(list,
                        device=cuda), whose pageable H2D synchronizes the stream every decode step (the spec path
                        got the same fix in h43)
  ARC_SPEC_AGREE_DEVICE=1|ab  distributed HiSparse spec guards (verify input, accept, snapshot claim) compare the
                        device tensors across the attention-TP pair on the GPU (hostcomm gather + device assert)
                        instead of .cpu() + gloo, which made run_batch wait for the draft and the verify; ab alternates
                        device / original guards every ARC_SPEC_AGREE_AB_BLOCK (500) verify cycles in the same server
                        and logs per-mode cycle medians (same host-communication speed for both arms)
  ARC_FWD_TIMING=1      whole-run GPU time accounting: CUDA events around every ModelRunner.forward (read back
                        without a host sync), logged per rank every ARC_FWD_TIMING_SECONDS (60) as GPU ms per
                        (runner, forward mode), the stream time between forwards, and EXTEND forwards by token bucket
                        (ARC_GPU_MONITOR=1 adds clocks / power / PCIe throughput, ARC_HISPARSE_MISS_STATS=1 the swap-in
                        counters of the fork's planar kernel; see install_fwd_timing)
  ARC_MALLOC_TRIM_SECONDS=<s>  every scheduler returns glibc's free heap memory to the OS with malloc_trim(0) every s
                        seconds (>= 60) on a daemon thread, logging ARC_MALLOC_TRIM per trim (see install_malloc_trim)
Every routing decision depends only on dtype/shape, so all ranks of a group agree.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_STATE = {"installed": False}


def _flag(name, default="0"):
    v = os.environ.get(name, default)
    if v not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {v!r}")
    return v == "1"


def _setup_group(coord, tag):
    sys.path.insert(0, str(Path(os.environ["ARC_HC_LIB"]).resolve().parent))
    from hostcomm import HostComm  # noqa: E402

    slot = int(float(os.environ.get("ARC_HC_SLOT_MB", "12")) * 2**20)
    pipe_stages = int(os.environ.get(f"ARC_HC_PIPE_STAGES{coord.world_size}", "0"))
    fold = _flag("ARC_HC_FOLD_EPOCH")
    n = coord.world_size
    cfg = {
        "ar_kind": os.environ.get(f"ARC_HC_AR{n}_KIND", "ar2_bf16" if n == 4 else "ar1_bf16"),
        "ar_blocks": int(os.environ.get(f"ARC_HC_AR{n}_BLOCKS", "8" if n == 4 else "32")),
        "ag_kind": os.environ.get("ARC_HC_AG_KIND", "ag_bf16"),
        "ag_blocks": int(os.environ.get("ARC_HC_AG_BLOCKS", "16")),
        "pipe_stages": pipe_stages,
        "fold_epoch": fold,
        "ar2_p2p": n == 2 and _flag("ARC_HC_AR2_P2P"),
    }
    if cfg["ar2_p2p"]:  # checked before any region is mapped
        if pipe_stages:
            raise ValueError("ARC_HC_AR2_P2P replaces the pair's (unpipelined) ar1_fp8v: unset ARC_HC_PIPE_STAGES2")
        if cfg["ar_kind"] != "ar1_fp8v":
            raise ValueError(f"ARC_HC_AR2_P2P is the ar1_fp8v kind, got ARC_HC_AR2_KIND={cfg['ar_kind']}")
    hc = HostComm(Path(os.environ["ARC_HC_LIB"]), coord.cpu_group, list(coord.ranks), tag, slot,
                  pipe_stages=pipe_stages, fold_epoch=fold)
    p2p = None
    if cfg["ar2_p2p"]:
        from hostcomm import PairP2P  # noqa: E402
        p2p = PairP2P(Path(os.environ["ARC_HC_LIB"]), coord.cpu_group, list(coord.ranks),
                      int(float(os.environ.get("ARC_HC_P2P_SLOT_MB", "2")) * 2**20))
    coord._arc_hc = {"hc": hc, "cfg": cfg, "p2p": p2p, "counts": {"ar_capture": 0, "ar_eager": 0, "ag_capture": 0,
                                                                   "ag_eager": 0, "fallback": 0}}
    logger.info("ARC_HOSTCOMM_INIT group=%s ranks=%s slot=%d cfg=%s placement=%s", coord.unique_name, list(coord.ranks),
                hc.slot_bytes, json.dumps(cfg), json.dumps(hc.placement))


_FP8_COLS = int(os.environ.get("ARC_HC_FP8_COLS", "2560"))
_BF16_OF = {"ar1_fp8v": "ar1_bf16", "ar1_fp8": "ar1_bf16", "ag_fp8v": "ag_bf16", "ag_fp8": "ag_bf16",
            "ag_fp8q": "ag_bf16"}


def _pick(kind, t):
    """Quantized kinds apply only to hidden-width activations (never logits)."""
    if kind in _BF16_OF and (t.dim() < 2 or t.shape[-1] != _FP8_COLS):
        return _BF16_OF[kind]
    return kind


def _eligible(state, t, kind):
    if t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous():
        return False
    n = t.numel()
    if n == 0 or n % (128 if "fp8" in kind else 8):
        return False
    need = n * 2 if kind.endswith("bf16") else n + (n // 128) * 4
    return need <= state["hc"].slot_bytes


def _fp8q_stash(dev, numel):
    """Device buffers the ag_fp8q gather writes its payload into (one set per process: the gathers and the MoE that
    reads them run in order on one stream, the routed-expert side stream joining before the next layer's gather)."""
    st = _STATE.get("fp8q")
    if st is None:
        rows = int(os.environ.get("ARC_HC_FP8Q_ROWS", "1024"))
        cap = rows * _FP8_COLS
        st = {"q": torch.empty(cap, dtype=torch.uint8, device=dev),
              "s": torch.empty(cap // 128, dtype=torch.float32, device=dev),
              "cap": cap, "last": None, "hits": 0, "misses": 0, "logged": False}
        _STATE["fp8q"] = st
    if numel > st["cap"]:
        raise RuntimeError(f"ARC_HC_AG_KIND=ag_fp8q: a {numel // _FP8_COLS}-row gather exceeds ARC_HC_FP8Q_ROWS="
                           f"{st['cap'] // _FP8_COLS}")
    if st["q"].device != dev:
        raise RuntimeError(f"ag_fp8q payload buffers on {st['q'].device}, gather on {dev}")
    return st


def _fp8q_forget():
    """The payload belongs only to the MoE call right after its gather: every other collective and every MoE call
    drop the record. A prefill MoE reads the same global DP buffer (filled by the AR4 dispatch), so a record left
    over from an earlier gather would otherwise match its address and feed stale rows."""
    st = _STATE.get("fp8q")
    if st is not None:
        st["last"] = None


def install(dp_attention_module):
    if _STATE["installed"]:
        return
    from sglang.srt.distributed import parallel_state as ps

    use_ar, use_ag, skip_rs2 = _flag("ARC_HC_AR"), _flag("ARC_HC_AG"), _flag("ARC_HC_SKIP_RS2")
    ag_anydtype = _flag("ARC_HC_AG_ANYDTYPE")  # PLE int64 ngram-id dp_gather otherwise falls back to NCCL
    ag_dp, ag_dp_check = _flag("ARC_HC_AG_DP"), _flag("ARC_HC_AG_DP_CHECK")
    moe_rs = os.environ.get("ARC_HC_MOE_RS", "0")
    if moe_rs not in ("0", "1", "owner", "owner_fp8"):
        raise ValueError(f"ARC_HC_MOE_RS must be 0, 1, owner or owner_fp8, got {moe_rs!r}")
    sumlen = os.environ.get("ARC_HC_SUMLEN", "0")
    if sumlen not in ("0", "1", "fp8"):
        raise ValueError(f"ARC_HC_SUMLEN must be 0, 1 or fp8, got {sumlen!r}")
    if sumlen != "0" and moe_rs == "0":
        raise ValueError("ARC_HC_SUMLEN replaces the collectives inside the ARC_HC_MOE_RS layer hook: set ARC_HC_MOE_RS")
    if os.environ.get("ARC_HC_AG_KIND") == "ag_fp8q" and (os.environ.get("ARC_HC_HUMMING_MOE") != "int3packed"
                                                          or os.environ.get("ARC_HC_HUMMING_MOE_ACT") != "fp8"
                                                          or moe_rs == "0"):
        # the ARC_HC_MOE_RS layer hook drops the payload record before each layer's gather (_run_qwen4_exp_mlp)
        raise ValueError("ARC_HC_AG_KIND=ag_fp8q feeds the int3packed W3A8 Humming MoE "
                         "(ARC_HC_HUMMING_MOE=int3packed, ARC_HC_HUMMING_MOE_ACT=fp8) under ARC_HC_MOE_RS")
    if (ag_dp or ag_dp_check) and not (skip_rs2 and use_ag):
        raise ValueError("ARC_HC_AG_DP needs ARC_HC_SKIP_RS2=1 and ARC_HC_AG=1")
    if ag_dp_check and not ag_dp:
        raise ValueError("ARC_HC_AG_DP_CHECK needs ARC_HC_AG_DP=1")
    ag_dp_fp8q = ag_dp and os.environ.get("ARC_HC_AG_KIND", "ag_bf16") == "ag_fp8q"
    if ag_dp and os.environ.get("ARC_HC_AG_KIND", "ag_bf16") not in ("ag_bf16", "ag_fp8q"):
        raise ValueError("ARC_HC_AG_DP gathers bf16 (ag_bf16) or FP8 with the payload (ag_fp8q), got ARC_HC_AG_KIND="
                         f"{os.environ['ARC_HC_AG_KIND']}")
    check_every = int(os.environ.get("ARC_HC_AG_DP_CHECK_EVERY", "1"))
    if check_every < 1 or (check_every != 1 and not ag_dp_check):
        raise ValueError(f"ARC_HC_AG_DP_CHECK_EVERY={check_every} needs ARC_HC_AG_DP_CHECK=1 and must be >= 1")
    _STATE["agdp_check"] = (ag_dp_check, check_every)
    cls = ps.GroupCoordinator
    orig_ar = cls.all_reduce
    orig_ag = cls.all_gather_into_tensor

    def all_reduce(self, input_):
        _fp8q_forget()
        st = getattr(self, "_arc_hc", None)
        if st is not None and use_ar and not torch.compiler.is_compiling():
            kind = _pick(st["cfg"]["ar_kind"], input_)
            if _eligible(st, input_, kind):
                cap = torch.cuda.is_current_stream_capturing()
                n = input_.numel()
                if kind == "ar1_fp8v" and st["p2p"] is not None and n + n // 32 <= st["p2p"].slot_bytes:
                    # both ranks of the pair see the same shape, so they take the same route
                    st["counts"]["ar_p2p_capture" if cap else "ar_p2p_eager"] = (
                        st["counts"].get("ar_p2p_capture" if cap else "ar_p2p_eager", 0) + 1)
                    return st["p2p"].all_reduce(input_, blocks=st["cfg"]["ar_blocks"])
                st["counts"]["ar_capture" if cap else "ar_eager"] += 1
                return st["hc"].all_reduce(input_, kind=kind, blocks=st["cfg"]["ar_blocks"])
            st["counts"]["fallback"] += 1
        return orig_ar(self, input_)

    def all_gather_into_tensor(self, output, input):
        _fp8q_forget()
        st = getattr(self, "_arc_hc", None)
        if st is not None and use_ag and not torch.compiler.is_compiling():
            kind = _pick(st["cfg"]["ag_kind"], input)
            if (ag_anydtype and input.dtype != torch.bfloat16 and input.dtype in (torch.int64, torch.int32, torch.float32)
                    and kind == "ag_bf16" and input.is_contiguous() and output.is_contiguous()
                    and output.dtype == input.dtype):
                # all-gather is a byte copy: gather e.g. PLE int64 ngram ids as bf16 lanes
                input = input.reshape(-1).view(torch.bfloat16)
                output = output.reshape(-1).view(torch.bfloat16)
            if (_eligible(st, input, kind) and output.is_contiguous() and output.dtype == input.dtype
                    and output.numel() == input.numel() * self.world_size):
                cap = torch.cuda.is_current_stream_capturing()
                st["counts"]["ag_capture" if cap else "ag_eager"] += 1
                if kind == "ag_fp8q":
                    stash = _fp8q_stash(output.device, output.numel())
                    st["hc"].ag_fp8q(input, output.view(-1), stash["q"], stash["s"], st["cfg"]["ag_blocks"])
                    stash["last"] = (output.data_ptr(), output.numel())
                    return
                st["hc"].launch(kind, input, output.view(-1), st["cfg"]["ag_blocks"])
                return
            st["counts"]["fallback"] += 1
        return orig_ag(self, output, input)

    cls.all_reduce = all_reduce
    cls.all_gather_into_tensor = all_gather_into_tensor

    orig_init = dp_attention_module.initialize_dp_attention

    def initialize_dp_attention(*args, **kwargs):
        out = orig_init(*args, **kwargs)
        tp = ps.get_tp_group()
        attn = ps.get_attn_tp_group()
        if tp.world_size == 4:
            _setup_group(tp, "tp4")
        if attn.world_size == 2 or (attn.world_size == 4 and attn is not tp):
            _setup_group(attn, f"attn{min(attn.ranks)}w{attn.world_size}")
        if moe_rs != "0":
            _install_moe_rs(dp_attention_module, ps, {"1": "rows", "owner": "owner", "owner_fp8": "owner_fp8"}[moe_rs],
                            sumlen)
        logger.info("ARC_HOSTCOMM_DP_AWARE rank=%d ag_dp=%s ag_dp_check=%s moe_rs=%s numa_node=%s", tp.rank_in_group,
                    ag_dp, ag_dp_check, moe_rs, os.environ.get("ARC_HC_NUMA_NODE"))
        install_vision_offload()
        install_embed_offload()
        install_online_fp8()
        install_moe_stats()
        install_w8a16()
        install_qsa_decode_knobs()
        install_hc_mix_rows()
        install_hc_mix_verify()
        install_lmhead_triton()
        install_hc_mix_w8()
        install_humming_dense()
        install_humming_moe()
        install_mrope_decode_pin()
        install_spec_agree_device()
        install_fwd_timing()
        install_malloc_trim()
        return out

    dp_attention_module.initialize_dp_attention = initialize_dp_attention

    if skip_rs2:
        orig_gather = dp_attention_module._dp_gather_via_all_gather

        def _dp_gather_via_all_gather(global_tokens, local_tokens, forward_batch, is_partial):
            attn_n = dp_attention_module.get_attn_tensor_model_parallel_world_size()
            if attn_n == 1 or is_partial or dp_attention_module.world_dp_gather_enabled():
                return orig_gather(global_tokens, local_tokens, forward_batch, is_partial)
            # Rows are replicated across attention-TP ranks: the reduce-scatter
            # (with zero-filled non-zero ranks) equals taking our own slice.
            r = dp_attention_module.get_attn_tensor_model_parallel_rank()
            tp = ps.get_tp_group()
            if ag_dp_fp8q and local_tokens.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
                st = tp._arc_hc
                if attn_n != 2 or tp.world_size != 4:
                    raise RuntimeError(f"ARC_HC_AG_DP is for 2 DP x attention-TP 2, got attn={attn_n} tp={tp.world_size}")
                cap = torch.cuda.is_current_stream_capturing()
                pm, fm = forward_batch.dp_padding_mode, forward_batch.forward_mode
                # MAX_LEN decode / verify only: a prefill (also MAX_LEN with an idle DP or a prefill graph) has rows
                # that differ in the pair. The batch flag is global; forward_mode is this rank's, and an IDLE rank
                # keeping ag_fp8q is fine (ag_dp_fp8q and ag_fp8q mix in one gather, pipe_collectives_test AGMIX).
                if (pm is not None and pm.is_max_len() and not forward_batch.is_extend_in_batch
                        and (fm.is_decode() or fm.is_target_verify()) and local_tokens.dim() == 2
                        and local_tokens.shape[1] == _FP8_COLS and local_tokens.shape[0] % 2 == 0
                        and local_tokens.is_contiguous() and global_tokens.is_contiguous()
                        and global_tokens.numel() == 2 * local_tokens.numel()
                        and _eligible(st, local_tokens.tensor_split(2)[r], "ag_fp8q")):
                    _fp8q_forget()
                    stash = _fp8q_stash(global_tokens.device, global_tokens.numel())
                    key = "agdp_fp8q" + ("_capture" if cap else "_eager")
                    st["counts"][key] = st["counts"].get(key, 0) + 1
                    check = _agdp_check_kwargs(local_tokens.device)
                    st["hc"].ag_dp_fp8q(local_tokens, global_tokens.view(-1), stash["q"], stash["s"],
                                        blocks=st["cfg"]["ag_blocks"], **check)
                    stash["last"] = (global_tokens.data_ptr(), global_tokens.numel())
                    if cap and st["counts"][key] == 1:
                        logger.info("ARC_HC_AG_DP_FP8Q first_capture rank=%d rows=%d mode=%s pipe_stages=%d check=%s",
                                    tp.rank_in_group, local_tokens.shape[0], fm.name, st["cfg"]["pipe_stages"],
                                    bool(check))
                    return
                key = "agdp_fp8q_skip" + ("_capture" if cap else "_eager")
                st["counts"][key] = st["counts"].get(key, 0) + 1  # the ag_fp8q gather below
            elif ag_dp and local_tokens.dtype == torch.bfloat16 and not torch.compiler.is_compiling():
                st = tp._arc_hc
                if attn_n != 2 or tp.world_size != 4:
                    raise RuntimeError(f"ARC_HC_AG_DP is for 2 DP x attention-TP 2, got attn={attn_n} tp={tp.world_size}")
                if (local_tokens.is_contiguous() and global_tokens.is_contiguous() and local_tokens.shape[0] % 2 == 0
                        and global_tokens.numel() == 2 * local_tokens.numel()
                        and local_tokens.numel() <= st["hc"].slot_bytes):  # shard = numel/2 bf16 = numel bytes
                    cap = torch.cuda.is_current_stream_capturing()
                    st["counts"]["agdp_capture" if cap else "agdp_eager"] = (
                        st["counts"].get("agdp_capture" if cap else "agdp_eager", 0) + 1)
                    st["hc"].ag_dp(local_tokens, global_tokens, blocks=st["cfg"]["ag_blocks"],
                                   mismatch=_agdp_mismatch(local_tokens.device))
                    return
                st["counts"]["fallback"] += 1
            mine = local_tokens.tensor_split(attn_n)[r]
            tp.all_gather_into_tensor(global_tokens, mine.contiguous())

        dp_attention_module._dp_gather_via_all_gather = _dp_gather_via_all_gather

    install_skinny()
    _STATE["installed"] = True
    logger.info("ARC_HOSTCOMM_INSTALLED ar=%s ag=%s skip_rs2=%s ag_dp=%s ag_dp_fp8q=%s ag_dp_check=%s check_every=%d "
                "moe_rs=%s numa_node=%s pipe_stages2=%s pipe_stages4=%s fold_epoch=%s ar2_p2p=%s rs_stage2=%s "
                "rs_sibwait=%s ag_blocks=%s", use_ar, use_ag, skip_rs2, ag_dp, ag_dp_fp8q, ag_dp_check, check_every,
                moe_rs, os.environ.get("ARC_HC_NUMA_NODE"), os.environ.get("ARC_HC_PIPE_STAGES2", "0"),
                os.environ.get("ARC_HC_PIPE_STAGES4", "0"), os.environ.get("ARC_HC_FOLD_EPOCH", "0"),
                os.environ.get("ARC_HC_AR2_P2P", "0"), os.environ.get("ARC_HC_RS_STAGE2", "bf16"),
                os.environ.get("ARC_HC_RS_SIBWAIT", "0"), os.environ.get("ARC_HC_AG_BLOCKS", "16"))


def _agdp_mismatch(dev):
    """bf16 ag_dp: the replication counter for this call under ARC_HC_AG_DP_CHECK (every ARC_HC_AG_DP_CHECK_EVERY-th
    call, counted in Python: at capture for a graph, so a graph checks fixed calls), else None."""
    check, every = _STATE["agdp_check"]
    if not check:
        return None
    n = _STATE["agdp_calls"] = _STATE.get("agdp_calls", 0) + 1
    if (n - 1) % every:
        return None
    return _agdp_counters(dev)["mismatch"]


def _agdp_check_kwargs(dev):
    """ag_dp_fp8q: {} without ARC_HC_AG_DP_CHECK, else the device counters and every (the kernel picks the checked
    calls from its call counter, so the position rotates under graph replay)."""
    check, every = _STATE["agdp_check"]
    if not check:
        return {}
    c = _agdp_counters(dev)
    return dict(mismatch=c["mismatch"], check_ctr=c["ctr"], check_every=every, checked=c["checked"])


def _agdp_counters(dev):
    """Device int64 counters: words of the sibling's shard that differ from the local copy, the ag_dp_fp8q call
    counter and the checked calls; logged every 60 s."""
    c = _STATE.get("agdp_counters")
    if c is not None:
        return c
    import threading
    import time
    c = {k: torch.zeros(1, dtype=torch.int64, device=dev) for k in ("mismatch", "ctr", "checked")}
    _STATE["agdp_counters"] = c

    def loop():
        torch.cuda.set_device(dev)
        side = torch.cuda.Stream(device=dev)
        while True:
            time.sleep(60)
            with torch.cuda.stream(side):
                vals = {k: int(v.cpu().item()) for k, v in c.items()}
            logger.info("ARC_HC_AG_DP_CHECK mismatched_words=%d checked_calls=%d calls=%d", vals["mismatch"],
                        vals["checked"], vals["ctr"])

    threading.Thread(target=loop, daemon=True, name="arc-agdp-check").start()
    return c


def _install_moe_rs(dpa, ps, mode, sumlen="0"):
    """MoE combine: skip the block's TP4 all-reduce and reduce only this DP's rows.

    Original: AR4 over all 2B global rows, then dp_scatter copies this DP's B rows.
    mode "rows": one host-mapped kernel summing just [start, start+count) (zero-fill the padded tail, as
    dp_scatter does) -- same bytes as the two-shot AR4, one stage fewer.
    mode "owner": rs_owner -- each rank reduces a quarter (its half of its DP's rows), then the pair exchanges
    the reduced halves; writes S / reads S per rank instead of S / 1.5S, bitwise equal to AR4 + dp_scatter.
    mode "owner_fp8": rs_owner with the partials sent to each quarter's owner in FP8 (e4m3, scale per 128; the
    owner's own partial exact, fp32 rank-order sum); the pair exchange stays bf16. Lossy (~1.8% relative per
    combine on random data, fp8_collectives_test.py); prefill (SUM_LEN) keeps the bf16 AR4 unless sumlen.
    sumlen "1" / "fp8" (ARC_HC_SUMLEN): SUM_LEN forwards gather with ag_sum and combine with rs_sum_owner (bf16 /
    FP8 partials) over the two DP segments of global_num_tokens instead of the zero-filled AR4 and AR4 + dp_scatter
    (det-g4 trace 10-11: 2 x 48 two-shot AR4s of ~3.1 ms in a 2k-token extend of ~830 ms).
    Fallback (slot too small): AR4 + dp_scatter.
    ARC_HC_RS_STAGE2=fp8 / ARC_HC_RS_SIBWAIT=1 (owner_fp8, MAX_LEN layout): the pair exchange in FP8 / waiting for
    the sibling only (HostComm.rs_owner stage2_fp8 / sib_wait).
    """
    import sglang.srt.models.qwen2_moe as q2
    import sglang.srt.models.qwen4_exp as q4

    stage2 = os.environ.get("ARC_HC_RS_STAGE2", "bf16")
    if stage2 not in ("bf16", "fp8"):
        raise ValueError(f"ARC_HC_RS_STAGE2 must be bf16 or fp8, got {stage2!r}")
    rs_stage2_fp8, rs_sibwait = stage2 == "fp8", _flag("ARC_HC_RS_SIBWAIT")
    if (rs_stage2_fp8 or rs_sibwait) and mode != "owner_fp8":
        raise ValueError("ARC_HC_RS_STAGE2=fp8 / ARC_HC_RS_SIBWAIT=1 are options of ARC_HC_MOE_RS=owner_fp8")

    orig_skip = q2.should_skip_post_experts_all_reduce

    def should_skip_post_experts_all_reduce(*, is_tp_path):
        return True if is_tp_path else orig_skip(is_tp_path=is_tp_path)

    q2.should_skip_post_experts_all_reduce = should_skip_post_experts_all_reduce

    owners = [c for c in vars(q4).values() if isinstance(c, type) and "_run_qwen4_exp_mlp" in vars(c)]
    if len(owners) != 1:
        raise RuntimeError(f"expected one class defining _run_qwen4_exp_mlp, got {owners}")
    owner = owners[0]
    orig_run = owner._run_qwen4_exp_mlp
    blocks = int(os.environ.get("ARC_HC_RS_BLOCKS", "8"))
    sumlen_blocks = int(os.environ.get("ARC_HC_SUMLEN_BLOCKS", "16"))

    def _sumlen_rows(st, forward_batch, global_hidden, hidden_states):
        """(n0, n1) of a SUM_LEN forward the ARC_HC_SUMLEN collectives take, else None."""
        pm = forward_batch.dp_padding_mode
        if sumlen == "0" or pm is None or not pm.is_sum_len():
            return None
        gnt = forward_batch.global_num_tokens_cpu
        if gnt is None or len(gnt) != 2:
            raise RuntimeError(f"ARC_HC_SUMLEN: expected two attention-DP token counts, got {gnt}")
        n0, n1 = int(gnt[0]), int(gnt[1])
        d = dpa.get_attention_dp_rank()
        if global_hidden.shape[0] != n0 + n1 or hidden_states.shape[0] != (n0, n1)[d]:
            raise RuntimeError(f"ARC_HC_SUMLEN: SUM_LEN buffers global {tuple(global_hidden.shape)} / local "
                               f"{tuple(hidden_states.shape)} do not match global_num_tokens {gnt} on DP {d}")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ARC_HC_SUMLEN: a SUM_LEN forward under CUDA-graph capture (host row counts)")
        if (n0 + n1) * global_hidden.shape[1] * 2 > st["hc"].slot_bytes:
            st["counts"]["sumlen_over_slot"] = st["counts"].get("sumlen_over_slot", 0) + 1
            return None
        return n0, n1

    def _run_qwen4_exp_mlp(self, hidden_states, forward_batch):
        if not self.config.num_experts or not self._qwen4_exp_use_dp_moe_gather():
            raise RuntimeError("ARC_HC_MOE_RS requires the DP-attention MoE gather path")
        tp = ps.get_tp_group()
        global_hidden = dpa.get_global_dp_buffer(tp)
        _fp8q_forget()  # only this layer's own gather may hand its payload to this layer's MoE
        st = tp._arc_hc
        rows = _sumlen_rows(st, forward_batch, global_hidden, hidden_states)
        if rows is not None:
            st["hc"].ag_sum(hidden_states, global_hidden, *rows, blocks=sumlen_blocks)
        else:
            dpa.dp_gather_replicate(global_hidden, hidden_states, forward_batch)
        out = self.mlp(global_hidden, forward_batch)  # post-experts TP all-reduce skipped
        local = dpa.get_local_dp_buffer(tp)
        if rows is not None:
            st["hc"].rs_sum_owner(out, local, *rows, fp8=sumlen == "fp8", blocks=sumlen_blocks)
            n = st["counts"]["sumlen"] = st["counts"].get("sumlen", 0) + 1
            if n == 1:
                logger.info("ARC_HC_SUMLEN first SUM_LEN layer rank=%d n0=%d n1=%d combine=%s", tp.rank_in_group,
                            rows[0], rows[1], "fp8" if sumlen == "fp8" else "bf16")
            return local
        if out.is_contiguous() and out.dtype == torch.bfloat16 and out.numel() * 2 <= st["hc"].slot_bytes:
            start, count = dpa.get_dp_local_info(forward_batch)
            cap = torch.cuda.is_current_stream_capturing()
            key = ("rs" if mode == "rows" else "rso") + ("_capture" if cap else "_eager")
            st["counts"][key] = st["counts"].get(key, 0) + 1
            start, count = start.reshape(1).to(torch.int64), count.reshape(1).to(torch.int64)
            if mode == "rows":
                st["hc"].rs_rows(out, local, start, count, blocks=blocks)
                return local
            if tp.world_size != 4:
                raise RuntimeError(f"ARC_HC_MOE_RS=owner is for TP4 (2 DP x attention-TP 2), got tp={tp.world_size}")
            if out.shape[0] == 2 * local.shape[0]:
                # MAX_LEN layout (decode / verify graphs): DP d at rows [d*L, (d+1)*L); the kernel traps if
                # start != d*L
                st["hc"].rs_owner(out, local, start, count, blocks=blocks, fp8=mode == "owner_fp8",
                                  stage2_fp8=rs_stage2_fp8, sib_wait=rs_sibwait)
                return local
            # SUM_LEN layout (prefill / extend batches): the DP segments are not aligned quarters; plain AR4 +
            # dp_scatter, as without the hook
            key = "rso_sumlen" + ("_capture" if cap else "_eager")
            st["counts"][key] = st["counts"].get(key, 0) + 1
            out = tp.all_reduce(out)
            dpa.dp_scatter(local, out, forward_batch)
            return local
        st["counts"]["fallback"] += 1
        out = tp.all_reduce(out)
        dpa.dp_scatter(local, out, forward_batch)
        return local

    owner._run_qwen4_exp_mlp = _run_qwen4_exp_mlp
    logger.info("ARC_HOSTCOMM_MOE_RS mode=%s sumlen=%s sumlen_blocks=%d installed on %s (orig %s)", mode, sumlen,
                sumlen_blocks, owner.__name__, orig_run)


def report():
    from sglang.srt.distributed import parallel_state as ps
    rows = {}
    for getter in (ps.get_tp_group, ps.get_attn_tp_group):
        c = getter()
        st = getattr(c, "_arc_hc", None)
        if st:
            rows[c.unique_name] = st["counts"]
    return rows

def install_skinny():
    """ARC_SKINNY_TABLE=<json>: bf16 dense GEMMs (M<=128, tuned (N,K)) and HC mix via Triton skinny GEMM."""
    table_path = os.environ.get("ARC_SKINNY_TABLE")
    if not table_path:
        return
    sys.path.insert(0, os.environ["ARC_SKINNY_DIR"])
    import skinny_gemm as sg
    import torch.nn.functional as F
    import sglang.srt.layers.quantization.unquant as uq
    import sglang.srt.layers.hyperconnection as hcmod

    table = sg.load_table(table_path)
    counts = {"dense": 0, "dense_fallback": 0, "hc": 0}
    orig = uq._bf16_gemm_dispatch_impl

    def _bf16_gemm_dispatch_impl(x, weight, bias, addend=None):
        m = x.numel() // x.shape[-1]
        cfg = table.get((weight.shape[0], weight.shape[1]))
        if cfg is not None and bias is None and m <= 128 and weight.is_contiguous():
            counts["dense"] += 1
            out = sg.skinny_linear(x, weight, cfg)
            if addend is not None:
                out = out.add_(addend)
            return out
        counts["dense_fallback"] += 1
        return orig(x, weight, bias, addend=addend)

    uq._bf16_gemm_dispatch_impl = _bf16_gemm_dispatch_impl

    orig_supported = hcmod.fused_hc_mix_supported

    def fused_hc_mix_supported(x, wd, wu):
        m = x.numel() // x.shape[-1]
        if (m <= 128 and x.dtype == torch.bfloat16 and (wd.shape[0], wd.shape[1]) in table
                and (wu.shape[0], wu.shape[1]) in table):
            return True
        return orig_supported(x, wd, wu)

    orig_fused = hcmod.fused_hc_mix

    def fused_hc_mix(x, wd, wu, hc, hs):
        m = x.numel() // x.shape[-1]
        cd, cu = table.get((wd.shape[0], wd.shape[1])), table.get((wu.shape[0], wu.shape[1]))
        if m > 128 or cd is None or cu is None:
            return orig_fused(x, wd, wu, hc, hs)
        counts["hc"] += 1
        t = F.silu(sg.skinny_linear(x, wd, cd) / hc)
        u = torch.sigmoid(sg.skinny_linear(t, wu, cu))
        return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)

    hcmod.fused_hc_mix_supported = fused_hc_mix_supported
    hcmod.fused_hc_mix = fused_hc_mix
    _STATE["skinny_counts"] = counts
    logger.info("ARC_SKINNY_INSTALLED shapes=%d table=%s", len(table), table_path)

def install_mrope_decode_pin():
    """ARC_MROPE_DECODE_PIN=1: ForwardBatch._compute_mrope_positions_decode builds the per-request M-RoPE deltas
    with torch.tensor(list, device=cuda). That pageable host-to-device copy synchronizes the stream, so on every
    decode step of a batch holding a multimodal request the scheduler CPU waits for the previous forward to drain
    and the overlap scheduler loses its lead (NoMTP C64 on Runpod 10-08: one 52 ms cudaStreamSynchronize per
    run_batch, GPU idle during the next get_next_batch_to_run). Same values through a pinned buffer and a
    non-blocking copy; every other case (text-only, precomputed positions, rl_on_policy_target) is the original."""
    if not _flag("ARC_MROPE_DECODE_PIN") or _STATE.get("mrope_decode_pin"):
        return
    _STATE["mrope_decode_pin"] = True
    from sglang.srt.model_executor import forward_batch_info as fbi

    cls = fbi.ForwardBatch
    original = cls._compute_mrope_positions_decode

    def _compute_mrope_positions_decode(self, model_runner, batch):
        mm_inputs = batch.multimodal_inputs
        seq_lens_cpu = self.seq_lens_cpu
        if (fbi.get_exec().deterministic.rl_on_policy_target is not None
                or not any(mm is not None for mm in mm_inputs)
                or any(mm is not None and mm.mrope_positions is not None
                       and mm.mrope_positions.shape[1] >= seq_lens_cpu[i] for i, mm in enumerate(mm_inputs))):
            return original(self, model_runner, batch)
        deltas = [0] * len(mm_inputs)
        for i, mm in enumerate(mm_inputs):
            if mm is not None:
                if not mm.mrope_position_delta.is_cpu:
                    raise RuntimeError("ARC_MROPE_DECODE_PIN: mrope_position_delta is on the device; .item() would sync")
                deltas[i] = mm.mrope_position_delta.item()
        host = torch.tensor(deltas, dtype=torch.int64).pin_memory()
        device_deltas = host.to(model_runner.device, non_blocking=True)
        positions_1d = (device_deltas - 1) + self.seq_lens.to(torch.int64)
        self.mrope_positions = positions_1d.unsqueeze(0).repeat(3, 1)

    cls._compute_mrope_positions_decode = _compute_mrope_positions_decode
    logger.info("ARC_MROPE_DECODE_PIN installed on %s.%s", cls.__module__, cls.__name__)


_FWD_TIMING_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


def install_malloc_trim():
    """ARC_MALLOC_TRIM_SECONDS=<s> (>= 60): a daemon thread calls glibc malloc_trim(0) every s seconds and logs
    ARC_MALLOC_TRIM rank=<r> n=<trims> rss_anon_before=<B> rss_anon_after=<B> s=<seconds>.

    10-12 ARC_HEAPDIAG on the L4 C96 deployment: each scheduler's [heap] grows from ~2.5 GiB of live C-level data
    (CUDA graph capture is most of it) to ~3.55 GiB in the first ~30 minutes of serving; the extra ~1.0 GiB is memory
    the server freed and glibc kept. malloc_trim(0) on a timer thread gave it back in 0.13-0.15 s per scheduler and it
    grew back only 0.06 GiB in the next 8 minutes: ~4 GiB of host over the four schedulers. ctypes releases the GIL
    for the call; a scheduler thread that mallocs during the trim waits for glibc's arena lock."""
    value = os.environ.get("ARC_MALLOC_TRIM_SECONDS")
    if value is None or _STATE.get("malloc_trim"):
        return
    if not value.isdigit() or int(value) < 60:
        raise ValueError(f"ARC_MALLOC_TRIM_SECONDS={value!r}: an integer number of seconds >= 60")
    _STATE["malloc_trim"] = True
    import ctypes
    import threading

    period = int(value)
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    rank = torch.distributed.get_rank()

    def rss_anon():
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        raise RuntimeError("/proc/self/status has no RssAnon line")

    def loop():
        n = 0
        while True:
            time.sleep(period)
            before = rss_anon()
            t0 = time.monotonic()
            libc.malloc_trim(0)
            seconds = time.monotonic() - t0
            n += 1
            logger.info("ARC_MALLOC_TRIM rank=%d n=%d rss_anon_before=%d rss_anon_after=%d s=%.3f",
                        rank, n, before, rss_anon(), seconds)

    threading.Thread(target=loop, name="arc-malloc-trim", daemon=True).start()
    logger.info("ARC_MALLOC_TRIM installed rank=%d every_s=%d", rank, period)


def install_fwd_timing():
    """ARC_FWD_TIMING=1: where the GPU time of a whole run goes, measured rather than modelled.

    Two CUDA events (enable_timing) around every ModelRunner.forward on the current stream, outside graph capture.
    They are read back lazily with Event.query() on later calls, never with a host sync. Every
    ARC_FWD_TIMING_SECONDS (default 60) each rank logs one line ARC_FWD_TIMING <json>:
      fwd     {"<runner>/<mode>": [GPU ms, forwards, tokens]}  runner target|draft, mode ForwardMode name
              (EXTEND = prefill chunk, DECODE, TARGET_VERIFY, IDLE = this DP had nothing while the other DP ran, ...)
      between [ms, count]  stream time from one forward's end to the next forward's start: GPU idle plus work outside
              ModelRunner.forward on that stream (sampling, spec accept, mamba commit, scheduler kernels)
      ext     {"<=N tokens": [GPU ms, forwards, tokens]}  target EXTEND / MIXED forwards by size, for the fixed +
              per-token cost of a prefill chunk
      wall_s  host seconds the window covers (forwards drained in it: GPU ms add up to about wall_s * 1000)
      shape   {"target/<mode>": {"bs<B>" | "eager": [GPU ms, forwards, live rows, executed rows, live requests,
              {"<2 ms bin>": forwards}]}}  per executed shape: a target forward that replayed a CUDA graph is keyed
              by the graph's padded batch (DecodeCudaGraphRunner.bs; executed rows = bs x captured_req_width), others
              by "eager"; live rows = input_ids rows. The ms histogram (2 ms bins) gives per-shape quantiles.
    A forward counts in the window in which its events are drained (seconds after it ran), so windows are exact
    in total and approximate at their edges. The four ranks of one forward are one sample (one distributed step),
    not four. ARC_GPU_MONITOR=1 (global rank 0): every ARC_GPU_MONITOR_SECONDS (default 10) nvidia-smi SM clock /
    power / temperature / clock-event reasons of every GPU, host PSI and cgroup CPU throttling, and every
    ARC_GPU_MONITOR_PCIE_SECONDS (default 0.5) NVML PCIe TX (GPU -> host) / RX (host -> GPU) throughput of every
    GPU (the driver's 20 ms window: the link load hostcomm shares, KV swap-in / backup / staging included); one
    ARC_GPU_MONITOR <json> line per ARC_FWD_TIMING_SECONDS with per-GPU p10 / median / p90 (PCIe: and mean, MB/s).
    ARC_HISPARSE_MISS_STATS=1 (the fork's coordinator then keeps per-layer swap-in counters, 10-13 review E5): at
    each window boundary ranks 0 and 2 (one per attention DP) copy the target coordinator's miss_stats to pinned
    memory (async, an event read back at the next boundary) and reset its max column on the stream; one
    ARC_HISPARSE_MISS <json> line per window, one window behind (t_from / t_to): per-layer and total deltas of
    misses, misses^2, slow / fast rows, ring and LRU hits, the interval's max misses of one row, record bytes
    (bytes loaded = misses x record bytes)."""
    if not _flag("ARC_FWD_TIMING") or _STATE.get("fwd_timing"):
        return
    _STATE["fwd_timing"] = True
    import collections
    from sglang.srt.model_executor.model_runner import ModelRunner

    period = float(os.environ.get("ARC_FWD_TIMING_SECONDS", "60"))
    if period < 10:
        raise ValueError(f"ARC_FWD_TIMING_SECONDS={period}: windows under 10 s are mostly edge effects")
    rank = torch.distributed.get_rank()
    pending = collections.deque()
    st = {"t0": time.monotonic(), "last_end": None}
    miss = _MissStatsReader(rank) if _flag("ARC_HISPARSE_MISS_STATS") and rank % 2 == 0 else None

    def fresh():
        st["fwd"] = collections.defaultdict(lambda: [0.0, 0, 0])
        st["ext"] = collections.defaultdict(lambda: [0.0, 0, 0])
        st["shape"] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0, 0, 0, 0, {}]))
        st["between"] = [0.0, 0]

    fresh()

    def drain():
        while pending and pending[0][1].query():
            start, end, key, ntok, shape = pending.popleft()
            ms = start.elapsed_time(end)
            f = st["fwd"][key]
            f[0] += ms
            f[1] += 1
            f[2] += ntok
            if shape is not None:
                tag, exec_rows, reqs = shape
                sh = st["shape"][key][tag]
                sh[0] += ms
                sh[1] += 1
                sh[2] += ntok
                sh[3] += exec_rows
                sh[4] += reqs
                b = str(int(ms // 2) * 2)
                sh[5][b] = sh[5].get(b, 0) + 1
            if st["last_end"] is not None:
                gap = st["last_end"].elapsed_time(start)
                if gap > 0:  # consecutive forwards on another stream may overlap; only count real gaps
                    st["between"][0] += gap
                    st["between"][1] += 1
            st["last_end"] = end
            if key in ("target/EXTEND", "target/MIXED"):
                bucket = next((b for b in _FWD_TIMING_BUCKETS if ntok <= b), None)
                e = st["ext"][f"<={bucket}" if bucket is not None else f">{_FWD_TIMING_BUCKETS[-1]}"]
                e[0] += ms
                e[1] += 1
                e[2] += ntok
        now = time.monotonic()
        if now - st["t0"] >= period:
            payload = {"rank": rank, "wall_s": round(now - st["t0"], 2),
                       "fwd": {k: [round(v[0], 1), v[1], v[2]] for k, v in sorted(st["fwd"].items())},
                       "between": [round(st["between"][0], 1), st["between"][1]],
                       "ext": {k: [round(v[0], 1), v[1], v[2]] for k, v in st["ext"].items()},
                       "shape": {k: {t: [round(v[0], 1), *v[1:5], v[5]] for t, v in sorted(d.items())}
                                 for k, d in sorted(st["shape"].items())},
                       "pending": len(pending)}
            logger.info("ARC_FWD_TIMING %s", json.dumps(payload, separators=(",", ":")))
            if miss is not None:
                miss.window(now)
            st["t0"] = now
            fresh()

    original = ModelRunner.forward

    def forward(self, forward_batch, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            return original(self, forward_batch, *args, **kwargs)
        mode = forward_batch.forward_mode
        if mode is None:
            raise RuntimeError("ARC_FWD_TIMING: forward_batch.forward_mode is None")
        ids = forward_batch.input_ids
        key = f"{'draft' if self.is_draft_worker else 'target'}/{mode.name}"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = original(self, forward_batch, *args, **kwargs)
        end.record()
        ntok = int(ids.shape[0]) if ids is not None else 0
        shape = None
        if not self.is_draft_worker:
            if miss is not None and miss.coord is None:
                miss.attach(self.hisparse_coordinator)
            if out.can_run_graph:
                runner = self.decode_cuda_graph_runner  # replay_prepare set bs / raw_bs for this replay
                shape = (f"bs{runner.bs}", runner.bs * runner.captured_req_width, runner.raw_bs)
            else:
                shape = ("eager", ntok, forward_batch.batch_size)
        pending.append((start, end, key, ntok, shape))
        drain()
        return out

    ModelRunner.forward = forward
    logger.info("ARC_FWD_TIMING installed rank=%d period_s=%.0f", rank, period)
    if miss is not None:
        logger.info("ARC_HISPARSE_MISS reader installed rank=%d", rank)
    if _flag("ARC_GPU_MONITOR") and rank == 0:
        _start_gpu_monitor(period)


class _MissStatsReader:
    """ARC_HISPARSE_MISS_STATS (see install_fwd_timing): the target coordinator's [layer, 8] swap-in counters
    (misses, misses^2, max, slow rows, ring hits, LRU hits, fast rows, -), read without a host sync. A snapshot is
    copied at each window boundary and read at the next one, so the line logged at boundary k covers the interval
    between the snapshots of boundaries k-2 and k-1 (t_from / t_to, monotonic seconds; the first starts at the first
    target forward)."""

    COLS = ("misses", "misses_sq", "max", "slow_rows", "ring_hits", "lru_hits", "fast_rows")

    def __init__(self, rank):
        self.rank = rank
        self.coord = None

    def attach(self, coord):
        if coord is None:
            raise RuntimeError("ARC_HISPARSE_MISS_STATS=1 but the target ModelRunner has no hisparse_coordinator")
        if coord.miss_stats is None:  # an AttributeError instead: a fork without the swap-in counters
            raise RuntimeError("ARC_HISPARSE_MISS_STATS=1 did not reach the coordinator (miss_stats is None)")
        self.coord = coord
        self.host = torch.empty(tuple(coord.miss_stats.shape), dtype=torch.int64, pin_memory=True)
        self.prev = None  # the snapshot the next delta starts from, taken at t_base
        self.t_base = None
        self._issue(time.monotonic())

    def window(self, now):
        if self.coord is None:
            return  # no target forward yet
        if not self.event.query():  # a copy issued a whole window ago and not done: log it, keep waiting
            logger.info("ARC_HISPARSE_MISS %s", json.dumps({"rank": self.rank, "pending": True}))
            return
        cur = self.host.clone()
        if self.prev is not None:
            d = cur - self.prev
            d[:, 2] = cur[:, 2]  # the max column is reset after each copy: already this interval's
            tot = d.sum(0)
            tot[2] = int(cur[:, 2].max())
            payload = {"rank": self.rank, "t_from": round(self.t_base, 2), "t_to": round(self.t_copy, 2),
                       "record_bytes": int(self.coord.miss_record_bytes),
                       "total": dict(zip(self.COLS, (int(v) for v in tot[:7]))),
                       "layer_misses": [int(v) for v in d[:, 0]],
                       "layer_slow_rows": [int(v) for v in d[:, 3]]}
            logger.info("ARC_HISPARSE_MISS %s", json.dumps(payload, separators=(",", ":")))
        cur[:, 2] = 0
        self.prev, self.t_base = cur, self.t_copy
        self._issue(now)

    def _issue(self, now):
        stats = self.coord.miss_stats
        self.host.copy_(stats, non_blocking=True)
        stats[:, 2].zero_()  # same stream as the swap-in kernels: ordered after the copy
        self.event = torch.cuda.Event()
        self.event.record()
        self.t_copy = now


def _start_gpu_monitor(period):
    """ARC_GPU_MONITOR (see install_fwd_timing): a daemon thread sampling nvidia-smi and the host's pressure files."""
    import subprocess
    import threading

    every = float(os.environ.get("ARC_GPU_MONITOR_SECONDS", "10"))
    if every < 2:
        raise ValueError(f"ARC_GPU_MONITOR_SECONDS={every}: nvidia-smi itself takes ~0.1-0.5 s")
    pcie_every = float(os.environ.get("ARC_GPU_MONITOR_PCIE_SECONDS", "0.5"))
    if not 0.25 <= pcie_every <= every:  # one NVML throughput read is a ~20 ms driver window; 8 of them per sample
        raise ValueError(f"ARC_GPU_MONITOR_PCIE_SECONDS={pcie_every}: must be in [0.25, ARC_GPU_MONITOR_SECONDS]")
    query = "index,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
    import pynvml

    pynvml.nvmlInit()
    handles = [(str(i), pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(pynvml.nvmlDeviceGetCount())]

    def host_counters():
        out = {}
        for name in ("cpu", "memory", "io"):
            # "some avg10=0.00 avg60=0.00 avg300=0.00 total=N" / "full ..."
            for line in Path(f"/proc/pressure/{name}").read_text().splitlines():
                kind, *fields = line.split()
                out[f"psi_{name}_{kind}_total_us"] = int(dict(f.split("=") for f in fields)["total"])
        stat = Path("/sys/fs/cgroup/cpu.stat")
        if stat.exists():  # cgroup v2; on v1 hosts the throttling counters live elsewhere and are not read
            for line in stat.read_text().splitlines():
                k, v = line.split()
                if k in ("nr_throttled", "throttled_usec", "usage_usec"):
                    out[f"cg_{k}"] = int(v)
        return out

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    def gbps(xs):  # NVML KiB/s -> GB/s: p10, median, p90, mean
        return [round(v * 1024 / 1e9, 3) for v in (pct(xs, 0.1), pct(xs, 0.5), pct(xs, 0.9), sum(xs) / len(xs))]

    def loop():
        samples, reasons, pcie = {}, {}, {}
        last_log = last_smi = time.monotonic()
        prev = host_counters()
        while True:
            time.sleep(pcie_every)
            for idx, h in handles:
                p = pcie.setdefault(idx, {"tx": [], "rx": []})
                p["tx"].append(pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES))
                p["rx"].append(pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES))
            if time.monotonic() - last_smi >= every:
                last_smi = time.monotonic()
                r = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode != 0:
                    raise RuntimeError(f"ARC_GPU_MONITOR: nvidia-smi rc {r.returncode}: {r.stderr[-300:]}")
                for row in r.stdout.strip().splitlines():
                    idx, sm, pw, temp, why = [c.strip() for c in row.split(",")]
                    g = samples.setdefault(idx, {"sm": [], "pw": [], "temp": []})
                    g["sm"].append(float(sm))
                    g["pw"].append(float(pw))
                    g["temp"].append(float(temp))
                    reasons.setdefault(idx, {})[why] = reasons.setdefault(idx, {}).get(why, 0) + 1
            if time.monotonic() - last_log >= period and samples:
                now = host_counters()
                payload = {"gpu": {i: {"sm_mhz": [pct(g["sm"], 0.1), pct(g["sm"], 0.5), pct(g["sm"], 0.9)],
                                       "power_w": [pct(g["pw"], 0.1), pct(g["pw"], 0.5), pct(g["pw"], 0.9)],
                                       "temp_max": max(g["temp"]), "reasons": reasons[i], "n": len(g["sm"]),
                                       "pcie_tx_gbps": gbps(pcie[i]["tx"]), "pcie_rx_gbps": gbps(pcie[i]["rx"]),
                                       "pcie_n": len(pcie[i]["tx"])}
                                   for i, g in sorted(samples.items())},
                           "host_delta": {k: now[k] - prev.get(k, 0) for k in now},
                           "wall_s": round(time.monotonic() - last_log, 1)}
                logger.info("ARC_GPU_MONITOR %s", json.dumps(payload, separators=(",", ":")))
                samples, reasons, pcie, prev, last_log = {}, {}, {}, now, time.monotonic()

    threading.Thread(target=loop, daemon=True, name="arc-gpu-monitor").start()
    logger.info("ARC_GPU_MONITOR installed every_s=%.0f pcie_every_s=%.2f gpus=%d", every, pcie_every, len(handles))


_AGREE_TAGS = {"spec-verify-input": 1, "spec-accept": 2, "spec-snapshot-claim": 3}


def _spec_accept_parts(coord, rows, before, counts, indices, predict):
    """agree_spec_accept's identity and invalid conditions on the device, no host sync: row and count ranges,
    completed-block bounds, the verify-overflow flag and unallocated (< 0) host addresses of the blocks this accept
    completes. Returns (packet parts, invalid as a 0-dim bool, (addr, live) for tests)."""
    i64 = torch.int64
    h0, h1 = coord.req_to_host_pool.shape[0], coord.req_to_host_pool.shape[1]
    vt, bt = coord.verify_tokens, coord.layout.block_tokens
    r, s, c = rows.to(i64), before.to(i64), counts.to(i64)
    ok = (r >= 1) & (r < h0) & (c >= 1) & (c <= vt)
    lo, hi = s // bt, (s + c) // bt
    ok &= (lo >= 0) & (lo <= hi) & (hi <= h1)
    cols = lo.unsqueeze(1) + torch.arange(-(-vt // bt), device=r.device)  # hi - lo <= ceil(vt / bt)
    live = ok.unsqueeze(1) & (cols < hi.unsqueeze(1))
    addr = coord.req_to_host_pool[r.clamp(0, h0 - 1).unsqueeze(1), cols.clamp(0, h1 - 1)].to(i64)
    addr = torch.where(live, addr, torch.zeros_like(addr))
    flag = coord.verify_overflow.reshape(-1)[:1].to(i64)
    bad = (~ok).any() | (live & (addr < 0)).any() | (flag != 0).any()
    return [r, s, c, indices, predict, addr, flag], bad, (addr, live)


def install_spec_agree_device():
    """ARC_SPEC_AGREE_DEVICE=1: the h29 guards of distributed HiSparse spec decode read device tensors with .cpu()
    inside run_batch (agree_spec_verify_input after the draft, agree_spec_accept after the verify, the snapshot
    claim's staging-busy MAX and agreement), so the scheduler CPU waits for the draft and for the whole verify
    forward; the overlap scheduler loses its lead and the GPU idles through process_batch_result and the next
    get_next_batch_to_run (Runpod L4 K4 C32 10-08: 100 + 11 ms of waiting per run_batch, ~15% GPU idle).

    Same guarantees without the wait: CPU-known identity (rids, shapes, dtypes, the CPU-side invalid conditions)
    still goes through the gloo consensus; device tensors are all-gathered over the attention-TP pair with hostcomm
    and compared word by word on the GPU (no hash), and a mismatch or any of the original invalid conditions fails a
    device assert queued before the mamba commit and host backup, so neither runs and the next host call raises.
    The staging-busy MAX is the same reduction on the device."""
    mode = os.environ.get("ARC_SPEC_AGREE_DEVICE", "0")
    if mode not in ("0", "1", "ab"):
        raise ValueError(f"ARC_SPEC_AGREE_DEVICE must be 0, 1 or ab, got {mode!r}")
    if mode == "0" or _STATE.get("spec_agree_device"):
        return
    _STATE["spec_agree_device"] = True
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.managers import qsa_hisparse_coordinator as qc
    from sglang.srt.mem_cache import qsa_hisparse_distributed as qd

    if not qd.distributed_spec_enabled():
        raise RuntimeError("ARC_SPEC_AGREE_DEVICE replaces the ARC_HISPARSE_DISTRIBUTED_SPEC guards, which are off")
    if qd.spec_phase_packed_enabled():
        raise RuntimeError("ARC_SPEC_AGREE_DEVICE replaces the phase-packed agreement; unset "
                           "ARC_HISPARSE_SPEC_PHASE_PACKED")
    attn = ps.get_attn_tp_group()
    # DP2: the attention-TP pair; pure TP4 (ARC_HISPARSE_TP4=1, no DP attention): the TP4 group, whose HostComm is
    # the tp4 one (the attention-TP group is the TP group itself)
    tp4 = os.environ.get("ARC_HISPARSE_TP4", "0") == "1"
    if attn.world_size != (4 if tp4 else 2) or (tp4 and attn is not ps.get_tp_group()):
        raise RuntimeError(f"ARC_SPEC_AGREE_DEVICE needs attention-TP {4 if tp4 else 2} (ARC_HISPARSE_TP4={int(tp4)}), "
                           f"got attention TP {attn.world_size}")
    coord_cls, snap_cls = qc.QSAHiSparseCoordinator, qc.QSAHiSparseSnapshots
    rank = ps.get_tp_group().rank_in_group
    methods = _spec_agree_device_methods(attn._arc_hc["hc"], qd, snap_cls.claim)
    if mode == "ab":
        block = int(os.environ.get("ARC_SPEC_AGREE_AB_BLOCK", "500"))
        if block < 50:
            raise ValueError(f"ARC_SPEC_AGREE_AB_BLOCK={block}: blocks under 50 cycles are mostly switch transients")
        methods = _spec_agree_ab(methods, (coord_cls.agree_spec_verify_input, coord_cls.agree_spec_accept,
                                           snap_cls.claim), block, rank)
    coord_cls.agree_spec_verify_input, coord_cls.agree_spec_accept, snap_cls.claim = methods
    logger.info("ARC_SPEC_AGREE_DEVICE installed rank=%d attn_ranks=%s mode=%s", rank, list(attn.ranks), mode)


def _spec_agree_ab(device, original, block, rank):
    """ARC_SPEC_AGREE_DEVICE=ab: device guards for `block` verify cycles, then the original ones, alternating. The
    switch counts non-idle verify inputs, which both attention-TP ranks of a DP see identically (same batches), so
    the pair switches on the same cycle and its gloo / hostcomm call sequences stay paired; the two DPs switch
    independently. accept and claim follow the mode of their cycle's verify input. Logs per-mode medians of the
    start-to-start time between verify inputs (in steady state the scheduler loop runs at the GPU cycle), so both
    arms are measured under the same host-communication speed."""
    import statistics

    st = {"n": 0, "device": True, "t": None, "win": {True: [], False: []}, "all": {True: [], False: []}}

    def verify_input(self, batch, verify_input_):
        if not self.distributed_spec or batch.forward_mode.is_idle():
            return
        now = time.perf_counter()
        if st["t"] is not None:
            dt = (now - st["t"]) * 1000
            st["win"][st["device"]].append(dt)
            st["all"][st["device"]].append(dt)
        st["t"] = now
        st["device"] = (st["n"] // block) % 2 == 0
        st["n"] += 1
        if st["n"] % (2 * block) == 0:
            med = {m: statistics.median(v) if v else float("nan") for m, v in st["win"].items()}
            tot = {m: statistics.median(v) if v else float("nan") for m, v in st["all"].items()}
            logger.info("ARC_SPEC_AGREE_AB rank=%d cycles=%d window device=%.2f original=%.2f ms (%.3fx) "
                        "cumulative device=%.2f original=%.2f ms (%.3fx) n=%d/%d", rank, st["n"], med[True],
                        med[False], med[False] / med[True], tot[True], tot[False], tot[False] / tot[True],
                        len(st["all"][True]), len(st["all"][False]))
            st["win"] = {True: [], False: []}
        return (device[0] if st["device"] else original[0])(self, batch, verify_input_)

    def accept(self, batch, predict, accepted, indices):
        return (device[1] if st["device"] else original[1])(self, batch, predict, accepted, indices)

    def claim(self, seq_lens, accept_lens):
        return (device[2] if st["device"] else original[2])(self, seq_lens, accept_lens)

    return verify_input, accept, claim


def _spec_agree_device_methods(hc, qd, original_claim):
    """(agree_spec_verify_input, agree_spec_accept, QSAHiSparseSnapshots.claim) for ARC_SPEC_AGREE_DEVICE; hc is
    the attention-TP group's HostComm (the pair, or TP4 under ARC_HISPARSE_TP4), qd the qsa_hisparse_distributed
    module (spec_agree_test.py drives these)."""
    i64 = torch.int64

    def device_check(parts, bad, op):
        flag = bad.reshape(1).to(i64) if bad is not None else torch.zeros(1, dtype=i64, device=parts[0].device)
        hc.pair_check_i64(torch.cat([x.reshape(-1).to(i64) for x in parts] + [flag]), _AGREE_TAGS[op])

    def agree_spec_verify_input(self, batch, verify_input):
        if not self.distributed_spec or batch.forward_mode.is_idle():
            return
        names = ("draft_token", "positions", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling")
        tensors = [batch.req_pool_indices, batch.seq_lens] + [getattr(verify_input, n) for n in names]
        cpu_identity, device = [], []
        for t in tensors:
            if t.is_floating_point():
                raise RuntimeError(f"ARC_SPEC_AGREE_DEVICE: {t.dtype} verify identity tensor")
            if t.is_cuda:
                cpu_identity.append([list(t.shape), str(t.dtype)])
                device.append(t)
            else:
                cpu_identity.append(t.tolist())
        qd.agree_spec_transition(self.spec_protocols["forward"], "spec-verify-input",
                                 [[req.rid for req in batch.reqs], cpu_identity],
                                 invalid=verify_input.draft_token_num != self.verify_tokens or verify_input.topk != 1)
        if device:
            device_check(device, None, "spec-verify-input")

    def agree_spec_accept(self, batch, predict, accepted, indices):
        if not self.distributed_spec or batch.forward_mode.is_idle():
            return
        rows, before, counts = batch.req_pool_indices, batch.seq_lens, accepted
        tensors = (rows, before, counts, indices, predict)
        if not all(t.is_cuda and not t.is_floating_point() for t in tensors):
            raise RuntimeError("ARC_SPEC_AGREE_DEVICE: spec-accept inputs must be integer device tensors, got "
                               f"{[(t.device.type, t.dtype) for t in tensors]}")
        qd.agree_spec_transition(self.spec_protocols["forward"], "spec-accept",
                                 [[list(t.shape), str(t.dtype)] for t in tensors],
                                 invalid=not (rows.dim() == before.dim() == counts.dim() == 1
                                              and rows.shape[0] == before.shape[0] == counts.shape[0]))
        parts, bad, _ = _spec_accept_parts(self, rows, before, counts, indices, predict)
        device_check(parts, bad, "spec-accept")

    def claim(self, seq_lens, accept_lens):
        if not self.coord.distributed_spec:
            return original_claim(self, seq_lens, accept_lens)
        t = self.interval
        pre = seq_lens.to(torch.int64)
        cross = pre // t != (pre + accept_lens.to(torch.int64)) // t
        n = len(self.staging_slots)
        fwd = self.coord.spec_protocols["forward"]
        fwd.agree("spec-snapshot-state", [list(self.staging_busy.shape), str(self.staging_busy.dtype)])
        busy = hc.ag_words(self.staging_busy).amax(0)  # spec_staging_busy's MAX over the attention-TP group
        free_rank = torch.cumsum((busy[:n] == 0).to(torch.int64), 0)
        k = torch.cumsum(cross.to(torch.int64), 0)  # 1-based rank among crossing rows
        ok = cross & (k <= free_rank[-1])
        idx = torch.searchsorted(free_rank, k)  # the k-th free entry
        idx = torch.where(ok, idx, torch.full_like(idx, n))
        device_check([pre, accept_lens, busy, idx, self.staging_ids[idx]], None, "spec-snapshot-claim")
        self.staging_busy.index_put_((idx,), torch.ones_like(idx, dtype=torch.int32))
        return self.staging_ids[idx]

    return agree_spec_verify_input, agree_spec_accept, claim


def install_vision_offload():
    """ARC_VISION_OFFLOAD=1: move the unused vision tower to CPU after load (text-only serving)."""
    if not _flag("ARC_VISION_OFFLOAD") or _STATE.get("vision_offload"):
        return
    _STATE["vision_offload"] = True
    from sglang.srt.model_executor.model_runner import ModelRunner
    original_load = ModelRunner.load_model

    def load_and_offload(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        # ARC2 never sends images (agentic_tir sends text only). Any multimodal request now fails
        # loudly with a device mismatch inside the CPU-resident vision tower.
        model = self.model
        vision = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
        if vision is None:
            raise RuntimeError("ARC_VISION_OFFLOAD: model has no .visual module")
        before = torch.cuda.memory_allocated()
        payload = sum(p.numel() * p.element_size() for p in vision.parameters())
        vision.to("cpu")
        torch.cuda.synchronize()
        if not all(p.device.type == "cpu" for p in vision.parameters()):
            raise RuntimeError("vision offload incomplete")
        freed = before - torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        logger.info("ARC_VISION_OFFLOAD payload_bytes=%d cuda_bytes_freed=%d", payload, freed)
        return result

    ModelRunner.load_model = load_and_offload



def _offload_embedding(model, model_path):
    """Serve embed_tokens from one full BF16 table in host-mapped shm shared by the node's ranks.

    Qwen4Exp vocab-shards embed_tokens over the attention-TP group (dp-attention) or the TP group and
    all-reduces the zero-filled partial rows. Gathering whole rows from the full table gives the same bits (x + 0 = x) without
    the all-reduce, and every rank of the group skips it alike.
    """
    from multiprocessing import shared_memory
    import ctypes
    import torch.distributed as dist
    from safetensors import safe_open
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel

    found = [(n, m) for n, m in model.named_modules()
             if n.endswith("embed_tokens") and "visual" not in n and "mtp" not in n]
    if len(found) != 1:
        raise RuntimeError(f"ARC_EMBED_OFFLOAD: expected one embed_tokens, got {[n for n, _ in found]}")
    name, emb = found[0]
    if type(emb) is not VocabParallelEmbedding or emb.num_added_embeddings:
        raise RuntimeError(f"ARC_EMBED_OFFLOAD: {name} is {type(emb).__name__}")
    w = emb.weight
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise RuntimeError(f"ARC_EMBED_OFFLOAD: weight {w.dtype} {tuple(w.shape)}")
    head = getattr(model, "lm_head", None)
    if head is not None and getattr(head, "weight", None) is not None and head.weight.data_ptr() == w.data_ptr():
        raise RuntimeError("ARC_EMBED_OFFLOAD: lm_head is tied to embed_tokens")
    key = "model.language_model.embed_tokens.weight"
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
    vocab, dim = emb.org_vocab_size, w.shape[1]
    nbytes = vocab * dim * 2
    tp = ps.get_tp_group()
    lead = tp.ranks[0]
    me = dist.get_rank()
    shm_name = [f"arc_embed_{os.getpid()}" if me == lead else None]
    if me == lead:
        shm = shared_memory.SharedMemory(name=shm_name[0], create=True, size=nbytes)
        with safe_open(str(Path(model_path) / index[key]), framework="pt") as f:
            full = f.get_tensor(key)
        if tuple(full.shape) != (vocab, dim) or full.dtype != torch.bfloat16:
            raise RuntimeError(f"ARC_EMBED_OFFLOAD: checkpoint {key} is {full.dtype} {tuple(full.shape)}")
        torch.frombuffer(shm.buf, dtype=torch.bfloat16, count=vocab * dim).view(vocab, dim).copy_(full)
        del full
    dist.broadcast_object_list(shm_name, src=lead, group=tp.cpu_group)
    if me != lead:
        shm = shared_memory.SharedMemory(name=shm_name[0])
    dist.barrier(group=tp.cpu_group)
    if me == lead:
        shm.unlink()
    host = torch.frombuffer(shm.buf, dtype=torch.bfloat16, count=vocab * dim).view(vocab, dim)
    # this rank's loaded shard must equal the corresponding rows of the shared table (every row, on GPU)
    lo = emb.shard_indices.org_vocab_start_index
    hi = emb.shard_indices.org_vocab_end_index
    step = 16384
    for r in range(lo, hi, step):
        e = min(hi, r + step)
        if not torch.equal(host[r:e].to(w.device), w.data[r - lo:e - lo]):
            raise RuntimeError(f"ARC_EMBED_OFFLOAD: shared table rows {r}:{e} differ from this rank's shard")
    lib = ctypes.CDLL(os.environ["ARC_HC_LIB"])
    lib.hc_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.hc_device_pointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    # register one rank at a time: concurrent pinning of the same shared pages by all TP ranks intermittently
    # failed on the creating rank with cudaErrorInvalidValue (3 of ~10 launches, always TP0)
    rc = 0
    for r in tp.ranks:
        if me == r:
            rc = lib.hc_register(host.data_ptr(), nbytes)
        dist.barrier(group=tp.cpu_group)
    if rc != 0:
        meminfo = {l.split(":")[0]: l.split()[1] for l in open("/proc/meminfo") if l.split(":")[0] in
                   ("MemAvailable", "Mlocked", "Shmem", "SwapFree")}
        raise RuntimeError(f"ARC_EMBED_OFFLOAD: cudaHostRegister({nbytes} B) failed, cudaError {rc}, meminfo kB {meminfo}")
    dev = ctypes.c_void_p()
    if lib.hc_device_pointer(host.data_ptr(), ctypes.byref(dev)) != 0:
        raise RuntimeError("ARC_EMBED_OFFLOAD: cudaHostGetDevicePointer failed")
    dev_ptr = dev.value
    block_d = 1 << (dim - 1).bit_length()
    device = w.device
    reference = emb.forward  # original gather + attention-TP all-reduce, for the self-check below
    probe = torch.randint(0, vocab, (512,), device=device, generator=torch.Generator(device).manual_seed(0))
    expect = reference(probe).clone()
    before = torch.cuda.memory_allocated()
    emb._arc_shm = shm  # keep the mapping alive
    del emb.weight
    del w  # last local reference to the GPU table; otherwise nothing is freed until return
    emb.weight = torch.nn.Parameter(host, requires_grad=False)  # CPU: any stray GPU use fails loudly

    def forward(input_ids):
        flat = input_ids.reshape(-1).long()
        out = torch.empty((*input_ids.shape, dim), dtype=torch.bfloat16, device=device)
        if flat.numel():
            _gather_ple_embedding_from_pinned_kernel[(flat.numel(),)](
                dev_ptr, flat, out, embedding_dim=dim, tp_vocab_start=0, tp_vocab_end=vocab,
                packed_row_bytes=0, is_fp8=False, is_nvfp4=False, is_int5=False, BLOCK_D=block_d)
        return out

    emb.forward = forward
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    freed = before - torch.cuda.memory_allocated()
    got = forward(probe)
    # exact against the table's own rows; the sharded reference is only bitwise when its all-reduce is: an FP8
    # hostcomm kind (ar1_fp8v on the attention-TP group, any hidden-width tensor) rounds the embedding to e4m3 x
    # per-128 scales (l4opt -eo2 died here), which the whole-row gather no longer does
    if not torch.equal(got.cpu(), host[probe.cpu()]):
        raise RuntimeError("ARC_EMBED_OFFLOAD: host-mapped gather differs from the shared table rows")
    rel = ((got.float() - expect.float()).norm() / expect.float().norm()).item()
    logger.info("ARC_EMBED_OFFLOAD module=%s tp=%d table_bytes=%d cuda_bytes_freed=%d rel_diff_vs_sharded=%.3e",
                name, emb.tp_size, nbytes, freed, rel)


def install_embed_offload():
    """ARC_EMBED_OFFLOAD=1: serve embed_tokens from a host-mapped table (frees ~1.18 GiB per card)."""
    if not _flag("ARC_EMBED_OFFLOAD") or _STATE.get("embed_offload"):
        return
    if "ARC_HC_LIB" not in os.environ:
        raise RuntimeError("ARC_EMBED_OFFLOAD needs ARC_HC_LIB (cudaHostRegister helper)")
    _STATE["embed_offload"] = True
    from sglang.srt.model_executor.model_runner import ModelRunner
    original_load = ModelRunner.load_model

    def load_and_offload_embed(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if not getattr(self, "is_draft_worker", False):
            _offload_embedding(self.model, self.model_config.model_path)
        return result

    ModelRunner.load_model = load_and_offload_embed


def install_online_fp8():
    """ARC_HC_ONLINE_FP8=<names>: arc-forge 0032 online block-FP8 on sm_89 under attention TP2.

    0032 refuses tp>1 and any dispatch but CUTLASS (what it was validated on, sm_120). Here the attention/GDN
    linears are TP2 shards whose split points are 128-aligned, so per-shard 128x128 blocks equal the blocks of
    the full matrix; sm_89's block-FP8 dispatch is sglang's Triton kernel (same per-token-group activation
    quant and block scales, different accumulation order).
    """
    names = os.environ.get("ARC_HC_ONLINE_FP8", "").strip()
    if not names or _STATE.get("online_fp8"):
        return
    _STATE["online_fp8"] = True
    from sglang.srt.layers.quantization import arc_online_fp8 as aof
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.model_executor.model_runner import ModelRunner

    if "lm_head" in names or names == "all":
        raise ValueError("ARC_HC_ONLINE_FP8: lm_head stays BF16 (vocab-parallel here, and judged too sensitive)")
    original_method = aof._method

    def method_sm89():
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
        m = Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                                      weight_block_size=[aof.BLOCK, aof.BLOCK]))
        if not m.block_quant or m.use_marlin or m.use_mxfp8:
            raise RuntimeError(f"ARC_HC_ONLINE_FP8: not plain block FP8 (marlin={m.use_marlin})")
        if m.w8a8_block_fp8_linear is not fp8_utils.triton_w8a8_block_fp8_linear:
            raise RuntimeError(f"ARC_HC_ONLINE_FP8: dispatch is {m.w8a8_block_fp8_linear.__name__}, expected Triton")
        return m

    original_load = ModelRunner.load_model

    def load_and_fp8(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        if os.environ.get(aof.ENV):
            raise RuntimeError(f"{aof.ENV} is set; use ARC_HC_ONLINE_FP8 only")
        before = torch.cuda.memory_allocated()
        os.environ[aof.ENV] = names
        aof._method = method_sm89
        try:
            aof.apply(self.model, is_draft_worker=False, tp_size=1, startup_overlap=False)
        finally:
            del os.environ[aof.ENV]
            aof._method = original_method
        torch.cuda.empty_cache()
        logger.info("ARC_HC_ONLINE_FP8 %s cuda_bytes_freed=%d", names, before - torch.cuda.memory_allocated())
        return result

    ModelRunner.load_model = load_and_fp8


_RING = 64  # ARC_MOE_STATS_DIR: decode / verify MoE calls kept per device


def install_moe_stats():
    """ARC_MOE_STATS=1: count distinct routed experts per decode MoE call (graph-capturable, measurement only).

    Per device: [decode calls, sum of distinct experts over all 320, sum of distinct local (this EP rank)
    experts, sum of real tokens]; a daemon thread logs the running totals every 60 s from a side stream
    (the values are a snapshot, not synchronized with the compute stream).

    ARC_MOE_STATS_DIR=<dir> also keeps the topk_ids of the last 64 decode / verify MoE calls per device in a ring
    (rows padded to 256 with -2, plus each call's row count) and saves a snapshot there every 60 s
    (routing_dev<i>_<n>.pt), so the microbench can replay the deployed routing (skinny_gemm/bench_humming_moe_replay.py,
    l4opt/moe_routing_stats.py). m1t 10-08: C64 verify touches a median 64 of the 80 local experts per call.
    """
    if not _flag("ARC_MOE_STATS") or _STATE.get("moe_stats"):
        return
    _STATE["moe_stats"] = True
    import threading
    import time
    from sglang.srt.layers.moe import topk as topk_mod

    original = topk_mod.TopK.forward_cuda
    state = {}

    def forward_cuda(self, hidden_states, router_logits, **kwargs):
        out = original(self, hidden_states, router_logits, **kwargs)
        ids = getattr(out, "topk_ids", None)
        if ids is None:
            raise RuntimeError(f"ARC_MOE_STATS: {type(out).__name__} has no topk_ids")
        if ids.shape[0] > 256:  # prefill chunks; decode batches are <= running
            return out
        dev = ids.device
        st = state.get(dev)
        if st is None:
            from sglang.srt.distributed import parallel_state as ps
            n_exp = router_logits.shape[-1]
            ep = ps.get_moe_expert_parallel_world_size()
            r = ps.get_moe_expert_parallel_rank()
            per = n_exp // ep
            st = {"buf": torch.zeros(n_exp + 1, dtype=torch.int32, device=dev),
                  "acc": torch.zeros(4, dtype=torch.int64, device=dev),
                  "n": n_exp, "lo": r * per, "hi": (r + 1) * per, "dir": os.environ.get("ARC_MOE_STATS_DIR")}
            if st["dir"]:
                st["ring"] = torch.full((_RING, 256, ids.shape[1]), -2, dtype=torch.int32, device=dev)
                st["rows"] = torch.zeros(_RING, dtype=torch.int32, device=dev)
                st["ctr"] = torch.zeros(1, dtype=torch.int64, device=dev)
            state[dev] = st
            _start_reporter(dev, st)
        buf, acc, n = st["buf"], st["acc"], st["n"]
        flat = ids.reshape(-1).long()
        flat = torch.where(flat < 0, torch.full_like(flat, n), flat)
        buf.zero_()
        buf.index_fill_(0, flat, 1)
        acc[0] += 1
        acc[1] += buf[:n].sum()
        acc[2] += buf[st["lo"]:st["hi"]].sum()
        acc[3] += (ids[:, 0] >= 0).sum()
        if st["dir"]:
            m = ids.shape[0]
            padded = torch.full((256, ids.shape[1]), -2, dtype=torch.int32, device=dev)
            padded[:m] = ids
            slot = st["ctr"] % _RING
            st["ring"].index_copy_(0, slot, padded.unsqueeze(0))
            st["rows"].index_copy_(0, slot, torch.full((1,), m, dtype=torch.int32, device=dev))
            st["ctr"] += 1
        return out

    def _start_reporter(dev, st):
        def loop():
            torch.cuda.set_device(dev)
            side = torch.cuda.Stream(device=dev)
            snap = 0
            while True:
                time.sleep(60)
                with torch.cuda.stream(side):
                    calls, ug, ul, tok = st["acc"].cpu().tolist()
                    if st["dir"]:
                        ring, rows, ctr = st["ring"].cpu(), st["rows"].cpu(), int(st["ctr"].cpu())
                if calls:
                    logger.info("ARC_MOE_STATS calls=%d tokens_per_call=%.1f unique_global=%.1f/%d unique_local=%.1f/%d",
                                calls, tok / calls, ug / calls, st["n"], ul / calls, st["hi"] - st["lo"])
                if st["dir"] and ctr:
                    out_dir = Path(st["dir"])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"ids": ring, "rows": rows, "calls": ctr, "experts": st["n"], "local": (st["lo"], st["hi"]),
                                "time": time.time()}, out_dir / f"routing_dev{dev.index}_{snap:03d}.pt")
                    snap += 1
        threading.Thread(target=loop, daemon=True, name="arc-moe-stats").start()

    topk_mod.TopK.forward_cuda = forward_cuda


def install_w8a16():
    """ARC_HC_W8A16=<names>: weight-only INT8 (per row x 128-column group, RTN) for chosen dense linears.

    Same layer selection as 0032 (gdn_in / gdn_out / qsa_qkv / qsa_o; lm_head refused). Activations stay BF16.
    The GEMM is the Triton kernel in probes/skinny_gemm/w8a16.py (ARC_SKINNY_DIR).
    """
    names = os.environ.get("ARC_HC_W8A16", "").strip()
    if not names or _STATE.get("w8a16"):
        return
    _STATE["w8a16"] = True
    from sglang.srt.layers.quantization import arc_online_fp8 as aof
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.model_executor.model_runner import ModelRunner

    wanted = [n.strip() for n in names.split(",") if n.strip()]
    bad = [n for n in wanted if n not in aof._SUFFIX]
    if bad:
        raise ValueError(f"ARC_HC_W8A16: unknown {bad}; choose from {list(aof._SUFFIX)} (lm_head is not allowed)")
    sys.path.insert(0, os.environ["ARC_SKINNY_DIR"])
    import w8a16 as wk

    class W8A16Method:
        def apply(self, layer, x, bias=None):
            y = wk.w8a16_linear(x, layer.w8_q, layer.w8_s)
            return y if bias is None else y + bias

    original_load = ModelRunner.load_model

    def load_and_w8(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
        before = torch.cuda.memory_allocated()
        counts = {n: 0 for n in wanted}
        for mod_name, module in self.model.named_modules():
            if "visual" in mod_name:
                continue
            for target in wanted:
                if not aof._SUFFIX[target].search(mod_name):
                    continue
                if not isinstance(module.quant_method, UnquantizedLinearMethod):
                    raise RuntimeError(f"ARC_HC_W8A16: {mod_name} is {type(module.quant_method).__name__}")
                w = module.weight
                if w.dtype != torch.bfloat16 or w.dim() != 2 or getattr(module, "bias", None) is not None:
                    raise RuntimeError(f"ARC_HC_W8A16: {mod_name} weight {w.dtype} {tuple(w.shape)} / bias")
                q, s = wk.quantize(w.data)
                del module.weight
                module.register_buffer("w8_q", q, persistent=False)
                module.register_buffer("w8_s", s, persistent=False)
                module.quant_method = W8A16Method()
                counts[target] += 1
            if "gdn_in" in wanted and isinstance(module, Qwen3_5GatedDeltaNet) and module._fused_in_proj_weight is not None:
                module.in_proj_ba.weight.data = module.in_proj_ba.weight.data.clone()
                module._fused_in_proj_weight = None
                module._fused_in_proj_qkvz_width = 0
        missing = [n for n, c in counts.items() if c == 0]
        if missing:
            raise RuntimeError(f"ARC_HC_W8A16: no layer matched {missing}")
        torch.cuda.empty_cache()
        logger.info("ARC_HC_W8A16 %s cuda_bytes_freed=%d", json.dumps(counts), before - torch.cuda.memory_allocated())
        return result

    ModelRunner.load_model = load_and_w8


def install_qsa_decode_knobs():
    """ARC_QSA_DECODE=<0|1|2> / ARC_QSA_CHUNK=<n>: override the direct-decode E2M1 variant and split chunk.

    decode 0 (arithmetic) and 1 (bit construction) give bit-identical values; 2 needs sm_100a/sm_120a.
    chunk changes the flash-decoding split count (combine order), not the math.
    """
    dec = os.environ.get("ARC_QSA_DECODE")
    chunk = os.environ.get("ARC_QSA_CHUNK")
    if (dec is None and chunk is None) or _STATE.get("qsa_knobs"):
        return
    _STATE["qsa_knobs"] = True
    import sglang.srt.layers.attention.qwen_sparse_attn_backend as qb
    original = qb.qsa_sparse_decode_direct

    def direct(*args, **kwargs):
        if dec is not None:
            kwargs["decode"] = int(dec)
        if chunk is not None:
            kwargs["chunk"] = int(chunk)
        return original(*args, **kwargs)

    qb.qsa_sparse_decode_direct = direct
    logger.info("ARC_QSA_DECODE_KNOBS decode=%s chunk=%s", dec, chunk)


def install_hc_mix_rows():
    """ARC_HC_MIX_ROWS=<n>: let the persistent fused HC-mix kernel (hc_mix_triton) serve up to n rows (<=128).

    Upstream caps it at 16 rows (ROWS=16 padding); decode batches of 40-80 rows per DP otherwise take the
    five-kernel torch.compile chain. ROWS is padded to the next power of two >= 16.
    """
    limit = os.environ.get("ARC_HC_MIX_ROWS")
    if not limit or _STATE.get("hc_mix_rows"):
        return
    limit = int(limit)
    if not 16 < limit <= 128:
        raise ValueError(f"ARC_HC_MIX_ROWS must be in (16, 128], got {limit}")
    _STATE["hc_mix_rows"] = True
    import sglang.srt.layers.hc_mix_triton as hm
    import sglang.srt.layers.hyperconnection as hcmod

    def supported(x, wd, wu):
        if hm._deterministic_inference() or hm._ARC_HC_MIX_DETERMINISTIC:
            return False
        return (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and wd.dtype == x.dtype
                and wu.dtype == x.dtype and x.dim() == 2 and x.shape[0] <= limit and x.shape[1] % 2048 == 0
                and x.is_contiguous() and wd.is_contiguous() and wu.is_contiguous())

    def fused(x, wd, wu, hc, hs):
        rows, k = x.shape
        lowrank = wd.shape[0]
        rows_pad = max(16, 1 << (rows - 1).bit_length())
        device = x.device
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
        t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
        out = torch.empty((rows, hs), dtype=x.dtype, device=device)
        if rows == 0:
            return out
        hm._hc_mix_persistent_kernel[(num_ctas,)](
            x, wd, wu, t_raw, out, hm._get_counters(device), k, lowrank, hs, rows, num_ctas, 1.0 / hc,
            ROWS=rows_pad, HC=hc, BLOCK_N=32, BLOCK_K=256 if rows_pad <= 32 else 128,
            BLOCK_J=16 if rows_pad >= 64 else 32, BLOCK_R=64, num_warps=8)
        return out

    hcmod.fused_hc_mix_supported = supported
    hcmod.fused_hc_mix = fused
    logger.info("ARC_HC_MIX_ROWS limit=%d", limit)


def install_hc_mix_verify():
    """ARC_HC_MIX_VERIFY=<lo>:<hi>: GatedResidual mixes of lo < rows <= hi (verify batches) on skinny_gemm/hc_mix_rows.py
    (deterministic split-K down projection, gate-mean fused into the up GEMM; three launches) instead of the
    five-kernel torch.compile chain. Tile configs per row bucket from ARC_SKINNY_DIR/hc_mix_rows_l4.json
    (bench_hc_mix_rows.py on an L4: {"buckets": [[max_rows, cfg], ...]}, ascending). BF16 weights; the same math with
    fp32 accumulation throughout, so not bitwise against the chain. Traced (torch.compile) mixes keep the chain.
    """
    spec = os.environ.get("ARC_HC_MIX_VERIFY")
    if not spec or _STATE.get("hc_mix_verify"):
        return
    if os.environ.get("ARC_HC_MIX_ROWS") or _flag("ARC_HC_MIX_W8"):
        raise RuntimeError("ARC_HC_MIX_VERIFY replaces the mixer path; it excludes ARC_HC_MIX_ROWS / ARC_HC_MIX_W8")
    lo, hi = (int(v) for v in spec.split(":"))
    if not 0 <= lo < hi:
        raise ValueError(f"ARC_HC_MIX_VERIFY must be <lo>:<hi> with 0 <= lo < hi, got {spec!r}")
    _STATE["hc_mix_verify"] = True
    import sglang.srt.layers.hyperconnection as hcmod

    skinny = os.environ["ARC_SKINNY_DIR"]
    sys.path.insert(0, skinny)
    import hc_mix_rows as hmr

    with open(os.path.join(skinny, "hc_mix_rows_l4.json")) as f:
        buckets = json.load(f)["buckets"]
    if not buckets or [b[0] for b in buckets] != sorted(b[0] for b in buckets):
        raise ValueError(f"hc_mix_rows_l4.json buckets must be non-empty and ascending: {buckets}")
    orig_supported, orig_fused = hcmod.fused_hc_mix_supported, hcmod.fused_hc_mix
    counts = {"calls": 0}

    def ours(x, wd, wu):
        return (lo < x.shape[0] <= hi and not torch.compiler.is_compiling() and x.dim() == 2
                and x.dtype == torch.bfloat16 and wd.dtype == torch.bfloat16 and wu.dtype == torch.bfloat16
                and x.is_contiguous() and wd.is_contiguous() and wu.is_contiguous())

    def supported(x, wd, wu):
        return ours(x, wd, wu) or orig_supported(x, wd, wu)

    def fused(x, wd, wu, hc, hs):
        if not ours(x, wd, wu):
            return orig_fused(x, wd, wu, hc, hs)
        cfg = next((c for m, c in buckets if x.shape[0] <= m), buckets[-1][1])
        if counts["calls"] == 0:
            logger.info("ARC_HC_MIX_VERIFY first mix rows=%d cfg=%s", x.shape[0], cfg)
        counts["calls"] += 1
        return hmr.hc_mix_rows(x, wd, wu, hc, hs, cfg={**hmr.DEFAULT_CFG, **cfg})

    hcmod.fused_hc_mix_supported = supported
    hcmod.fused_hc_mix = fused
    logger.info("ARC_HC_MIX_VERIFY rows (%d, %d] buckets=%s", lo, hi, [b[0] for b in buckets])


def install_lmhead_triton():
    """ARC_LMHEAD_TRITON=<lo>:<hi>: the LM head GEMM of lo < rows <= hi (verify batches) on skinny_gemm/lm_head_gemm.py
    instead of LogitsProcessor's torch.matmul (L4 at 192 rows x 124,160 x 2560: 2,765 vs 3,016 us). Only the plain
    BF16 path (no bias, LoRA, quant method, fp32 head or RL on-policy cast); everything else keeps the original. fp32
    accumulation like cuBLAS, another kernel: not bitwise.
    """
    spec = os.environ.get("ARC_LMHEAD_TRITON")
    if not spec or _STATE.get("lmhead_triton"):
        return
    lo, hi = (int(v) for v in spec.split(":"))
    if not 0 <= lo < hi:
        raise ValueError(f"ARC_LMHEAD_TRITON must be <lo>:<hi> with 0 <= lo < hi, got {spec!r}")
    _STATE["lmhead_triton"] = True
    from sglang.srt.layers import logits_processor as lp

    sys.path.insert(0, os.environ["ARC_SKINNY_DIR"])
    import lm_head_gemm as lg

    orig = lp.LogitsProcessor._compute_lm_head
    counts = {"calls": 0}

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        w = getattr(lm_head, "weight", None)
        if (embedding_bias is None and w is not None and not self.use_fp32_lm_head
                and getattr(self, "rl_on_policy_target", None) is None and not hasattr(lm_head, "set_lora")
                and not lp.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
                and hidden_states.dim() == 2 and lo < hidden_states.shape[0] <= hi and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
                and hidden_states.is_contiguous() and w.is_contiguous()):
            if counts["calls"] == 0:
                logger.info("ARC_LMHEAD_TRITON first call rows=%d vocab=%d", hidden_states.shape[0], w.shape[0])
            counts["calls"] += 1
            return lg.lm_head_gemm(hidden_states, w)
        return orig(self, hidden_states, lm_head, embedding_bias)

    lp.LogitsProcessor._compute_lm_head = _compute_lm_head
    logger.info("ARC_LMHEAD_TRITON rows (%d, %d] cfg=%s", lo, hi, json.dumps(lg.CFG, sort_keys=True))


def install_hc_mix_w8():
    """ARC_HC_MIX_W8=1: hyper-connection mixers (GatedResidual.mix) with INT8 weight-only down/up projections.

    Same math as GatedResidual._mix_compute (silu(x Wd^T / hc) -> sigmoid(. Wu^T) gates -> mean over the hc
    streams) with W8A16 GEMMs (per-row x group scales, bf16 activations) for every row count; the BF16 mixer
    weights (1.19 GiB/card, replicated) are freed. Needs ARC_SKINNY_DIR.
    """
    if not _flag("ARC_HC_MIX_W8") or _STATE.get("hc_mix_w8"):
        return
    _STATE["hc_mix_w8"] = True
    import types
    import torch.nn.functional as F
    from sglang.srt.layers.hyperconnection import GatedResidual
    from sglang.srt.model_executor.model_runner import ModelRunner

    sys.path.insert(0, os.environ["ARC_SKINNY_DIR"])
    import w8a16 as wk

    @torch.compile(dynamic=True)
    def _gate_mean(u, x, hc: int, hs: int):
        g = torch.sigmoid(u).unflatten(-1, (hc, hs))
        return (g * x.unflatten(-1, (hc, hs))).mean(dim=-2)

    def mix(self, hyper_input):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed = hyper_input.new_empty((*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype)
            return mixed, (hyper_input, hyper_input)
        if self.config.hc_per_branch_norm:
            normed = self.hc_norm(hyper_input)
        else:
            normed = self.hc_norm(hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))).flatten(-2)
        t = F.silu(wk.w8a16_linear(normed, self._w8_dq, self._w8_ds) / self.hc_count)
        u = wk.w8a16_linear(t.contiguous(), self._w8_uq, self._w8_us)
        mixed = _gate_mean(u, normed, self.hc_count, self.hidden_size).to(self.params_dtype)
        return mixed, (hyper_input, normed)

    original_load = ModelRunner.load_model

    def load_and_hc_w8(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        before = torch.cuda.memory_allocated()
        n = 0
        for name, module in self.model.named_modules():
            if not isinstance(module, GatedResidual):
                continue
            wd, wu = module.input_mix_weight_down.weight, module.input_mix_weight_up.weight
            if wd.dtype != torch.bfloat16 or wu.dtype != torch.bfloat16:
                raise RuntimeError(f"ARC_HC_MIX_W8: {name} weights {wd.dtype}/{wu.dtype}")
            module._w8_dq, module._w8_ds = wk.quantize(wd.data)
            module._w8_uq, module._w8_us = wk.quantize(wu.data)
            del module.input_mix_weight_down.weight, module.input_mix_weight_up.weight
            module.mix = types.MethodType(mix, module)
            n += 1
        if n == 0:
            raise RuntimeError("ARC_HC_MIX_W8: no GatedResidual module found")
        torch.cuda.empty_cache()
        logger.info("ARC_HC_MIX_W8 mixers=%d cuda_bytes_freed=%d", n, before - torch.cuda.memory_allocated())
        return result

    ModelRunner.load_model = load_and_hc_w8


def install_humming_dense():
    """ARC_HC_HUMMING_DENSE='<dtype>:<group>' (e.g. int4:128) with ARC_HC_HUMMING_LAYERS=<names>: weight-only
    low-bit dense linears through Humming (BF16 activations). Layer names as 0032 (gdn_in/gdn_out/qsa_qkv/qsa_o).
    Needs ARC_HUMMING_DIR (unpacked humming-kernels) on the path.
    """
    spec = os.environ.get("ARC_HC_HUMMING_DENSE", "").strip()
    if not spec or _STATE.get("humming_dense"):
        return
    _STATE["humming_dense"] = True
    dtype, group = spec.split(":")
    wanted = [n.strip() for n in os.environ["ARC_HC_HUMMING_LAYERS"].split(",") if n.strip()]
    sys.path.insert(0, os.environ["ARC_HUMMING_DIR"])
    from humming.layer import HummingLayer
    from sglang.srt.layers.quantization import arc_online_fp8 as aof
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.model_executor.model_runner import ModelRunner

    bad = [n for n in wanted if n not in aof._SUFFIX]
    if bad:
        raise ValueError(f"ARC_HC_HUMMING_LAYERS: unknown {bad}")

    class HummingDenseMethod:
        def apply(self, layer, x, bias=None):
            lead = x.shape[:-1]
            y = layer._hm(x.reshape(-1, x.shape[-1]).contiguous())
            y = y.view(*lead, y.shape[-1])
            return y if bias is None else y + bias

    original_load = ModelRunner.load_model

    def load_and_humming(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
        before = torch.cuda.memory_allocated()
        counts = {n: 0 for n in wanted}
        for mod_name, module in self.model.named_modules():
            if "visual" in mod_name:
                continue
            for target in wanted:
                if not aof._SUFFIX[target].search(mod_name):
                    continue
                if not isinstance(module.quant_method, UnquantizedLinearMethod):
                    raise RuntimeError(f"ARC_HC_HUMMING_DENSE: {mod_name} is {type(module.quant_method).__name__}")
                w = module.weight
                n, k = w.shape
                hm = HummingLayer(shape_n=n, shape_k=k, weight_config={"b_dtype": dtype, "group_size": int(group)},
                                  torch_dtype=torch.bfloat16)
                hm.load_from_unquantized(w.data)
                hm = hm.to(w.device)
                hm.transform()
                del module.weight, w
                module._hm = hm
                module.quant_method = HummingDenseMethod()
                counts[target] += 1
            if "gdn_in" in wanted and isinstance(module, Qwen3_5GatedDeltaNet) and module._fused_in_proj_weight is not None:
                module.in_proj_ba.weight.data = module.in_proj_ba.weight.data.clone()
                module._fused_in_proj_weight = None
                module._fused_in_proj_qkvz_width = 0
        missing = [n for n, c in counts.items() if c == 0]
        if missing:
            raise RuntimeError(f"ARC_HC_HUMMING_DENSE: no layer matched {missing}")
        torch.cuda.empty_cache()
        logger.info("ARC_HC_HUMMING_DENSE %s %s cuda_bytes_freed=%d", spec, json.dumps(counts),
                    before - torch.cuda.memory_allocated())
        return result

    ModelRunner.load_model = load_and_humming


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _dequant_nvfp4(w_u8, scale_e4m3, scale_2):
    """modelopt NVFP4 [N, K/2] uint8 (low nibble = even column), e4m3 [N, K/16], fp32 scalar -> fp32 [N, K]."""
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=w_u8.device)
    lo = (w_u8 & 0x0F).long()
    hi = (w_u8 >> 4).long()
    vals = torch.stack((lut[lo], lut[hi]), dim=-1).reshape(w_u8.shape[0], -1)
    return vals * scale_e4m3.float().repeat_interleave(16, dim=1) * scale_2.float()


_TOPK_SUM = {}


def _topk_weighted_sum(down, topk_weights, topk_ids, scale, out_dtype):
    """out[m] = scale * sum_k topk_weights[m, k] * down[m * topk + k] over the local experts (topk_ids >= 0), fp32
    accumulation, one Triton kernel. Replaces down.float() * w.float() -> sum(1) -> .to(bf16) (a float copy of all
    M*topk rows, a multiply, a reduce and a cast per MoE layer) and needs no zero-filled `down`: rows of non-local
    experts (never written by the indexed GEMM) are masked by their id instead."""
    if "kernel" not in _TOPK_SUM:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(down_ptr, w_ptr, ids_ptr, out_ptr, N, scale, TOPK: tl.constexpr, BLOCK_N: tl.constexpr):
            m = tl.program_id(0).to(tl.int64)
            offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < N
            acc = tl.zeros([BLOCK_N], dtype=tl.float32)
            for k in tl.static_range(TOPK):
                live = tl.load(ids_ptr + m * TOPK + k) >= 0
                wk = tl.load(w_ptr + m * TOPK + k).to(tl.float32)
                x = tl.load(down_ptr + (m * TOPK + k) * N + offs, mask=mask & live, other=0.0)
                acc += x.to(tl.float32) * wk
            # round to nearest even, as torch's float -> bf16 cast in the expression this replaces
            tl.store(out_ptr + m * N + offs, (acc * scale).to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
                     mask=mask)

        _TOPK_SUM["kernel"] = kernel
        _TOPK_SUM["cdiv"] = triton.cdiv
    M, topk = topk_ids.shape
    N = down.shape[-1]
    if down.shape[0] != M * topk or topk_weights.shape != topk_ids.shape or not down.is_contiguous():
        raise RuntimeError(f"_topk_weighted_sum: down {tuple(down.shape)}, weights {tuple(topk_weights.shape)}, "
                           f"ids {tuple(topk_ids.shape)}")
    out = torch.empty((M, N), dtype=out_dtype, device=down.device)
    if M:
        block_n = 512
        _TOPK_SUM["kernel"][(M, _TOPK_SUM["cdiv"](N, block_n))](
            down, topk_weights.contiguous(), topk_ids.contiguous(), out, N, float(scale), TOPK=topk, BLOCK_N=block_n)
    return out


def install_humming_moe():
    """ARC_HC_HUMMING_MOE='<dtype>:<group>' (e.g. int3:128, int4:128) or 'nvfp4': routed experts through Humming's
    indexed MoE GEMM instead of Marlin. Expert weights are read from the NVFP4 checkpoint, dequantized and
    requantized at load; Marlin's copies are freed. Each call takes one heuristic tuning config per (GEMM,
    power-of-two bucket of a row count); its block_shape_m is the moe_align block size, since the indexed kernel walks
    sorted_ids in block_shape_m chunks with one expert per chunk.

    ARC_HC_HUMMING_MOE_TILE (unset = global): the row count the tuning heuristic gets. Humming picks block_shape_m from
    rows / num_experts, the rows each expert is expected to hold. 'global' passes M*top_k, the rows of all experts
    of every EP rank, so with 80 of 320 experts here it expects 4x the rows a local expert gets (and rounds up to a
    power of two): K4 verify at C64 (M=256) gets block 64 for ~8 rows per local expert. 'local' passes
    M*top_k*E_local/E_global. L4 back-to-back microbench (10-08, INT3 g128, BF16): M=256 731 -> 513 us per layer, but
    that replay runs at the 72 W power cap (~1.05 GHz) where block 64's padded MMA is compute-bound; serving runs at
    2040 MHz (nvidia-smi). Deployed C64 verify (h1t vs e2t traces): 625 -> 574 us per layer, routed MoE 29.6 -> 27.2
    ms per verify step; bench_humming_moe_replay with the deployed routing at the serving clock: 599 -> 526. Prefill
    chunks get block 128 instead of 96 (gate_up ~6% slower).

    ARC_HC_HUMMING_MOE_ACT (unset = bf16): 'int8' quantizes each GEMM's input per token inside Humming (W3A8, INT8
    MMA). The microbench gains (prefill-sized calls 2285 -> 1353 us) come from the power-capped clock: deployed (h2t)
    the verify GEMMs are unchanged (395 / 171 us) plus 14 us per layer of input quantization, +0.7 ms per verify
    step; the routed MoE GEMMs are ~5% of a 2048-token prefill chunk (communication ~65%). Relative output difference
    ~0.8% per GEMM.

    ARC_HC_HUMMING_MOE=int3packed: the checkpoint is the packed INT3 GPTQ deployment (config arc_int3, layout
    humming_uint3_g128_nozp: routed experts already in Humming's uint3 g128 format with signed BF16 group scales;
    GDN in/out and attention q/k/v block FP8; o_proj BF16; the G4 W3A8 model). sglang cannot load those experts, so
    _install_int3_packed_carrier gives them a placeholder method on the Marlin path this hook intercepts, the dense
    FP8 / BF16 modules their methods, and drops the expert tensors from the load; after the load each rank reads its
    own E_local experts' rows from the safetensors straight into Humming (no requantization). ARC_HC_HUMMING_MOE_ACT
    is bf16 (W3A16) or fp8 (W3A8: dynamic per-token e4m3, the G4 deployment's ARC_INT3_ACT=fp8); int8 is refused, it
    would make Humming fold the signed BF16 group scales into integer scales (a different weight numerics path).

    ARC_HC_AG_KIND=ag_fp8q (int3packed W3A8 only): gate_up takes e4m3 inputs with one scale per 128 values (input
    group 128 instead of per token). A call whose hidden_states are the rows the last ag_fp8q gather wrote (decode /
    verify: the MoE dispatch gather) passes that gather's payload as it travelled (no requantization, one FP8
    rounding instead of two); other calls (prefill, whose dispatch is the bf16 AR4) let Humming quantize the bf16 rows
    in groups of 128. A captured call without its gather raises. down keeps per-token inputs. 4090 layer-21 checks
    (10-10): gate_up alone group-128 prequantized 2.11% vs per-token inside 2.55% relative error (W3A16 0.21%); the
    whole MoE layer fed hostcomm's payload 4.39-4.45% vs per-token W3A8 4.47-4.53% (check_int3packed_layer.py).
    """
    spec = os.environ.get("ARC_HC_HUMMING_MOE", "").strip()
    if not spec or _STATE.get("humming_moe"):
        return
    _STATE["humming_moe"] = True
    tile = os.environ.get("ARC_HC_HUMMING_MOE_TILE", "global")
    if tile not in ("global", "local"):
        raise ValueError(f"ARC_HC_HUMMING_MOE_TILE={tile!r}: global or local")
    act = os.environ.get("ARC_HC_HUMMING_MOE_ACT", "bf16")
    packed = spec == "int3packed"
    fp8q = os.environ.get("ARC_HC_AG_KIND") == "ag_fp8q"
    if fp8q and not (packed and act == "fp8"):
        raise ValueError("ARC_HC_AG_KIND=ag_fp8q needs ARC_HC_HUMMING_MOE=int3packed with ARC_HC_HUMMING_MOE_ACT=fp8")
    if packed:
        if act not in ("bf16", "fp8"):
            raise ValueError(f"ARC_HC_HUMMING_MOE_ACT={act!r}: the packed INT3 checkpoint runs bf16 or fp8 activations")
        fmt = None
    elif act not in ("bf16", "int8"):
        raise ValueError(f"ARC_HC_HUMMING_MOE_ACT={act!r}: bf16 or int8")
    elif spec == "nvfp4":
        if act != "bf16":
            raise ValueError("ARC_HC_HUMMING_MOE_ACT: INT8 activations take integer weights, not nvfp4")
        fmt = {"b_dtype": "float4e2m1", "group_size": 16, "bs_dtype": "float8e4m3", "scale_2_type": "tensor"}
    else:
        dtype, group = spec.split(":")
        fmt = {"b_dtype": dtype, "group_size": int(group)}
    input_cfg = {"dtype": "int8"} if act == "int8" else None
    sys.path.insert(0, os.environ["ARC_HUMMING_DIR"])
    from humming.layer import HummingLayer
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.tune import get_heuristics_config
    from safetensors import safe_open
    import sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe as fmm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.model_executor.model_runner import ModelRunner

    table = {}  # placeholder w13 data_ptr -> (hm_up, hm_down, E_local, E_global)
    original_fmm = fmm.fused_marlin_moe
    compute_cfg = json.dumps({"gemm_type": "indexed"})
    tuning_cache = {}  # (shape_n, shape_k, bucket) -> (tuning json, block_shape_m)

    def tuning_for(hm, rows, device):
        bucket = 1 << max(4, (rows - 1).bit_length())
        key = (hm.shape_n, hm.shape_k, bucket)
        ent = tuning_cache.get(key)
        if ent is None:
            t = get_heuristics_config(hm.humming_config, shape_m=bucket, gemm_type="indexed", device=device)
            if not isinstance(t, dict):
                raise RuntimeError(f"ARC_HC_HUMMING_MOE: expected one tuning dict, got {type(t).__name__}")
            ent = (json.dumps(t), int(t["block_shape"][0]))
            tuning_cache[key] = ent
            logger.info("ARC_HC_HUMMING_MOE tuning n=%d k=%d rows<=%d %s", hm.shape_n, hm.shape_k, bucket, ent[0])
        return ent

    def align(local, E, bsm):
        sorted_ids, expert_ids, ntp = moe_align_block_size(local, bsm, E + 1)
        n_dummy = (local == E).sum().to(torch.int32)
        return sorted_ids, expert_ids, (ntp - (n_dummy + bsm - 1) // bsm * bsm).to(torch.int32)

    def fused_marlin_moe(hidden_states, w1, w2, *args, **kwargs):
        ent = table.get(w1.data_ptr())
        if ent is None:
            _fp8q_forget()  # e.g. the drafter's Marlin MoE
            return original_fmm(hidden_states, w1, w2, *args, **kwargs)
        hm_up, hm_dn, E, E_global = ent
        if (kwargs.get("activation", "silu") != "silu" or not kwargs.get("is_gated", True)
                or kwargs.get("gemm1_alpha") is not None or kwargs.get("clamp_limit") is not None
                or kwargs.get("inplace", False)):
            raise RuntimeError("ARC_HC_HUMMING_MOE: only plain gated SiLU, non-inplace MoE is implemented")
        scale = kwargs.get("routed_scaling_factor") or 1.0
        topk_ids = kwargs["topk_ids"]
        topk_weights = kwargs["topk_weights"]
        M, topk = topk_ids.shape
        if not torch.cuda.is_current_stream_capturing() and M > 0:
            # Device-side assert: a host read here (int(topk_ids.max())) synced every eager MoE layer, 48 per prefill
            # chunk, so the CPU could never launch ahead of the GPU (d1 trace 10-08).
            torch._assert_async(topk_ids.max() < E, f"ARC_HC_HUMMING_MOE: topk_ids >= local experts {E}; ids are not local")
        local = torch.where(topk_ids < 0, torch.full_like(topk_ids, E), topk_ids).to(torch.int32)
        dev = hidden_states.device
        rows = M * topk if tile == "global" else max(1, M * topk * E // E_global)
        t_up, bsm_up = tuning_for(hm_up, rows, dev)
        t_dn, bsm_dn = tuning_for(hm_dn, rows, dev)
        a_up = align(local, E, bsm_up)
        a_dn = a_up if bsm_dn == bsm_up else align(local, E, bsm_dn)
        # Rows of non-local experts are never written (dummy expert E, cut from num_tokens_padded) nor read by the
        # down GEMM, and _topk_weighted_sum masks them, so neither buffer needs zero-filling.
        up = torch.empty((M * topk, hm_up.shape_n), dtype=hidden_states.dtype, device=dev)
        x_up, x_scale = hidden_states, None
        if fp8q and M > 0:
            stash = _STATE.get("fp8q")
            K = hidden_states.shape[1]
            if stash is not None and stash["last"] == (hidden_states.data_ptr(), hidden_states.numel()):
                x_up = stash["q"][:M * K].view(torch.float8_e4m3fn).view(M, K)
                x_scale = stash["s"][:M * K // 128].view(M, K // 128)
                stash["hits"] += 1
                stash["last"] = None
                if torch.cuda.is_current_stream_capturing() and not stash["logged"]:
                    stash["logged"] = True
                    logger.info("ARC_HC_FP8Q captured_hit rank=%d rows=%d", torch.distributed.get_rank(), M)
            elif torch.cuda.is_current_stream_capturing():
                raise RuntimeError("ARC_HC_AG_KIND=ag_fp8q: a captured MoE call whose rows are not the last ag_fp8q "
                                   f"gather's (M={M}, last={None if stash is None else stash['last']})")
            elif stash is not None:
                stash["misses"] += 1
        hm_up(x_up, outputs=up, input_scale=x_scale, sorted_ids=a_up[0], expert_ids=a_up[1],
              num_tokens_padded=a_up[2], top_k=topk, compute_config=compute_cfg, tuning_config=t_up)
        act = torch.empty((M * topk, hm_up.shape_n // 2), dtype=hidden_states.dtype, device=dev)
        fmm.silu_and_mul(up, act)
        down = torch.empty((M * topk, hm_dn.shape_n), dtype=hidden_states.dtype, device=dev)
        hm_dn(act, outputs=down, sorted_ids=a_dn[0], expert_ids=a_dn[1], num_tokens_padded=a_dn[2], top_k=1,
              compute_config=compute_cfg, tuning_config=t_dn)
        return _topk_weighted_sum(down, topk_weights, topk_ids, scale, hidden_states.dtype)

    fmm.fused_marlin_moe = fused_marlin_moe
    carrier = _install_int3_packed_carrier() if packed else None

    original_load = ModelRunner.load_model

    def load_and_humming_moe(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        model_path = Path(self.model_config.model_path)
        index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
        handles = {}
        load_dev = f"cuda:{torch.cuda.current_device()}"

        def handle(name):
            f = index[name]
            if f not in handles:
                # straight to this rank's GPU: CPU copies of ~11k expert tensors leave GBs of malloc'd heap
                # resident per rank, which the HiSparse host admission then counts against the node
                handles[f] = safe_open(str(model_path / f), framework="pt", device=load_dev)
            return handles[f]

        def get(name):
            return handle(name).get_tensor(name)

        def packed_layer(prefix, sub, n, k, E, g0, dev):
            """This rank's experts [g0, g0+E) of one packed bank, loaded as stored (no requantization)."""
            group = 128 if fp8q and sub == "w13" else 0  # ag_fp8q: gate_up takes the dispatch payload's groups
            hm = HummingLayer(shape_n=n, shape_k=k, weight_config=HummingWeightSchema(
                                  b_dtype="uint3", weight_scale_group_size=128, has_zero_point=False),
                              input_config=(HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=group)
                                            if act == "fp8" else None),
                              num_experts=E, torch_dtype=torch.bfloat16).to(dev)
            params = dict(hm.named_parameters())
            if set(params) != {"weight", "weight_scale"}:
                raise RuntimeError(f"ARC_HC_HUMMING_MOE int3packed: Humming uint3 g128 holds {sorted(params)}")
            for key, ckpt in (("weight", f"{prefix}.{sub}_qweight"), ("weight_scale", f"{prefix}.{sub}_scales")):
                sl = handle(ckpt).get_slice(ckpt)
                full = sl.get_shape()
                t = sl[g0:g0 + E].to(dev)
                p = params[key]
                if full[0] != E_global or t.shape != p.shape or t.dtype != p.dtype:
                    raise RuntimeError(f"ARC_HC_HUMMING_MOE int3packed: {ckpt} {full} {t.dtype} -> rows [{g0}, {g0 + E}) "
                                       f"{tuple(t.shape)}, Humming wants {tuple(p.shape)} {p.dtype}")
                p.data.copy_(t)
                del t
            hm.transform()
            return hm

        before = torch.cuda.memory_allocated()
        n_layers = 0
        want_method = "_Int3PackedPlaceholderMoE" if packed else "ModelOptNvFp4FusedMoEMethod"
        for mod_name, module in self.model.named_modules():
            if not (hasattr(module, "w13_weight") and hasattr(module, "_num_local_routed")
                    and type(module.quant_method).__name__ == want_method):
                continue
            if "mtp" in mod_name:
                continue
            lid = module.layer_id
            E = module._num_local_routed
            E_global = module._num_global_routed
            if E_global % E:
                raise RuntimeError(f"ARC_HC_HUMMING_MOE: {E_global} routed experts do not split into {E} per rank")
            g0 = module._expert_storage_rank * E
            dev = module.w13_weight.device
            if dev.type != "cuda":
                raise RuntimeError(f"ARC_HC_HUMMING_MOE: {mod_name} expert placeholder is on {dev}")
            prefix = f"model.language_model.layers.{lid}.mlp.experts"
            if packed:
                inter, hidden = module.intermediate_size_per_partition, module.hidden_size
                hm_up = packed_layer(prefix, "w13", 2 * inter, hidden, E, g0, dev)
                hm_dn = packed_layer(prefix, "w2", hidden, inter, E, g0, dev)
                for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                    setattr(module, name, torch.nn.Parameter(torch.empty(16, dtype=torch.uint8, device=dev),
                                                             requires_grad=False))
                module._arc_hm = (hm_up, hm_dn)
                table[module.w13_weight.data_ptr()] = (hm_up, hm_dn, E, E_global)
                n_layers += 1
                last_moe = module
                torch.cuda.empty_cache()
                continue

            def deq(e, proj):
                base = f"{prefix}.{e}.{proj}"
                return _dequant_nvfp4(get(base + ".weight").to(dev), get(base + ".weight_scale").to(dev),
                                      get(base + ".weight_scale_2").to(dev))

            w13 = torch.stack([torch.cat([deq(g0 + e, "gate_proj"), deq(g0 + e, "up_proj")], 0)
                               for e in range(E)]).to(torch.bfloat16)
            hm_up = HummingLayer(shape_n=w13.shape[1], shape_k=w13.shape[2], weight_config=dict(fmt),
                                 input_config=dict(input_cfg) if input_cfg else None, num_experts=E,
                                 torch_dtype=torch.bfloat16)
            hm_up.load_from_unquantized(w13)
            del w13
            hm_up = hm_up.to(dev)
            hm_up.transform()
            w2 = torch.stack([deq(g0 + e, "down_proj") for e in range(E)]).to(torch.bfloat16)
            hm_dn = HummingLayer(shape_n=w2.shape[1], shape_k=w2.shape[2], weight_config=dict(fmt),
                                 input_config=dict(input_cfg) if input_cfg else None, num_experts=E,
                                 torch_dtype=torch.bfloat16)
            hm_dn.load_from_unquantized(w2)
            del w2
            hm_dn = hm_dn.to(dev)
            hm_dn.transform()
            for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                setattr(module, name, torch.nn.Parameter(torch.empty(16, dtype=torch.uint8, device=dev),
                                                         requires_grad=False))
            module._arc_hm = (hm_up, hm_dn)
            table[module.w13_weight.data_ptr()] = (hm_up, hm_dn, E, E_global)
            n_layers += 1
            last_moe = module
            torch.cuda.empty_cache()
        if n_layers == 0:
            raise RuntimeError(f"ARC_HC_HUMMING_MOE: no {want_method} FusedMoE layer found")
        if packed:
            carrier["check_loaded"](self.model, n_layers)
        freed = before - torch.cuda.memory_allocated()
        # compile every row bucket now (all layers share shapes, so one layer covers them); otherwise the first
        # prefill of a new size stalls serving on an NVRTC compile
        t0 = time.time()
        topk = last_moe.top_k
        E, dev, hm_up = table[last_moe.w13_weight.data_ptr()][2], last_moe.w13_weight.device, last_moe._arc_hm[0]
        with torch.no_grad():
            for b in (1 << i for i in range(4, 18)):
                m = max(1, b // topk)
                ids = torch.randint(0, E, (m, topk), device=dev, dtype=torch.int32)
                wts = torch.full((m, topk), 1.0 / topk, device=dev, dtype=torch.float32)
                x = torch.zeros((m, hm_up.shape_k), dtype=torch.bfloat16, device=dev)
                fmm.fused_marlin_moe(x, last_moe.w13_weight, last_moe.w2_weight, topk_ids=ids, topk_weights=wts)
                if fp8q and m <= int(os.environ.get("ARC_HC_FP8Q_ROWS", "1024")):
                    # the dispatch-payload path (prequantized group-128 input) is a separate compile
                    stash = _fp8q_stash(dev, x.numel())
                    stash["q"][:x.numel()].zero_()
                    stash["s"][:x.numel() // 128].fill_(1.0)
                    stash["last"] = (x.data_ptr(), x.numel())
                    fmm.fused_marlin_moe(x, last_moe.w13_weight, last_moe.w2_weight, topk_ids=ids, topk_weights=wts)
                    stash["last"] = None
            if fp8q:
                stash = _STATE["fp8q"]
                prewarm_hits = stash["hits"]
                stash["hits"] = stash["misses"] = 0
                logger.info("ARC_HC_FP8Q installed rank=%d up_input_group=128 rows_cap=%d prewarm_hits=%d",
                            torch.distributed.get_rank(), stash["cap"] // hm_up.shape_k, prewarm_hits)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        handles.clear()
        import ctypes, gc
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        # the expert reads leave active page cache, which HiSparse's host admission counts as used memory
        for f in sorted(set(index.values())):
            fd = os.open(str(model_path / f), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        logger.info("ARC_HC_HUMMING_MOE %s tile=%s act=%s layers=%d cuda_bytes_freed=%d prewarm_s=%.1f", spec, tile, act,
                    n_layers, freed, time.time() - t0)
        return result

    ModelRunner.load_model = load_and_humming_moe


_INT3_EXPERT = None  # compiled lazily: ^model.language_model.layers.<i>.mlp.experts.<w13|w2>_<qweight|scales>$


def _int3_runtime_prefix(name):
    """HF module name -> Qwen4-Exp runtime prefix, fused dense linears included (as work/qwen38fn_eval/
    int3_deploy_runtime.runtime_prefix, the G4 loader of the same checkpoint)."""
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


def _install_int3_packed_carrier():
    """ARC_HC_HUMMING_MOE=int3packed: let sglang build and load the packed INT3 checkpoint under EP.

    The checkpoint's quantization_config is modelopt_mixed whose quantized_layers name only the PLE table; the
    experts (arc_int3.main_expert_modules) and the dense FP8 / BF16 modules are listed in arc_int3, which
    sglang does not read. Here (same contract as the G4 loader, work/qwen38fn_eval/int3_deploy_runtime.py):
      experts  -> _Int3PackedPlaceholderMoE: 16-byte placeholders on sglang's Marlin path, which
                  install_humming_moe's fused_marlin_moe intercepts (EP id mapping and dispatch unchanged)
      FP8      -> block-FP8 Fp8LinearMethod (sm_89: sglang's Triton kernel, checked here)
      BF16     -> UnquantizedLinearMethod
    and Qwen4ExpForConditionalGeneration.load_weights drops the 192 packed expert tensors (each rank reads its own
    rows after the load). A config without arc_int3 (the drafter) is left alone; the target must carry it.
    Returns {"check_loaded": fn(model, n_layers)} for after the load.
    """
    global _INT3_EXPERT
    import re
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.moe_runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    _INT3_EXPERT = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(w13_qweight|w13_scales|w2_qweight|w2_scales)$")
    contract = {"schema_version": 1, "bits": 3, "group_size": 128, "num_experts": 320, "scale_dtype": "bfloat16",
                "layout": "humming_uint3_g128_nozp", "w13_rows": "gate_then_up"}
    state = {"meta": None, "fp8": set(), "bf16": set(), "experts": set()}

    class _Int3PackedPlaceholderMoE(FusedMoEMethodBase):
        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                layer.register_parameter(name, torch.nn.Parameter(torch.empty(16, dtype=torch.uint8),
                                                                  requires_grad=False))

        def process_weights_after_loading(self, layer):
            pass

        def create_moe_runner(self, layer, moe_runner_config):
            self.moe_runner_config = moe_runner_config
            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

        def apply(self, layer, dispatch_output):
            # as ModelOptNvFp4FusedMoEMethod.get_marlin_quant_info; the tensors are install_humming_moe's
            # placeholders (its table key is w13_weight's data_ptr), never read by a Marlin kernel
            expert_map = getattr(layer.dispatcher, "local_expert_mapping", None)
            quant_info = MarlinMoeQuantInfo(
                w13_qweight=layer.w13_weight, w2_qweight=layer.w2_weight, w13_scales=layer.w13_weight_scale,
                w2_scales=layer.w2_weight_scale, w13_g_idx_sort_indices=None, w2_g_idx_sort_indices=None,
                weight_bits=4, expert_map=expert_map,
                global_num_experts=self.moe_runner_config.num_experts if expert_map is not None else -1)
            return self.runner.run(dispatch_output, quant_info)

    def fp8_method():
        m = Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                                      weight_block_size=[128, 128]))
        if not m.block_quant or m.use_marlin or m.use_mxfp8:
            raise RuntimeError(f"int3packed: dense FP8 is not plain block FP8 (marlin={m.use_marlin})")
        if m.w8a8_block_fp8_linear is not fp8_utils.triton_w8a8_block_fp8_linear:
            raise RuntimeError(f"int3packed: dense FP8 dispatch is {m.w8a8_block_fp8_linear.__name__}, expected Triton")
        return m

    original_from = ModelOptMixedPrecisionConfig.from_config.__func__
    original_method = ModelOptMixedPrecisionConfig.get_quant_method
    original_load = Qwen4ExpForConditionalGeneration.load_weights

    def from_config(cls, config):
        result = original_from(cls, config)
        meta = config.get("arc_int3")
        if meta is not None:
            got = {k: meta[k] for k in contract}
            if got != contract:
                raise ValueError(f"int3packed: arc_int3 contract {got}, expected {contract}")
            experts = {_int3_runtime_prefix(n) for n in meta["main_expert_modules"]}
            fp8 = {_int3_runtime_prefix(n) for n in meta["dense_fp8_modules"]}
            bf16 = {_int3_runtime_prefix(n) for n in meta["dense_bf16_modules"]}
            if len(experts) != 48 or fp8 & bf16 or (fp8 | bf16) & experts:
                raise ValueError(f"int3packed: {len(experts)} expert banks, fp8&bf16={sorted(fp8 & bf16)[:3]}")
            result._arc_int3 = {"experts": experts, "fp8": fp8, "bf16": bf16}
            state["meta"] = result._arc_int3
        return result

    def get_quant_method(self, layer, prefix):
        meta = getattr(self, "_arc_int3", None)
        if meta is None:
            return original_method(self, layer, prefix)
        name = _int3_runtime_prefix(prefix)
        if name in meta["experts"]:
            if not isinstance(layer, FusedMoE):
                raise TypeError(f"int3packed: expert prefix {prefix} is {type(layer).__name__}, not FusedMoE")
            if layer.moe_tp_size != 1 or layer.num_fused_shared_experts != 0 or layer.top_k != 10:
                raise ValueError(f"int3packed: {prefix} moe_tp={layer.moe_tp_size} fused_shared="
                                 f"{layer.num_fused_shared_experts} top_k={layer.top_k}; needs EP-only, unfused, top-10")
            state["experts"].add(name)
            return _Int3PackedPlaceholderMoE()
        if name in meta["fp8"] or name in meta["bf16"]:
            if not isinstance(layer, LinearBase):
                raise TypeError(f"int3packed: dense prefix {prefix} is {type(layer).__name__}, not LinearBase")
            if name in meta["bf16"]:
                state["bf16"].add(name)
                return UnquantizedLinearMethod()
            state["fp8"].add(name)
            return fp8_method()
        return original_method(self, layer, prefix)

    def load_weights(self, weights):
        if getattr(self.quant_config, "_arc_int3", None) is None:
            return original_load(self, weights)
        dropped = set()

        def without_experts():
            for name, tensor in weights:
                if _INT3_EXPERT.fullmatch(name):
                    if name in dropped:
                        raise RuntimeError(f"int3packed: {name} yielded twice")
                    dropped.add(name)
                    continue
                yield name, tensor

        result = original_load(self, without_experts())
        if len(dropped) != 4 * 48:
            raise RuntimeError(f"int3packed: {len(dropped)} packed expert tensors in the load, expected 192")
        return result

    def check_loaded(model, n_layers):
        meta = state["meta"]
        if meta is None:
            raise RuntimeError("int3packed: the target's quantization_config carries no arc_int3 (not the packed "
                               "INT3 checkpoint)")
        missing = {k: sorted(meta[k] - state[k])[:3] for k in ("experts", "fp8", "bf16") if meta[k] - state[k]}
        if missing or n_layers != len(meta["experts"]):
            raise RuntimeError(f"int3packed: modules without their method {missing}; Humming banks {n_layers}")
        fp8 = [m for m in model.modules() if isinstance(getattr(m, "quant_method", None), Fp8LinearMethod)]
        bad = [type(m).__name__ for m in fp8 if getattr(m, "weight", None) is None
               or m.weight.dtype != torch.float8_e4m3fn or getattr(m, "weight_scale_inv", None) is None]
        if len(fp8) != len(meta["fp8"]) or bad:
            raise RuntimeError(f"int3packed: {len(fp8)} FP8 linears loaded (expected {len(meta['fp8'])}), bad {bad[:3]}")
        logger.info("ARC_HC_INT3_PACKED experts=%d fp8_linears=%d bf16_linears=%d", n_layers, len(fp8),
                    len(state["bf16"]))

    ModelOptMixedPrecisionConfig.from_config = classmethod(from_config)
    ModelOptMixedPrecisionConfig.get_quant_method = get_quant_method
    Qwen4ExpForConditionalGeneration.load_weights = load_weights
    return {"check_loaded": check_loaded}
