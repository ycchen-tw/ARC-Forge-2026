"""Materialize a fresh, bounded stream run (rollout replicas + trainer GPUs) and a matched baseline config.

Topologies: tp2x2 = two TP2 replicas on GPUs 0-1 / 2-3 (ports 30571-30572, the FP8 line), tp1x4 = four TP1 replicas on
GPUs 0-3 (ports 30571-30574; the gEAN320 INT3 W3A8 deployment runs on one card) -- both train on GPUs 4-7; tp1x6 = six
TP1 replicas on GPUs 0-5 (ports 30571-30576) and a 2-rank trainer on GPUs 6-7; tp1x5 = five TP1 replicas on GPUs 0-4
(ports 30571-30575) and a 3-rank trainer on GPUs 5-7; tp1x3 = three TP1 replicas on GPUs 0-2 (ports 30571-30573) and a
3-rank trainer on GPUs 3-5; tp1x2 = half the machine.
launch_b200_stream.sh reads the topology back from plan.json.

Datasets: training1000 / eval120 = ARC-AGI-2 (held-out = a seeded draw from training1000); arc3_levels = ARC-AGI-3 single-level
episodes (work/arc_code/run_arc3_level.py): the task dir holds the stubs of work/arc_code/arc3_levels.py, train and held-out
come from --train-ids-file / --heldout-ids-file, and the template's harness config must be kind arc3_level; arc2_dir = ARC-AGI-2
tasks from any directory (--tasks-dir, e.g. $ARC_DATA/arc2_pools/largegrid30-20261022/tasks; --train-ids-file picks a subset) with
the held-out from --heldout-dir (default the public evaluation 120) / --heldout-ids-file, so a pool that has no evaluation
parent keeps the eval set clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: harness config kind of work/arc_code/run_arc3_level.py (orchestrator.ARC3_LEVEL_KIND)
ARC3_LEVEL_KIND = "arc3_level"
#: whole-game episodes (RL v4): the same run_arc3_level.py with game stubs; prepare trains every context window
#: (prepare.allow_compaction)
ARC3_GAME_KIND = "arc3_game"
#: gen-2 single-level episodes (10-26): run_arc3_level.py with arc3_levels.py g2seeds stubs; the harness scores
#: R = 1[cleared] x min(1, (baseline / actions)^2) itself, one context window per episode
ARC3_G2_KIND = "arc3_g2seed"
ARC3_KINDS = (ARC3_LEVEL_KIND, ARC3_GAME_KIND, ARC3_G2_KIND)
#: run_arm's GEN2_LATEST_BRIEF (tests compare the two copies): the phrase the arm config's prompts brief must contain when
#: packet_level_docs is latest
GEN2_LATEST_BRIEF = "the names of the older ones"
TOPOLOGIES = {
    "tp2x2": {"tp": 2, "replica_gpus": ["0,1", "2,3"], "replica_ports": [30571, 30572], "default_sessions": 64,
              "trainer_gpus": "4,5,6,7", "trainer_world": 4},
    # sessions: 32 per TP1 card = the per-GPU batch of the tp2x2 line (64 per TP2 replica); one card holds 816 mamba slots and 3.8M fp8-KV tokens at MEMFRAC 0.75
    "tp1x4": {"tp": 1, "replica_gpus": ["0", "1", "2", "3"], "replica_ports": [30571, 30572, 30573, 30574],
              "default_sessions": 32, "trainer_gpus": "4,5,6,7", "trainer_world": 4},
    # half the machine (GPUs 0-3): two TP1 replicas on 0-1 and a 2-rank trainer (FSDP over 2) on 2-3 -- for toys and A/Bs
    # while the other half is busy; rollout supply and trainer throughput both halve
    "tp1x2": {"tp": 1, "replica_gpus": ["0", "1"], "replica_ports": [30571, 30572], "default_sessions": 32,
              "trainer_gpus": "2,3", "trainer_world": 2},
    # rollout-heavy (10-15): the eval28 run's trainer was busy 34% of the wall on tp1x4; six TP1 replicas and a 2-rank
    # trainer (FSDP over 2) on GPUs 6-7
    # balanced (10-15): six replicas generated ~20k tok/s while the 2-rank trainer took ~11k (25 min per 256-episode
    # step); five replicas and a 3-rank trainer on GPUs 5-7 come out at ~17k / ~16.5k
    "tp1x5": {"tp": 1, "replica_gpus": ["0", "1", "2", "3", "4"], "replica_ports": [30571, 30572, 30573, 30574, 30575],
              "default_sessions": 32, "trainer_gpus": "5,6,7", "trainer_world": 3},
    # six cards (10-21, node1 v3.1, the other two busy): three TP1 replicas on GPUs 0-2 and a 3-rank trainer on GPUs 3-5
    "tp1x3": {"tp": 1, "replica_gpus": ["0", "1", "2"], "replica_ports": [30571, 30572, 30573], "default_sessions": 32,
              "trainer_gpus": "3,4,5", "trainer_world": 3},
    # one replica and a 1-rank trainer (10-25 toy next to a 6-card run): with --gpus 6,7 and its own --replica-port-base /
    # --control-port / --trainer-port so the two stacks share no port
    "tp1x1": {"tp": 1, "replica_gpus": ["0"], "replica_ports": [30571], "default_sessions": 32,
              "trainer_gpus": "1", "trainer_world": 1},
    # seven cards (10-30 arc2-hq741, GPU 7 kept for the held-out evaluator): four TP1 replicas on GPUs 0-3 and a 3-rank
    # trainer on 4-6; arc2 r3/r4 (tp1x3) had the trainer busy 60-66% of the wall, so the extra card goes to rollout
    "tp1x4t3": {"tp": 1, "replica_gpus": ["0", "1", "2", "3"], "replica_ports": [30571, 30572, 30573, 30574],
                "default_sessions": 32, "trainer_gpus": "4,5,6", "trainer_world": 3},
    # seven cards, trainer-heavy (10-31 arc2-gdv3, the user): three TP1 replicas on GPUs 0-2 (128 sessions each) and a
    # 4-rank trainer on 3-6; GPU 7 kept for the held-out evaluator
    "tp1x3t4": {"tp": 1, "replica_gpus": ["0", "1", "2"], "replica_ports": [30571, 30572, 30573],
                "default_sessions": 32, "trainer_gpus": "3,4,5,6", "trainer_world": 4},
    "tp1x6": {"tp": 1, "replica_gpus": ["0", "1", "2", "3", "4", "5"],
              "replica_ports": [30571, 30572, 30573, 30574, 30575, 30576], "default_sessions": 32,
              "trainer_gpus": "6,7", "trainer_world": 2},
}
#: --trainer-fast-kernels: the kernels of work/qwen38fn_lora/trainer_kernels.py (KERNELS there; the trainer re-validates)
FAST_KERNELS = {"hc", "hc_inject", "lora", "lora_grad", "gdn_norm", "act_quant", "qsa_select", "qsa_dkdv", "gdn_gva",
                "qsa_union"}
NAMED_ONLY_KERNELS = {"qsa_union"}   # not in the trainer's "all": name it ("all,qsa_union"); it needs qsa_dkdv
#: --serve-profile: serving env/args added to every replica on top of the launcher's base flags, and the fork lock target
#: whose tree has the code. plan.json records the resolved profile; launch_b200_stream.sh serves exactly that and checks it.
#: perf = the B200 perf opt-ins of fork exp/rl-int3-perf (decode, drafter, prefill graph, QSA prefill), all default off in
#: the tree: +12.6% gen tok/s on the RL load (A3 vs A0, 2 rounds); INT3 TP1 correctness gates in
#: $ARC_RUNS/qwen38fn_rl/20261011-int3-perf-gates. Validated on INT3 TP1 only.
#: --sampling-mask-max-tokens of the rollout servers when the rollouts truncate (launch_b200_stream.sh serves the same
#: value): the packed width of a recorded support; top-k must fit under it (flashinfer's top-k keeps exactly k)
SAMPLING_MASK_MAX_TOKENS = 32
SERVE_PROFILES = {
    "base": None,
    "perf": {
        "model_kind": "int3", "tp": 1, "lock_target": "b200-rl-int3-perf",
        "env": ["ARC_RS_FUSED_SOFTMAX=1", "ARC_DFLASH_SPARSE_Q=1", "ARC_QSA_FP8_DIRECT=1", "ARC_GDN_FOLD_IN_VERIFY=1",
                "ARC_GDN_FOLD_COMMIT_COMPACT=1", "ARC_HC_GATE_PRECOMPUTE=1", "ARC_DFLASH_RING_OUT_T=1",
                "ARC_DFLASH_RING_TILE=auto", "ARC_DFLASH_BATCHED_COMMIT=1", "ARC_DFLASH_FUSED_PROPOSE=1",
                "ARC_DFLASH_FAST_TOPK=1", "ARC_DFLASH_GRAPH_SIZES=1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96",
                "ARC_QWEN4_EXP_PREFILL_BCG=1", "ARC_MAMBA_TRACK_HOST=1", "ARC_MROPE_EXTEND_PINNED=1",
                "ARC_QSA_INDEXER_LOCAL=1", "ARC_QSA_UNION_PREFILL=1"],
        # GDN_FOLD_IN_VERIFY keeps the verify window's state in the extra mamba buffer
        "args": ["--mamba-radix-cache-strategy", "extra_buffer"],
    },
}
#: perf-main-fp16 (10-15) = the perf opt-ins on fork main (lock target main = ed837597; all-merge, which contains the perf
#: tree and was A/B'd on the B200 RL load: +0.3% gen tok/s, ledger compare passed) with an fp16 SSM state. FlashInfer GDN
#: decode/verify on SM100 takes bf16 state only, so linear-attention decode goes to triton (verify follows it).
SERVE_PROFILES["perf-main-fp16"] = {
    **SERVE_PROFILES["perf"], "lock_target": "main",
    "env": SERVE_PROFILES["perf"]["env"] + ["SSM_DTYPE=float16", "LINEAR_DECODE_BACKEND=triton"],
}
#: perf-main-fp16-ser (10-18) = perf-main-fp16 + session exact resume (every position forwarded once per rollout) +
#: own-last-turn replay (the session's last generated message replays as its sampled ids), on fork main f31bd1bcf
#: (lock target b200-rl-ser). The two go together: exact resume alone turns codex's lossy re-render of the last turn
#: into rewinds whose D_cross episodes prepare_rl drops (8/96 > the 2% gate on the f7 load); both on: 0
#: (docs/plans/session_exact_resume.md, section "how to enable"; replay_own_last_turn.md, section "GPU 7"). The drafter park
#: needs EXACT_PREFILL.
#: A session holds about 5 mamba slots under spec, the --mamba-slots floor below (v3 c64 on lazyfix, 10-20: 448 slots
#: for 64 sessions peaked at 0.71 = 318, 5.0 per session); the default is 6 x sessions (10-28: auto took ~800-975 slots).
SERVE_PROFILES["perf-main-fp16-ser"] = {
    **SERVE_PROFILES["perf-main-fp16"], "lock_target": "b200-rl-ser",
    "env": SERVE_PROFILES["perf-main-fp16"]["env"] + ["ARC_DFLASH_EXACT_PREFILL=1", "ARC_DFLASH_SESSION_PARK=1"],
    "args": SERVE_PROFILES["perf-main-fp16"]["args"] + ["--session-exact-resume", "--session-replay-own-last-turn"],
}
#: perf-main-fp16-ser-opt (10-20, the node1 kverify session: work/qwen38fn_kverify/allopt, arm allopt-nolazy) =
#: perf-main-fp16-ser on fork exp/kverify-dev 3140fd578 (lock target b200-rl-ser-opt = f31bd1bcf + 3 kverify commits, QSA
#: unpad) with the B200 Humming schedule table and the Kaggle media flags RELEASE_AFTER_PREFILL / NO_SYNC. Not LAZY_MEDIA:
#: its rematerialize resend skips the ledger intake (appends without logprobs, exact-resume contract broken). The dense
#: CUTLASS FP8 GEMM goes with it but is a server flag, not profile env: launch with RL_FP8_GEMM=flashinfer_cutlass (the
#: round-2 launcher sets it). On the arc3 RL load at conc 96: decode x1.15, generation x1.08-1.17, AL unchanged; verify
#: parity, weight sync (mid-decode in place too) and gate 5 (23/23, ledger_compare) passed.
SERVE_PROFILES["perf-main-fp16-ser-opt"] = {
    **SERVE_PROFILES["perf-main-fp16-ser"], "lock_target": "b200-rl-ser-opt",
    "env": SERVE_PROFILES["perf-main-fp16-ser"]["env"] + [
        f"ARC_INT3_TUNING={ROOT / 'work' / 'qwen38fn_eval' / 'int3_tuning' / 'b200-w3a8-k456-humming0112.json'}",
        "ARC_MM_RELEASE_AFTER_PREFILL=1", "ARC_MM_NO_SYNC=1"],
}
#: perf-main-fp16-ser-opt-lazyfix (10-20, node1 kverify handover) = perf-main-fp16-ser-opt + ARC_MM_LAZY_MEDIA on fork
#: exp/kverify-dev-lazyfix a6c7fddbb (lock target b200-rl-ser-opt-lazyfix = 3140fd578 + c1e702334, a6c7fddbb): the lazy
#: media park keeps the ledger intake, forced logprobs / routing and the exact-resume contract. Soak on one B200 (128
#: sessions, 90 min, 4 in-place syncs): 29,924 appends all with logprobs / routing (1,201 missing before), 0 window
#: violations (349 before). No retract was hit there: a retract in a run's server.log means checking those requests' ledger.
SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix"] = {
    **SERVE_PROFILES["perf-main-fp16-ser-opt"], "lock_target": "b200-rl-ser-opt-lazyfix",
    "env": SERVE_PROFILES["perf-main-fp16-ser-opt"]["env"] + ["ARC_MM_LAZY_MEDIA=1"],
}
#: perf-main-fp16-ser-opt-lazyfix-l8 (10-21, v3.1) = perf-main-fp16-ser-opt-lazyfix on fork main a5c67c41c (all-merge-1020,
#: lock target b200-rl-main-1020: a6c7fddbb + plan B drafter serving + the B200 FP8 drafter + plan B opt-ins, all opt-in)
#: with the 8-layer plan B drafter's FP8 deploy and the opt-ins that go with it. kverify replay on B200 INT3 TP1 K4 (step
#: 25562, $ARC_RUNS/qwen38fn_planb_serve/am-b200-s25562): AL +1.6% [+0.7, +2.5] vs D-line ext9425, draft 3.02 -> 2.36 ms,
#: decode x1.03 at bs 48-64. "drafter" = what --dflash-dir must hold (plain residual, injected context, FP8 linears).
SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8"] = {
    **SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix"], "lock_target": "b200-rl-main-1020",
    "env": SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix"]["env"] + [
        "ARC_DFLASH_FP8_GEMM=deep_gemm", "ARC_DFLASH_FP8_OP=1", "ARC_DFLASH_MERGE=1", "ARC_DFLASH_FUSE_ONCE=1",
        "ARC_DFLASH_INJECT_KV=gemm"],
    "drafter": {"residual": "plain", "context": "inject", "ffn_type": "dense", "fp8": True},
}
#: perf-main-fp16-ser-opt-lazyfix-l8-a16 (10-21) = perf-main-fp16-ser-opt-lazyfix-l8 with the routed experts on BF16
#: activations (W3A16) on Humming 0.1.18 (HUMMING_VERSION: serve_b200.sh lays $ARC_DATA/humming-overlays/0.1.18, built by
#: work/sglang_fork/humming_overlay.py) and fork main c61e351b7 (lock target b200-rl-main-1021 = a5c67c41c + w2 aligned to
#: its own block M, which 0.1.18's prefill defaults need). No ARC_INT3_TUNING: that table is W3A8 on 0.1.12, and a 0.1.18
#: bf16 sweep found no faster decode schedule. kverify replay on B200 INT3 TP1 K4: decode x1.17-1.18 at bs 48-64, wall -28%,
#: AL unchanged (docs/research/humming_0118_b200_int3_20261021.md). The trainer follows the serving activations:
#: launch_b200_stream.sh reads RL_INT3_ACT from this env and runs the trainer with --int3-act to match.
SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16"] = {
    **SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8"], "lock_target": "b200-rl-main-1021",
    "env": [e for e in SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8"]["env"] if not e.startswith("ARC_INT3_TUNING=")]
           + ["HUMMING_VERSION=0.1.18", "RL_INT3_ACT=bf16"],
}
# the same serving on the fork with per-token sampling supports (rl-topp-mask bdf43e509: --session-ledger-sampling-mask,
# ARC_RESPONSES_DEFAULT_TOP_P/_TOP_K) for top-k / top-p rollouts (--rollout-top-p / --rollout-top-k)
SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp"] = {
    **SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16"], "lock_target": "b200-rl-topp-1024",
}
# + inline developer messages (10-26, arc3 RL v5: the gen-2 harness g2b): ARC_RESPONSES_INLINE_DEVELOPER renders a
# mid-conversation developer message (the g2 hooks' "window full" and friends) where it stands instead of moving it to the
# system prompt, which would rewrite the prefix of every later request. Fork rl-topp-mask f9cf529b0 (= bdf43e509 + the
# inline-developer commit 45c4dc2b1), lock target b200-rl-topp-inldev-1026; the seed collection served g2 the same way
SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp-inldev"] = {
    **SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp"], "lock_target": "b200-rl-topp-inldev-1026",
    "env": SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp"]["env"] + ["ARC_RESPONSES_INLINE_DEVELOPER=1"],
}
# + NVFP4 target KV (10-29, arc3 RL v7b, user: "when I tell you to use nv4 kv, use it"): QSA reads the packed FP4 pool through
# the sm_100 trtllm-gen path the fork already has (_trtllm_sparse_nvfp4_sm100, patch 0035's layout); prefill expands to FP8 as
# serve_qwen38fn_nvfp4_0520.sh does; the drafter KV stays as before. B200 09-17: slots +63.6%, decode / PPL within noise
# (docs/research/qwen38fn_qsa_nvfp4_b200.md). The pool goes to more sessions per replica, not to a longer context.
# No ARC_QSA_UNION_PREFILL: it reads FP8/BF16 keys straight from the pool and raises on the NVFP4 pool at the first prefix
# chunk (10-29 v7b launch: all four replicas died on "ARC_QSA_UNION_PREFILL covers the plain FP8/BF16 KV pool without
# QSA HiSparse"); the prefix chunk prefill takes the FP8 expansion instead.
SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp-inldev-nvfp4kv"] = {
    **SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp-inldev"],
    "env": [e for e in SERVE_PROFILES["perf-main-fp16-ser-opt-lazyfix-l8-a16-topp-inldev"]["env"]
            if not e.startswith("ARC_QSA_UNION_PREFILL=")]
           + ["ARC_QSA_NVFP4=1", "ARC_QSA_NVFP4_PREFILL=fp8", "KV_CACHE_DTYPE=nvfp4"],
}


def check_drafter(directory: Path, want: dict | None) -> None:
    """--dflash-dir: a complete dense drafter deploy directory; with ``want`` (a profile's "drafter"), its config must be that
    kind -- the profile's opt-ins only serve that drafter. Raises ValueError."""
    config = directory / "config.json"
    if not config.is_file():
        raise ValueError(f"{directory}: no config.json (not a drafter deploy directory)")
    c = json.loads(config.read_text())
    if not json.loads((directory / "initialization_manifest.json").read_text()).get("complete"):
        raise ValueError(f"{directory}: initialization_manifest.json does not say complete")
    if c.get("ffn_type") not in ("dense", "dense_shared"):
        raise ValueError(f"{directory}: ffn_type {c.get('ffn_type')!r} is not a dense drafter")
    if want is None:
        return
    have = {"residual": c.get("residual"), "context": c.get("context"), "ffn_type": c.get("ffn_type"),
            "fp8": bool((c.get("quantization") or {}).get("modules"))}
    if have != want:
        raise ValueError(f"{directory}: drafter {have}, the serve profile needs {want}")


def profile_env(profile: dict, sessions: int) -> list[str]:
    """The profile's serving env for replicas of ``sessions`` running requests: the drafter's CUDA graph sizes above it
    are dropped (the server refuses sizes past max_running_requests, the perf list goes up to 96)."""
    env = []
    for item in profile["env"]:
        key, value = item.split("=", 1)
        if key == "ARC_DFLASH_GRAPH_SIZES":
            sizes = [size for size in value.split(",") if int(size) <= sessions]
            if not sizes:
                raise RuntimeError(f"{item}: no size fits {sessions} sessions per replica")
            item = f"{key}={','.join(sizes)}"
        env.append(item)
    return env


def config_fingerprint(cfg: dict) -> str:
    """Hash all config semantics except the device selector bound to a UUID at launch."""
    command = list(cfg["prepare"]["command"])
    if not command[2].startswith("CUDA_VISIBLE_DEVICES="):
        raise RuntimeError("prepare.command[2] is not the expected CUDA device selector")
    command[2] = "CUDA_VISIBLE_DEVICES=<launch-bound-device>"
    normalized = dict(cfg, prepare=dict(cfg["prepare"], command=command))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def init_policy(checkpoint: Path, merged: Path, trainer_base: Path | None) -> dict:
    """--init-checkpoint / --init-merged: a trainer checkpoint-<N> and the merged/v<N> export of the same run and step
    (merged_manifest.json names the run's trainer directory, the step and the base). Raises on any mismatch."""
    checkpoint, merged = checkpoint.resolve(), merged.resolve()
    name = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if name is None or not (checkpoint / "trainer_state.pt").is_file() or not (checkpoint / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"{checkpoint}: not a trainer checkpoint-<N> with trainer_state.pt and adapter_model.safetensors")
    step = int(name.group(1))
    if not (merged / "DONE").is_file():
        raise RuntimeError(f"{merged}: no DONE marker (an unfinished merged export)")
    manifest = json.loads((merged / "merged_manifest.json").read_text())
    if manifest["format"] != "fn-lora-merged-v2" or manifest["step"] != step or str(manifest["weight_version"]) != str(step):
        raise RuntimeError(f"{merged}: format {manifest['format']!r}, step {manifest['step']!r}, weight_version "
                           f"{manifest['weight_version']!r}; want fn-lora-merged-v2 at step {step} ({checkpoint.name})")
    if Path(manifest["run"]).resolve() != checkpoint.parent:
        raise RuntimeError(f"{merged}: exported by {manifest['run']}, the checkpoint is in {checkpoint.parent}")
    if trainer_base is not None and Path(manifest["base_model"]).resolve() != trainer_base.resolve():
        raise RuntimeError(f"{merged}: base {manifest['base_model']}, this run's trainer base is {trainer_base}")
    return {"checkpoint": str(checkpoint), "merged": str(merged), "step": step, "source_run": str(checkpoint.parent.parent)}


def arc3_note_module():
    """arcforge.arc3_note by file path (stdlib only): the launcher runs this with the system python3, without src on its path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("arc3_note", Path(__file__).resolve().parents[2] / "src" / "arcforge" /
                                                  "arc3_note.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def claude_judge_pin(ap: argparse.ArgumentParser, claude_bin: Path) -> dict:
    """The v8b Claude Code judge binary (ensemble, claude-code), pinned at configure: resolved (readlink -f) and its
    ``--version``. It runs on the user's subscription login (user 11-01), so ANTHROPIC_API_KEY must not be set (claude
    would bill the API).
    -> note_judge.claude_bin / claude_version."""
    import subprocess
    exe = claude_bin.resolve()
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        ap.error(f"--note-judge-claude-bin {claude_bin} ({exe}) is not an executable file")
    if os.environ.get("ANTHROPIC_API_KEY"):
        ap.error("ANTHROPIC_API_KEY is set: the claude judge runs on the Claude Code subscription login, unset it")
    run = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
    if run.returncode != 0 or not run.stdout.strip():
        ap.error(f"`{exe} --version` exited {run.returncode}: {(run.stderr or run.stdout)[-500:]}")
    return {"claude_bin": str(exe), "claude_version": run.stdout.strip()}


def codex_judge_pin(ap: argparse.ArgumentParser, codex_bin: Path) -> dict:
    """The codex judge's binary, pinned at configure: codex_bin resolved (readlink -f) and its ``--version``. The judge
    runs it with the default CODEX_HOME (~/.codex, the user's login; ~/.codex/auth.json must exist -- only checked, never
    read) and --ignore-user-config. -> note_judge.codex_bin / codex_version."""
    import subprocess
    exe = codex_bin.resolve()
    if not (exe.is_file() and os.access(exe, os.X_OK)):
        ap.error(f"--note-judge-codex-bin {codex_bin} ({exe}) is not an executable file")
    auth = Path.home() / ".codex" / "auth.json"
    if not auth.is_file():
        ap.error(f"the codex judge needs the ChatGPT login {auth} (codex login)")
    run = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
    if run.returncode != 0 or not run.stdout.strip():
        ap.error(f"`{exe} --version` exited {run.returncode}: {(run.stderr or run.stdout)[-500:]}")
    return {"codex_bin": str(exe), "codex_version": run.stdout.strip()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--dataset", choices=("training1000", "eval120", "arc3_levels", "arc2_dir"), default="training1000")
    ap.add_argument("--train-ids-file", type=Path, default=None,
                    help="train on only these task ids (one per line), a subset of the dataset's train split; default: all "
                         "(arc3_levels: required)")
    ap.add_argument("--tasks-dir", type=Path, default=None,
                    help="arc3_levels: the stub directory (work/arc_code/arc3_levels.py stub --out); arc2_dir: the ARC-AGI-2 "
                         "task directory to train on")
    ap.add_argument("--heldout-ids-file", type=Path, default=None,
                    help="arc3_levels / arc2_dir only: the held-out task ids (one per line; arc3_levels: stubs in --tasks-dir, "
                         "arc2_dir: tasks in --heldout-dir); replaces --heldout")
    ap.add_argument("--heldout-dir", type=Path, default=None,
                    help="arc2_dir only: where the held-out tasks are (default $ARC_DATA/ARC-AGI-2/data/evaluation)")
    steps = ap.add_mutually_exclusive_group()
    steps.add_argument("--optimizer-steps", dest="optimizer_steps", type=int, default=400,
                       help="optimizer updates; each update publishes one rollout weight version (default: 400)")
    steps.add_argument("--versions", dest="optimizer_steps", type=int,
                       help="compatibility alias for --optimizer-steps; one version equals one optimizer update")
    ap.add_argument("--global-optimizer-batch", type=int, default=32,
                    help="target live episodes before eligibility filtering per global optimizer update (default: 32)")
    ap.add_argument("--save-every-steps", type=int, default=20)
    ap.add_argument("--eval-every-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--heldout", type=int, default=64)
    ap.add_argument("--fork", type=Path, required=True)
    ap.add_argument("--prepare-media-device", choices=("cpu-exact", "cuda"), default="cpu-exact",
                    help="prepare_rl's image processing: cpu-exact (default) uses no GPU and reproduces the deployed CUDA "
                         "bicubic bit for bit (cuda_bicubic_aa.py); cuda runs it on --prepare-gpu. 10-21 a16t08 (node1): "
                         "cuda on the last trainer card wedged that rank's FSDP all-gather three times")
    ap.add_argument("--prepare-gpu", default=None,
                    help="--prepare-media-device cuda only (then required): physical index; UUID resolved by launch preflight")
    ap.add_argument("--gpus", default=None,
                    help="physical GPUs the topology's GPUs 0..N-1 map to, comma separated (default: identity); e.g. "
                         "--topology tp1x2 --gpus 4,5,6,7 runs on the upper half of a shared machine")
    ap.add_argument("--replica-port-base", type=int, default=None,
                    help="replica i serves on this port + i instead of the topology's 30571.. (a second set of servers on "
                         "the other half of the machine, e.g. a prescreen next to a run); must keep clear of the control "
                         "proxy and the trainer ports. Default: the topology's ports")
    ap.add_argument("--control-port", type=int, default=30570,
                    help="port of the replica control proxy (rollout.base_url; plan.json control_port). Default: 30570")
    ap.add_argument("--trainer-port", type=int, default=30591,
                    help="port of the trainer service (plan.json trainer_port). Default: 30591")
    ap.add_argument("--topology", choices=sorted(TOPOLOGIES), default="tp2x2",
                    help="rollout replica layout and trainer GPUs: tp2x2 / tp1x4 with the trainer on GPUs 4-7, tp1x2 = two "
                         "TP1 replicas on GPUs 0-1 and a 2-rank trainer on GPUs 2-3 (default: tp2x2)")
    ap.add_argument("--template", type=Path, default=ROOT / "work/qwen38fn_rl/configs/eval120-b32k-stream-dflash.json",
                    help="stream config the run config is derived from (harness, audit, prepare settings)")
    ap.add_argument("--tokenizer-dir", type=Path, default=None,
                    help="tokenizer/config directory (prepare_rl reads num_experts from its config.json); default: the template's")
    ap.add_argument("--trainer-base-path", type=Path, default=None,
                    help="declare the trainer's base model when it is not the directory the servers load (QAT: BF16 master vs "
                         "deployed package); written to rollout.trainer_base_path")
    ap.add_argument("--sessions-per-replica", type=int, default=None,
                    help="rollout sessions (MAXREQ) per replica (default: 64 for tp2x2, 32 for tp1x4); a multiple of the "
                         "group size, groups are split evenly over the replicas")
    ap.add_argument("--group-size", type=int, default=8,
                    help="stream.group_size: samples per task per group (GRPO group), also the harness concurrency of one "
                         "group (default: 8)")
    ap.add_argument("--adv-normalization", choices=("std", "none"), default="std",
                    help="prepare_rl --adv-normalization: std = (R - mean) / std per group, none = R - mean (default: std)")
    ap.add_argument("--loss-reduction", choices=("sequence", "prompt"), default="sequence",
                    help="prepare_rl --loss-reduction: sequence = coef A/(E*L_i), prompt = coef A/(H*T_group) "
                         "(default: sequence)")
    ap.add_argument("--algo", choices=("bpo", "mis"), default="bpo",
                    help="prepare_rl --algo, the trainer's loss (SPEC §1): bpo = the template's eps/cap/eps_low/eps_high; "
                         "mis = direct token importance sampling, r = pi/mu kept in [--mis-low, --mis-high] and 0 outside "
                         "(default: bpo)")
    ap.add_argument("--mis-low", type=float, default=None, help="mis: lower ratio bound (default: 0.2)")
    ap.add_argument("--mis-high", type=float, default=None, help="mis: upper ratio bound (default: 5.0)")
    ap.add_argument("--efficiency-beta", type=float, default=0.0,
                    help="prepare_rl --efficiency-beta: among fully correct episodes prefer shorter ones, e = 1 - output "
                         "tokens / the harness output_token_budget (default: 0 = off; needs --adv-normalization none)")
    ap.add_argument("--efficiency-guard", type=float, default=0.25,
                    help="prepare_rl --efficiency-guard: in a mixed group the longest correct episode loses at most this "
                         "fraction of its task advantage (default: 0.25)")
    ap.add_argument("--keep-train-order", action="store_true",
                    help="train in the --train-ids-file order (tasks.train.shuffle false: the stream takes the list in order "
                         "and wraps) instead of sorting and shuffling it")
    ap.add_argument("--tool-hand-weight", type=float, default=0.0,
                    help="prepare_rl --tool-hand-weight: the arc2 tool-use term (arcforge.arc2_tooluse) penalises "
                         "min(hand_chars / --tool-hand-scale, --tool-hand-cap) with this weight (default 0 = off; needs the "
                         "harness config's tool_use_stats and --adv-normalization none)")
    ap.add_argument("--tool-toolkit-weight", type=float, default=0.0,
                    help="prepare_rl --tool-toolkit-weight: rewards the share of code commands that call a toolkit function")
    ap.add_argument("--tool-hand-scale", type=float, default=20000.0, help="hand-computation chars per unit (default 20000)")
    ap.add_argument("--tool-hand-cap", type=float, default=3.0, help="cap of the hand-computation term (default 3)")
    ap.add_argument("--arc3-reward", choices=("score", "eff", "c0", "official", "rank", "hinge", "note"), default="score",
                    help="prepare_rl --arc3-reward (arc3_levels only for eff / c0): score = the 0/1 clear reward; eff = within-group "
                         "relative clear + efficiency, R = cleared x (1 + w(p) E) (the four eff --arc3-* values below); c0 = a "
                         "smooth efficiency term centred among the cleared episodes, R = cleared x (1 + s w(p) D), clear-vs-fail "
                         "gap fixed at 1 (the six c0 --arc3-* values below); official = the official level score "
                         "min(100 (baseline/actions)^2, 115) mapped piecewise linearly to R, no token term (the four official "
                         "--arc3-* values below; arcforge.arc3_official); rank = the episode's score x (1 - a r_act - b r_tok), "
                         "r = average rank of actions / tokens_at_clear among the group's cleared episodes in [0, 1] (the two "
                         "rank --arc3-* values below; arcforge.arc3_rank; g2seed); hinge = the score plus a cost centred among "
                         "the cleared, past per-level thresholds frozen from the warm-up (the three hinge --arc3-* values "
                         "below; arcforge.arc3_hinge; g2seed, RL v7; the refs file is <run>/arc3_hinge_refs.json, written by "
                         "hinge_refs.py); note = RL v8: an episode ends at its first window change (--stop-after-first-window), "
                         "a clear scores the official level score with an exponential penalty past the baseline, a window "
                         "change at the threshold scores its note as graded by the note judge (the --note-judge-* values "
                         "below; arcforge.arc3_note; g2seed). All need --adv-normalization none, no defaults. Default: score")
    ap.add_argument("--arc3-beta-actions", type=float, default=None,
                    help="eff: beta on ln(actions) among cleared episodes (>= 0; the term is doubled: score ~ 1/actions^2)")
    ap.add_argument("--arc3-beta-tokens", type=float, default=None,
                    help="eff: beta on ln(tokens_at_clear) among cleared episodes (>= 0)")
    ap.add_argument("--arc3-w-start", type=float, default=None,
                    help="eff: the group clear rate where the efficiency weight w starts rising from 0 to 1 at rate 1 ([0, 1))")
    ap.add_argument("--arc3-eff-clip", type=float, default=None,
                    help="eff: clip of the efficiency term E before re-centering (> 0; <= 0.5 keeps every cleared R > 0)")
    ap.add_argument("--arc3-lambda-actions", type=float, default=None, help="c0: weight of the action quality (>= 0)")
    ap.add_argument("--arc3-kappa-actions", type=float, default=None,
                    help="c0: sigmoid slope on ln(actions) vs the cleared episodes' mean (> 0)")
    ap.add_argument("--arc3-lambda-tokens", type=float, default=None,
                    help="c0: weight of the tokens_at_clear quality (>= 0)")
    ap.add_argument("--arc3-kappa-tokens", type=float, default=None,
                    help="c0: sigmoid slope on ln(tokens_at_clear) vs the cleared episodes' mean (> 0)")
    ap.add_argument("--arc3-success-floor", type=float, default=None,
                    help="c0: every cleared episode's R stays >= this ((0, 1))")
    ap.add_argument("--arc3-top-score", type=float, default=None,
                    help="official: level score >= this -> R = 1 (<= 115, the official cap)")
    ap.add_argument("--arc3-knee-score", type=float, default=None, help="official: the level score of the knee")
    ap.add_argument("--arc3-knee-reward", type=float, default=None, help="official: R at the knee ((0, 1))")
    ap.add_argument("--arc3-zero-score", type=float, default=None,
                    help="official: a clear with level score <= this -> R = 0, like a failure")
    ap.add_argument("--arc3-rank-actions", type=float, default=None,
                    help="rank: weight a of the actions rank (>= 0, a + b < 1; v6: 0.3)")
    ap.add_argument("--arc3-rank-tokens", type=float, default=None,
                    help="rank: weight b of the tokens_at_clear rank (>= 0, a + b < 1; v6: 0.2)")
    ap.add_argument("--arc3-rank-pass-scale", type=int, default=None,
                    help="rank: 1 = the efficiency term times the group's cleared fraction rho, 0 = not (v6: 0, v6b: 1)")
    ap.add_argument("--arc3-rank-warmup-versions", type=int, default=None,
                    help="rank: groups dispatched at a policy version below this get no efficiency term, R = score "
                         "(>= 0; > 0 needs --group-ready; v6: 0, v6b: 5)")
    ap.add_argument("--arc3-hinge-tokens", type=float, default=None,
                    help="hinge: weight of the tokens_at_clear cost past T (full at 3T) (>= 0; v7: 0.25)")
    ap.add_argument("--arc3-hinge-actions", type=float, default=None,
                    help="hinge: weight of the actions cost past A (full at 2A) (>= 0, sum < 1; v7: 0.25)")
    ap.add_argument("--arc3-hinge-warmup-versions", type=int, default=None,
                    help="hinge: the thresholds come from the groups dispatched below this version (>= 1; needs "
                         "--group-ready; v7: 5)")
    ap.add_argument("--window-change", action="store_true",
                    help="kind g2seed: harness window_change (no context_token_limit): an episode changes window as deployed, "
                         "every window a segment (prepare.allow_compaction) (RL v7)")
    ap.add_argument("--stop-after-first-window", action="store_true",
                    help="kind g2seed with --window-change, harness-train only (training groups and the in-stream held-out; "
                         "harness-eval keeps whole windows): harness stop_after_first_window, an episode ends at its first "
                         "window change (stop window) and its handoff note goes to the note judge. Goes with --arc3-reward "
                         "note, and only with it (RL v8)")
    ap.add_argument("--note-judge-backend", choices=("api", "codex", "ensemble", "claude-code"), default=None,
                    help="note: api = the Anthropic API (needs --note-judge-env-file); codex = codex exec with the ChatGPT "
                         "login in ~/.codex (needs --note-judge-codex-bin; no key); ensemble = v8b, Claude Code on the "
                         "subscription login + codex, one pass each per round (needs --note-judge-mapping, "
                         "--note-judge-claude-{bin,model,effort}, --note-judge-codex-{bin,model,effort}); claude-code = "
                         "v8b with Claude Code alone, one pass (needs --note-judge-mapping, --note-judge-claude-{bin,model,"
                         "effort}). -> config note_judge.backend. No default")
    ap.add_argument("--note-judge-mapping", choices=("tier", "tier+rank", "band", "rank"), default=None,
                    help="note, backends ensemble / claude-code: the v8b pass score mapping "
                         "(arcforge.arc3_note.V8B_MAPPINGS). No default")
    ap.add_argument("--note-judge-claude-bin", type=Path, default=None,
                    help="note, backends ensemble / claude-code: the claude executable, resolved and recorded with its "
                         "--version (note_judge.claude_bin / claude_version). No default")
    ap.add_argument("--note-judge-claude-model", default=None, help="note, backends ensemble / claude-code. No default")
    ap.add_argument("--note-judge-claude-effort", choices=("low", "medium", "high", "xhigh", "max"), default=None,
                    help="note, backends ensemble / claude-code. No default")
    ap.add_argument("--note-judge-codex-model", default=None, help="note, backend ensemble only. No default")
    ap.add_argument("--note-judge-codex-effort", choices=("minimal", "low", "medium", "high", "xhigh"), default=None,
                    help="note, backend ensemble only. No default")
    ap.add_argument("--note-judge-python", type=Path, default=None,
                    help="note: the python that runs arc3_note_judge.py (api: the venv with the anthropic SDK, e.g. "
                         "$ARC_DATA/venvs/arc3-judge/bin/python; codex: any python 3) -> config note_judge.python. No default")
    ap.add_argument("--note-judge-model", default=None,
                    help="note: the judge model of the backend (api --model, codex --codex-model). No default")
    ap.add_argument("--note-judge-effort", choices=("minimal", "low", "medium", "high", "xhigh", "max"), default=None,
                    help="note: the backend's effort (api: low..max; codex model_reasoning_effort: minimal..xhigh). No default")
    ap.add_argument("--note-judge-env-file", type=Path, default=None,
                    help="note, backend api only: the file launch_orchestrator.sh sources before it execs the orchestrator "
                         "(the judge's ANTHROPIC_API_KEY_2; never printed). No default")
    ap.add_argument("--note-judge-concurrency", type=int, default=4,
                    help="note: at most this many judge processes at once (note_judge.max_concurrent). Default: 4")
    ap.add_argument("--note-judge-call-timeout-seconds", type=float, default=900.0,
                    help="note: one backend request's limit (note_judge.call_timeout_seconds -> arc3_note_judge.py "
                         "--call-timeout; api: per SDK request, codex: per codex exec). Default: 900")
    ap.add_argument("--note-judge-timeout-seconds", type=float, default=None,
                    help="note: one judge attempt's limit (note_judge.timeout_seconds), at least "
                         "arcforge.arc3_note.judge_attempt_seconds(backend, call timeout) (every call of every pass timing "
                         "out, plus api's overload-retry budget, x1.1 + 600 s margin); the run stops after "
                         "orchestrator.JUDGE_ATTEMPTS failed attempts of one group. Default: exactly that bound")
    ap.add_argument("--note-judge-codex-bin", type=Path, default=None,
                    help="note, backend codex only: the codex executable, resolved (readlink -f) and recorded with its "
                         "--version in note_judge.codex_bin / codex_version (the orchestrator checks the version at stream "
                         "start). No default")
    ap.add_argument("--retire-full-groups", action="store_true",
                    help="stream.retire_full_groups (needs --arc3-reward official): a task whose group has R = 1 on every "
                         "episode is not dispatched again (an all-equal group trains nothing)")
    ap.add_argument("--init-checkpoint", type=Path, default=None,
                    help="start this run from another run's trainer checkpoint-<N> (its LoRA; AdamW starts fresh): the "
                         "trainer stage passes INIT_LORA, needs --init-merged")
    ap.add_argument("--init-merged", type=Path, default=None,
                    help="that run's merged/v<N> (the same policy for the servers): rollout.initial_weights_source; the "
                         "launcher's init-policy stage syncs the replicas to it as weight_version 0")
    ap.add_argument("--max-pooled-groups", type=int, default=128,
                    help="back-pressure: stop dispatching while this many completed live groups wait in the pool")
    ap.add_argument("--prefill-delayer-ratio", type=float, default=0.10,
                    help="--prefill-delayer-queue-min-ratio; queue_min = int(running x ratio), aim for ~5 waiting")
    ap.add_argument("--min-free-gb", type=int, default=140,
                    help="dispatcher stops dispatching below this free space on the disk-guard paths (default: 140)")
    ap.add_argument("--hard-min-free-gb", type=int, default=110, help="learner stops the run below this (default: 110)")
    ap.add_argument("--prepare-max-gb", type=int, default=60, help="prepare_rl output cap (default: 60)")
    ap.add_argument("--prepare-reserve-gb", type=int, default=50, help="free space prepare_rl keeps in reserve (default: 50)")
    ap.add_argument("--model-kind", choices=("fp8", "int3"), default="fp8",
                    help="what the rollout servers and the trainer run (launch_b200_stream.sh reads it from plan.json): "
                         "fp8 = Qwen3.8-Flash-Next-FP8, int3 = gEAN320 INT3 W3A8 deployment package + QAT trainer")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="trainer AdamW learning rate; the launcher's trainer stage reads it from plan.json (default: 1e-4)")
    ap.add_argument("--lr-warmup-steps", type=int, default=0,
                    help="train_rl_service.py --lr-warmup-steps: optimizer step k <= N runs at lr*k/(N+1); plan.json "
                         "trainer_lr_warmup_steps, checked against the trainer's config after it starts (default: 0, constant)")
    ap.add_argument("--adam-beta2", type=float, default=0.98,
                    help="trainer AdamW beta2 (beta1 0.9); the launcher's trainer stage reads it from plan.json (default: 0.98; "
                         "Tinker uses 0.95)")
    ap.add_argument("--adam-eps", type=float, default=1e-12,
                    help="trainer AdamW eps, recorded in plan.json as trainer_adam_eps and checked against the trainer's "
                         "config after it starts (plans without it ran 1e-8). The audit's P4: 1e-8 damps the LoRA-A steps "
                         "to 0.58-0.74x; 1e-12 as Tinker and every arc3 launcher since 10-18 (default: 1e-12 since 10-22, "
                         "1e-8 before)")
    ap.add_argument("--ple-media", choices=("semantic", "deployed"), default="semantic",
                    help="PLE n-gram ids at media pads, read from plan.json by the launcher for both sides: semantic = the "
                         "servers run --ple-mm-placeholder-ids (<|image_pad|>/<|video_pad|>, as HF / vLLM / the G4 deployment) "
                         "and the trainer --ple-media semantic; deployed = upstream SGLang's clamp to vocab_size - 1 on both "
                         "(the runs configured before 10-11). Default: semantic")
    ap.add_argument("--trainer-fast-kernels", default="none",
                    help="trainer --fast-kernels (work/qwen38fn_lora/trainer_kernels.py): none, all, all,qsa_union (the "
                         "query-union QSA forward / dQ, named only, needs qsa_dkdv), or a comma list of the kernels; "
                         "recorded in plan.json, read by the launcher's trainer stage. Default: none (the reference kernels)")
    ap.add_argument("--trainer-dense-qat-gemm", choices=("triton", "cutlass"), default="triton",
                    help="int3 trainer --dense-qat-gemm (work/qwen38fn_lora/fp8_dense_qat.py): the dense FP8 QAT forward GEMM, "
                         "triton or flashinfer's SM100 CUTLASS kernel (same bits; train_rl_service.sh adds the pinned flashinfer "
                         "overlay). Recorded in plan.json, read by the launcher's trainer stage. Default: triton")
    ap.add_argument("--trainer-qsa-indexer", choices=("fp32", "bf16", "fp8"), default="fp32",
                    help="trainer --qsa-indexer (work/qwen38fn_lora/sparse_qsa.py): QSA indexer scoring, fp32 SIMT, bf16 tensor "
                         "cores (the BF16-index rollout's own GEMM; v3.1 world 3: -0.9 s per micro step, |dlogp| to the "
                         "rollout unchanged) or fp8 (only for a serve profile with ARC_QSA_INDEX_FP8=1, which needs it). "
                         "Recorded in plan.json, read by the launcher's trainer stage. Default: fp32")
    ap.add_argument("--prefetch-prepare", action="store_true",
                    help="stream.prefetch_prepare: form and prepare batch b+1 while batch b trains (same groups, earlier; "
                         "orchestrator.py stream_learner_loop_prefetch)")
    ap.add_argument("--group-ready", action="store_true",
                    help="stream.group_ready: accumulate each group's packet as soon as it completes, /step when the batch "
                         "is sealed (SPEC §10); excludes --prefetch-prepare")
    ap.add_argument("--rollout-temperature", type=float, default=1.0,
                    help="sampling temperature of the rollouts (top_p 1, no top-k): written to prepare.temperature, which "
                         "prepare_rl checks every ledger append against and launch_b200_stream.sh serves through "
                         "ARC_RESPONSES_DEFAULT_TEMPERATURE (codex sends none; the /v1/responses class default is 0.7). "
                         "Default: 1.0")
    ap.add_argument("--rollout-top-p", type=float, default=1.0,
                    help="top_p of the rollouts: prepare.top_p (prepare_rl checks every append against it), served "
                         "through ARC_RESPONSES_DEFAULT_TOP_P. Below 1 the ledger records each sampled token's support "
                         "(--session-ledger-sampling-mask) and the trainer replays it (SPEC §1 replay: mask); needs "
                         "--rollout-top-k (the support is bounded by top-k). Default: 1.0 (off)")
    ap.add_argument("--rollout-top-k", type=int, default=-1,
                    help="top_k of the rollouts: prepare.top_k, served through ARC_RESPONSES_DEFAULT_TOP_K; on, the "
                         "ledger records the supports and the trainer replays them. Default: -1 (off)")
    ap.add_argument("--agent-runtime", type=Path,
                    help="the agent's python runtime in the jail (a venv with bin/python), written into harness.agent_python "
                         "as an absolute path: the orchestrator unit re-sources env/env.sh, so an ARC_AGENT_RUNTIME exported "
                         "only by the launcher never reaches the harness. Default: the template's ($ARC_AGENT_RUNTIME)")
    ap.add_argument("--serve-profile", choices=sorted(SERVE_PROFILES), default="base",
                    help="extra serving opt-ins on every replica, recorded in plan.json: base = none; perf = the B200 perf "
                         "opt-ins (fork lock target b200-rl-int3-perf); perf-main-fp16 = the same on fork main with an fp16 "
                         "SSM state (lock target main). Both int3 + TP1 only. Default: base")
    ap.add_argument("--dflash-dir", type=Path, default=None,
                    help="the rollout servers' DFlash drafter deploy directory, recorded in plan.json (dflash_dir) and the "
                         "only one launch_b200_stream.sh serves for this run; required by a profile with a drafter kind "
                         "(perf-main-fp16-ser-opt-lazyfix-l8). Default: none in the plan (the launcher's RL_DFLASH_DIR or "
                         "acc-v4)")
    ap.add_argument("--max-groups-in-flight", type=int, default=None,
                    help="stream.max_groups_in_flight, the global rollout concurrency in groups (default: replicas x "
                         "sessions per replica / group size). Lower it to leave headroom per replica: the dispatcher sends each group "
                         "to the least-loaded replica, so one replica holds at most ceil(groups / replicas) of them")
    ap.add_argument("--memfrac", type=float, default=None,
                    help="rollout server --mem-fraction-static, recorded in plan.json as serve_memfrac (default: 0.96 on TP1 "
                         "replicas, 6-9 GB left per B200 INT3 replica; 0.75 on TP2, the FP8 line's 10-08 value for 512 "
                         "rows x K=8 verify scratch)")
    ap.add_argument("--rollout-slots", type=int, default=None,
                    help="stream.rollout_slots: global rollout concurrency in episodes. With it the dispatcher sends a new "
                         "group while running episodes + group size <= slots (no waiting for whole groups) and each sample "
                         "goes to the replica running the fewest episodes; max groups in flight becomes a process cap "
                         "(default then 2 x slots / group size). Must be replicas x sessions per replica. Default: the "
                         "group-count rule")
    ap.add_argument("--trainer-memory-fraction", type=float, default=None,
                    help="train_rl_service.py --gpu-memory-fraction: cap each trainer rank's CUDA caching allocator at this "
                         "fraction of its card, so the rest is free for prepare_rl's media on a trainer card (plan.json "
                         "trainer_memory_fraction). Default: no cap")
    ap.add_argument("--trainer-offload-keep-gib", type=float, default=None,
                    help="train_rl_service.py --offload-keep-gib: the newest checkpointed layer inputs stay on the trainer GPU "
                         "while they fit in this many GiB instead of going to host memory and back (same bits). 10-21 v3.1 "
                         "p001 world 3: 8 (the last 2 layers at 171k tokens) is -0.23 s per micro step (-1.1%%), peak "
                         "allocated unchanged; the --trainer-memory-fraction cap still bounds the reserved memory "
                         "(docs/research/qwen38fn_rl_trainer_a16_20261021.md). plan.json trainer_offload_keep_gib. "
                         "Default: none (the trainer's 0)")
    ap.add_argument("--trainer-trainable", choices=("lora", "full"), default="lora",
                    help="train_rl_service.py --trainable: lora, or full = the same projections as FP32 full weights "
                         "(dense_full_rl.py; int3 only). plan.json trainer_trainable. Default: lora")
    ap.add_argument("--trainer-freeze-first-layers", type=int, default=0,
                    help="full: train_rl_service.py --freeze-first-layers (decoder layers 0..N-1 stay frozen and run no "
                         "backward; 2 = through the PLE layer). plan.json trainer_freeze_first_layers. Default: 0")
    ap.add_argument("--trainer-checkpoint-store", type=Path, default=None,
                    help="full: train_rl_service.py --checkpoint-store, where the ~35 GB checkpoint DCPs go (linked from "
                         "<run>/trainer/checkpoint-<step>/dcp; e.g. tmpfs). plan.json trainer_checkpoint_store. Default: "
                         "inside the trainer directory")
    ap.add_argument("--trainer-keep-checkpoints", type=int, default=0,
                    help="full: train_rl_service.py --keep-checkpoints (newest N kept; 0 = all). plan.json "
                         "trainer_keep_checkpoints. Default: 0")
    ap.add_argument("--prepare-drop-stops", default="",
                    help="prepare_rl --drop-stops: outcome stops dropped as infra failures instead of trained with their "
                         "reward (arc2: wall,deadline,cancelled,proxy_fatal). Default: none")
    ap.add_argument("--ramp-seconds", type=float, default=None,
                    help="stream.ramp_seconds: after every dispatcher start (resume included) the running-episode cap rises "
                         "linearly from one group to min(rollout slots, max groups in flight x group size) over this many "
                         "seconds, so the episodes' KV does not peak together. Default: none (no ramp)")
    ap.add_argument("--mamba-slots", type=int, default=None,
                    help="rollout server --max-mamba-cache-size (MAMBA_CACHE_SLOTS), recorded in plan.json as "
                         "serve_mamba_slots; default 6 x sessions per replica. A running request holds 5 slots (fewer than "
                         "5 x sessions and SGLang silently caps max_running_requests); the rest keep the radix cache's "
                         "states of sessions waiting on a tool, and what is not given to mamba goes to the KV pool. SGLang's "
                         "own sizing (the old default, 'auto') took ~800-975 slots = 42+ GB of the KV pool")
    ap.add_argument("--output-token-budget", type=int, default=None,
                    help="harness output_token_budget (the per-episode output quota the proxy counts, reasoning included; "
                         "stubs may only lower it); default: the template harness config's")
    ap.add_argument("--context-token-limit", type=int, default=None,
                    help="harness context_token_limit (an episode's thread ends there as a failure, stop file context; "
                         "run_arc3_level.py checks it against codex's compaction point); default: the template's")
    ap.add_argument("--efficiency-free-fraction", type=float, default=None,
                    help="kind g2seed: harness efficiency_free_fraction kappa in (0, 1], R = 1[cleared] x min(1, (kappa x "
                         "baseline / actions)^2) (v5 10-26: 0.7); default: no key = kappa 1")
    ap.add_argument("--action-token-charge", type=int, default=None,
                    help="kind g2seed, harness-train only: action_token_charge C > 0, each counted action takes C / baseline "
                         "tokens off the output budget (after the first f x baseline, --action-charge-free-fraction); "
                         "harness-eval (offline held-out, baselines) stays without it; default: no charge")
    ap.add_argument("--action-charge-free-fraction", type=float, default=None,
                    help="with --action-token-charge: action_charge_free_fraction f in [0, 1], the first f x baseline "
                         "actions are free; default: no key = 0")
    ap.add_argument("--train-output-token-budget", type=int, default=None,
                    help="kind g2seed, harness-train only: output_token_budget N with output_token_budget_override, every "
                         "training episode (and the in-stream held-out) gets N whatever the stubs' quotas; harness-eval "
                         "keeps the template's / --output-token-budget (v5b 10-26: 64000); default: the stubs' quotas")
    ap.add_argument("--seed-note-done", action="store_true",
                    help="kind g2seed, harness-train and harness-eval: seed_note_done, each episode's seed gets "
                         ".note_gate/L(K-1).done at load so level K starts without ./act note_done (the g2seed snapshots "
                         "are taken at note_done); default: the gate stays")
    ap.add_argument("--packet-level-docs", choices=("all", "latest"), default=None,
                    help="kind g2seed, harness-train and harness-eval: packet_level_docs for the gen-2 SessionStart packet, "
                         "latest = only the newest levels/LNN.md, the others by name -- one setting for training, "
                         "evaluation and the Kaggle deployment (10-27); default: no key = all")
    ap.add_argument("--pace-line", action="store_true",
                    help="kind g2seed, harness-train and harness-eval: pace_line, every ./act do and ./act status ends "
                         "with 'This level: X output tokens used of Y.' (the proxy writes the workspace's .pace.json); "
                         "default: no pace line")
    args = ap.parse_args(argv)
    if min(args.optimizer_steps, args.heldout, args.global_optimizer_batch,
           args.save_every_steps, args.eval_every_steps) < 1:
        ap.error("optimizer steps, heldout, batch and checkpoint/eval intervals must be positive")
    if (args.prepare_media_device == "cuda") != (args.prepare_gpu is not None):
        ap.error("--prepare-gpu goes with --prepare-media-device cuda, and only with it")
    topo = TOPOLOGIES[args.topology]
    if args.gpus is not None:
        phys = args.gpus.split(",")
        used = {int(g) for gs in topo["replica_gpus"] + [topo["trainer_gpus"]] for g in gs.split(",")}
        if sorted(used) != list(range(len(phys))) or not all(g.isdigit() for g in phys) or len(set(phys)) != len(phys):
            ap.error(f"--gpus {args.gpus!r}: topology {args.topology} uses {len(used)} GPUs, give that many distinct indices")
        if args.prepare_gpu is not None and args.prepare_gpu not in phys:
            ap.error(f"--prepare-gpu {args.prepare_gpu} is not one of --gpus {args.gpus}")

        def remap(gs: str) -> str:
            return ",".join(phys[int(g)] for g in gs.split(","))
        topo = dict(topo, replica_gpus=[remap(gs) for gs in topo["replica_gpus"]], trainer_gpus=remap(topo["trainer_gpus"]))
    if args.replica_port_base is not None:
        ports = [args.replica_port_base + i for i in range(len(topo["replica_ports"]))]
        if ports[0] < 1024 or ports[-1] > 65535:
            ap.error(f"--replica-port-base {args.replica_port_base}: replica ports {ports} must be valid")
        topo = dict(topo, replica_ports=ports)
    service_ports = {args.control_port, args.trainer_port}
    if len(service_ports) != 2 or not all(1024 <= p <= 65535 for p in service_ports) \
            or service_ports & set(topo["replica_ports"]):
        ap.error(f"--control-port {args.control_port} / --trainer-port {args.trainer_port}: two distinct valid ports, "
                 f"neither a replica port {topo['replica_ports']}")
    profile = SERVE_PROFILES[args.serve_profile]
    for item in (profile or {}).get("env", ()):
        key, value = item.split("=", 1)
        if key == "ARC_INT3_TUNING" and not Path(value).is_file():
            ap.error(f"serve profile {args.serve_profile}: {key} {value} is not a file (the servers would load no table)")
    want_drafter = (profile or {}).get("drafter")
    if want_drafter is not None and args.dflash_dir is None:
        ap.error(f"serve profile {args.serve_profile} serves a {want_drafter} drafter: give --dflash-dir")
    if args.dflash_dir is not None:
        try:
            check_drafter(args.dflash_dir, want_drafter)
        except ValueError as e:
            ap.error(f"--dflash-dir: {e}")
    fast = args.trainer_fast_kernels
    names = fast.split(",")
    named = (FAST_KERNELS - NAMED_ONLY_KERNELS) | set(names[1:]) if names[0] == "all" else set(names)
    if fast != "none" and (not fast or not named <= FAST_KERNELS or len(set(names)) != len(names)
                           or (names[0] == "all" and not set(names[1:]) <= NAMED_ONLY_KERNELS)
                           or ("qsa_union" in named and "qsa_dkdv" not in named)):
        ap.error(f"--trainer-fast-kernels {fast!r}: none, all, all,qsa_union, or a comma list of {sorted(FAST_KERNELS)} "
                 "(qsa_union needs qsa_dkdv)")
    index_fp8 = profile is not None and "ARC_QSA_INDEX_FP8=1" in profile["env"]
    if index_fp8 != (args.trainer_qsa_indexer == "fp8"):
        ap.error(f"--trainer-qsa-indexer {args.trainer_qsa_indexer}: the trainer scores the QSA index in fp8 exactly when "
                 f"the serve profile sets ARC_QSA_INDEX_FP8=1 ({args.serve_profile} {'does' if index_fp8 else 'does not'})")
    if args.trainer_dense_qat_gemm != "triton" and args.model_kind != "int3":
        ap.error("--trainer-dense-qat-gemm is the int3 dense QAT trainer's GEMM")
    if args.trainer_trainable == "full":
        if args.model_kind != "int3":
            ap.error("--trainer-trainable full trains the int3 package's projections (--model-kind int3)")
        if args.init_checkpoint is not None:
            ap.error("--trainer-trainable full has no LoRA checkpoint to start from (--init-checkpoint)")
        if args.trainer_freeze_first_layers < 0 or args.trainer_keep_checkpoints < 0:
            ap.error("--trainer-freeze-first-layers / --trainer-keep-checkpoints must be >= 0")
    elif args.trainer_freeze_first_layers or args.trainer_checkpoint_store or args.trainer_keep_checkpoints:
        ap.error("--trainer-freeze-first-layers / --trainer-checkpoint-store / --trainer-keep-checkpoints need "
                 "--trainer-trainable full")
    if profile is not None and (args.model_kind, topo["tp"]) != (profile["model_kind"], profile["tp"]):
        ap.error(f"serve profile {args.serve_profile} is validated for model kind {profile['model_kind']} on TP{profile['tp']} "
                 f"replicas only (got {args.model_kind}, {args.topology})")
    if args.sessions_per_replica is None:
        args.sessions_per_replica = topo["default_sessions"]
    if args.memfrac is None:
        args.memfrac = 0.96 if topo["tp"] == 1 else 0.75
    if args.mamba_slots is None:
        args.mamba_slots = 6 * args.sessions_per_replica
    n_replicas = len(topo["replica_ports"])
    gs = args.group_size
    if gs < 2:
        ap.error("group size must be at least 2")
    if args.sessions_per_replica < 1:
        ap.error("sessions per replica must be positive")
    # the group-count rule (no --rollout-slots) puts a whole group on one replica; with rollout slots every sample goes
    # to the least-loaded replica on its own, so the sessions need not be a multiple of the group size
    if args.rollout_slots is None and (args.sessions_per_replica < gs or args.sessions_per_replica % gs):
        ap.error(f"sessions per replica must be a positive multiple of the group size {gs} (without --rollout-slots)")
    if args.trainer_memory_fraction is not None and not 0.5 <= args.trainer_memory_fraction <= 1.0:
        ap.error(f"trainer memory fraction must be in [0.5, 1], not {args.trainer_memory_fraction}")
    if args.trainer_offload_keep_gib is not None and not 0 <= args.trainer_offload_keep_gib <= 64:
        ap.error(f"trainer offload keep must be in [0, 64] GiB, not {args.trainer_offload_keep_gib}")
    if args.ramp_seconds is not None and not (args.ramp_seconds >= 0 and math.isfinite(args.ramp_seconds)):
        ap.error(f"ramp seconds must be a finite number >= 0, not {args.ramp_seconds}")
    max_groups = n_replicas * args.sessions_per_replica // gs
    if args.rollout_slots is not None:
        if args.rollout_slots != n_replicas * args.sessions_per_replica:
            ap.error(f"rollout slots must be replicas x sessions per replica ({n_replicas * args.sessions_per_replica}): "
                     "each replica then holds at most its sessions")
        if args.max_groups_in_flight is None:
            args.max_groups_in_flight = 2 * max_groups
        if args.max_groups_in_flight < max_groups:
            ap.error(f"with rollout slots, max groups in flight is a process cap: at least {max_groups}")
    else:
        if args.max_groups_in_flight is None:
            args.max_groups_in_flight = max_groups
        if not 1 <= args.max_groups_in_flight <= max_groups:
            ap.error(f"max groups in flight must be in [1, {max_groups}] (replicas x sessions per replica / group size)")
        if -(-args.max_groups_in_flight // n_replicas) * gs > args.sessions_per_replica:
            ap.error("the least-loaded dispatcher can put ceil(groups in flight / replicas) groups on one replica: "
                     "that many x group size sessions must fit the sessions per replica")
    if not 0.5 <= args.memfrac <= 0.96:     # 0.94 left ~9.5 GB free per B200 in v3 c64; 0.96 leaves ~6 GB
        ap.error("memfrac must be in [0.5, 0.96]")
    if args.mamba_slots < 5 * args.sessions_per_replica:
        ap.error(f"mamba slots must be at least 5 x sessions per replica ({5 * args.sessions_per_replica}): SGLang admits "
                 "a running request on 5 slots and silently caps max_running_requests below that")
    for key in ("output_token_budget", "context_token_limit"):
        value = getattr(args, key)
        if value is not None and value < 1:
            ap.error(f"{key.replace('_', ' ')} must be positive, not {value}")
    if args.max_pooled_groups * gs < args.global_optimizer_batch:
        ap.error("max pooled groups x group size must cover one global optimizer batch")
    if not 0 < args.prefill_delayer_ratio < 1:
        ap.error("prefill delayer ratio must be in (0, 1)")
    if not 0 < args.lr < 1:
        ap.error("learning rate must be in (0, 1)")
    if args.lr_warmup_steps < 0:
        ap.error("lr warmup steps must be >= 0")
    if not 0 < args.adam_beta2 < 1:
        ap.error("adam beta2 must be in (0, 1)")
    if not 0 < args.adam_eps < 1e-4:
        ap.error("adam eps must be in (0, 1e-4)")
    if not 0 < args.rollout_temperature <= 2:
        ap.error("rollout temperature must be in (0, 2]")
    if not 0 < args.rollout_top_p <= 1 or not (args.rollout_top_k == -1 or 1 <= args.rollout_top_k <= SAMPLING_MASK_MAX_TOKENS):
        ap.error(f"--rollout-top-p must be in (0, 1] and --rollout-top-k -1 or in [1, {SAMPLING_MASK_MAX_TOKENS}]")
    if args.rollout_top_p < 1 and args.rollout_top_k == -1:
        ap.error("--rollout-top-p < 1 needs --rollout-top-k: the recorded support is bounded by top-k")
    if args.efficiency_beta < 0 or not 0 < args.efficiency_guard < 1:
        ap.error("efficiency beta must be >= 0 and the guard in (0, 1)")
    if args.efficiency_beta > 0 and args.adv_normalization != "none":
        ap.error("--efficiency-beta needs --adv-normalization none (the std division would cancel beta)")
    if args.global_optimizer_batch % gs:
        ap.error(f"global optimizer batch must contain complete groups of {gs} episodes")
    if args.optimizer_steps % args.save_every_steps:
        ap.error("optimizer steps must be divisible by save-every-steps so the final checkpoint is saved")
    arc3 = args.dataset == "arc3_levels"
    from_dir = args.dataset in ("arc3_levels", "arc2_dir")
    if from_dir != (args.tasks_dir is not None) or from_dir != (args.heldout_ids_file is not None):
        ap.error("--tasks-dir and --heldout-ids-file go with --dataset arc3_levels / arc2_dir, and they need both")
    if args.heldout_dir is not None and args.dataset != "arc2_dir":
        ap.error("--heldout-dir goes with --dataset arc2_dir")
    if arc3 and args.train_ids_file is None:
        ap.error("--dataset arc3_levels needs --train-ids-file (the levels to train on)")
    if arc3 and args.efficiency_beta > 0:   # its B is one number; arc3 stubs carry a quota per level
        ap.error("--efficiency-beta is not for arc3_levels: each level has its own quota (stub output_token_budget)")
    if args.algo == "mis":
        args.mis_low = 0.2 if args.mis_low is None else args.mis_low
        args.mis_high = 5.0 if args.mis_high is None else args.mis_high
        if not (0 < args.mis_low <= 1 <= args.mis_high < math.inf):
            ap.error(f"--mis-low {args.mis_low} / --mis-high {args.mis_high}: need 0 < low <= 1 <= high < inf")
    elif args.mis_low is not None or args.mis_high is not None:
        ap.error("--mis-low / --mis-high go with --algo mis")
    # the prepare keys of each efficiency mode (orchestrator.ARC3_MODE_KEYS) and their flags
    arc3_keys = {"eff": ("arc3_beta_actions", "arc3_beta_tokens", "arc3_w_start", "arc3_eff_clip"),
                 "c0": ("arc3_lambda_actions", "arc3_kappa_actions", "arc3_lambda_tokens", "arc3_kappa_tokens",
                        "arc3_w_start", "arc3_success_floor"),
                 "official": ("arc3_top_score", "arc3_knee_score", "arc3_knee_reward", "arc3_zero_score"),
                 "rank": ("arc3_rank_actions", "arc3_rank_tokens", "arc3_rank_pass_scale", "arc3_rank_warmup_versions"),
                 # + arc3_hinge_refs = <run>/arc3_hinge_refs.json (set below, not a flag)
                 "hinge": ("arc3_hinge_tokens", "arc3_hinge_actions", "arc3_hinge_warmup_versions"),
                 # note: no prepare parameters; the judge's settings are --note-judge-* (top-level note_judge in the config)
                 "note": ()}
    arc3_flags = {k: "--" + k.replace("_", "-") for keys in arc3_keys.values() for k in keys}
    want = arc3_keys.get(args.arc3_reward, ())
    foreign = [arc3_flags[k] for k in arc3_flags if getattr(args, k) is not None and k not in want]
    if foreign:
        ap.error(f"{foreign} do not go with --arc3-reward {args.arc3_reward}")
    arc3_params = {k: getattr(args, k) for k in want}
    if args.arc3_reward != "score":
        mode = args.arc3_reward
        if not arc3:
            ap.error(f"--arc3-reward {mode} is for --dataset arc3_levels")
        if args.adv_normalization != "none":
            ap.error(f"--arc3-reward {mode} needs --adv-normalization none (the std division would cancel the "
                     "efficiency scale)")
        missing = [arc3_flags[k] for k, v in arc3_params.items() if v is None]
        if missing:
            ap.error(f"--arc3-reward {mode} needs {missing} (no defaults)")
        v = arc3_params
        if mode == "official":
            ok = (all(math.isfinite(x) for x in v.values()) and 0 <= v["arc3_zero_score"] < v["arc3_knee_score"]
                  < v["arc3_top_score"] <= 115 and 0 < v["arc3_knee_reward"] < 1)
            if not ok:           # = arcforge.arc3_official.check_params (orchestrator / prepare_rl check again)
                ap.error("--arc3-* official: want 0 <= zero score < knee score < top score <= 115 and 0 < knee reward < 1")
        elif mode == "rank":
            ok = (all(math.isfinite(x) for x in v.values()) and v["arc3_rank_actions"] >= 0 and v["arc3_rank_tokens"] >= 0
                  and v["arc3_rank_actions"] + v["arc3_rank_tokens"] < 1 and v["arc3_rank_pass_scale"] in (0, 1)
                  and v["arc3_rank_warmup_versions"] >= 0)
            if not ok:           # = arcforge.arc3_rank.check_params (orchestrator / prepare_rl check again)
                ap.error("--arc3-rank-actions / --arc3-rank-tokens: want both >= 0 and their sum < 1 (all finite); "
                         "--arc3-rank-pass-scale 0 or 1; --arc3-rank-warmup-versions >= 0")
            if v["arc3_rank_warmup_versions"] > 0 and not args.group_ready:
                ap.error("--arc3-rank-warmup-versions > 0 needs --group-ready (only its packets know each group's "
                         "launch version)")
        elif mode == "hinge":
            ok = (all(math.isfinite(x) for x in v.values()) and v["arc3_hinge_tokens"] >= 0
                  and v["arc3_hinge_actions"] >= 0 and v["arc3_hinge_tokens"] + v["arc3_hinge_actions"] < 1
                  and isinstance(v["arc3_hinge_warmup_versions"], int) and v["arc3_hinge_warmup_versions"] >= 1)
            if not ok:           # = arcforge.arc3_hinge.check_params (orchestrator / prepare_rl check again)
                ap.error("--arc3-hinge-tokens / --arc3-hinge-actions: want both >= 0 and their sum < 1 (finite); "
                         "--arc3-hinge-warmup-versions >= 1")
            if not args.group_ready:
                ap.error("--arc3-reward hinge needs --group-ready (only its packets know each group's launch version)")
        elif mode == "note":
            pass                 # no prepare parameters; the --note-judge-* flags are checked below
        else:
            if mode == "eff":
                ok = v["arc3_beta_actions"] >= 0 and v["arc3_beta_tokens"] >= 0 and v["arc3_eff_clip"] > 0
                rule = "--arc3-beta-actions / --arc3-beta-tokens must be >= 0, --arc3-eff-clip > 0"
            else:
                ok = (v["arc3_lambda_actions"] >= 0 and v["arc3_lambda_tokens"] >= 0 and v["arc3_kappa_actions"] > 0
                      and v["arc3_kappa_tokens"] > 0 and 0 < v["arc3_success_floor"] < 1)
                rule = "--arc3-lambda-* must be >= 0, --arc3-kappa-* > 0, --arc3-success-floor in (0, 1)"
            if not (all(math.isfinite(x) for x in v.values()) and ok and 0 <= v["arc3_w_start"] < 1):
                ap.error(f"{rule}, --arc3-w-start in [0, 1) (all finite)")
    judge_flags = {"--note-judge-backend": args.note_judge_backend, "--note-judge-python": args.note_judge_python,
                   "--note-judge-model": args.note_judge_model, "--note-judge-effort": args.note_judge_effort,
                   "--note-judge-env-file": args.note_judge_env_file, "--note-judge-codex-bin": args.note_judge_codex_bin,
                   "--note-judge-timeout-seconds": args.note_judge_timeout_seconds,
                   "--note-judge-mapping": args.note_judge_mapping, "--note-judge-claude-bin": args.note_judge_claude_bin,
                   "--note-judge-claude-model": args.note_judge_claude_model,
                   "--note-judge-claude-effort": args.note_judge_claude_effort,
                   "--note-judge-codex-model": args.note_judge_codex_model,
                   "--note-judge-codex-effort": args.note_judge_codex_effort}
    ensemble_only = {"--note-judge-mapping", "--note-judge-claude-bin", "--note-judge-claude-model",
                     "--note-judge-claude-effort", "--note-judge-codex-model", "--note-judge-codex-effort"}
    backend_only = {"api": {"--note-judge-env-file", "--note-judge-model", "--note-judge-effort"},
                    "codex": {"--note-judge-codex-bin", "--note-judge-model", "--note-judge-effort"},
                    "ensemble": {"--note-judge-codex-bin", *ensemble_only},
                    "claude-code": {"--note-judge-mapping", "--note-judge-claude-bin", "--note-judge-claude-model",
                                    "--note-judge-claude-effort"}}
    note_judge = None
    if args.arc3_reward == "note":
        # the env file (API key) is the api backend's only; codex has its pinned binary (ChatGPT login in ~/.codex)
        other = set().union(*backend_only.values()) - backend_only.get(args.note_judge_backend, set())
        want = [f for f in judge_flags if f not in other and f != "--note-judge-timeout-seconds"]
        missing = [flag for flag in want if judge_flags[flag] is None]
        if missing:
            ap.error(f"--arc3-reward note needs {missing} (no defaults)")
        stray = sorted(f for f in other if judge_flags[f] is not None)
        if stray:
            ap.error(f"{stray} are not flags of --note-judge-backend {args.note_judge_backend}")
        if args.note_judge_backend not in ("ensemble", "claude-code"):
            efforts = (("low", "medium", "high", "xhigh", "max") if args.note_judge_backend == "api"
                       else ("minimal", "low", "medium", "high", "xhigh"))      # = orchestrator.NOTE_JUDGE_EFFORTS
            if args.note_judge_effort not in efforts:
                ap.error(f"--note-judge-effort {args.note_judge_effort} is not one of {efforts} (backend "
                         f"{args.note_judge_backend})")
        for flag in ("--note-judge-python", "--note-judge-env-file"):
            if judge_flags[flag] is not None and not judge_flags[flag].is_file():
                ap.error(f"{flag} {judge_flags[flag]} is not a file")
        call = args.note_judge_call_timeout_seconds
        if args.note_judge_concurrency < 1 or not (math.isfinite(call) and call > 0):
            ap.error("--note-judge-concurrency must be >= 1 and --note-judge-call-timeout-seconds a finite number > 0")
        need = arc3_note_module().judge_attempt_seconds(args.note_judge_backend, call)
        attempt = need if args.note_judge_timeout_seconds is None else args.note_judge_timeout_seconds
        if not (math.isfinite(attempt) and attempt >= need):
            ap.error(f"--note-judge-timeout-seconds {attempt:g} < {need:g}: a {args.note_judge_backend} judge attempt can "
                     f"take that long at --note-judge-call-timeout-seconds {call:g} (arcforge.arc3_note."
                     f"judge_attempt_seconds: every call of every pass timing out, api's overload budget, x1.1 + 600 s)")
        codex = {}
        if args.note_judge_backend in ("codex", "ensemble"):
            codex = codex_judge_pin(ap, args.note_judge_codex_bin)
        if not args.stop_after_first_window:
            ap.error("--arc3-reward note needs --stop-after-first-window (only an episode that ends at its first window "
                     "change has a note to judge)")
        # the env file's path only: launch_orchestrator.sh sources it, nothing here reads it
        common = {"backend": args.note_judge_backend, "python": str(args.note_judge_python.absolute()),
                  "max_concurrent": args.note_judge_concurrency, "timeout_seconds": attempt,
                  "call_timeout_seconds": call}
        if args.note_judge_backend == "ensemble":
            note_judge = {**common, "mapping": args.note_judge_mapping, **codex,
                          "codex_model": args.note_judge_codex_model, "codex_effort": args.note_judge_codex_effort,
                          **claude_judge_pin(ap, args.note_judge_claude_bin),
                          "claude_model": args.note_judge_claude_model, "claude_effort": args.note_judge_claude_effort}
        elif args.note_judge_backend == "claude-code":
            note_judge = {**common, "mapping": args.note_judge_mapping, **claude_judge_pin(ap, args.note_judge_claude_bin),
                          "claude_model": args.note_judge_claude_model, "claude_effort": args.note_judge_claude_effort}
        else:
            note_judge = {**common, "model": args.note_judge_model, "effort": args.note_judge_effort,
                          **({"env_file": str(args.note_judge_env_file.absolute())} if args.note_judge_backend == "api"
                             else codex)}
    else:
        given = [flag for flag, value in judge_flags.items() if value is not None]
        if given:
            ap.error(f"{given} go with --arc3-reward note")
        if args.stop_after_first_window:
            ap.error("--stop-after-first-window goes with --arc3-reward note (its window episodes are scored only there)")
    if args.retire_full_groups and args.arc3_reward != "official":
        ap.error("--retire-full-groups needs --arc3-reward official")
    init = None
    if (args.init_checkpoint is None) != (args.init_merged is None):
        ap.error("--init-checkpoint and --init-merged go together")
    if args.init_checkpoint is not None:
        init = init_policy(args.init_checkpoint, args.init_merged, args.trainer_base_path)
    if arc3:
        pool = train_dir = args.tasks_dir.absolute()
        train = sorted(p.stem for p in pool.glob("*.json"))
        heldout = args.heldout_ids_file.read_text().split()
        if len(set(heldout)) != len(heldout) or not set(heldout) <= set(train):
            raise RuntimeError(f"{args.heldout_ids_file}: ids must be unique stubs in {pool} "
                               f"(not found: {sorted(set(heldout) - set(train))})")
        heldout = sorted(heldout)
        train = [t for t in train if t not in heldout]
    elif args.dataset == "arc2_dir":
        train_dir = args.tasks_dir.absolute()
        pool = (args.heldout_dir or Path(os.environ["ARC_DATA"]) / "ARC-AGI-2/data/evaluation").absolute()
        train = sorted(p.stem for p in train_dir.glob("*.json"))
        heldout = args.heldout_ids_file.read_text().split()
        missing = [t for t in heldout if not (pool / f"{t}.json").is_file()]
        if not train or len(set(heldout)) != len(heldout) or missing:
            raise RuntimeError(f"arc2_dir: {len(train)} tasks in {train_dir}; {args.heldout_ids_file}: ids must be unique "
                               f"tasks in {pool} (not found: {missing})")
        heldout = sorted(heldout)
    else:
        data = Path(os.environ["ARC_DATA"])
        pool = data / "arc2_pools/training1000"
        ids = sorted(p.stem for p in pool.glob("*.json"))
        if len(ids) != 1000 or args.heldout >= len(ids):
            raise RuntimeError(f"training1000 has {len(ids)} tasks, heldout={args.heldout}")
        random.Random(args.seed).shuffle(ids)
        heldout = sorted(ids[:args.heldout])
        train_dir = pool if args.dataset == "training1000" else data / "ARC-AGI-2/data/evaluation"
        train = ids[args.heldout:] if args.dataset == "training1000" else sorted(p.stem for p in train_dir.glob("*.json"))
    if args.train_ids_file is not None:
        subset = args.train_ids_file.read_text().split()
        if len(set(subset)) != len(subset) or not set(subset) <= set(train):
            raise RuntimeError(f"{args.train_ids_file}: ids must be unique and in the {args.dataset} train split "
                               f"(not found: {sorted(set(subset) - set(train))})")
        train = subset if args.keep_train_order else sorted(subset)
    elif args.keep_train_order:
        ap.error("--keep-train-order needs --train-ids-file")
    if not train or set(train) & set(heldout):
        raise RuntimeError("training set empty or overlaps heldout")
    run = args.run_root.absolute()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise RuntimeError(f"{run}/config.json already exists; use a new run name")
    tasks = run / "task_lists"
    tasks.mkdir()
    for name, values in (("train", train), ("heldout", heldout)):
        (tasks / f"{name}.txt").write_text("\n".join(values) + "\n")
    cfg = json.loads(args.template.read_text())
    if "quarantine" not in cfg["stream"]:          # the orchestrator refuses to run a stream without it (no default)
        ap.error(f"template {args.template}: stream.quarantine is missing (the quarantine circuit breaker, orchestrator "
                 f"QUARANTINE_KEYS; the configs/ stream templates carry it)")
    if not 0 < args.hard_min_free_gb <= args.min_free_gb:
        ap.error("hard-min-free-gb must be positive and at most min-free-gb")
    if args.model_kind == "int3" and topo["tp"] != 1:
        ap.error("model kind int3 runs one card per replica: use a tp1 topology "
                 f"({', '.join(sorted(k for k, t in TOPOLOGIES.items() if t['tp'] == 1))})")
    cfg.update(run_root=str(run), seed=args.seed, rounds=args.optimizer_steps, min_free_gb=args.min_free_gb)
    if args.tokenizer_dir is not None:
        cfg["tokenizer_dir"] = str(args.tokenizer_dir.absolute())
    cfg["stream"].update(group_size=gs, max_groups_in_flight=args.max_groups_in_flight,
                         max_pooled_groups=args.max_pooled_groups, max_staleness=8,
                         batch_live_episodes=args.global_optimizer_batch,
                         eval_keep_every=args.eval_every_steps, hard_min_free_gb=args.hard_min_free_gb)
    if args.prefetch_prepare and args.group_ready:
        ap.error("--group-ready and --prefetch-prepare exclude each other (group_ready already prepares while it trains)")
    if args.prefetch_prepare:
        cfg["stream"]["prefetch_prepare"] = True
    if args.group_ready:
        cfg["stream"]["group_ready"] = True
    if args.rollout_slots is not None:
        cfg["stream"]["rollout_slots"] = args.rollout_slots
    if args.ramp_seconds is not None:
        cfg["stream"]["ramp_seconds"] = args.ramp_seconds
    if args.retire_full_groups:
        cfg["stream"]["retire_full_groups"] = True
    cfg["rollout"].update(base_url=f"http://127.0.0.1:{args.control_port}/v1",
                          replica_urls=[f"http://127.0.0.1:{port}/v1" for port in topo["replica_ports"]])
    if args.trainer_base_path is not None:
        cfg["rollout"]["trainer_base_path"] = str(args.trainer_base_path.absolute())
    if init is not None:
        cfg["rollout"]["initial_weights_source"] = init["merged"]
    cfg["trainer"].update(endpoint_file=str(run / "trainer/trainer_endpoint.json"),
                          save_every_rounds=args.save_every_steps)
    cfg["tasks"] = {"train": {"dir": str(train_dir), "ids_file": str(tasks / "train.txt"),
                              "shuffle": not args.keep_train_order},
                    "heldout": {"dir": str(pool), "ids_file": str(tasks / "heldout.txt")}}
    prepare_gpu = "" if args.prepare_media_device == "cpu-exact" else args.prepare_gpu
    cfg["prepare"].update(command=["env", f"MEDIA_DEVICE={args.prepare_media_device}", f"CUDA_VISIBLE_DEVICES={prepare_gpu}",
                                  f"ARC_SGLANG_FORK={args.fork.absolute()}", "bash",
                                  "work/qwen38fn_lora/prepare_rl.sh", "--workers", "4",
                                  *(["--drop-stops", args.prepare_drop_stops] if args.prepare_drop_stops else [])],
                            minibatches=1, max_gb=args.prepare_max_gb, reserve_gb=args.prepare_reserve_gb,
                            adv_normalization=args.adv_normalization, loss_reduction=args.loss_reduction,
                            temperature=args.rollout_temperature, world=topo["trainer_world"])
    if args.rollout_top_p < 1 or args.rollout_top_k != -1:
        cfg["prepare"].update(top_p=args.rollout_top_p, top_k=args.rollout_top_k)
    harness = json.loads((ROOT / cfg["harness"]["config"]).read_text())
    if (harness.get("kind") in ARC3_KINDS) != arc3:
        ap.error(f"template {args.template}: harness config kind {harness.get('kind')!r} does not match --dataset "
                 f"{args.dataset} (arc3_levels <-> kind {ARC3_KINDS})")
    if harness.get("kind") == ARC3_GAME_KIND:
        cfg["prepare"]["allow_compaction"] = True      # every context window of a whole game is a segment
        # a whole game's budget comes from the harness file (launch_arc3_round2_int3qat.sh's default 90000 once silently
        # overrode the 800k whole-game budget to 90k)
        if args.output_token_budget is not None and args.output_token_budget != harness["output_token_budget"]:
            ap.error(f"template {args.template}: kind {ARC3_GAME_KIND} runs the harness config's output_token_budget "
                     f"{harness['output_token_budget']}, not --output-token-budget {args.output_token_budget}")
        # whole-game R = official game score / 100; c0 / eff / official / rank are shapings of the single-level reward
        if args.arc3_reward != "score":
            ap.error(f"kind {ARC3_GAME_KIND} needs --arc3-reward score (R = official game score / 100), "
                     f"not {args.arc3_reward}")
    if harness.get("kind") == ARC3_G2_KIND and args.arc3_reward not in ("score", "rank", "hinge", "note"):
        # c0 / eff / official shape a 0/1 level-clear reward; g2seed's score is already 1[cleared] × efficiency² (prepare_rl
        # raises on partial credit); rank multiplies onto score (arcforge.arc3_rank); hinge adds onto score (arcforge.arc3_hinge)
        ap.error(f"kind {ARC3_G2_KIND} needs --arc3-reward score (R = the harness's 1[cleared] x min(1, (b/a)^2)), rank "
                 f"(that score x the in-group rank discount) or hinge, not {args.arc3_reward}")
    if args.arc3_reward == "note" and harness.get("kind") != ARC3_G2_KIND:
        ap.error(f"--arc3-reward note is for harness kind {ARC3_G2_KIND}")
    if args.arc3_reward == "hinge" and harness.get("kind") != ARC3_G2_KIND:
        ap.error(f"--arc3-reward hinge is for harness kind {ARC3_G2_KIND} (task ids <game>_L<NN>_<seed>)")
    if args.arc3_reward == "hinge":
        arc3_params["arc3_hinge_refs"] = str(run / "arc3_hinge_refs.json")
    if harness.get("window_change") and not args.window_change:
        ap.error(f"template {args.template}: its harness config changes window (window_change): pass --window-change so "
                 "prepare keeps every window (allow_compaction)")
    if args.window_change:
        if harness.get("kind") != ARC3_G2_KIND:
            ap.error(f"--window-change is for harness kind {ARC3_G2_KIND}, not {harness.get('kind')!r}")
        if args.context_token_limit is not None or args.train_output_token_budget is not None:
            ap.error("--window-change has no context limit: it does not go with --context-token-limit / "
                     "--train-output-token-budget")
        harness.pop("context_token_limit", None)
        harness["window_change"] = True
        cfg["prepare"]["allow_compaction"] = True      # every window of an episode is a segment
    elif args.stop_after_first_window:
        ap.error("--stop-after-first-window needs --window-change")
    if args.output_token_budget is not None:
        harness["output_token_budget"] = args.output_token_budget
    if args.context_token_limit is not None:
        if "context_token_limit" not in harness:
            ap.error(f"template {args.template}: its harness config has no context_token_limit to override")
        harness["context_token_limit"] = args.context_token_limit
    if args.efficiency_free_fraction is not None:
        if harness.get("kind") != ARC3_G2_KIND:
            ap.error(f"--efficiency-free-fraction is for harness kind {ARC3_G2_KIND}, not {harness.get('kind')!r}")
        if not 0 < args.efficiency_free_fraction <= 1:
            ap.error(f"--efficiency-free-fraction {args.efficiency_free_fraction} must be in (0, 1]")
        harness["efficiency_free_fraction"] = args.efficiency_free_fraction
    # the per-action charge goes only into harness-train (used by training groups and the stream loop's held-out): harness-eval
    # is for offline held-out and baseline, and without the charge it matches the conditions of v5 / v5k's held-out
    train_only: dict = {}
    if args.action_charge_free_fraction is not None and args.action_token_charge is None:
        ap.error("--action-charge-free-fraction goes with --action-token-charge")
    if args.action_token_charge is not None:
        if harness.get("kind") != ARC3_G2_KIND:
            ap.error(f"--action-token-charge is for harness kind {ARC3_G2_KIND}, not {harness.get('kind')!r}")
        if args.action_token_charge <= 0:
            ap.error(f"--action-token-charge {args.action_token_charge} must be > 0")
        train_only["action_token_charge"] = args.action_token_charge
        if args.action_charge_free_fraction is not None:
            if not 0 <= args.action_charge_free_fraction <= 1:
                ap.error(f"--action-charge-free-fraction {args.action_charge_free_fraction} must be in [0, 1]")
            train_only["action_charge_free_fraction"] = args.action_charge_free_fraction
    # ending at the first window change (RL v8) likewise goes only into harness-train: offline held-out's harness-eval plays the
    # whole level as deployed
    if args.stop_after_first_window:
        train_only["stop_after_first_window"] = True
    # the training budget likewise goes only into harness-train (overriding the stub's quota); offline held-out's harness-eval
    # keeps the original budget
    if args.train_output_token_budget is not None:
        if harness.get("kind") != ARC3_G2_KIND:
            ap.error(f"--train-output-token-budget is for harness kind {ARC3_G2_KIND}, not {harness.get('kind')!r}")
        if not 0 < args.train_output_token_budget < harness["context_token_limit"]:
            ap.error(f"--train-output-token-budget {args.train_output_token_budget} must be in (0, context_token_limit "
                     f"{harness['context_token_limit']})")
        train_only.update(output_token_budget=args.train_output_token_budget, output_token_budget_override=True)
    # the note_gate marker at game start and the status packet's level files (10-27): both harness-train and harness-eval carry
    # them. The status packet is a setting shared by training, evaluation and the Kaggle deployment (the user, 10-27); new runs
    # train from scratch, their starting point is measured by heldout_at_v0, and offline held-out is no longer compared with
    # v5 / v5k. The pace line likewise goes into both: the act output the model sees is the same in training and evaluation
    for flag, key, value in (("--seed-note-done", "seed_note_done", True if args.seed_note_done else None),
                             ("--packet-level-docs", "packet_level_docs", args.packet_level_docs),
                             ("--pace-line", "pace_line", True if args.pace_line else None)):
        if value is not None:
            if harness.get("kind") != ARC3_G2_KIND:
                ap.error(f"{flag} is for harness kind {ARC3_G2_KIND}, not {harness.get('kind')!r}")
            harness[key] = value
    # the status packet's level files must agree with what the brief says (10-28): run_arm checks the brief at the start of every
    # game and SystemExits on a mismatch; this catches it at configure time instead of after the server is up and the first game
    # stops
    if harness.get("kind") == ARC3_G2_KIND:
        arm = json.loads((ROOT / harness["arm_config"]).read_text())
        brief = " ".join((ROOT / "work/arc_code/prompts" / arm["prompts"] / "brief.md").read_text().split())
        docs = harness.get("packet_level_docs", "all")
        if (docs == "latest") != (GEN2_LATEST_BRIEF in brief):
            ap.error(f"packet_level_docs {docs} but the brief of prompts {arm['prompts']} ({harness['arm_config']}) "
                     f"{'does not say' if docs == 'latest' else 'says'} {GEN2_LATEST_BRIEF!r}: run_arm would stop "
                     f"every episode at its start")
    tool_on = args.tool_hand_weight != 0 or args.tool_toolkit_weight != 0
    if tool_on:
        if args.adv_normalization != "none" or args.arc3_reward != "score":
            ap.error("--tool-*-weight needs --adv-normalization none and no --arc3-reward")
        if harness.get("tool_use_stats") is not True:
            ap.error(f"--tool-*-weight needs the harness config {cfg['harness']['config']} to set tool_use_stats true")
        # by file path (stdlib only): the launcher runs this with the system python3, without src on its path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "arc2_tooluse", Path(__file__).resolve().parents[2] / "src" / "arcforge" / "arc2_tooluse.py")
        tooluse = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tooluse)
        check_tool_params = tooluse.check_params
        tool = {"hand_weight": args.tool_hand_weight, "toolkit_weight": args.tool_toolkit_weight,
                "hand_scale": args.tool_hand_scale, "hand_cap": args.tool_hand_cap}
        try:
            check_tool_params(**tool)
        except ValueError as error:
            ap.error(str(error))
        cfg["prepare"].update({"tool_" + k: v for k, v in tool.items()})
    # the stub's quota (10-28 external review): g2seed stubs carry their own output_token_budget, which takes precedence when there
    # is no override (run_arc3_level.episode_budget), so raising the harness's budget while the stub still has the old 80000
    # silently runs as before. Each stub's quota must equal its harness's budget, or that harness must turn on the override
    if harness.get("kind") == ARC3_G2_KIND:
        for name, ids, hc in (("harness-train", train, dict(harness, **train_only)), ("harness-eval", heldout, harness)):
            if hc.get("output_token_budget_override"):
                continue
            off = sorted({json.loads((pool / f"{t}.json").read_text()).get("output_token_budget",
                                                                            hc["output_token_budget"]) for t in ids}
                         - {hc["output_token_budget"]})
            if off:
                ap.error(f"{name}: stubs in {pool} carry output_token_budget {off} but the harness budget is "
                         f"{hc['output_token_budget']} (no override): the episodes would silently run at the stubs' quota")
    if args.efficiency_beta > 0:          # e = 1 - L / B with B = the line the proxy draws (its own token count)
        cfg["prepare"].update(efficiency_beta=args.efficiency_beta, efficiency_guard=args.efficiency_guard,
                              efficiency_budget=int(harness["output_token_budget"]))
    if args.arc3_reward != "score":       # score = prepare_rl's default: no key
        cfg["prepare"].update(arc3_reward=args.arc3_reward, **arc3_params)
    if note_judge is not None:
        cfg["note_judge"] = note_judge
    if args.algo == "mis":                # bpo = no algo key and the template's four BPO parameters
        for key in ("eps", "cap", "eps_low", "eps_high"):
            del cfg["prepare"][key]
        cfg["prepare"].update(algo="mis", mis_low=args.mis_low, mis_high=args.mis_high)
    harness.update(concurrency=gs, note=f"stream groups: {gs} samples/group; {args.sessions_per_replica // gs} groups per "
                                        f"TP{topo['tp']} replica ({args.topology}), {args.sessions_per_replica} sessions each.")
    for name, salt in (("harness-train", "episode"), ("harness-eval", "none")):
        hc = dict(harness, cache_salt=salt, **(train_only if name == "harness-train" else {}))
        (run / f"{name}.json").write_text(json.dumps(hc, indent=1) + "\n")
    cfg["harness"]["config"] = str(run / "harness-train.json")
    if args.agent_runtime is not None:
        runtime = args.agent_runtime.absolute()
        if not ((runtime / "pyvenv.cfg").is_file() and (runtime / "bin/python").exists()):
            ap.error(f"--agent-runtime {runtime}: not a venv (needs pyvenv.cfg and bin/python)")
        cfg["harness"]["agent_python"] = str(runtime / "bin/python")
    (run / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    plan = {"dataset": args.dataset,
            "train_ids_file": None if args.train_ids_file is None else str(args.train_ids_file.absolute()),
            "train_tasks": len(train), "heldout_tasks": len(heldout),
            "seed": args.seed, "versions": args.optimizer_steps, "optimizer_steps": args.optimizer_steps,
            "cadence": {"round_unit": "optimizer_update_and_rollout_publication",
                        "optimizer_steps_per_version": 1, "replay_passes": 1,
                        "target_live_episodes_per_optimizer_step": args.global_optimizer_batch,
                        "target_live_groups_per_optimizer_step": args.global_optimizer_batch // gs,
                        "eligible_episodes": "actual post-filter count recorded in each batch; may be below target",
                        "max_staleness_unit": "optimizer_updates", "max_staleness": cfg["stream"]["max_staleness"],
                        "save_every_optimizer_steps": args.save_every_steps,
                        "eval_every_optimizer_steps": args.eval_every_steps},
            "checkpoint_versions": list(range(args.save_every_steps, args.optimizer_steps + 1, args.save_every_steps)),
            "evaluation_versions": sorted(set(range(args.eval_every_steps, args.optimizer_steps + 1,
                                                      args.eval_every_steps)) | {args.optimizer_steps}),
            "config_sha256": config_fingerprint(cfg),
            "model_kind": args.model_kind,
            "rollout_gpus": topo["replica_gpus"], "trainer_gpus": topo["trainer_gpus"],
            "sessions_per_replica": args.sessions_per_replica,
            "topology": {"name": args.topology, "tp": topo["tp"], "replica_gpus": topo["replica_gpus"],
                         "replica_ports": topo["replica_ports"]},
            "control_port": args.control_port, "trainer_port": args.trainer_port,
            "prefill_delayer_queue_min_ratio": args.prefill_delayer_ratio, "trainer_lr": args.lr,
            "trainer_lr_warmup_steps": args.lr_warmup_steps, "trainer_adam_beta2": args.adam_beta2,
            "trainer_adam_eps": args.adam_eps, "serve_memfrac": args.memfrac,
            "serve_mamba_slots": args.mamba_slots, "group_size": gs,
            "rollout_slots": args.rollout_slots, "ramp_seconds": args.ramp_seconds,
            "retire_full_groups": args.retire_full_groups, "init": init,
            "trainer_memory_fraction": args.trainer_memory_fraction,
            "trainer_offload_keep_gib": args.trainer_offload_keep_gib,
            "output_token_budget": harness["output_token_budget"],
            "context_token_limit": harness.get("context_token_limit"),
            "window_change": harness.get("window_change", False),
            "stop_after_first_window": train_only.get("stop_after_first_window", False),
            "note_judge": note_judge,
            "efficiency_free_fraction": harness.get("efficiency_free_fraction"),
            "action_token_charge": train_only.get("action_token_charge"),
            "action_charge_free_fraction": train_only.get("action_charge_free_fraction"),
            "train_output_token_budget": train_only.get("output_token_budget"),
            "seed_note_done": harness.get("seed_note_done", False),
            "packet_level_docs": harness.get("packet_level_docs"),
            "pace_line": harness.get("pace_line", False),
            "agent_python": cfg["harness"]["agent_python"],
            "adv_normalization": args.adv_normalization, "loss_reduction": args.loss_reduction,
            "loss_algo": {"algo": args.algo, **{k: cfg["prepare"][k] for k in (
                ("eps", "cap", "eps_low", "eps_high") if args.algo == "bpo" else ("mis_low", "mis_high"))}},
            "efficiency": None if args.efficiency_beta == 0 else {
                "beta": args.efficiency_beta, "guard": args.efficiency_guard,
                "budget": cfg["prepare"]["efficiency_budget"]},
            "tool_reward": None if not tool_on else {k.removeprefix("tool_"): v for k, v in cfg["prepare"].items()
                                                    if k.startswith("tool_")},
            "arc3_reward": None if args.arc3_reward == "score" else {
                "mode": args.arc3_reward, **{k.removeprefix("arc3_"): v for k, v in arc3_params.items()}},
            "ple_media": args.ple_media, "trainer_fast_kernels": args.trainer_fast_kernels,
            "trainer_dense_qat_gemm": args.trainer_dense_qat_gemm, "trainer_qsa_indexer": args.trainer_qsa_indexer,
            "trainer_trainable": args.trainer_trainable,
            "trainer_freeze_first_layers": args.trainer_freeze_first_layers,
            "trainer_checkpoint_store": (None if args.trainer_checkpoint_store is None
                                         else str(args.trainer_checkpoint_store.absolute())),
            "trainer_keep_checkpoints": args.trainer_keep_checkpoints,
            "serve_profile": {"name": args.serve_profile} if profile is None else {
                "name": args.serve_profile, "lock_target": profile["lock_target"],
                "env": profile_env(profile, args.sessions_per_replica), "args": profile["args"]},
            "dflash_dir": None if args.dflash_dir is None else str(args.dflash_dir.absolute()),
            "dflash_k": 4, "learning_rate": 1e-4,
            "heldout_samples": 4, "baseline_budget": 32000,
            "train_ids_sha256": hashlib.sha256((tasks / "train.txt").read_bytes()).hexdigest(),
            "heldout_ids_sha256": hashlib.sha256((tasks / "heldout.txt").read_bytes()).hexdigest()}
    (run / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps(plan, indent=1))
    print(f"config: {run / 'config.json'}")


if __name__ == "__main__":
    main()
