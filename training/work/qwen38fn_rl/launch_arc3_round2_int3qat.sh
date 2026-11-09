#!/bin/bash
# ARC-AGI-3 single-level RL round 2 on node2 (docs/plans/arc3_level_rl_round2_20261019.md): gEAN320 INT3 W3A8 QAT on the 19
# training games' L1..L(last-1) (the plan's 20 minus sc25); levels 2+ start from a seed workspace replayed from a real v0
# full game (stub seed* fields; work/arc_code/arc3_levels.py seed / lists), in a fresh codex conversation with the whole
# game's hand-over message (prompts v14seed). harness-arc3-round2: 100k output and 180k context per level, either one used
# up ends the episode as a failure; window 262144, so codex's compaction point (0.9 x window = 235929) is never reached.
# Reward = prepare_rl --arc3-reward c0 (v3; round 2 ran eff): cleared - p̂ plus w(p̂) x an efficiency term among the
# cleared episodes, D = L_a (q_a - mean q_a) + L_t (q_t - mean q_t) with q = sigmoid(k (mean ln cost - ln cost)) on
# actions and tokens_at_clear, no clip, scaled down only as far as keeping every cleared R >= the floor; the clear-vs-fail
# gap stays 1. Defaults L_a 2 k_a 4 L_t 0.5 k_t 2 w_start 0 (w = p̂) floor 0.1: on round 2's groups the token part is ~15%
# of the efficiency term and reverses ~3% of the action order. The last level of the training games (E1) and every level of the 5 held-out games (E2) are evaluation only
# (heldout.txt); they are not evaluated in the loop (arc3_version_eval.py on kept versions).
# Everything else as the overfit10 run (launch_arc3_overfit10_int3qat.sh): tp1x4 INT3 + 4-rank trainer, group_ready,
# routing-gc, lr 2e-4, AdamW (0.9, 0.95) eps 1e-12, adv none, loss prompt, T=1.0. Two changes from the node1 kverify
# session (10-20): serve profile perf-main-fp16-ser-opt-lazyfix (configure_b200.py: fork exp/kverify-dev-lazyfix
# a6c7fddbb = lock target b200-rl-ser-opt-lazyfix, the B200 Humming table, ARC_MM_LAZY_MEDIA / RELEASE_AFTER_PREFILL /
# NO_SYNC; with RL_FP8_GEMM=flashinfer_cutlass; round 2 ran perf-main-fp16-ser-opt, the same without LAZY_MEDIA) and
# codex p8 ($ARC_CODEX_P8_PREFIX: mp4-header video estimate + features.server_reasoning_included, which
# b200-level-v14seed sets; p7 refuses that key).
#
# Knobs (env, read at configure time only; config.json / plan.json record what was used):
#   RL_TASKS_DIR       stub dir (seeded stubs + the unseeded L1s)        default $ARC_DATA/arc3_levels/round2-x2-20261020
#                      (what round 2 ran: lists/train.txt = all 221, no difficulty filter -- not train_feasible.txt, the
#                      198-task filter; lists/heldout.txt = 99, E2 = dc22 ft09 lp85 re86 s5i5. round2-v00-20261020 splits
#                      the games differently and trains on four of x2's held-out games)
#   RL_TRAIN_IDS       train ids file                                   default $RL_TASKS_DIR/lists/train.txt
#   RL_HELDOUT_IDS     held-out ids file (E1 + E2)                      default $RL_TASKS_DIR/lists/heldout.txt
#   RL_GROUP_SIZE 16   RL_SESSIONS 80 (per replica; v7 10-28: kv_sim c80 mean 0.70 / p99 0.82 of the 6.84M pool at
#                      memfrac 0.96 with 480 slots; a full pool stalls exact resume; round 2 ran 32 on a 4.6M pool)
#                      RL_BATCH 256 (episodes/step)
#   RL_MAMBA_SLOTS 6 x RL_SESSIONS (configure_b200.py --mamba-slots; a running request holds 5 slots and fewer than
#                      5 x sessions silently caps max_running_requests; v3 c64 peaked at 5.0 slots per session; round 2
#                      passed none and SGLang gave 982 slots = 54 GB of mamba state, at most 16% used, out of the KV pool)
#   RL_OUTPUT_BUDGET 90000  RL_CONTEXT_LIMIT 140000 (harness output_token_budget / context_token_limit, configure_b200.py
#                      --output-token-budget / --context-token-limit; round 2 and v3 c64 ran the template's 100k / 180k;
#                      none = no --context-token-limit, for a harness without one: the whole-game arc3_game compacts)
#   RL_TEMPLATE        configure template (default work/qwen38fn_rl/configs/arc3-round2-int3qat.json; v4 whole game:
#                      work/qwen38fn_rl/configs/arc3-fullgame-v4.json with RL_TASKS_DIR = arc3_levels.py games stubs)
#   RL_STEPS 100       RL_SAVE_EVERY 10      RL_ROLLOUT_TEMPERATURE 1.0      RL_LR 2e-4      RL_MEMFRAC 0.96
#   RL_LR_WARMUP_STEPS 0 (configure_b200.py --lr-warmup-steps: optimizer step k <= N runs at lr*k/(N+1); 0 = constant)
#   RL_ROLLOUT_TOP_P 1.0  RL_ROLLOUT_TOP_K -1 (configure_b200.py --rollout-top-p / --rollout-top-k; truncated sampling
#                      needs the -topp serve profile: the ledger records each token's support, the trainer replays it)
#   RL_AGENT_RUNTIME   the agent's python venv in the jail (configure_b200.py --agent-runtime -> harness.agent_python;
#                      default $ARC_AGENT_RUNTIME). The selftest reads it back from config.json, so a resume runs the same one
#   RL_EVAL_EVERY RL_SAVE_EVERY (configure_b200.py --eval-every-steps; also the merged exports the stream keeps, so a
#                      checkpoint every step keeps evaluation and merged retention at 10)
#   RL_INIT_CHECKPOINT / RL_INIT_MERGED (configure_b200.py --init-checkpoint / --init-merged): start from another run's
#                      trainer checkpoint-<N> and its merged/v<N>; the init-policy stage syncs the replicas to it
#   RL_LOSS_REDUCTION prompt (configure_b200.py --loss-reduction: prompt = A/(H*T_group), sequence = A/(E*L_i))
#   RL_EFFICIENCY_FREE_FRACTION (unset = kappa 1; configure_b200.py --efficiency-free-fraction, g2seed only): R = 1[cleared]
#                      x min(1, (kappa x baseline / actions)^2)
#   RL_ACTION_TOKEN_CHARGE RL_ACTION_CHARGE_FREE_FRACTION (unset = no charge; configure_b200.py --action-token-charge /
#                      --action-charge-free-fraction, g2seed, harness-train only): each counted action takes C / baseline
#                      tokens off the 48k output budget after the first f x baseline actions
#   RL_TRAIN_OUTPUT_BUDGET (unset = the stubs' quotas; configure_b200.py --train-output-token-budget, g2seed,
#                      harness-train only): every training / in-stream held-out episode gets this budget (v5b: 64000)
#   RL_SEED_NOTE_DONE=1 (unset = the gate stays; --seed-note-done) / RL_PACKET_LEVEL_DOCS all|latest (unset = all;
#                      --packet-level-docs): g2seed, harness-train and harness-eval -- level K starts without
#                      ./act note_done; the SessionStart packet carries only the newest levels/LNN.md (latest is the one
#                      setting for training, evaluation and the Kaggle deployment, 10-27)
#   RL_PACE_LINE=1 (unset = no pace line; --pace-line): g2seed, harness-train and harness-eval -- every ./act do and
#                      ./act status ends with "This level: X output tokens used of Y." (the proxy writes .pace.json)
#   RL_WINDOW_CHANGE=1 (unset = one window; --window-change, g2seed, needs RL_CONTEXT_LIMIT=none): episodes change window as
#                      deployed, every window a segment (RL v7)
#   RL_STOP_AFTER_FIRST_WINDOW=1 (unset = whole windows; --stop-after-first-window, needs RL_WINDOW_CHANGE=1 and
#                      RL_ARC3_REWARD=note; harness-train only): an episode ends at its first window change (RL v8)
#   RL_ARC3_REWARD c0  RL_LAMBDA_ACTIONS 2  RL_KAPPA_ACTIONS 4  RL_LAMBDA_TOKENS 0.5  RL_KAPPA_TOKENS 2  RL_W_START 0
#                      RL_SUCCESS_FLOOR 0.1;  eff (round 2): RL_BETA_ACTIONS 1  RL_BETA_TOKENS 0.5  RL_W_START 0.5  RL_EFF_CLIP 0.5;
#                      official (v3.3): R = f(official level score), RL_TOP_SCORE 115 (-> 1)  RL_KNEE_SCORE 100
#                      RL_KNEE_REWARD 0.7  RL_ZERO_SCORE 75 (-> 0), no token term (arcforge.arc3_official);
#                      rank (v6): R = score x (1 - RL_RANK_ACTIONS r_act - RL_RANK_TOKENS r_tok), r = in-group rank of
#                      actions / tokens_at_clear among the cleared, RL_RANK_ACTIONS 0.3  RL_RANK_TOKENS 0.2 (arcforge.arc3_rank);
#                      RL_RANK_PASS_SCALE 0 (v6b 1: the efficiency term x the group's cleared fraction)  RL_RANK_WARMUP_VERSIONS 0
#                      hinge (v7): RL_HINGE_TOKENS RL_HINGE_ACTIONS RL_HINGE_WARMUP_VERSIONS, no defaults (arcforge.arc3_hinge;
#                      the thresholds are frozen per level by hinge_refs.py into <run>/arc3_hinge_refs.json)
#                      (v6b 5: groups dispatched before version 5 get R = score)
#                      note (v8): a clear gets the official level score with an exponential penalty past the baseline, a
#                      window change at the threshold the note judge's grade (arcforge.arc3_note); needs
#                      RL_STOP_AFTER_FIRST_WINDOW=1. RL_JUDGE_BACKEND api|codex, RL_JUDGE_MODEL, RL_JUDGE_EFFORT (no
#                      defaults; plan §8 is still open; model / effort are the chosen backend's: api claude-*, low..max;
#                      codex e.g. gpt-6-astra, minimal..xhigh), RL_JUDGE_PYTHON (no default; api: the venv with the
#                      anthropic SDK, e.g. $ARC_DATA/venvs/arc3-judge/bin/python), api only: RL_JUDGE_ENV_FILE (no default,
#                      e.g. $HOME/work/arc-forge/.env: launch_orchestrator.sh sources it for the orchestrator unit; it holds
#                      ANTHROPIC_API_KEY_2, never printed); codex only: the binary is `readlink -f $(command -v codex)` of
#                      this shell, pinned with its --version in config/plan; the login is the default ~/.codex
#                      (~/.codex/auth.json must exist; no CODEX_HOME, codex exec --ignore-user-config). RL_JUDGE_CONCURRENCY
#                      (configure default 4),
#                      RL_JUDGE_CALL_TIMEOUT_SECONDS (one backend request; configure default 900), RL_JUDGE_TIMEOUT_SECONDS
#                      (one judge attempt, >= arcforge.arc3_note.judge_attempt_seconds; configure default = that bound)
#   RL_RETIRE_FULL_GROUPS 0 (1: stream.retire_full_groups, official only: a task with an all-R = 1 group is not dispatched
#                      again; retired.jsonl in the run root)
#   RL_RAMP_SECONDS 1200 (configure_b200.py --ramp-seconds: the running-episode cap rises from one group to all rollout
#                      slots over 20 min after every dispatcher start, so the KV does not peak together; v3 c64 opened
#                      with every slot at once and peaked at 0.86-0.91; 0 = no ramp. A resumed run reads its config.json:
#                      add stream.ramp_seconds there while the orchestrator is stopped)
#   RL_WANDB_PROJECT qwen38fn-rl-arc3 (launch_orchestrator.sh; round 2 and v3 c64 logged to the arc2 project qwen38fn-rl)
#   RL_ALGO mis        (v3: the MIS loss of SPEC §1, ratio band [0.2, 5]; bpo = round 2; configure_b200.py --algo)
#   RL_MIN_FREE_GB 80  RL_HARD_MIN_FREE_GB 60  RL_PREPARE_MAX_GB 60  RL_PREPARE_RESERVE_GB 20   (GiB on $ARC_RUNS and tmpfs)
#   RL_TOPOLOGY tp1x4 (configure_b200.py --topology; tp1x3 = 3 replicas on GPUs 0-2 + a 3-rank trainer on 3-5, v3.1)
#   RL_GPUS 0,...,N-1 (the topology's N GPUs)   rollout slots = replicas x RL_SESSIONS
#   RL_DFLASH_DIR      the drafter (configure_b200.py --dflash-dir, recorded in plan.json; a profile with a drafter kind
#                      needs it). Unset: none in the plan, and launch_b200_stream.sh serves acc-v4 as before
#   v3.1 (10-21, node1): RL_TOPOLOGY tp1x3 RL_SERVE_PROFILE perf-main-fp16-ser-opt-lazyfix-l8 RL_LOCK_TARGET b200-rl-main-1020
#                      RL_FORK_TREE <fork worktree at a5c67c41c> RL_DFLASH_DIR <8-layer plan B s25562 deploy-fp8>
#   RL_TRAINER_TRAINABLE full (configure_b200.py --trainer-trainable: the projections as FP32 full weights instead of the
#                      LoRA; v8 dense full FT) with RL_TRAINER_CHECKPOINT_STORE (where the ~35 GB checkpoint DCPs go, e.g. tmpfs),
#                      RL_TRAINER_KEEP_CHECKPOINTS (newest N kept), RL_TRAINER_FREEZE_FIRST_LAYERS (e.g. 2); unset = the
#                      configure defaults (lora; full needs no --init-checkpoint)
#   RL_TRAINER_MEMORY_FRACTION 0.92 (configure_b200.py --trainer-memory-fraction: each trainer rank's caching allocator
#                      capped at 0.92 x 178 GiB = 164 GiB; v3 c64 peaked at 141 GiB allocated but held 175 GiB, and
#                      v3 c84 left GPU 7 1 GB, so prepare_rl on that card died of a CUDA OOM)
#   RL_TRAINER_FAST_KERNELS all  RL_TRAINER_DENSE_QAT_GEMM triton  RL_TRAINER_QSA_INDEXER fp32  RL_TRAINER_OFFLOAD_KEEP_GIB
#                      (unset = 0): configure_b200.py --trainer-*. The 10-21 trainer speed-ups (bit-identical except union):
#                      all,qsa_union / cutlass / bf16 / 8, with RL_SERVE_PROFILE perf-main-fp16-ser-opt-lazyfix-l8-a16
#                      RL_LOCK_TARGET b200-rl-main-1021 (W3A16 rollout; docs/research/humming_0118_b200_int3_20261021.md)
#   RL_PREPARE_MEDIA cpu-exact (configure_b200.py --prepare-media-device): prepare_rl reproduces the deployed CUDA
#                      bicubic on the CPU bit for bit and uses no GPU. cuda (with RL_PREPARE_GPU, default the last of
#                      RL_GPUS) ran it on a trainer card: 10-21 a16t08 that rank's FSDP all-gather wedged three times while
#                      a prepare was running there; on GPU 0 it shared r0's card and v3 c88 at memfrac 0.96 died of a CUDA OOM
#   RL_SERVE_PROFILE perf-main-fp16-ser-opt-lazyfix with RL_FORK_TREE $HOME/work/sglang-arc-2-kverify-dev-lazyfix (a
#   worktree of the fork at a6c7fddbb, branch exp/kverify-dev-lazyfix), RL_LOCK_TARGET b200-rl-ser-opt-lazyfix,
#   RL_FP8_GEMM flashinfer_cutlass; round 2's serving (no LAZY_MEDIA) is perf-main-fp16-ser-opt with
#   $HOME/work/sglang-arc-2-kverify-opt and b200-rl-ser-opt; for the one before that set all four
#   (perf-main-fp16-ser, $HOME/work/sglang-arc-2-ser-b200, b200-rl-ser, flashinfer_trtllm)
# Smoke: launch_arc3_round2_smoke.sh (same script, small values).
#
# Disk (10-20: /data had ~60 GiB free): batches/ (the prepared payload, mostly routing at ~1 KB per token: 26 GiB for the
# first 256-episode step, deleted after it) is a symlink into the run's tmpfs like replicas/, and once the orchestrator runs,
# unit <prefix>-group-pack (group_pack.py) moves every consumed or abandoned group's episodes/ + harness/ (~25 MB a group)
# into a verified <tmpfs>/groups-packed/<group>.tar.zst (~5 MB). Both are lost on a reboot (as is the ledger): copy
# groups-packed/ off the tmpfs before one. The prepare reservation (max_gb + reserve_gb) is checked on
# /data too, so a full run here wants RL_PREPARE_MAX_GB 40 RL_PREPARE_RESERVE_GB 3 RL_MIN_FREE_GB 41 RL_HARD_MIN_FREE_GB 22
# RL_DISK_FLOOR_GIB 22 (min_free_gb in GiB must cover max_gb + reserve_gb in GB). max_gb caps one packet's routing: v3 at 64
# sessions packed the 8 groups that completed during a step into one 23.1 GB packet and prepare_rl refused it at max_gb 20
# (a packet holds at most 16 groups, ~2.2 GB of routing each at the 100k output cap).
#
# public-25 is the bench: everything this run produces is contaminated -- do not evaluate its LoRA on public-25, do not feed
# its trajectories to DFlash / QAD / SFT. RL_RUN must say so (its name contains public25-contaminated).
#
#   RL_RUN=$ARC_RUNS/qwen38fn_rl/<YYYYMMDD>-arc3-round2-int3qat-public25-contaminated \
#       bash work/qwen38fn_rl/launch_arc3_round2_int3qat.sh            # configure (once) + selftest + the stages
#   RL_DRY_RUN=1 RL_RUN=... bash ...                                   # print every command, start nothing
#   RL_STAGES="check servers" RL_RUN=... bash ...                      # a subset, in this order
# Stop: touch $RL_RUN/STOP (drains), then systemctl --user stop <unit prefix>-{orchestrator,routing-gc,control,trainer,r0..r3}.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh
: "${RL_RUN:?set RL_RUN=\$ARC_RUNS/qwen38fn_rl/<YYYYMMDD>-arc3-round2-int3qat-public25-contaminated}"
case "$(basename "$RL_RUN")" in
    *public25-contaminated*) ;;
    *) echo "RL_RUN $RL_RUN: the name must contain public25-contaminated (it trains on the bench)" >&2; exit 1 ;;
esac
export RL_RUN
export RL_FORK_TREE=${RL_FORK_TREE:-$HOME/work/sglang-arc-2-kverify-dev-lazyfix}
export RL_LOCK_TARGET=${RL_LOCK_TARGET:-b200-rl-ser-opt-lazyfix}
export RL_FP8_GEMM=${RL_FP8_GEMM:-flashinfer_cutlass}     # read by launch_b200_stream.sh servers (--fp8-gemm-backend)
export RL_SERVE_MODEL=${RL_SERVE_MODEL:-/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-gptq}
export RL_TRAIN_MODEL=${RL_TRAIN_MODEL:-/dev/shm/uid-$UID/models/qwen38fn-gean320-int3-bf16dense}
export RL_DISK_FLOOR_GIB=${RL_DISK_FLOOR_GIB:-50}
export RL_WANDB_PROJECT=${RL_WANDB_PROJECT:-qwen38fn-rl-arc3}   # arc3 runs apart from the arc2 ones (qwen38fn-rl)
RL_TMPFS=${RL_TMPFS:-/dev/shm/uid-$UID/arcforge-runs/qwen38fn_rl/$(basename "$RL_RUN")-tmpfs}
dry=${RL_DRY_RUN:-0}
stages=${RL_STAGES-check servers control init-policy trainer orchestrator routing-gc}  # RL_STAGES="" = configure and the tmpfs link only
py=${RL_PYTHON:-python3}
tasks=${RL_TASKS_DIR:-$ARC_DATA/arc3_levels/round2-x2-20261020}
gs=${RL_GROUP_SIZE:-16}
sessions=${RL_SESSIONS:-80}
profile=${RL_SERVE_PROFILE:-perf-main-fp16-ser-opt-lazyfix}
mamba_slots=${RL_MAMBA_SLOTS:-$((6 * sessions))}
reward=${RL_ARC3_REWARD:-c0}
case "$reward" in
    c0)  reward_args=(--arc3-lambda-actions "${RL_LAMBDA_ACTIONS:-2}" --arc3-kappa-actions "${RL_KAPPA_ACTIONS:-4}"
                      --arc3-lambda-tokens "${RL_LAMBDA_TOKENS:-0.5}" --arc3-kappa-tokens "${RL_KAPPA_TOKENS:-2}"
                      --arc3-w-start "${RL_W_START:-0}" --arc3-success-floor "${RL_SUCCESS_FLOOR:-0.1}") ;;
    eff) reward_args=(--arc3-beta-actions "${RL_BETA_ACTIONS:-1}" --arc3-beta-tokens "${RL_BETA_TOKENS:-0.5}"
                      --arc3-w-start "${RL_W_START:-0.5}" --arc3-eff-clip "${RL_EFF_CLIP:-0.5}") ;;
    official) reward_args=(--arc3-top-score "${RL_TOP_SCORE:-115}" --arc3-knee-score "${RL_KNEE_SCORE:-100}"
                           --arc3-knee-reward "${RL_KNEE_REWARD:-0.7}" --arc3-zero-score "${RL_ZERO_SCORE:-75}") ;;
    rank) reward_args=(--arc3-rank-actions "${RL_RANK_ACTIONS:-0.3}" --arc3-rank-tokens "${RL_RANK_TOKENS:-0.2}"
                       --arc3-rank-pass-scale "${RL_RANK_PASS_SCALE:-0}" --arc3-rank-warmup-versions "${RL_RANK_WARMUP_VERSIONS:-0}") ;;
    hinge) reward_args=(--arc3-hinge-tokens "${RL_HINGE_TOKENS:?hinge: no default}"
                        --arc3-hinge-actions "${RL_HINGE_ACTIONS:?hinge: no default}"
                        --arc3-hinge-warmup-versions "${RL_HINGE_WARMUP_VERSIONS:?hinge: no default}") ;;
    note) judge_backend=${RL_JUDGE_BACKEND:?note: RL_JUDGE_BACKEND api, codex, ensemble or claude-code, no default}
          reward_args=(--note-judge-backend "$judge_backend"
                       --note-judge-python "${RL_JUDGE_PYTHON:?note: RL_JUDGE_PYTHON, no default}"
                       ${RL_JUDGE_CONCURRENCY:+--note-judge-concurrency "$RL_JUDGE_CONCURRENCY"}
                       ${RL_JUDGE_CALL_TIMEOUT_SECONDS:+--note-judge-call-timeout-seconds "$RL_JUDGE_CALL_TIMEOUT_SECONDS"}
                       ${RL_JUDGE_TIMEOUT_SECONDS:+--note-judge-timeout-seconds "$RL_JUDGE_TIMEOUT_SECONDS"})
          [ "$judge_backend" = ensemble ] || [ "$judge_backend" = claude-code ] \
              || reward_args+=(--note-judge-model "${RL_JUDGE_MODEL:?note: no default}"
                               --note-judge-effort "${RL_JUDGE_EFFORT:?note: no default}")
          case "$judge_backend" in
              api) reward_args+=(--note-judge-env-file "${RL_JUDGE_ENV_FILE:?note api: RL_JUDGE_ENV_FILE, no default}") ;;
              ensemble)
                     # v8b: Claude Code (the subscription login, user 11-01) + codex (the ChatGPT login), both binaries
                     # resolved now; configure records them with their --version
                     [ -z "${RL_JUDGE_ENV_FILE:-}${ANTHROPIC_API_KEY:-}" ] || {
                         echo "note ensemble: no RL_JUDGE_ENV_FILE / ANTHROPIC_API_KEY (claude runs on the subscription login)" >&2
                         exit 1; }
                     judge_codex=$(readlink -f "$(command -v codex)") || { echo "note ensemble: no codex on PATH" >&2; exit 1; }
                     judge_claude=$(readlink -f "$(command -v claude)") || { echo "note ensemble: no claude on PATH" >&2; exit 1; }
                     [ -f "$HOME/.codex/auth.json" ] || {
                         echo "note ensemble: $HOME/.codex/auth.json missing (codex login first)" >&2; exit 1; }
                     reward_args+=(--note-judge-codex-bin "$judge_codex" --note-judge-claude-bin "$judge_claude"
                                   --note-judge-mapping "${RL_JUDGE_MAPPING:?note ensemble: no default}"
                                   --note-judge-claude-model "${RL_JUDGE_CLAUDE_MODEL:?note ensemble: no default}"
                                   --note-judge-claude-effort "${RL_JUDGE_CLAUDE_EFFORT:?note ensemble: no default}"
                                   --note-judge-codex-model "${RL_JUDGE_CODEX_MODEL:?note ensemble: no default}"
                                   --note-judge-codex-effort "${RL_JUDGE_CODEX_EFFORT:?note ensemble: no default}") ;;
              claude-code)
                     # v8b, Claude Code alone (user 11-01, after the test set): the subscription login, the binary
                     # resolved now; configure records it with its --version
                     [ -z "${RL_JUDGE_ENV_FILE:-}${ANTHROPIC_API_KEY:-}" ] || {
                         echo "note claude-code: no RL_JUDGE_ENV_FILE / ANTHROPIC_API_KEY (claude runs on the subscription login)" >&2
                         exit 1; }
                     judge_claude=$(readlink -f "$(command -v claude)") || { echo "note claude-code: no claude on PATH" >&2; exit 1; }
                     reward_args+=(--note-judge-claude-bin "$judge_claude"
                                   --note-judge-mapping "${RL_JUDGE_MAPPING:?note claude-code: no default}"
                                   --note-judge-claude-model "${RL_JUDGE_CLAUDE_MODEL:?note claude-code: no default}"
                                   --note-judge-claude-effort "${RL_JUDGE_CLAUDE_EFFORT:?note claude-code: no default}") ;;
              codex) [ -z "${RL_JUDGE_ENV_FILE:-}" ] || {
                         echo "RL_JUDGE_ENV_FILE is for RL_JUDGE_BACKEND=api (codex uses the ChatGPT login)" >&2; exit 1; }
                     # the judge's codex: this shell's binary, resolved now (configure records it and its --version);
                     # the login is the user's own ~/.codex (checked to exist, never read)
                     judge_codex=$(command -v codex) || { echo "note codex: no codex on this shell's PATH" >&2; exit 1; }
                     judge_codex=$(readlink -f "$judge_codex")
                     [ -f "$HOME/.codex/auth.json" ] || {
                         echo "note codex: $HOME/.codex/auth.json missing (codex login first)" >&2; exit 1; }
                     reward_args+=(--note-judge-codex-bin "$judge_codex") ;;
              *) echo "RL_JUDGE_BACKEND $judge_backend: api, codex, ensemble or claude-code" >&2; exit 1 ;;
          esac ;;
    score) reward_args=() ;;
    *) echo "RL_ARC3_REWARD $reward: c0, eff, official, rank, hinge, note or score" >&2; exit 1 ;;
esac
retire_args=()
case "${RL_RETIRE_FULL_GROUPS:-0}" in
    1) retire_args=(--retire-full-groups) ;;
    0) ;;
    *) echo "RL_RETIRE_FULL_GROUPS ${RL_RETIRE_FULL_GROUPS}: 0 or 1" >&2; exit 1 ;;
esac
topology=${RL_TOPOLOGY:-tp1x4}
# replica count and GPU count of the topology (configure_b200.TOPOLOGIES)
read -r n_replicas n_gpus < <("$py" - "$topology" <<'PY'
import sys
sys.path.insert(0, "work/qwen38fn_rl")
from configure_b200 import TOPOLOGIES
t = TOPOLOGIES[sys.argv[1]]
print(len(t["replica_gpus"]), len({g for gs in t["replica_gpus"] + [t["trainer_gpus"]] for g in gs.split(",")}))
PY
)
[ -n "${n_gpus:-}" ] || { echo "RL_TOPOLOGY $topology: not a configure_b200.py topology" >&2; exit 1; }
gpus=${RL_GPUS:-$(seq -s, 0 $((n_gpus - 1)))}
drafter_args=()
[ -z "${RL_DFLASH_DIR:-}" ] || drafter_args=(--dflash-dir "$RL_DFLASH_DIR")
prepare_media=${RL_PREPARE_MEDIA:-cpu-exact}
prepare_args=(--prepare-media-device "$prepare_media")
case "$prepare_media" in
    cuda) prepare_args+=(--prepare-gpu "${RL_PREPARE_GPU:-${gpus##*,}}") ;;
    *) [ -z "${RL_PREPARE_GPU:-}" ] || { echo "RL_PREPARE_GPU needs RL_PREPARE_MEDIA=cuda" >&2; exit 1; } ;;
esac

context_args=(--context-token-limit "${RL_CONTEXT_LIMIT:-140000}")
[ "${RL_CONTEXT_LIMIT:-}" != none ] || context_args=()

if [ ! -f "$RL_RUN/config.json" ]; then
    cmd=("$py" work/qwen38fn_rl/configure_b200.py --run-root "$RL_RUN"
         --template "${RL_TEMPLATE:-work/qwen38fn_rl/configs/arc3-round2-int3qat.json}" --topology "$topology" --model-kind int3
         --gpus "$gpus" "${prepare_args[@]}"
         --fork "$RL_FORK_TREE" --dataset arc3_levels --tasks-dir "$tasks"
         --train-ids-file "${RL_TRAIN_IDS:-$tasks/lists/train.txt}" --heldout-ids-file "${RL_HELDOUT_IDS:-$tasks/lists/heldout.txt}"
         --sessions-per-replica "$sessions" --group-size "$gs" --rollout-slots $((n_replicas * sessions))
         --ramp-seconds "${RL_RAMP_SECONDS:-1200}" --trainer-memory-fraction "${RL_TRAINER_MEMORY_FRACTION:-0.92}"
         --output-token-budget "${RL_OUTPUT_BUDGET:-90000}" "${context_args[@]}"
         --memfrac "${RL_MEMFRAC:-0.96}" --mamba-slots "$mamba_slots" --optimizer-steps "${RL_STEPS:-100}" --save-every-steps "${RL_SAVE_EVERY:-10}"
         --eval-every-steps "${RL_EVAL_EVERY:-${RL_SAVE_EVERY:-10}}" --global-optimizer-batch "${RL_BATCH:-256}"
         --serve-profile "$profile" "${drafter_args[@]}" --group-ready
         --rollout-temperature "${RL_ROLLOUT_TEMPERATURE:-1.0}" --lr "${RL_LR:-2e-4}" --lr-warmup-steps "${RL_LR_WARMUP_STEPS:-0}"
         --rollout-top-p "${RL_ROLLOUT_TOP_P:-1.0}" --rollout-top-k "${RL_ROLLOUT_TOP_K:--1}"
         ${RL_AGENT_RUNTIME:+--agent-runtime "$RL_AGENT_RUNTIME"}
         --adam-beta2 0.95 --adam-eps 1e-12 --adv-normalization none --loss-reduction "${RL_LOSS_REDUCTION:-prompt}"
         --trainer-fast-kernels "${RL_TRAINER_FAST_KERNELS:-all}" --trainer-dense-qat-gemm "${RL_TRAINER_DENSE_QAT_GEMM:-triton}"
         --trainer-qsa-indexer "${RL_TRAINER_QSA_INDEXER:-fp32}"
         ${RL_TRAINER_OFFLOAD_KEEP_GIB:+--trainer-offload-keep-gib "$RL_TRAINER_OFFLOAD_KEEP_GIB"}
         ${RL_TRAINER_TRAINABLE:+--trainer-trainable "$RL_TRAINER_TRAINABLE"}
         ${RL_TRAINER_CHECKPOINT_STORE:+--trainer-checkpoint-store "$RL_TRAINER_CHECKPOINT_STORE"}
         ${RL_TRAINER_KEEP_CHECKPOINTS:+--trainer-keep-checkpoints "$RL_TRAINER_KEEP_CHECKPOINTS"}
         ${RL_TRAINER_FREEZE_FIRST_LAYERS:+--trainer-freeze-first-layers "$RL_TRAINER_FREEZE_FIRST_LAYERS"}
         --algo "${RL_ALGO:-mis}"
         --arc3-reward "$reward" "${reward_args[@]}" "${retire_args[@]}"
         --min-free-gb "${RL_MIN_FREE_GB:-80}" --hard-min-free-gb "${RL_HARD_MIN_FREE_GB:-60}"
         --prepare-max-gb "${RL_PREPARE_MAX_GB:-60}" --prepare-reserve-gb "${RL_PREPARE_RESERVE_GB:-20}"
         --tokenizer-dir "$RL_SERVE_MODEL" --trainer-base-path "$RL_TRAIN_MODEL"
         ${RL_INIT_CHECKPOINT:+--init-checkpoint "$RL_INIT_CHECKPOINT"} ${RL_INIT_MERGED:+--init-merged "$RL_INIT_MERGED"}
         ${RL_EFFICIENCY_FREE_FRACTION:+--efficiency-free-fraction "$RL_EFFICIENCY_FREE_FRACTION"}
         ${RL_ACTION_TOKEN_CHARGE:+--action-token-charge "$RL_ACTION_TOKEN_CHARGE"}
         ${RL_TRAIN_OUTPUT_BUDGET:+--train-output-token-budget "$RL_TRAIN_OUTPUT_BUDGET"}
         ${RL_SEED_NOTE_DONE:+--seed-note-done}
         ${RL_PACKET_LEVEL_DOCS:+--packet-level-docs "$RL_PACKET_LEVEL_DOCS"}
         ${RL_PACE_LINE:+--pace-line} ${RL_WINDOW_CHANGE:+--window-change}
         ${RL_STOP_AFTER_FIRST_WINDOW:+--stop-after-first-window}
         ${RL_ACTION_CHARGE_FREE_FRACTION:+--action-charge-free-fraction "$RL_ACTION_CHARGE_FREE_FRACTION"})
    if [ "$dry" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
fi
if [ "$dry" = 1 ] && [ ! -f "$RL_RUN/plan.json" ]; then
    echo "(dry run: $RL_RUN/plan.json does not exist yet; run the configure command above, then repeat for the stages)"
    exit 0
fi
# Anti-cheat controls (no model, no GPU) in the jail the episodes use: every launch, a failure stops here.
if [ -n "$stages" ]; then
    # the agent python the orchestrator will run (config.json harness.agent_python, expanded in this shell's env.sh)
    agent_py=$("$py" -c 'import json, os, sys; print(os.path.expandvars(json.load(open(sys.argv[1]))["harness"]["agent_python"]))' \
               "$RL_RUN/config.json")
    case "$agent_py" in *'$'*) echo "config.json harness.agent_python $agent_py does not expand" >&2; exit 1 ;; esac
    selftest=(env PYTHONPATH="$PWD/src" "$agent_py" work/arc_code/run_arc3_level.py --selftest
              --config "$RL_RUN/harness-train.json" --run-dir "$RL_RUN/selftest/$(date +%Y%m%d-%H%M%S)"
              --agent-python "$agent_py" --driver-bin "$ARC_CODEX_P8_PREFIX")
    if [ "$dry" = 1 ]; then printf '%q ' "${selftest[@]}"; echo; else "${selftest[@]}"; fi
fi
for sub in replicas batches; do
    if [ -L "$RL_RUN/$sub" ]; then
        [ "$(readlink "$RL_RUN/$sub")" = "$RL_TMPFS/$sub" ] || {
            echo "$RL_RUN/$sub points to $(readlink "$RL_RUN/$sub"), not $RL_TMPFS/$sub" >&2; exit 1; }
    elif [ -e "$RL_RUN/$sub" ]; then
        echo "$RL_RUN/$sub exists and is not the tmpfs symlink" >&2; exit 1
    elif [ "$dry" = 1 ]; then
        echo "mkdir -p $RL_TMPFS/$sub && ln -s $RL_TMPFS/$sub $RL_RUN/$sub"
    else
        mkdir -p "$RL_TMPFS/$sub"
        ln -s "$RL_TMPFS/$sub" "$RL_RUN/$sub"
    fi
done
for stage in $stages; do
    echo "=== stage $stage"
    bash work/qwen38fn_rl/launch_b200_stream.sh "$stage"
done
case " $stages " in
    *" orchestrator "*)
        pack_unit=${RL_UNIT_PREFIX:-rl-b200-$(basename "$RL_RUN")}-group-pack
        pack=(systemd-run --user --unit="$pack_unit" --collect
              --working-directory="$PWD" -p Nice=10 -p StandardOutput=append:"$RL_RUN/group_pack.log"
              -p StandardError=append:"$RL_RUN/group_pack.log"
              "$py" work/qwen38fn_rl/group_pack.py --run-root "$RL_RUN" --archive-dir "$RL_TMPFS/groups-packed")
        if [ "$dry" = 1 ]; then
            printf '%q ' "${pack[@]}"; echo
        elif systemctl --user is-active --quiet "$pack_unit"; then
            # it outlives the orchestrator: an orchestrator-only restart (RL_STAGES=orchestrator) keeps the running one
            echo "· $pack_unit already runs: kept"
        else
            "${pack[@]}"
        fi ;;
esac
