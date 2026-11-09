---
name: agentic_distill
area: [teacher, corpus]
status: active
machines: [B200]
headline: "Collect a token-in/token-out distillation corpus from Qwen3.8-Flash-Next with agentic_tir's compact v2 harness (50k context, rolling compaction) on all 103k NVARC tasks (training 47,337 + full 55,886): the full messages before each compaction are saved as one segment, and the raw tokens are concatenated directly into parquet with the same schema as qwen38fn_distill; each task deterministically draws effort / ctx_cap / K / fold fraction / gen budget. Pilot: recovers 68.0% of the 100 tasks naive did not solve"
---

# agentic_distill

[`agentic_tir`](../agentic_tir/)'s compact v2 (ARC-AGI-2 120 tasks 91.94%, peak context 42k) is used as the teacher harness
to collect a distillation corpus on all 103,223 NVARC tasks: `nvarc_training` (47,337; both seeds are ARC-AGI-2 training tasks, eval-clean) and `nvarc_full`
(55,886; **parent1 is the ARC-AGI-2 public eval 120 tasks, every single one, so it contaminates this project's evaluation** -- the user decided on 2026-09-03 to ignore the contamination for now and collect everything). On 09-03 we first ran 5,501 training tasks (the hard ones naive did not solve);
from 09-04 the **two splits are mixed and shuffled together** (`lists/mix_{a,b}.txt`, excluding completed tasks); each task's `source` is
recorded in `tasks/<uid>.json` and in the parquet `source` column. `nvarc_full` is downloaded from Kaggle
`sorokin/nvarc-synthetic-puzzles` (no login needed) and packed in the same format as `$ARC_DATA/nvarc_synthetic/nvarc_full.jsonl.zst` (132 MB).
**Scores on ARC-AGI-2 public eval of a student trained on `nvarc_full` are not trustworthy**; tag them separately (parquet `run` column = `nvfull-*`). The goal is for the student to learn the protocol of "continuing
within a 50k context from sandbox state + `notes` + harness notices", not just the teacher's long chains. Plan and trade-offs in
`docs/plans/agentic_nvarc_distill_plan.md`.

## Relation to the naive corpus ([`qwen38fn_distill`](../qwen38fn_distill/))

- **Same uid, same pairs**: `agentic_tir.choose_pairs` matches `rollout_runner.choose_pairs` exactly (`sha256(uid:attempt)` picks 4 train + 1 test;
  verified identical on 12,000 cases), so every task can be paired with its naive record (solve rate, tokens).
- **Same schema**: the parquet columns from `render_segments.py` have the same names and types as `OUT_SCHEMA` (`ids` + `span_{start,end,role}`), and the role sequence is the same as naive's
  `0 (1 2)* 1?` (the cue is folded into the role-2 span); downstream `qwen35_9b_sft/convert_distill_shards.py` relaxes only one rule (the `<|im_end|>` appended to a truncated turn sits at the start of the next span);
  300 tasks / 2,703 rows converted in practice. The extra columns (`episode` / `segment_idx` / `ctx_cap` / `task_correct` / `ep_peak_total` ...) are listed below.
- **Same token-in / token-out**: `--backend tokens` ([`agentic_tir/token_backend.py`](../agentic_tir/token_backend.py)) renders + tokenizes itself,
  calls `/generate` with `input_ids`, and appends `output_ids` to the context as-is; seam rules are the same as `rollout_runner.py` (G stops at `<|im_end|>`, the next message starts with `\n`,
  the cue is `<|im_start|>assistant\n<think>\n`). Each message's tokens are stored in `_ids` and travel with the segments when written; `render_segments.py` concatenates them directly without re-tokenizing.
  `retok_ok` here is an informational column, "does re-render + re-tokenize equal the raw tokens" (in the audit only 39% of 2,703 segments matched; all differences are in the
  BPE split of assistant text and the `</think>` of truncated turns -- exactly why we take the token path; zero differences in prompt / tool spans); `prompt_match` holds by construction.
  Tool definitions are rendered in the shape of sglang `Tool.model_dump()` (extra `strict: false` / `defer_loading: null`), so the training prompt matches token-for-token what the student
  sees later when evaluated through the chat API (the chat API smoke test differed by 11 tokens precisely because of this).
- The chat API path for comparison (`BACKEND=openai`) is kept: the 200-task pilot ran on it.

## What a segment is

When the context fills, the compact harness folds the oldest tool rounds, inserts a notice, and continues. **The full messages the model saw before each fold are saved as one segment**
(`--save-segments` → `segments/<uid>.jsonl.gz`); the last segment is the messages at episode end. Within a segment messages are append-only,
so one segment is one multi-turn training row: prompt / harness messages role 0, assistant (reasoning + content + tool call) role 1, tool output role 2.
A folded segment starts from `[system, task, notice, the most recent rounds kept...]` -- this is the "folded context" on which the student learns to continue.

## Per-task hyperparameters (`agentic_tir.HPARAM_MENU`, each axis hashed independently)

| Axis | Values (weights) |
|---|---|
| reasoning_effort | xhigh .6 / medium .3 / low .1 |
| ctx_cap | 50k .6 / 40k .2 / 65k .2 |
| n_explorers | 2 .7 / 3 .3 (base 2: the 3rd only opens when the first two disagree) |
| compact_target_frac | 0.6 .6 / 0.5 .2 / 0.7 .2 |
| compact_gen_budget | 100k .7 / 150k .3 |

Everything else fixed: `--compact --max-hops 1 --max-turns 200`; checkpoint / rich / structured notes / delegate are off (all within MDE; the fewer protocols the better).
The drawn values are stored in `hparams` of `tasks/<uid>.json` and become per-row columns in the parquet after rendering.

## How to run

```bash
source env/env.sh
# Lists: mix = training + full mixed shuffle (excluding the 5,501 tasks already done in train-a/b); train = naive-unsolved first; nvfull = random; all interleaved a/b
#   $ARC_RUNS/agentic_distill/lists/{pilot,train,nvfull,mix}_{a,b}.txt, naive_qwen38fn.json, source_uids.json (uid → split)
# Two servers: GPU 0-3 :30031, GPU 4-7 :30032 (serve_qwen38fn.sh, CTXLEN=131072; preflight's two KV-size checks are false FAILs from a hardcoded 262k)
TAG=pilot SHARD=a PORT=30031 LIVE=96 TLIVE=48 bash work/agentic_distill/run_shard.sh
# Full run: the watchdog starts two servers + two shards, TAGS defaults to "mix" (both corpora fed to the harness together), restarts dead ones with --resume (PID files only, no pkill -f; stays out of the session cgroup)
LIVE=160 TLIVE=80 setsid nohup bash work/agentic_distill/supervise.sh > $ARC_RUNS/agentic_distill/supervisor.nohup 2>&1 < /dev/null &
python work/agentic_distill/summarize.py $ARC_RUNS/agentic_distill/{train,mix}-{a,b} --list-total 103223   # stratified by source
# render → parquet
$HOME/work/dsv4/venv-qwen4/bin/python work/agentic_distill/render_segments.py \
    --run $ARC_RUNS/agentic_distill/{train,mix}-{a,b} \
    --corpus $HOME/work/dsv4/nvarc_distill/nvarc_training.jsonl.zst $ARC_DATA/nvarc_synthetic/nvarc_full.jsonl.zst \
    --out $ARC_DATA/agentic_nvarc_segments --tokenizer $ARC_MODELS/Qwen3.8-Flash-Next-FP8 --workers 32
```

## Results

**Pilot (chat API, 200 tasks = 100 naive-unsolved + 100 solved, hyperparameters drawn from the menu, 96 parallel per server; 2026-09-03)**

| | Solved (pass@2) | gen tok/task | Notes |
|---|---|---|---|
| All | 84.0% (168/200) | mean 79k / p50 48k / p90 220k | peak ctx 53.9k (a 65k-cap task); only 9 of 613 episodes did not finish answering on their own |
| 100 tasks the naive teacher did not solve | **68.0%** | 125k | naive was 0% on the same tasks (by construction); naive spent only 26k on average before hitting the cap / answering wrong |
| 100 tasks naive solved | 100% | 34k | naive 11k on the same tasks |
| effort low / medium / xhigh | 100% (18) / 74.6% (63) / 86.6% (119) | 54k / 68k / 89k | medium clearly worse, low has too few samples (first 179 train tasks show the same pattern: 94 / 68 / 92%) |
| K=2 / K=3 | 83.7% / 84.5% | 72k / 93k | the 3rd explorer bought no score on synthetic tasks |
| ctx 40k / 50k / 65k | 89.6% / 81.7% / 84.4% | 63k / 88k / 71k | all within noise |

Solved tasks average 61k, unsolved 174k (all three explorers run to the budget); segments 124 KB/task (gz). This batch ran on the chat API and only gives solve rate / cost; it does not go into the corpus.

**Full run (token backend, TAGS="train nvfull", 160 parallel per server, started 11:49)**: the first 179 tasks (all naive-unsolved) 84.9% solved,
66k gen/task; each server has 113–130 in flight, 7–8.7k tok/s, KV 0.20–0.22, mamba 0.4. 131 tasks finished in the first 26 minutes (including warm-up);
steady-state estimate 8–10k tasks/day, about 10–12 days for 103k (the 21k naive-unsolved tasks come first and are slowest). Progress: `summarize.py train-a train-b --list-total 47337`.

## Audit (2026-09-03, first 4,499 tasks)

`agentic_distill_audit_20260903.md`: 0 issues on the collection side; on the training side fixed the render span layout,
the truncated-turn `<|im_end|>` is kept out of the loss, the handoff loop no longer runs tools after STOP (20/14,039 episodes in old data affected; filter with `ep_peak_total > ctx_cap`),
the notice's function list no longer cuts off its head, refiner notes accounting.

## Files

| | |
|---|---|
| [`run_shard.sh`](run_shard.sh) | one shard against one server; all fixed flags live here |
| [`supervise.sh`](supervise.sh) | watchdog: two servers + two shards, restart cap, cooldown, kills only processes from its PID files whose cmdline matches |
| [`render_segments.py`](render_segments.py) | segments → parquet; append-only asserted per message, `prompt_match` against server usage |
| [`summarize.py`](summarize.py) | solve rate (overall / stratified by naive difficulty / per hyperparameter axis), tokens, segments size |
| `agentic_tir.py`'s `--task-source nvarc --corpus --hparam-menu --save-segments --backend tokens` | new harness flags; the token backend is in [`agentic_tir/token_backend.py`](../agentic_tir/token_backend.py) |

## Status (2026-09-05)

The mix batch paused 09-05 00:00 (GPUs handed to NVFP4 evaluation): `train-{a,b}` + `mix-{a,b}` total **18,261 tasks / 56,219 episodes / 1.84B generated tokens** (reasoning 1.58B),
segments 7.6 GB; solve rate 79.4% (nvarc_full 83.1%, nvarc_training 76.9%); mix batch 85.0% solved, 610 tasks/hour, 86k gen/task.
Resume: `LIVE=160 TLIVE=80 setsid nohup bash work/agentic_distill/supervise.sh ...` (the runner carries `--resume` and skips finished tasks).
