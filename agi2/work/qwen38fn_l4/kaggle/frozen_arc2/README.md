# Original ARC2 agentic TIR

The ten files in [ORIGINAL_MANIFEST.json](ORIGINAL_MANIFEST.json) are unchanged
notebook cell bodies (audited notebook SHA `4738fa3a…`), except `work/agentic_tir/agentic_tir.py`
which carries the v15 patch below (its manifest entry records `patched_from_sha256` = the audited original;
`extract_original.py` refuses to overwrite it). Regenerate with
`python extract_original.py --notebook <audited notebook>`; startup checks every hash.

Scoring entry:
`python hisparse_bridge.py --hisparse-sessions <all original agentic_tir arguments>`.
Without the opt-in flag, the original transport runs unchanged. The bridge adds one
SID per original episode, routes each episode to the DP with the fewest live episodes
(the original pins whole tasks, which skewed DP load in Kaggle v6), records native IDs/cache usage, avoids
uncertain retries, and closes only that SID. It uses the original HTTP timeout.
It changes no prompts, explorer/refiner decisions, compaction, budgets or scoring.

Keep the original Config invocation: xhigh/K3/50k context; explorer75k/refiner40k;
task180k/lineage90k/min-session20k/handoff12k; hops3/turns200; report reserve10k/margin2500.
Root supplies the original deadline from total705min minus elapsed startup/reserve25min,
and Save&Run `--limit 8`. Hidden-test rerun omits the validation limit.

The original scorer uses public `test[i].output`; pass a validation challenge copy
annotated with public solutions. Hidden test has no truth and reports no score.
Original artifacts: `args.json`, `tasks/`, `pairs/`, `summary.txt`, `notes/`, optional
`segments/`, `failures.jsonl`, and the specified `--submission-out`. `--limit` truncates
the original submission roster; label that output validation-only. Transport receipts
are separate in `hisparse-transport/`. No Kaggle Submit API is present.

## v15 changes to `agentic_tir.py` (both opt-in; defaults keep the v13 behavior)

- A. `--deadline-rate-adaptive`: every completed model call records completion tokens / wall seconds
  (one hook: `args.chat` is wrapped in `main_async`, outside the bridge transport); `_time_room` uses
  `min(--deadline-tok-per-s, median of the last 200 calls)` once >= 30 samples exist. v14 (C80) measured ~6.2 tok/s per
  stream against the fixed 30, so the last task finished 900 s after the deadline.
- B. `--time-aware-budget`: when a task passes `task_live`, after 1800 s it shrinks that task's budget to
  `clamp(1.8 * 0.9 * rate_agg * time_left / (unstarted + 0.5 * inflight), 40k, task_budget)`; below 150k only 2
  explorers (no E3), below 90k no refiner (`refine_skipped = "budget"`); explorer budget = 0.42 * cap
  (>= min_session_budget) and, when a measured stream rate exists, <= 0.8 * measured * time_left. One
  `TIME_BUDGET tid=... cap=... explorers=... refine=on|off` line per task. Requires `--deadline-min`, `--compact`,
  `--lineage-budget`.
