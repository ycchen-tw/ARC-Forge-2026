# Preparation, launch and analysis scripts for arc3 RL v8 (2026-10-29 onward, node2)

One-off probe, not maintained. The plan is in `docs/plans/arc3_rl_v8_window_note_judge_20261029.md`;
the settings actually run and the results are in the research log. The full-game evaluation scripts are in `../arc3_v8_fullgame/`.

| File | What it does |
|---|---|
| `v8.env`, `launch-v8.sh`, `apply-v8-config.py` | settings and launch (`configure` / `start`). `RL_STAGES=...` starts only some stages. Required: `RL_JUDGE_BACKEND` (api = Opus 5.5 using only key 2; codex = gpt-6-astra), `RL_JUDGE_MODEL`, `RL_JUDGE_EFFORT`, `RL_RUN_DATE` |
| `seeds/build_v8_dir.py` | builds the seed set `$ARC_DATA/arc3_levels/g2seed-v8-20261029` (18 games, 235 ids: 5 / 3 / 2 copies per level per epoch for hard / wall / medium) |
| `smoke.env`, `smoke/` | smoke: same topology, window 100k / window-change point 50k, 4 ids × 16 games, 2 steps |
| `build_judge_set.py`, `validate_judge.py` | offline judge validation: takes the first window's handoff messages from the v7 archive and grades them per group; `--fakes` mixes in three kinds of fake notes |
| `train_trend.py` | training trend (read-only): per-step R, level-clear rate (hard / medium), judge grade distribution, actions of window-changing games, tokens per action, kl / entropy / grad norm; before/after pairing on the same id |

Resume (from checkpoint-N) follows the stage order of `launch-v8.sh`: servers → control → `/update_weights_from_disk` loads merged/v&lt;N&gt; →
start the trainer with `RL_TRAINER_RESUME=<run>/trainer/checkpoint-N` (the resume gate re-exports the restored master and compares sha256 tensor by tensor against merged/v&lt;N&gt;, stopping on any difference) →
start the orchestrator and routing-gc with `RL_ORCH_ARGS=--abandon-incomplete-round`.
