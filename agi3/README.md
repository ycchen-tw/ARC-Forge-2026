# ARC-AGI-3 track — arc-code arm on one RTX PRO 6000 (submission 57846476)

This tree is the code behind ARC-AGI-3 submission 57846476 (kernel
`arc-code-hs110-v14tk5an-cards-swv8`): every file the notebook carries, plus the builder.
Submission 57678190 (kernel `arc-code-hs110-click`) ran an earlier revision; its exact files are
in its notebook (`../submissions/extract_code.py`), and its notebook config and arm configs are
included here.

## What runs on Kaggle

1. The notebook stages the runtime archive (`qwen38fn-sglang-0520-sm120-cp312`) and lays the
   sglang fork tree (`sglang-arc-2-tree-allmerge-1020`, tag `kaggle-final`) over it, then starts
   one server (TP 1) for the model `qwen38fn-gean320-int3-gptq` v3: context 240,000, 110
   concurrent requests, NVFP4 KV cache, Humming INT3 MoE, the DFlash plan-B drafter (4 draft
   tokens, draft vocabulary from `hs110-draft-vocab`).
2. `work/arc_code/run_arm.py` plays every game with one codex session per game (single pass),
   against the official gateway in the scored rerun and against a stand-in gateway in Save & Run.
3. Idle slots run read-only reviewers on games that have spent at least 45 minutes on their
   current level; each review's `REVIEW.md` is delivered into the player's workspace. In
   57846476 the reviewers also read game cards.
4. 21,600 s into the rerun, `kaggle/notebook/04_main.py` pauses the server and loads the RL delta
   `qwen38fn-int3-arc3v8-v20-delta` in place (`v8@20`), then play continues.

## Layout

| Path | Role |
|---|---|
| `work/arc_code/kaggle/build_notebook.py` | Builds the notebook: carries every file below in one deterministic tarball (`PAYLOAD`) with per-file sha256 (`BUILD["carried"]`). |
| `work/arc_code/kaggle/notebook/0{1,2,3,4}_*.py` | The notebook cells: prelude, staging of inputs, server start-up and checks, the main run (gateway, weight switch, final notice). |
| `work/arc_code/kaggle/notebook-…-v40-….json`, `notebook-…-v35-….json` | Notebook configs of the two finals: model, drafter, server arguments, runtime pins, arm configs. |
| `work/arc_code/configs/` | Arm configs; the finals use `kaggle-competition-…-click(-final).json` (rerun) and `kaggle-saverun-1x90m-…` (Save & Run). |
| `work/arc_code/run_arm.py` | The arm driver: game queue, wall clock, per-game output-token accounting, resumes, notices, reviewer and card scheduling. |
| `work/arc_code/arms/` | How each agent is launched (`codex.py` for the finals); `__init__.py` defines the driver protocol. |
| `work/arc_code/prompts/`, `prompts.py` | Player prompt sets (`v14tk5an` in 57846476) and their assembly. |
| `work/arc_code/review_scheduler.py`, `reviewer.py`, `review_notice.py`, `review_prompts/` | Read-only reviewers: who is reviewed and when, one review session, delivery. |
| `work/arc_code/card_library.py`, `game_cards/` | Game cards: 25 static cards of the public games and live cards written during the run. |
| `work/arc_code/sampling_proxy.py`, `reasoning_trim.py`, `compaction_tail.py`, `act_images.py` | Proxy that writes explicit sampling parameters into every request; context and image helpers. |
| `work/arc_code/gen2/hooks/` | codex hooks for session start, tool use and stop. |
| `work/arc_code/posix_fence.py`, `fence_posix.sh`, `jail.sh` | Per-game isolation (POSIX uids on Kaggle, bubblewrap elsewhere). |
| `src/arcforge/arc_code/` | The game bridge and `act` client, gateway client, ledgers, scoring, middleware notices, review snapshots. |
| `work/qwen38fn_eval/` | Serving helpers carried into the notebook: INT3 deploy runtime and its server-process patch, Humming kernel tuning tables, page-cache prefetch. |
| `work/arc_agi3/` | Builders of the `arc-code-kaggle-assets` and `pip-offline-wheel` datasets, and the offline patch for `jerber/arc-code`. |

## Rebuild the notebook

```bash
python work/arc_code/kaggle/build_notebook.py --out /tmp/v40 \
  --config work/arc_code/kaggle/notebook-reap320-0520-v12-hs110-dense-k8-v40-int3-w3a8-tb24k-planb-l8s25562-allmerge-qadv3-swv8at20-dyn-v512r2-scipy-v14tk5an-review9-cards-click.json
```

Requires Python 3.12 with `numpy`; no GPU or network.
