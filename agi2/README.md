# ARC-AGI-2 track — 4×L4 stack (submission 57848658)

This tree is the code behind ARC-AGI-2 submission 57848658 (kernel
`qwen38fn-l4x4-c128-v10-rl30p8-x39`): the codex harness carried in the notebook, the L4 serving
launcher shipped in the payload dataset, and the builder that assembles the notebook. Submission
57747104 (kernel `qwen38fn-l4x4-c128-ext300-arcv6`) ran an earlier revision of the same harness;
its exact files are in its notebook (`../submissions/extract_code.py`), and its preset is
included here.

## What runs on Kaggle

1. The notebook stages the runtime archive (`qwen38fn-sglang-0520-sm120-cp312`), the L4 payload
   and the sglang fork's source tree (`qwen38fn-l4-v16-…-l4opt48`, unpacked by
   `work/qwen38fn_l4/kaggle/runtime_setup.py`), and the model (`qwen38fn-gean320-int3-gptq` v3;
   57848658 also applies the RL weight patch `qwen38fn-int3-arc2r4-v30-patch` at load time).
2. It starts sglang across the four L4s with the settings recorded in the notebook's `CONFIG`:
   128 concurrent requests, context 179,968 (153,600 for 57747104), HiSparse KV offload with 3,392
   hot blocks, Humming INT3 MoE with FP8 activations, the DFlash plan-B drafter at K=4 with
   a 16k draft vocabulary, host-mapped collectives (`hostcomm`), and the runtime hooks in
   `work/qwen38fn_l4/kaggle/v16_patch/sitecustomize.py`.
3. It runs `work/arc_code/arc2/kaggle_run.py`, which solves every task with codex sessions and
   writes `/kaggle/working/submission.json` before the 700-minute deadline.

## Layout

| Path | Role |
|---|---|
| `work/qwen38fn_l4/kaggle/build_notebook.py` | Builds the notebook: embeds the harness bundle, the asset manifest, `CONFIG`, and the runtime cell. |
| `work/qwen38fn_l4/kaggle/l4opt/` | The two submission presets (`…-ext300-arcv6.json` → 57747104, `…-rl30p8-x39.json` → 57848658). |
| `work/qwen38fn_l4/kaggle/pack_asset.py`, `runtime_setup.py`, `stage_fork_candidate.py` | Build the payload dataset and stage the sglang fork tree into it. |
| `work/qwen38fn_l4/kaggle/v16_patch/`, `v16_gates.py`, `hc_inject/` | Code loaded into the server process on Kaggle: RL weight patch, private host pools, hostcomm injection; start-up gates. |
| `work/qwen38fn_l4/kaggle/*.json` | Pins: drafter dataset, model identity (shard sizes and sha256), draft/target vocabulary subsets. |
| `work/qwen38fn_l4/probes/` | The L4 launcher payload: server launch and runtime (`runpod_*.py`), the host-memory collectives hook and its CUDA source (`hostcomm/`), skinny GEMM kernels (`skinny_gemm/`). |
| `work/qwen38fn_l4/kaggle/frozen_arc2/`, `reference/` | The earlier agentic-TIR harness and its original notebook. The builder and payload manifest pin their sha256, so they are kept verbatim; the finals run the codex harness (`"harness": "codex"` in the preset). |
| `work/arc_code/arc2/kaggle_run.py` | Kaggle entry point: verifies the codex binary, closes `/kaggle` to the agents, runs the scheduler, writes the submission. |
| `work/arc_code/arc2/run_arc2_seq.py` | Per-task scheduler: parallel solver sessions (`s0`, `s2`, extra solvers), refiners that pick the two attempts, time and token budgets. |
| `work/arc_code/arc2/run_arc2.py` | One codex episode: workspace, budget, resume, answer harvest. |
| `work/arc_code/arc2/model_proxy.py` | Per-episode proxy between codex and sglang: exact output-token budget, accounting, DP routing, wrap-up, image re-attachment after compaction. |
| `work/arc_code/arc2/arc_v6.py`, `arc_v10.py`, `*_TOOLS.md`, `check.py` | The workspace toolkit the agent imports, its reference card, and the train-pair checker. |
| `work/arc_code/arc2/prompts/`, `configs/`, `priors/` | Prompt sets, harness configs, and the scheduler's difficulty priors. |
| `work/arc_code/arc2/fence_arc2.sh`, `netns_inner.sh`, `sandbox_runner.py`, `harvest_sandbox.py` | Per-task isolation: one uid per task, no network, answers unreadable. |
| `work/arc_code/arms/`, `prompts.py`, `posix_fence.py`, `fence_posix.sh`, `jail.sh` | The codex driver and fences shared with the ARC-AGI-3 arm. |
| `src/arcforge/` | Shared package: `arc_code` (bridge, ledgers), `tir` (sandbox, submission validation), sglang tree/lock digests. |

## Rebuild the notebook

```bash
python work/qwen38fn_l4/kaggle/build_notebook.py \
  --preset work/qwen38fn_l4/kaggle/l4opt/kaggle-1101-c128-planb-l8s25562-qadR-dual-idlefix-plenvfp4-ext300-v10-skipagree-rl30p8-x39.json \
  --asset-manifest ../submissions/agi2-57848658-x39-rl/asset-manifest.json \
  --out-dir /tmp/x39 --kernel-id threerabbits/qwen38fn-l4x4-c128-v10-rl30p8-x39 \
  --title qwen38fn-l4x4-c128-v10-rl30p8-x39 --harness-root .
```

The asset manifest names the Kaggle inputs and pins their sha256; to rebuild the payload
dataset itself, run `pack_asset.py` (see its `--help`) against a checkout of the sglang fork at
tag `kaggle-final`.
