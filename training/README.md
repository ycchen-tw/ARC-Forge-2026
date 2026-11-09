# Training — how the weights of the finals were produced

All four finals serve one base checkpoint (QAD R) and one DFlash drafter; two of them add an RL
weight delta. Every step below ran on two 8×B200 nodes, `node1` and `node2` in the run records and
`TRAINING_MAP.json`, with the sglang fork for serving and capture.

```
Qwen3.8-Flash-Next (official FP8 / BF16; non-expert BF16 tensors via the NVIDIA NVFP4 checkpoint)
 ├─ A1  calibration ledgers: the FP8 512-expert model running the codex ARC harnesses
 ├─ A2  gEAN expert saliency on those ledgers → 512 → 320 experts per layer
 ├─ A3  prune
 ├─ A4  INT3 GPTQ of the 48 × 320 routed experts (one shot)
 ├─ A5  deploy package                                   = Kaggle qwen38fn-gean320-int3-gptq v1
 └─ B1–B6  QAD R: student trajectories, reverse KL to the FP8 teacher's top-32
                                                         = Kaggle qwen38fn-gean320-int3-gptq v3  (all four finals)
      ├─ C1–C3  ARC-AGI-2 RL r4, step 30 → weight patch  = Kaggle qwen38fn-int3-arc2r4-v30-patch   (57848658)
      └─ D1–D4  ARC-AGI-3 RL v8, step 20 → delta         = Kaggle qwen38fn-int3-arc3v8-v20-delta   (57846476)
 E1–E6  DFlash drafter: official MTP init → D line → codex1015 extension → plan B → 8 layers, step 25,562
                                                         = Kaggle qwen38fn-dflash-planb-l8-s25562-fp8 (all four finals)
 F1–F5  training-data producers: codex trace ledgers, Nemotron-prompt raw tokens, NVARC teacher
        rollouts, agentic distillation segments, ARC-AGI-2 task pools
```

Training data (task pools, ARC-AGI-3 RL seeds, the QAD set, drafter corpora and plans):
[huggingface.co/datasets/ycchen/arc-2026-training-data](https://huggingface.co/datasets/ycchen/arc-2026-training-data).
The gEAN expert masks are in [`masks/`](masks/).

[`TRAINING_MAP.json`](TRAINING_MAP.json) is the step-by-step record: for each step its purpose,
entry points, configs, the commands exactly as written in the original runbooks (with the source
file and line), inputs, outputs, and what is not in this tree.

## Steps

| Step | What | Main entry points |
|---|---|---|
| A1 | Collect TITO session ledgers (exact token ids, media, MoE routing) from the FP8 model playing ARC-AGI-2 pools and the 25 public ARC-AGI-3 games | `work/arc_code/arc2/run_arc2.py`, `work/arc_code/run_arm.py`, `src/arcforge/tito/` |
| A2 | Replay ledgers through a capture server, accumulate per-expert saliency, select the gEAN 320-expert mask | `work/qwen38fn_dflash/reap_*.py`, `work/qwen38fn_eval/reap_mm_select.py` |
| A3 | Prune the 512-expert checkpoint with the mask | `work/qwen38fn_eval/prune_experts.py` |
| A4 | INT3 GPTQ (Humming uint3, group 128) with Hessians from 50 replayed ARC-AGI-2 segments (2.0 M tokens, 389 images) | `work/qwen38fn_eval/run_gean_int3_mm.py`, `quantize_int3_gptq.py`, `gptq_int3.py` |
| A5 | Package INT3 experts + block-FP8 dense + BF16 rest into the deploy checkpoint (18 × 5 GiB shards, no MTP) | `work/qwen38fn_eval/build_int3_deploy.py`, `export_int3_sidecar.py` |
| B1 | Student base with BF16 dense masters | `work/qwen38fn_eval/build_int3_bf16dense.py` |
| B2 | Student trajectories: the INT3 model on ARC-AGI-2 pools (6 prompt variants) and ARC-AGI-3 games (10 arms) | `work/qwen38fn_qad/probes/student_traj_1022/collect/` |
| B3 | Filter degenerate segments, merge into `qadmix-1019-1022` (20,200 segments, 1.15 B tokens) | `work/qwen38fn_qad/probes/student_traj_1022/filter/merge_filter.py` |
| B4 | Teacher targets: FP8 512-expert top-32 logprobs + tail mass at every supervised position | `work/qwen38fn_qad/train_qad.sh --mode teacher` |
| B5 | QAD R: 1,000 steps of reverse-KL distillation training INT3 group scales, router, dense and shared weights | `work/qwen38fn_qad/probes/student_traj_1022/train/qadmix_chain.sh`, `train_qad.py` |
| B6 | Materialize QAD R as a servable package | `work/qwen38fn_qad/materialize_package.py` |
| C1 | ARC-AGI-2 RL task pool (60 eval120fix + 30 audited hard tasks) | `work/qwen38fn_rl/probes/arc2_ctx64k_20261029/build_pool.py` |
| C2 | RL r4: LoRA Dr.GRPO from QAD R, 24 groups × 24 episodes per step, all-tests reward, 40k output / 64k context | `work/qwen38fn_rl/launch_arc2_largegrid_int3qat.sh`, `orchestrator.py`, `work/qwen38fn_lora/train_rl_service.py` |
| C3 | Step 30 → weight patch of the 516 changed tensors | `work/qwen38fn_rl/rl_overlay_package.py`, `rl_weight_patch.py` |
| D1 | Collect gen-2 level seeds (workspace snapshots at level changes) on the public games | `work/arc_code/gen2/seed_collect/` |
| D2 | Build the v8 seed set (18 games, 235 seeds) | `work/qwen38fn_rl/probes/arc3v8_20261029/seeds/build_v8_dir.py` |
| D3 | RL v8: dense full fine-tune from QAD R, episodes from a level seed to the first window change, reward from clears and a codex note judge | `work/qwen38fn_rl/probes/arc3v8_20261029/launch-v8.sh`, `orchestrator.py`, `arc3_note_judge.py` |
| D4 | Step 20 → one-file delta of the 516 changed tensors | `work/qwen38fn_lora/merged_export.py` |
| E1 | Drafter init from the official MTP layer and target layers 35/39/43/47 | `work/qwen38fn_dflash/probes/native_mtp_dense/` |
| E2 | Data plans over the trajectory corpora | `work/qwen38fn_dflash/make_data_plan.py`, `data_plan.py` |
| E3 | D-line training on INT3 captures (34,404 steps), extension on codex1015 (9,425 steps) | `work/qwen38fn_dflash/probes/native_mtp_dense/launch_dline_int3.sh`, `pipeline.sh`, `train.py` |
| E4 | Plan B: convert to plain residual + target-KV injection, train 8,251 steps | `work/qwen38fn_dflash/convert_plain.py` |
| E5 | Expand to 8 layers, train to step 25,562 on 2.25 B anchors | `work/qwen38fn_dflash/expand_layers.py` |
| E6 | Export and convert to the FP8 deploy format | `work/qwen38fn_dflash/export_trained.py`, `make_dense_deploy.py` |
| F1 | Codex trace ledgers on FP8 + DFlash (codex1015) | `work/tito/collect/run_20261015/` |
| F2 | Nemotron-v2 prompts → raw-token corpus (300,010 rows) | `work/qwen38fn_specdata/` |
| F3 | Teacher rollouts on NVARC tasks (CoT : TIR = 1 : 3) | `work/qwen38fn_distill/` |
| F4 | Agentic distillation segments on all 103k NVARC tasks | `work/agentic_distill/`, `work/agentic_tir/` |
| F5 | ARC-AGI-2 task-pool build and validation | `work/seed_bank/` |

## Environment

- `source env/env.sh` after creating `env/local.sh` from `env/local.sh.example`, which lists every
  machine location the scripts read (data, models, run roots, the sglang fork and its venv, the
  codex builds, the ARC-AGI-3 corpus and agent runtime). `$ARC_NFS` in commands and records is the
  shared NFS archive that mirrors runs/, data/ and models/.
- sglang: [ycchen-tw/sglang-arc-2026](https://github.com/ycchen-tw/sglang-arc-2026). Captures and
  calibration used tags `train-tito-collect`, `train-gean-reap`, `train-int3-gptq`,
  `train-rl-int3`; RL rollouts used `rl-rollout-node1` (ARC-AGI-2 r4) and `rl-rollout-node2`
  (ARC-AGI-3 v8). `deploy/sglang.lock.json` lists every fork target with its tree digest.
- Python packages of the B200 environments: `env/b200-train.freeze.txt` (trainers),
  `env/b200-serve.freeze.txt` (serving), `env/b200-venv-torch.freeze.txt`, and
  `env/b200-train.not-in-lock.txt` (git-pinned and source-built packages).

## Run records

[`run_records/`](run_records/) holds the launch-side files of the actual runs, one directory per step.
Each has a `README.md` (run, host, UTC dates, the main command line, what was not recorded), `launch/`
(scripts, units, journal `Started` lines, command lines extracted from logs), `config/` (resolved configs,
plans, manifests), `lists/` (task, seed and batch lists) and `code/` (run-dir scripts; trainer files that
differ from the released `work/qwen38fn_lora`, as run). [`PROVENANCE.tsv`](run_records/PROVENANCE.tsv) maps
each file to its source path, sha256, size, mtime and whether it was edited (text translated to English,
or machine names and personal paths replaced by `node1` / `node2` and the `$ARC_*` roots).

## Code versions

This tree is arc-forge `main` at `dc8cccf40`. The runs used earlier revisions of the same files;
where that matters (the r4 harness, the v8 branch fixes, the QAD worktree) `TRAINING_MAP.json`
says so under `gaps`.
