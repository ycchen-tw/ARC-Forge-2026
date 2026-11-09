# ARC Prize 2026 — code for the final submissions (ARC-AGI-2 and ARC-AGI-3)

Both tracks run the same model and the same agent:

- **Model** — Qwen3.8-Flash-Next with the routed experts pruned from 512 to 320 per layer
  (gEAN saliency), experts quantized to INT3 (GPTQ, Humming uint3 g128 format), then
  quantization-aware distillation against the 512-expert FP8 teacher on the student's own
  trajectories ("QAD R"). One final per track adds an RL weight delta on top.
- **Serving** — an sglang v0.5.20 fork ([ycchen-tw/sglang-arc-2026](https://github.com/ycchen-tw/sglang-arc-2026),
  tag `kaggle-final`) with a DFlash speculative drafter (8 layers, verify width K=4) and
  Humming INT3 MoE kernels, fully offline inside Kaggle; the 4×L4 stack adds HiSparse KV
  offload to host memory and host-mapped collectives.
- **Agent** — OpenAI's codex CLI 0.154.0 with our patches, driving the model through a
  tool-using workspace: ARC-AGI-2 tasks are solved by writing and checking programs;
  ARC-AGI-3 games are played through an `act` command line.

ARC-AGI-2 runs on Kaggle's 4×L4 machine (128 concurrent sessions); ARC-AGI-3 runs on one
RTX PRO 6000 (110 concurrent games).

## The four final submissions

| Track | Submission | Notebook (Kaggle kernel @ version) | Weights | Code in this repo |
|---|---|---|---|---|
| ARC-AGI-2 | 57747104 | [qwen38fn-l4x4-c128-ext300-arcv6 @362595216](https://www.kaggle.com/code/threerabbits/qwen38fn-l4x4-c128-ext300-arcv6?scriptVersionId=362595216) | QAD R | [submissions/agi2-57747104-ext300-arcv6](submissions/agi2-57747104-ext300-arcv6) |
| ARC-AGI-2 | 57848658 | [qwen38fn-l4x4-c128-v10-rl30p8-x39 @364318725](https://www.kaggle.com/code/threerabbits/qwen38fn-l4x4-c128-v10-rl30p8-x39?scriptVersionId=364318725) | QAD R + RL patch `arc2-r4@30` | [submissions/agi2-57848658-x39-rl](submissions/agi2-57848658-x39-rl), [agi2/](agi2) |
| ARC-AGI-3 | 57678190 | [arc-code-hs110-click @362591255](https://www.kaggle.com/code/threerabbits/arc-code-hs110-click?scriptVersionId=362591255) | QAD R | [submissions/agi3-57678190-v35-click](submissions/agi3-57678190-v35-click) |
| ARC-AGI-3 | 57846476 | [arc-code-hs110-v14tk5an-cards-swv8 @364285204](https://www.kaggle.com/code/threerabbits/arc-code-hs110-v14tk5an-cards-swv8?scriptVersionId=364285204) | QAD R, switched in place to RL `v8@20` after 6 h | [submissions/agi3-57846476-v40-cards](submissions/agi3-57846476-v40-cards), [agi3/](agi3) |

## Layout

| Path | What it is |
|---|---|
| [`submissions/`](submissions) | The four notebooks exactly as they ran on Kaggle, their Kaggle input lists, and `extract_code.py`, which unpacks the code each notebook carries and checks it against the notebook's own manifest. |
| [`agi2/`](agi2) | Self-contained snapshot of the ARC-AGI-2 stack behind 57848658: codex harness, L4 serving launcher, notebook builder. |
| [`agi3/`](agi3) | Self-contained snapshot of the ARC-AGI-3 stack behind 57846476: the arc-code arm, reviewers, game cards, notebook builder. |
| [`runtime/`](runtime) | How the offline Kaggle inputs are built: the sm_120 sglang runtime archive, the sglang fork tree, the patched codex binary (patch sets p7 and p8). |
| [`training/`](training) | How the weights are produced: expert pruning, INT3 GPTQ, QAD, the two RL runs, the DFlash drafter, and the data collection behind them; `run_records/` holds the launch scripts, configs and logs of the actual runs. |

Each top-level directory keeps the internal repository layout (`src/arcforge/`, `work/<unit>/`),
so every path pinned in a builder, config, or notebook resolves with that directory as the root.
`work/<unit>` names are internal research-unit names: `qwen38fn` = Qwen3.8-Flash-Next,
`arc_code` = the codex agent harness, `qwen38fn_l4` = the 4×L4 serving stack, `hs110` = the
ARC-AGI-3 configuration with 110 concurrent games.

## Reproducing

**Exact code of any final.** Every notebook embeds the code it ran:

```bash
python submissions/extract_code.py submissions/agi2-57747104-ext300-arcv6/notebook.ipynb /tmp/agi2-arcv6
```

**Rebuild a notebook from source** (Python 3.12 with `numpy` and `httpx`; no GPU, no network):

```bash
cd agi3
python work/arc_code/kaggle/build_notebook.py --out /tmp/v40 \
  --config work/arc_code/kaggle/notebook-reap320-0520-v12-hs110-dense-k8-v40-int3-w3a8-tb24k-planb-l8s25562-allmerge-qadv3-swv8at20-dyn-v512r2-scipy-v14tk5an-review9-cards-click.json

cd agi2
python work/qwen38fn_l4/kaggle/build_notebook.py \
  --preset work/qwen38fn_l4/kaggle/l4opt/kaggle-1101-c128-planb-l8s25562-qadR-dual-idlefix-plenvfp4-ext300-v10-skipagree-rl30p8-x39.json \
  --asset-manifest ../submissions/agi2-57848658-x39-rl/asset-manifest.json \
  --out-dir /tmp/x39 --kernel-id threerabbits/qwen38fn-l4x4-c128-v10-rl30p8-x39 \
  --title qwen38fn-l4x4-c128-v10-rl30p8-x39 --harness-root .
```

Both commands reproduce the submitted notebooks: identical configuration, asset manifest, generated
cells and list of carried files (264 for ARC-AGI-2, 660 for ARC-AGI-3). The carried files in this
repository have English comments, docstrings and operator messages, so they differ from the submitted
ones in that text only (same Python AST once docstrings are blanked); the files exactly as submitted
come out of the notebooks in `submissions/` with `extract_code.py`. 57747104 and 57678190 were built
from earlier revisions of the same code; use `extract_code.py` for them.

**Kaggle inputs.** The notebooks run offline and mount these inputs (all owned by `threerabbits`):

| Input | Used by | Contents | Built by |
|---|---|---|---|
| model `qwen38fn-gean320-int3-gptq` v3 (88.98 GB) | all four | QAD R checkpoint: 320 INT3 routed experts per layer, block-FP8 GDN/attention projections, BF16 rest, FP8 PLE table, no MTP head | [`training/`](training) |
| model `qwen38fn-int3-arc2r4-v30-patch` v1 (3.35 GB) | 57848658 | RL weight patch `arc2-r4@30`: the 516 tensors that differ from v3, swapped in at load time (`ARC_L4_WEIGHT_PATCH`) | [`training/`](training) |
| model `qwen38fn-int3-arc3v8-v20-delta` v1 (3.35 GB) | 57846476 | RL `v8@20`: the 516 changed tensors, loaded in place 21,600 s into the run | [`training/`](training) |
| `qwen38fn-dflash-planb-l8-s25562-fp8` (1.45 GB) | all four | DFlash drafter, 8 layers, step 25,562, FP8 | [`training/`](training) |
| `qwen38fn-sglang-0520-sm120-cp312` v2 (3.50 GB) | all four | Python 3.12 runtime: sglang v0.5.20 rebuilt with `runtime/patches/sglang-0.5.20`, FlashInfer JIT cache, FlashAttention 2, CUDA 13.1 compiler subset | `runtime/work/qwen38fn_kaggle/package_runtime_sm120_0520.sh` |
| `sglang-arc-2-tree-allmerge-1020` (9.4 MB) | ARC-AGI-3 | Python tree of the sglang fork at `kaggle-final` (4,943 files), laid over the runtime | `runtime/work/sglang_fork/pack_kaggle_fork_tree.py` |
| `qwen38fn-l4-v16-arcdensevocabtp4-gean-qad3-l4opt48` (14.5 MB) | ARC-AGI-2 | L4 launcher payload (sha256 `83843f18…`), the fork's source tree at `kaggle-final`, `runtime_setup.py` | `agi2/work/qwen38fn_l4/kaggle/pack_asset.py` |
| `arcforge-codex-0154-p7` | 57747104 | static musl codex 0.154.0 + patches 0001–0007, sglang Responses-API overlay, vision fixtures | `runtime/work/arc_code/build_codex_musl.sh` |
| `arcforge-codex-0154-p8` | other three | static musl codex 0.154.0 + patch set p8 (0001 revised, 0008 added) | same, with `runtime/patches/codex-0.154.0-p8` |
| `humming-kernels-0116` | all four | `humming-kernels==0.1.16` wheel from PyPI, unmodified | `pip download` |
| `arc-code-kaggle-assets` v2 | ARC-AGI-3 | `jerber/arc-code` @6b33c1f with our offline patch, the stock `@openai/codex` 0.154.0 npm tree, `bwrap` 0.6.1 | `agi3/work/arc_agi3/arc_code_qwen/probes/build_kaggle_assets.py` |
| `agent-wheels-scipy-1-18-1` | ARC-AGI-3 | `scipy==1.18.1` cp312 wheel from PyPI | `pip download` |
| `pip-offline-wheel` | ARC-AGI-3 | `pip==26.2.1` wheel | `agi3/work/arc_agi3/arc_code_qwen/probes/build_pip_dataset.py` |
| `hs110-draft-vocab` | ARC-AGI-3 | `draft_vocab_16384.json`: the 16,384 most frequent output token ids of 110 ARC-AGI-3 games (99.92 % coverage), the drafter's output vocabulary | counted from the arm's own rollouts |
| `zstd-x86-64-kaggle` | all four | `zstd` 1.5.5 binary with `liblz4`, `liblzma`, `libz` | prebuilt binaries |

Runtime build details: [`runtime/README.md`](runtime/README.md). Weight and data lineage:
[`training/README.md`](training/README.md). Training data:
[huggingface.co/datasets/ycchen/arc-2026-training-data](https://huggingface.co/datasets/ycchen/arc-2026-training-data).

## Licenses

All code and documentation we wrote in this repository is released under **MIT-0**
([LICENSE](LICENSE)). The patches in `runtime/patches/` are ours (MIT-0); applied, they modify
Apache-2.0 projects. Third-party components and their licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The model weights are derivatives of Qwen3.8-Flash-Next and are distributed under the
Qwen Community License 1.0.
