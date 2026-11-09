# Third-party components

Everything in this repository that we wrote is MIT-0 (see `LICENSE`). The solution also
depends on the components below. Source-form third-party code inside this repository is
listed first; the rest is fetched or mounted at build/run time.

## Model weights

| Component | License | Role |
|---|---|---|
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (official BF16 / FP8) | Qwen Community License 1.0 | Base model. Every checkpoint we publish (pruned INT3 model, QAD R, RL deltas, DFlash drafter) is a derivative and is distributed under the same license; its copyright and license notice must accompany copies. |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | NVIDIA Open Model License | Source of the non-expert BF16 tensors copied into the pruned checkpoint shell (`prune_manifest.json` records it as the pruning source). |

## Code we modify (fork / patches)

| Component | Version | License | Where |
|---|---|---|---|
| [sglang](https://github.com/sgl-project/sglang) | v0.5.20 (release branch commit `94602c9c`) | Apache-2.0 | Fork: [ycchen-tw/sglang-arc-2026](https://github.com/ycchen-tw/sglang-arc-2026); patch set `runtime/patches/sglang-0.5.20/` |
| [OpenAI codex CLI](https://github.com/openai/codex) | rust-v0.154.0 (`6b9826e3`) | Apache-2.0 | Patch sets `runtime/patches/codex-0.154.0/` (p7) and `runtime/patches/codex-0.154.0-p8/` (p8) |
| [jerber/arc-code](https://github.com/jerber/arc-code) | `6b33c1f` | MIT (Copyright (c) 2026 Jeremy Berman) | Offline patch `agi3/work/arc_agi3/harness_patches/arc_code_offline.patch`; packed into Kaggle dataset `arc-code-kaggle-assets` |

## Runtime components (mounted on Kaggle, unmodified)

| Component | Version | License | Kaggle input |
|---|---|---|---|
| [Humming kernels](https://github.com/vllm-project/humming) | 0.1.16 (PyPI wheel) | Apache-2.0 | `humming-kernels-0116` |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) + JIT cache | 0.6.18 (+cu130) | Apache-2.0 | `qwen38fn-sglang-0520-sm120-cp312` |
| [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) | 2.x cp312 wheel | BSD-3-Clause | `qwen38fn-sglang-0520-sm120-cp312` |
| PyTorch, Triton, Transformers, xgrammar, sgl-kernel and the rest of the Python 3.12 environment | as packaged | their own licenses (BSD / MIT / Apache-2.0, MPL-2.0 for certifi / orjson / tqdm, LGPL for easydict / pycountry) | `qwen38fn-sglang-0520-sm120-cp312` |
| NVIDIA CUDA 13.1 components and NVIDIA pip packages (cuBLAS, cuDNN, NCCL, NVSHMEM, nvJitLink, nvcc/ptxas subset, CUTLASS DSL) | 13.1 | NVIDIA proprietary licenses (CUDA EULA, NVIDIA Software License) | `qwen38fn-sglang-0520-sm120-cp312` |
| [OpenAI codex CLI](https://github.com/openai/codex) npm package with its vendored `rg`, `zsh`, `bwrap`, code-mode host | 0.154.0 | Apache-2.0 (codex); bundled tools under their own licenses | `arc-code-kaggle-assets` |
| [bubblewrap](https://github.com/containers/bubblewrap) | 0.6.1 (Ubuntu jammy build) | LGPL-2.0-or-later; source: upstream tag v0.6.1 and Ubuntu package `bubblewrap 0.6.1-1ubuntu0.1` | `arc-code-kaggle-assets` |
| [SciPy](https://github.com/scipy/scipy) | 1.18.1 cp312 wheel | BSD-3-Clause (bundled OpenBLAS / LAPACK BSD-3-Clause, libgfortran GPL-3.0-or-later WITH GCC-exception-3.1, libquadmath LGPL-2.1-or-later) | `agent-wheels-scipy-1-18-1` |
| [pip](https://github.com/pypa/pip) | 26.2.1 wheel | MIT | `pip-offline-wheel` |
| [Zstandard](https://github.com/facebook/zstd), [LZ4](https://github.com/lz4/lz4), [XZ Utils](https://tukaani.org/xz/), [zlib](https://zlib.net) | zstd 1.5.5, lz4 1.9.4, xz 5.4.5 | BSD-3-Clause or GPL-2.0 (zstd), BSD-2-Clause (lz4 library), public domain (liblzma 5.4.x), zlib License | `zstd-x86-64-kaggle` |

## Data

| Data | License | Role |
|---|---|---|
| [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) public training and evaluation tasks | Apache-2.0 | Training data, calibration, evaluation |
| ARC-AGI-3 public games | as published by ARC Prize | Training environments, game cards |
| [NVARC synthetic puzzles](https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles) | as published by the authors | Training / calibration tasks |
| [google/ARC-GEN](https://github.com/google/ARC-GEN) | Apache-2.0 | Task generators (large-grid pool) |
| [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | as published by NVIDIA | Prompts for drafter training data |
