# Runtime — building the offline Kaggle inputs

Kaggle runs the finals without network access, so everything the server and the agents
execute is mounted from datasets. This tree holds the scripts and patch sets that build the
datasets shared by both tracks. Run scripts from this directory after `source env/env.sh`
(create `env/local.sh` from `env/local.sh.example` first).

## sglang

| Step | Script | Output |
|---|---|---|
| 1. Build the sglang v0.5.20 environment: clone tag `v0.5.20`, apply `patches/sglang-0.5.20/` (base commit in `BASE`), create a Python 3.12 venv, install, check that every patch is present | `work/qwen38fn_eval/build_sglang_0520.sh` (patch markers in `work/qwen38fn_eval/sglang_0520_markers.py`, NVFP4 scale check `tools/verify_nvfp4_graph_scale.py`) | `$VENV` (default `$HOME/work/dsv4/venv-0520`) |
| 2. Package it as a relocatable runtime for Kaggle: re-clone the tag and apply the committed patches, add the FlashAttention 2 wheel and the FlashInfer sm_120 JIT cache (`work/qwen38fn_eval/install_flashinfer_jit_cache_sm120.py`), copy the CUDA 13.1 compiler subset, write `runtime-provenance.json` | `work/qwen38fn_kaggle/package_runtime_sm120_0520.sh` | `sglang-0520-94602c9-sm120-cp312.tar.zst` → dataset `qwen38fn-sglang-0520-sm120-cp312` |
| 3. Pack the fork's Python tree at the deployed commit (`deploy/sglang.lock.json`, target `l4-all-merge-1020` = tag `kaggle-final` of [sglang-arc-2026](https://github.com/ycchen-tw/sglang-arc-2026)) | `work/sglang_fork/pack_kaggle_fork_tree.py` (uses `export_tree.py` and `work/qsa_hisparse/probes/pack_kaggle_tree.py`; `lock.py` maintains the lock file) | `sglang-a5c67c41cd22.tar.zst` → dataset `sglang-arc-2-tree-allmerge-1020`; the packer checks the tree digest (`30a05674…`, 4,943 files) against the lock, the archive bytes depend on the zstd build |

On Kaggle the notebook unpacks the runtime archive and replaces its `python/sglang/` with the
fork tree, after checking the whole-package digest (`src/arcforge/sglang_tree.py`). The
ARC-AGI-2 notebooks get the same fork tree from the L4 payload dataset instead (see
`../agi2/`).

## codex

`work/arc_code/build_codex_musl.sh` clones `openai/codex` at `rust-v0.154.0`, applies the
patches in `patches/codex-0.154.0/`, and cross-compiles a static musl binary (it does not need
root; the toolchain is rustup + zig). Two patch sets were deployed:

| Patch set | Directory | Dataset | Used by |
|---|---|---|---|
| p7: 0001–0007 | `patches/codex-0.154.0/` | `arcforge-codex-0154-p7` | 57747104 |
| p8: 0001 (revised) + 0002–0007 + 0008 | `patches/codex-0.154.0-p8/` | `arcforge-codex-0154-p8` | 57848658, 57678190, 57846476 |

The script reads its patches from `patches/codex-0.154.0/`; to build p8, put the p8 set there.
Each directory's `README.md` describes the patches.

## Other inputs

| Dataset | Source |
|---|---|
| `humming-kernels-0116` | `pip download humming-kernels==0.1.16` (wheel `humming_kernels-0.1.16-cp310-abi3-manylinux_2_28_x86_64.whl`, sha256 `68321269…`) |
| `zstd-x86-64-kaggle` | Prebuilt `zstd` 1.5.5 with its shared libraries (`liblz4.so.1`, `liblzma.so.5`, `libz.so.1`), used to unpack the runtime archive |
| `arc-code-kaggle-assets`, `pip-offline-wheel`, `agent-wheels-scipy-1-18-1` | ARC-AGI-3 only; see `../agi3/work/arc_agi3/` |
