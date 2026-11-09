# The four final submissions

Each directory holds one notebook exactly as Kaggle ran it (`notebook.ipynb`, pulled with
`kaggle kernels pull`), its Kaggle inputs (`kernel-metadata.json`), and for ARC-AGI-2 the asset
manifest embedded in the notebook (`asset-manifest.json`, also the builder's `--asset-manifest`).

| Directory | Track | Submission | Kernel @ scriptVersionId | Machine | Built from (arc-forge commit) | Carried files |
|---|---|---|---|---|---|---|
| `agi2-57747104-ext300-arcv6` | ARC-AGI-2 | 57747104 | `threerabbits/qwen38fn-l4x4-c128-ext300-arcv6` @362595216 | 4×L4 | harness `e823c225` (clean) | 183 |
| `agi2-57848658-x39-rl` | ARC-AGI-2 | 57848658 | `threerabbits/qwen38fn-l4x4-c128-v10-rl30p8-x39` @364318725 | 4×L4 | harness `f445b452` (clean) | 264 |
| `agi3-57678190-v35-click` | ARC-AGI-3 | 57678190 | `threerabbits/arc-code-hs110-click` @362591255 | RTX PRO 6000 | `00a62b51` (clean) | 539 |
| `agi3-57846476-v40-cards` | ARC-AGI-3 | 57846476 | `threerabbits/arc-code-hs110-v14tk5an-cards-swv8` @364285204 | RTX PRO 6000 | `3b70782b` + uncommitted files, committed as `1db8e4b3` | 660 |

`extract_code.py` unpacks the code a notebook carries and checks it against the notebook's own
manifest (`BUNDLE_FILES` for ARC-AGI-2, per-file sha256 in `BUILD["carried"]` for ARC-AGI-3):

```bash
python submissions/extract_code.py submissions/agi3-57846476-v40-cards/notebook.ipynb /tmp/v40
```

The ARC-AGI-2 notebooks carry the harness only; the serving launcher is the payload archive
inside the dataset `qwen38fn-l4-v16-arcdensevocabtp4-gean-qad3-l4opt48` (its files are in
`agi2/work/qwen38fn_l4/probes/` and `agi2/src/arcforge/`, sha256 listed in
`asset-manifest.json` under `payload.files_sha256`).
