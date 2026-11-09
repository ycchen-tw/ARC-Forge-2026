# gEAN expert masks (steps A2 → A3)

- `gean320.pt`: the 320-of-512 routed-expert keep list per layer (gEAN saliency, `sum gate * expert_output_norm`), sha256 `5757e975…ef669`; `deploy_manifest.json` of the shipped checkpoint carries the same keep list.
- `mtp320.pt`: the MTP-layer keep list (reused from the REAP320 incumbent). Two path strings in its metadata were rewritten for this release, so its sha256 is `0088dee0…0dee` instead of the run's `76db5289…cd7a`; the mask tensors are identical.
- `pruning_calibration.json`: selection settings (sources, ARC-AGI-3 split, top-k).

`work/qwen38fn_eval/prune_experts.py` (step A3, see `TRAINING_MAP.json`) applies them.
