# A2 run record: REAP saliency capture and gEAN mask
Host node2, 2026-10-08 17:15Z to 22:14Z, run dir $ARC_RUNS/qwen38fn_eval/20261009-reap-mm.
Plan: `config/inventory_meta.json` and `config/plan_meta.json` hold the reap_plan.py arguments; `lists/plan.jsonl` is plan-v1 over the A1 ledgers.
Capture: `launch/run_capture.sh` (reap_capture.py over diag, calib, heldout) against a server on GPUs 4,5 port 30690 started by the released work/qwen38fn_dflash/reap_server.sh with PROFILE = `../A1/config/p30241.server.json`. `launch/capture-server.txt` is the flag line that server start printed (sglang-arc-2 4b57f3c45, not in this tree); the log of the start that served the capture is not in the archive.
Selection: `lists/groups_A.txt` / `groups_B.txt` are the mix-v1 game split. Output masks: variants-v3/masks/ean_all.pt (sha256 5757e975...) and, for MTP, select-v1/masks/reap320_incumbent_mtp.pt.
Not recorded: the reap_mm_select.py / reap_mix_mask.py / reap_mask_variants.py command lines.
