# C1 run record: ARC-AGI-2 r4 task pool
Host node1, 2026-10-27 to 2026-10-28.
`code/` holds the scripts of $ARC_RUNS/arc2_synthcheck/scripts: convert.py (synthetic sets to task files), final.py (per-task table), build_audit.py / build_audit2.py (review packets) and true_hard.py (writes true_hard.csv).
`lists/truehard558-20261028.train_order.txt` and `.heldout20.txt` are the true_hard pool. `lists/ctx64k90v2-20261029.{train.txt,selection.json}` are the r4 pool written by the released work/qwen38fn_rl/probes/arc2_ctx64k_20261029/build_pool.py (seed 20261029); `.untrained_eval120fix.txt` is the r4 held-out list.
Not recorded: the step from true_hard.csv to the truehard558 lists, and the builder of the held-out list.
