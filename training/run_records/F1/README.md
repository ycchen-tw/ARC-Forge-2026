# F1 run record: codex1015 trace ledgers on node1 (FP8 + DFlash)
Host node1, 2026-10-14 to 2026-10-16. Four TP2 replicas: r0 :30811 on GPUs 0,1 for arc2, r1-r3 :30812-30814 on GPUs 2-7 for arc3 (ARC_THINK_BUDGET 24576). From 2026-10-15 06:19Z the top-32 ledgers came from the gate p30811 (arc2) and r1 p30812 (arc3) started by gate/serve_gate2.sh (fork 2da3f25f, not in this tree).
The feeders, run_batch.sh and serve_gate2.sh are released in work/tito/collect/run_20261015/. `lists/arc2_batches.tsv` is the 17 arc2 batches launched from the 2,099-batch queue.
Not recorded: the first-phase server script gate/serve_gate.sh and a list of the arc3 rounds.
