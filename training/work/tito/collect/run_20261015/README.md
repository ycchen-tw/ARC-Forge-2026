# 10-15 node1 FP8 + DFlash TITO collection: script snapshot

A verbatim copy of the scripts used in the collection run directory (node1 `$ARC_RUNS/tito/20261015-fp8-dflash-collect/`), kept for the record; paths are hardcoded to node1,
so these are not reusable tools; change the paths before rerunning. Results and write-up in the research note,
segment-2 inventory in `inventory_tk/SUMMARY.md`.

- `gate/serve_gate2.sh`: server start-up for the top-32 ledger (lock `b200-tito-topk`).
- `arc2/feeder.py` + `run_batch.sh`, `arc3/feeder.py` + `corpora.json`, `arc3_stopgap/`: open batches / rounds based on KV and the number running.
  `arc2/queue.json` (2,099 batches, 2.2 MB) is not copied; it is in the run directory.
- `offload2.py`, `move_ledger_tk.sh`: moving to NFS (built on `work/tito/collect/nfs_move.py`).
- `watch_anomalies.py`: anomaly watch and disk brake every 5 minutes.
- `topk_test/`: top-32 correctness checks and throughput A/B.
- `inventory_tk/`, `rebuild_manifests.py`: shutdown inventory.
