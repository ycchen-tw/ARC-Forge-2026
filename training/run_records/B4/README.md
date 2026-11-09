# B4 run record: FP8 teacher top-32 targets
Host node1, systemd unit qadcol-teacher, 2026-10-22 13:52Z to 17:25Z, GPUs 0-3, MASTER_PORT 29641 (the unit environment is not in the journal).
Main command line: `launch/qadcol-teacher.journal.txt` (train_qad.sh --mode teacher --model Qwen3.8-Flash-Next-FP8 --data qadmix-1019-1022).
Output: teacher-qadmix/targets/{train,validation} under $ARC_RAM_RUNS/tito/20261022-int3-qad-collect (NFS copy under $ARC_NFS/runs/tito/20261022-int3-qad-collect); `config/topk.json` is its format header (qad-topk-v1, k 32).
