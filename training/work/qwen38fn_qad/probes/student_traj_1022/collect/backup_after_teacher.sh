#!/bin/bash
# After the teacher (unit qadcol-teacher) finishes successfully and both splits have DONE, O_DIRECT-copy targets and
# qadmix to NFS (shm not deleted).
set -euo pipefail
R=$HOME/work/runs/tito/20261022-int3-qad-collect
S=/dev/shm/uid-$UID/arcforge-runs/tito/20261022-int3-qad-collect
D=$ARC_NFS/runs/tito/20261022-int3-qad-collect
MV="python3 $HOME/work/runs/data-cleanup-20261021/nfs_move.py --run-dir $R"
while systemctl --user is-active -q qadcol-teacher; do sleep 60; done
res=$(systemctl --user show -p Result --value qadcol-teacher); echo "teacher result=$res $(date -u)"
[ "$res" = success ] || { echo "teacher failed; no backup"; exit 1; }
for sp in train validation; do [ -f $S/teacher-qadmix/targets/$sp/DONE ] || { echo "no DONE for $sp"; exit 1; }; done
cd $S/teacher-qadmix
$MV files --src-root $S/teacher-qadmix --dst-root $D/teacher-qadmix $(find targets -type f | sort)
$MV dirs --src-root $S --dst-root $D qadmix-1019-1022
echo "backup done $(date -u)"
