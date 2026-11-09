#!/bin/bash
# Wrap-up of a finished collection run (collect.py starts unit g2seed-fin-<tag> after the run's unit ends normally):
#   1. seeds.py gate     (replay-gate the level-change seeds, prefix-check the window-change seeds; writes
#                         <run>/seeds/gate.jsonl)
#   2. seeds.py promote  (hard-link the passing ones into the seed store $STORE, read-only, index.jsonl)
#   3. nfs_move.py dirs --delete (the whole run directory as tar.zst, verified with tar -d, O_DIRECT-written to NFS,
#      read back and sha256-compared, only then deleted locally)
#   4. seeds.py archive (once the seed store has a full batch, archive it to $NFS/store and delete the local
#      window-change seeds)
# Any failing step stops (set -e) and the local run directory stays; only one runs at a time (flock): one NFS write
# stream at a time.
#   bash finish.sh <collection root> <run tag>        # run in the pinned worktree
set -euo pipefail
ROOT=${1:?collection root} TAG=${2:?run tag}
source env/env.sh >/dev/null
STORE=${STORE:-$HOME/work/data/arc3_seeds/g2seed-20261021}
NFS=${NFS:-$ARC_NFS/arc3_seeds/g2seed-20261021}
OUT=$ROOT/runs/$TAG
[ -d "$OUT/run" ] || { echo "no $OUT/run" >&2; exit 64; }
exec 9>"$ROOT/collect/finish.lock"
flock 9
echo "$(date -u +%FT%TZ) gate $TAG"
.venv/bin/python work/arc_code/gen2/seed_collect/seeds.py gate "$OUT"
echo "$(date -u +%FT%TZ) promote $TAG -> $STORE"
.venv/bin/python work/arc_code/gen2/seed_collect/seeds.py promote --store "$STORE" "$OUT"
mkdir -p "$ROOT/collect/lanes"
echo "$(date -u +%FT%TZ) offload $TAG -> $NFS/runs"
.venv/bin/python work/tito/collect/nfs_move.py --run-dir "$ROOT/collect" dirs --src-root "$ROOT/runs" \
    --dst-root "$NFS/runs" --delete "$TAG"
# Once the seed store has 300 unarchived seeds, archive a batch (at most 1000) to NFS and delete the local window-change
# seeds (seeds.py archive)
echo "$(date -u +%FT%TZ) archive store -> $NFS/store"
.venv/bin/python work/arc_code/gen2/seed_collect/seeds.py archive --store "$STORE" --nfs "$NFS" \
    --nfs-run-dir "$ROOT/collect"
echo "$(date -u +%FT%TZ) FINISHED $TAG"
