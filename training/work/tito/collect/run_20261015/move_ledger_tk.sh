#!/bin/bash
# 10-16: after shutdown, move the files the two ledger_tk instances left locally (media, events, last topk, server.json) to NFS (O_DIRECT, per-file sha256 check, delete local);
# once moved, the local instance directory holds only symlinks and is replaced by a symlink to the NFS instance directory.
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source "$REPO/env/env.sh" >/dev/null
C=$ARC_RUNS/tito/20261015-fp8-dflash-collect
nm="/usr/bin/python3 $REPO/work/tito/collect/nfs_move.py --run-dir $C --pause 0"
for pair in "gate gate/ledger_tk" "r1 replicas/r1/ledger_tk"; do
    set -- $pair; tag=$1; root=$C/$2
    for inst in $root/*-p[0-9]*-*; do
        [ -d "$inst" ] && [ ! -L "$inst" ] || continue
        name=$(basename $inst)
        cd $root
        mapfile -t rels < <(find $name -type f ! -name server.json | sort -t/ -k2,2r; echo $name/server.json)
        $nm files --src-root $root --dst-root $ARC_NFS/runs/tito/20261015-fp8-dflash-collect/ledger_tk/$tag --delete "${rels[@]}"
        left=$(find $name -type f | wc -l)
        [ "$left" = 0 ] || { echo "$name: $left regular files left"; exit 1; }
        for l in $name/topk.*.bin; do [ "$(readlink $l)" = "$ARC_NFS/runs/tito/20261015-fp8-dflash-collect/ledger_tk/$tag/$l" ] || { echo "bad link $l"; exit 1; }; done
        rm -r $name && ln -s $ARC_NFS/runs/tito/20261015-fp8-dflash-collect/ledger_tk/$tag/$name $name
        echo "LINKED $root/$name -> $ARC_NFS/runs/tito/20261015-fp8-dflash-collect/ledger_tk/$tag/$name"
    done
done
echo MOVE_LEDGER_TK_DONE
