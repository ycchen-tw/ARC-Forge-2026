#!/bin/bash
# The D-line INT3 plan's sources (20261012-int3-plan: v1r's corpora) from NFS into SRC (default
# $ARC_RAM_RUNS/qwen38fn_dflash/dline-int3-sources), laid out as make_data_plan.py reads them:
#   $SRC/agentic_nvarc_segments/part-*.parquet          copied from $ARC_NFS/data/agentic_nvarc_segments
#   $SRC/naive/out/part-*.parquet                        qwen38fn_distill/out.tar.zst unpacked
#   $SRC/nemotron/<batch>/tokens/part-*.parquet          qwen38fn_specdata/<batch>/tokens.tar.zst unpacked (<batch> is
#                                                        the plan's subset name: keep it)
#   $SRC/ledger/<instance>/                              server.json, events.0.jsonl copied; media/ from media.tar.zst;
#                                                        routing.*.bin symlinked to NFS (capture and the plan builder
#                                                        only stat them; 257 GB)
# then loads every ledger with the strict reader and checks its media blobs' sha256 (about 1.5 min).
#
#   [SRC=<new dir>] bash work/qwen38fn_dflash/probes/native_mtp_dense/stage_dline_int3_sources.sh
#
# NFS is only read, one file at a time. Next: build_dline_int3_plan.sh.
set -euo pipefail
main() {  # the whole body is a function, so an edit to this file cannot change a running copy
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
ROOT=$PWD
export ARC_ROOT=$ROOT
source env/env.sh
SRC=${SRC:-$ARC_RAM_RUNS/qwen38fn_dflash/dline-int3-sources}
NFS=$ARC_NFS
LEDGERS=(node1-p30241-20261005T205435-2524457 node1-p30242-20261005T211903-2542675)
BATCHES=(20260907-nemotron-100k-v1 20260908-nemotron-8k-v2 20260908-nemotron-short4k-v1)
[ ! -e "$SRC" ] || { echo "✗ $SRC exists" >&2; exit 1; }
mkdir -p "$SRC/agentic_nvarc_segments" "$SRC/naive" "$SRC/ledger"
t0=$SECONDS
for f in "$NFS"/data/agentic_nvarc_segments/part-*.parquet; do cp "$f" "$SRC/agentic_nvarc_segments/"; done
zstd -dc "$NFS/runs/qwen38fn_distill/out.tar.zst" | tar -xf - -C "$SRC/naive"
for b in "${BATCHES[@]}"; do
    mkdir -p "$SRC/nemotron/$b"
    zstd -dc "$NFS/runs/qwen38fn_specdata/$b/tokens.tar.zst" | tar -xf - -C "$SRC/nemotron/$b"
done
for inst in "${LEDGERS[@]}"; do
    from=$NFS/runs/tito/20261006-node1-collect/ledger/$inst
    d=$SRC/ledger/$inst
    mkdir "$d"
    cp "$from/server.json" "$from/events.0.jsonl" "$d/"
    for r in "$from"/routing.*.bin; do ln -s "$r" "$d/"; done
    zstd -dc "$from/media.tar.zst" | tar -xf - -C "$d"  # the archive holds media/<sha256>
done
echo "· copied in $((SECONDS - t0)) s: agentic $(ls "$SRC/agentic_nvarc_segments" | wc -l) (92), naive $(ls "$SRC/naive/out" | wc -l) (474), nemotron $(for b in "${BATCHES[@]}"; do ls "$SRC/nemotron/$b/tokens" | wc -l; done | paste -sd/) (1867/6251/1265)"
PYTHONPATH=src "$ARC_SGLANG_VENV/bin/python" - "$SRC" "${LEDGERS[@]}" <<'PY'
import os
import sys
from pathlib import Path

from arcforge.tito.export import verify_media_blobs
from arcforge.tito.ledger import Vocab, load_instance

src = Path(sys.argv[1])
vocab = Vocab(Path(os.environ["ARC_NFS"]) / "models/Qwen3.8-Flash-Next-NVFP4-nvidia")  # the same tokenizer files as the INT3 package
for name in sys.argv[2:]:
    inst = load_instance(src / "ledger" / name, vocab)
    bad = verify_media_blobs(inst)
    if bad:
        raise SystemExit(f"✗ {name}: {len(bad)} media blobs do not match their sha256")
    print(f"· {name}: {len(inst.segments)} segments, media ok")
PY
echo "· staged in $SRC"
}
main "$@"
exit
