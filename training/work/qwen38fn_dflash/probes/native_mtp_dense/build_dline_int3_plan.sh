#!/bin/bash
# make_data_plan.py with 20261012-int3-plan's settings (v1r's corpora, fractions, passes, seed) on the sources staged by
# stage_dline_int3_sources.sh and the INT3 target profile recorded by record_int3_profile.sh.
#
#   SRC=<staged sources> PROFILE=<profile json> OUT=<new dir> [SMALL=1] \
#       bash work/qwen38fn_dflash/probes/native_mtp_dense/build_dline_int3_plan.sh
#
# The full plan must come out with the original plan.jsonl bytes (sha256 below; 2026-10-12 on node1, rebuilt twice): jobs
# name sources by index and subset / instance names, not by path, and the INT3 package's tokenizer is the NVFP4 one. A
# different sha means other sources or other plan code. SMALL=1 is the pool resume check's plan (v1r's resume-check
# settings: 2e5 held-out tokens a corpus, 248 jobs). Both shas were checked on node1 from the NFS-staged sources with another
# profile (the NVFP4 one): plan.jsonl depends on neither the source paths nor the target. The shas are of plans built
# before this release renamed the ledger corpus (now node1) and the ledger instances (now node1-p...): job keys carry
# both names, so a plan built with the renamed ones has other bytes and fails this check.
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
ROOT=$PWD
export ARC_ROOT=$ROOT
source env/env.sh
: "${SRC:?SRC=<stage_dline_int3_sources.sh dir>}" "${PROFILE:?PROFILE=<record_int3_profile.sh json>}" "${OUT:?OUT=<new dir>}"
WANT=19365363201d9ef683c80a820035fe429e990564faaa50da298a435b70152038
[ "${SMALL:-0}" = 1 ] && WANT=5a82539dca05a697ad37531c1892642a4ddbed77ac352d8de5d4c647d91e2cdd
args=(--node1 "$SRC/ledger/node1-p30241-20261005T205435-2524457" "$SRC/ledger/node1-p30242-20261005T211903-2542675"
      --agentic "$SRC/agentic_nvarc_segments" --naive "$SRC/naive/out"
      --nemotron "$SRC/nemotron/20260907-nemotron-100k-v1/tokens" "$SRC/nemotron/20260908-nemotron-8k-v2/tokens"
                 "$SRC/nemotron/20260908-nemotron-short4k-v1/tokens")
if [ "${SMALL:-0}" = 1 ]; then
    args+=(--heldout-tokens 2e5 --fraction node1=0.004 agentic=0.0002 naive=0.0005 nemotron=0.0001
           --passes node1=2 agentic=1 naive=1 nemotron=1)
else
    args+=(--fraction node1=1 agentic=0.35 naive=0.8 nemotron=0.25 --passes node1=4 agentic=1 naive=1 nemotron=1)
fi
PYTHONPATH=src "$ARC_SGLANG_VENV/bin/python" work/qwen38fn_dflash/make_data_plan.py --out "$OUT" \
    --target-profile "$PROFILE" --seed 0 "${args[@]}" --workers 32
got=$(sha256sum "$OUT/plan.jsonl" | cut -d' ' -f1)
[ "$got" = "$WANT" ] || { echo "✗ $OUT/plan.jsonl sha256 $got, the original plan has $WANT" >&2; exit 1; }
echo "· $OUT/plan.jsonl is the original plan ($got)"
}
main "$@"
exit
