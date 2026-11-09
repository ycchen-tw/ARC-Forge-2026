"""``python -m arcforge.tito {inspect,audit,export-sft,export-dflash,export-rl} LEDGER... --tokenizer DIR``

LEDGER is an instance directory (has server.json) or a ledger root DIR (every instance under it is taken).
``--tokenizer`` is a model directory or tokenizer.json, with no default: pad and special-token ids are always read from
it.

    python -m arcforge.tito inspect $DIR --tokenizer $MODEL [--segment <instance>:<seg>]
    python -m arcforge.tito audit $DIR --tokenizer $MODEL --manifest $RUN/tito_manifest.json \\
        --ledger-status $RUN/ledger_status.json [--json report.json]       # exit 1 if not passed
    python -m arcforge.tito export-sft $DIR --tokenizer $MODEL --out $OUT [--manifest …] [--inline-media]
    python -m arcforge.tito export-dflash $DIR --tokenizer $MODEL --out $OUT [--manifest …]
    python -m arcforge.tito export-rl $DIR --tokenizer $MODEL --out $OUT [--manifest …] [--inline-media]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from . import audit as audit_mod
from . import export, roles
from .join import Joiner, load_manifests
from .ledger import Vocab, instance_dirs, iter_instances, load_instance, load_ledger


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("ledger", nargs="+", help="instance directory or ledger root DIR")
    p.add_argument("--tokenizer", required=True, help="model directory or tokenizer.json")


def _manifests(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", action="append", default=[],
                   help="the harness's tito_manifest.json (repeatable); when given, only segments that match it are used")


def _prefixes(p: argparse.ArgumentParser) -> None:
    p.add_argument("--harness-prefix", action="append", default=[],
                   help="extra user-message prefix counted as harness_user (added after the codex defaults)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m arcforge.tito", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="summary; --segment decodes one segment span_kind by span_kind")
    _common(p)
    _prefixes(p)
    p.add_argument("--segment", help="<instance>:<seg>")
    p.add_argument("--no-strict", action="store_true",
                   help="mark broken segments broken and keep reading (default: stop at the first invariant violation)")

    p = sub.add_parser("audit", help="collection audit; exit 1 if it does not pass")
    _common(p)
    _manifests(p)
    p.add_argument("--manifest-threads-only", action="store_true",
                   help="rebuild only the manifests' threads (needs --manifest); global event format / stream metadata "
                        "are still checked")
    p.add_argument("--ledger-status", action="append", default=[],
                   help="snapshot JSON of /session_ledger/status (object or list); lossy is only there")
    p.add_argument("--expected-splits", default=",".join(audit_mod.DEFAULT_EXPECTED_SPLITS),
                   help="comma-separated; empty string = no split is expected")
    p.add_argument("--no-verify-media", action="store_true", help="do not recompute the sha256 of media blobs")
    p.add_argument("--json", help="write the full report to this file")

    for name in ("export-sft", "export-dflash", "export-rl"):
        p = sub.add_parser(name)
        _common(p)
        _manifests(p)
        p.add_argument("--out", required=True, help="output directory (must not exist)")
        p.add_argument("--rows-per-shard", type=int, default=256)
        if name != "export-dflash":
            p.add_argument("--inline-media", action="store_true", help="put media into the parquet as base64 (sha256 verified)")
        if name == "export-sft":
            _prefixes(p)
    return ap


def _inspect(args, vocab: Vocab) -> int:
    instances = load_ledger(args.ledger, vocab, strict=not args.no_strict)
    prefixes = roles.HARNESS_USER_PREFIXES + tuple(args.harness_prefix)
    if args.segment:
        name, _, seg_id = args.segment.rpartition(":")
        inst = {i.name: i for i in instances}[name]
        seg = inst.segments[int(seg_id)]
        gen = [(s.start, s.end) for s in seg.gen_spans]
        kinds = roles.token_kinds(seg.ids, gen, vocab, prefixes)
        print(f"{seg.key} sid={seg.sid} status={seg.status} cause={seg.cause} close={seg.close and seg.close['reason']} "
              f"len={len(seg)} gen={seg.n_gen} appends={len(seg.appends)} media={len(seg.media)}")
        for s, e, r in roles.role_spans(len(seg), gen):
            print(f"  role {r} [{s:7d}:{e:7d}]")
        for s, e, k in roles.kind_spans(kinds):
            text = vocab.decode(seg.ids[s:e])
            head = text[:70].replace("\n", "\\n")
            tail = text[-30:].replace("\n", "\\n") if len(text) > 100 else ""
            print(f"  {roles.KINDS[k]:<18} [{s:7d}:{e:7d}] {head}{' … ' + tail if tail else ''}")
        return 0
    for inst in instances:
        segs = list(inst.segments.values())
        appends = [a for s in segs for a in s.appends]
        print(f"{inst.name}: events={inst.n_events} segments={len(segs)} {dict(Counter(s.status for s in segs))} "
              f"appends={len(appends)} attempts={len(inst.attempts)} tokens={sum(len(s) for s in segs)} "
              f"gen={sum(s.n_gen for s in segs)} media={sum(len(s.media) for s in segs)} "
              f"logprobs={sum(a.logprobs is not None for a in appends)}/{len(appends)} "
              f"routing={sum(a.routing is not None for a in appends)}/{len(appends)}"
              + (" TRUNCATED_TAIL" if inst.truncated_tail else ""))
        for err in inst.errors:
            print(f"  ! {err}")
        for s in segs:
            close = "-" if s.close is None else s.close["reason"]
            print(f"  seg {s.seg:5d} sid={s.sid} {s.status:6s} cause={s.cause} close={close} appends={len(s.appends)} "
                  f"len={len(s)} gen={s.n_gen} media={len(s.media)}" + (f" ! {s.error}" if s.error else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "audit" and args.manifest_threads_only and not args.manifest:
        parser.error("--manifest-threads-only requires at least one --manifest")
    vocab = Vocab(args.tokenizer)
    if args.cmd == "inspect":
        return _inspect(args, vocab)
    manifests = load_manifests(args.manifest)
    if args.cmd == "audit":
        if args.manifest_threads_only:
            threads = {thread for manifest in manifests for thread in manifest.threads}
            instances = [load_instance(d, vocab, strict=False, threads=threads) for d in instance_dirs(args.ledger)]
        else:
            instances = load_ledger(args.ledger, vocab, strict=False)
        status = audit_mod.load_status(args.ledger_status) if args.ledger_status else None
        expected = [s for s in args.expected_splits.split(",") if s]
        report = audit_mod.audit(instances, vocab, manifests=manifests, ledger_status=status,
                                 expected_splits=expected, verify_media=not args.no_verify_media)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=1, ensure_ascii=False, default=str)
        for line in report["failures"]:
            print(f"FAIL {line}")
        for line in report["warnings"]:
            print(f"WARN {line}")
        print(f"segments={report['segments']} splits={report['split_reasons']} logprobs={report['logprobs']} "
              f"routing={report['routing']} top_logprobs={report['top_logprobs']} usage={report['usage']}")
        w = report["window"]
        print(f"window n={w['n']} unknown={w['unknown']}"
              + ("" if not w["n"] else f" p50={w['p50']:g} p90={w['p90']:g} p99={w['p99']:g} max={w['max']} "
                 f"=0:{w['eq0']} <=0:{w['le0']} >0:{w['gt0']} reforward/new={w['reforward_over_new']}")
              + f" positions={w['positions']} resume={report['resume']}")
        att = w["attempts"]
        print("attempt windows " + " ".join(
            f"{k}: n={att[k]['n']}" + (f" p50={att[k]['p50']:g} p99={att[k]['p99']:g} max={att[k]['max']} >0:{att[k]['gt0']}"
                                       if att[k]["n"] else "") for k in ("prefix_len", "vs_S_upper_bound"))
              + f" not_admitted={att['not_admitted']} unknown={att['unknown']}")
        print("collection_passed=" + str(report["collection_passed"]).lower())
        return 0 if report["collection_passed"] else 1
    instances = iter_instances(args.ledger, vocab, strict=True)      # one instance at a time
    joiner = Joiner(manifests) if manifests else None
    if args.cmd == "export-sft":
        doc = export.export_sft(instances, vocab, args.out, joiner=joiner, inline_media=args.inline_media,
                                rows_per_shard=args.rows_per_shard,
                                harness_prefixes=roles.HARNESS_USER_PREFIXES + tuple(args.harness_prefix))
    elif args.cmd == "export-dflash":
        doc = export.export_dflash(instances, vocab, args.out, joiner=joiner, rows_per_shard=args.rows_per_shard)
    else:
        doc = export.export_rl(instances, vocab, args.out, joiner=joiner, inline_media=args.inline_media,
                               rows_per_shard=args.rows_per_shard)
    print(json.dumps(doc["stats"], ensure_ascii=False))
    for key in ("media_without_bytes", "skipped_media", "unjoined"):
        if doc["stats"][key]:
            print(f"!! {key}={doc['stats'][key]} (see {args.out}/tito_export.json)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
