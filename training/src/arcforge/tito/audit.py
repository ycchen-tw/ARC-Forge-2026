"""Collection audit of a batch of ledgers (+ harness manifests).

``collection_passed`` (spec audit section: rid join complete, ``lossy=0``, at least one segment per task / game, split
reasons as expected) + two that should hold anyway:

1. No invariant violation at all (read with ``strict=False`` so that every broken segment is listed, instead of stopping
   at the first).
2. ``lossy == 0`` and the writer did not die: **the on-disk files do not have this number** (the fork reports it only in
   ``/session_ledger/status``), so ``ledger_status`` has to be fed in (the status JSON the harness fetches at the end of
   the run, a single object or a list, matched by ``instance``). Not fed = "unknown", which does not pass.
3. rid join complete (``join.rid_join``), and the ledger instances the manifests record were loaded. Without a manifest
   there is nothing to join against, which does not pass.
4. Every manifest item (task / game) maps to at least one segment. An item the harness itself recorded as failed and
   that has no thread is only a warning (the harness blew up and there was nothing to collect; that is a harness
   failure, not something the ledger missed).
5. Split reasons (on both seg_open and seg_close) are all in ``expected_splits``. The default
   ``DEFAULT_EXPECTED_SPLITS`` is an **assumption**: retry / concurrent (resends) and no_end (truncated turns) count as
   normal; text_differs, stop, mm_*, evicted… all mean replay did not catch it or the ledger is too small.
6. The sha256 of each media blob's content equals its file name (``verify_media``).
7. logprob / routing are not "partially present": within one batch some appends have it and some lack it (``*_error``
   or null) = collection lost it partway.
8. Session resume contract v2.1 (``ledger.resume_violations`` / ``attempt_violations``,
   ``docs/plans/session_exact_resume.md``): no session_id means none; none has no reason / pos; exact has no reason,
   ``pos == cached_tokens``, and extension turns with ``rewind == 0`` have ``window ∈ {0, −1}`` (``rewind > 0`` is not
   judged; counted in ``resume.exact_rewinds`` by ``ledger.rewind_cause``); fallback has a reason and
   ``pos == cached_tokens`` (retract: only needs a pos); an attempt without admission has no pos. A flag that does not
   match resume (an append with the flag on but no resume, exact / fallback with the flag off) is a ledger format error
   and raises at read time.
   A violation = a fork bug; the failure line starts with ``RESUME_CONTRACT_FAILURE`` and lists the sid / k / numbers
   of the first few (the orchestrator's excusal rules only recognise the split and rid join prefixes, so this line is
   never excused).

Reported only, not blocking: segments without seg_close (crash), a half-written last line in events, the number of
records whose usage disagrees with the reconstructed length, cached_tokens / prefix_len of extension requests,
tokenizer identity (pad ids compared when server.json records them; whether tokenizer.json's sha256 appears in
server.json -- the spec fixes no key name), the distribution of the recompute window ``window`` (``reforward``: p50 /
p90 / p99 / max, counts of =0 / ≤0 / >0, the source of the recomputed positions (previous turn's gen or prefill) and
their weight-version class, split by resume mode; attempts listed apart: with prefix_len it is their recompute window,
without it is measured against S at the time and is an upper bound), resume mode and fallback reason counts (appends
and attempts apart).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .export import verify_media_blobs
from .join import Manifest, rid_join
from .ledger import (ATTEMPT_RESUME_ABSENT, ATTEMPT_RESUME_NULL, REWIND_CAUSES, Instance, Vocab, attempt_violations,
                     resume_violations, rewind_cause, segment_reforward, split_sid)

DEFAULT_EXPECTED_SPLITS = ("retry", "concurrent", "no_end")
#: start of the contract-violation failure line ("session resume contract violated"; work/qwen38fn_rl/test_orchestrator.py
#: uses the same string to check that it is never excused)
RESUME_CONTRACT_FAILURE = "session 接續契約違反"
#: count name for a missing resume key (old fork, flag off)
RESUME_ABSENT = "absent"
#: count name for an attempt whose resume is null (no meta_info)
RESUME_NULL = "null"


def window_summary(windows: list[int]) -> dict:
    """Distribution of the window of extension appends. Quantiles are numpy's default linear interpolation."""
    w = np.asarray(windows, dtype=np.int64)
    if not w.size:
        return {"n": 0}
    p50, p90, p99 = (float(x) for x in np.percentile(w, [50, 90, 99]))
    return {"n": int(w.size), "p50": p50, "p90": p90, "p99": p99, "max": int(w.max()), "min": int(w.min()),
            "mean": float(w.mean()), "eq0": int((w == 0).sum()), "le0": int((w <= 0).sum()), "gt0": int((w > 0).sum()),
            "ge512": int((w >= 512).sum()), "reforward_tokens": int(np.clip(w, 0, None).sum())}


def load_status(paths: Iterable[str | Path]) -> dict[str, dict]:
    """Snapshot files of ``/session_ledger/status`` → {instance: status}."""
    out: dict[str, dict] = {}
    for p in paths:
        doc = json.loads(Path(p).read_text())
        for st in doc if isinstance(doc, list) else [doc]:
            for key in ("instance", "lossy", "writer_error"):
                if key not in st:
                    # "<p>: status lacks <key> (not a /session_ledger/status response?)"
                    raise ValueError(f"{p}: status 缺 {key!r}（不是 /session_ledger/status 的回應？）")
            if st["instance"] in out:
                # "<p>: duplicate status for instance <instance>"
                raise ValueError(f"{p}: instance {st['instance']} 的 status 重複")
            out[st["instance"]] = st
    return out


def _contains(value, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(_contains(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(_contains(v, needle) for v in value)
    return False


def audit(instances: list[Instance], vocab: Vocab, *, manifests: Iterable[Manifest] = (),
          ledger_status: dict[str, dict] | None = None,
          expected_splits: Iterable[str] = DEFAULT_EXPECTED_SPLITS, verify_media: bool = True) -> dict:
    manifests = list(manifests)
    expected = set(expected_splits)
    failures: list[str] = []
    warnings: list[str] = []

    status = Counter()
    invariants = Counter()
    violations = []
    splits = Counter()
    lp = Counter()
    rt = Counter()
    tk = Counter()
    usage = Counter()
    replayed = Counter()
    closure_tokens = 0                            # own_turn spans: tokens added after a cut generation (prompt)
    cached_ratio = []
    media_bad = []
    per_instance = []
    resume_modes = Counter()
    fallback_reasons = Counter()
    windows: dict[str, list[int]] = {}            # resume mode (or RESUME_ABSENT) → extension windows
    window_unknown = Counter()                    # extension appends without cached_tokens, by mode
    new_tokens = 0                                # tail + gen of the extension appends with a known window
    reforward = Counter()
    contract: list[str] = []
    exact_rewinds = {c: Counter() for c in REWIND_CAUSES}   # exact extensions with rewind > 0 (not judged), by cause
    attempt_modes = Counter()
    attempt_reasons = Counter()
    attempt_windows: dict[str, list[int]] = {"prefix_len": [], "vs_S_upper_bound": []}
    attempt_other = Counter()                     # not admitted / window not measurable
    for inst in instances:
        for err in inst.errors:
            invariants[err.code] += 1
            violations.append(str(err))
        segs = list(inst.segments.values())
        for seg in segs:
            status[seg.status] += 1
            if seg.error is not None:
                invariants[seg.error.code] += 1
                violations.append(str(seg.error))
            contract += resume_violations(seg)
            for a in seg.appends:
                mode = RESUME_ABSENT if a.resume is None else a.resume.mode
                resume_modes[mode] += 1
                if mode == "fallback":
                    fallback_reasons[a.resume.reason] += 1
                if a.k == 0:
                    continue
                if mode == "exact" and a.rewind > 0 and a.window is not None:
                    exact_rewinds[rewind_cause(seg, a)].update(
                        appends=1, window_gt0=int(a.window > 0), window_lt_minus1=int(a.window < -1),
                        positions=max(a.window, 0))
                if a.window is None:
                    window_unknown[mode] += 1
                    continue
                windows.setdefault(mode, []).append(a.window)
                new_tokens += a.tail_len + a.gen_len
            if seg.appends:
                rf = segment_reforward(seg)
                for w in rf.windows:
                    reforward.update(positions=w.hi - w.lo, from_gen=w.from_gen, from_prefill=w.from_prefill,
                                     D_same=w.same, D_cross=w.cross, D_unknown=w.unknown,
                                     appends_cross=int(w.cross > 0))
                reforward["segments_cross"] += int(any(w.cross for w in rf.windows))
            if seg.split_reason is not None:
                splits[f"open:{seg.split_reason}"] += 1
            if seg.close is not None and seg.close["split_reason"] is not None:
                splits[f"close:{seg.close['split_reason']}"] += 1
            for a in seg.appends:
                lp["with" if a.logprobs is not None else ("error" if a.logprobs_error else "none")] += 1
                rt["with" if a.routing is not None else ("error" if a.routing_error else "none")] += 1
                tk["with" if a.top_logprobs is not None else ("error" if a.top_logprobs_error else "none")] += 1
                replayed.update(a.replay_kinds)
                closure_tokens += a.replay_closure
                pt, ct = a.usage["prompt_tokens"], a.usage["completion_tokens"]   # may be null (the fork copies meta_info)
                if pt is None or ct is None:
                    usage["missing"] += 1
                elif pt != a.prefix_len + a.tail_len or ct != a.gen_len:
                    usage["mismatch"] += 1
                else:
                    usage["ok"] += 1
                if a.k > 0 and a.cached_tokens is not None and a.prefix_len:
                    cached_ratio.append(a.cached_tokens / a.prefix_len)
        for at in inst.attempt_records:
            if at.resume_state == ATTEMPT_RESUME_ABSENT:
                mode = RESUME_ABSENT                    # old fork
            elif at.resume_state == ATTEMPT_RESUME_NULL:
                mode = RESUME_NULL                      # no meta_info (orphaned, aborted before the scheduler, flag off)
            else:
                mode = at.resume.mode
            attempt_modes[mode] += 1
            if mode == "fallback":
                attempt_reasons[at.resume.reason] += 1
            contract += attempt_violations(inst.name, at)
            if at.admitted is False:
                attempt_other["not_admitted"] += 1
            elif at.window is not None:
                attempt_windows["prefix_len"].append(at.window)
            elif at.window_vs_s is not None:
                attempt_windows["vs_S_upper_bound"].append(at.window_vs_s)
            else:
                attempt_other["unknown"] += 1           # old fork (no usage), or no open segment for its sid
        if verify_media:
            media_bad += [(inst.name, sha, why) for sha, why in verify_media_blobs(inst)]
        if inst.truncated_tail:
            # warning: "<instance>: the last line of events is half-written (server crash?)"
            warnings.append(f"{inst.name}: events 最後一行寫到一半（server crash？）")
        tok = {"sha256_recorded": _contains(inst.server, vocab.sha256)}
        for key in ("image_pad", "video_pad"):
            if key in inst.server:      # the fork WIP records them; the spec does not list them
                tok[key] = inst.server[key] == getattr(vocab, key)
                if not tok[key]:
                    # failure: "<instance>: server.json <key>=<id>, the tokenizer has <id>"
                    failures.append(f"{inst.name}: server.json {key}={inst.server[key]}，tokenizer 是 {getattr(vocab, key)}")
        if not tok["sha256_recorded"]:
            # warning: "<instance>: tokenizer.json's sha256 <sha>… not found in server.json (cannot prove the same tokenizer)"
            warnings.append(f"{inst.name}: server.json 裡找不到 tokenizer.json 的 sha256 {vocab.sha256[:12]}…（無法證明同一個 tokenizer）")
        per_instance.append({"instance": inst.name, "exact_resume": inst.exact_resume, "segments": len(segs),
                             "events": inst.n_events,
                             "attempts": dict(Counter(a["kind"] for a in inst.attempts)),
                             "open": sum(s.status == "open" for s in segs), "tokenizer": tok,
                             "truncated_tail": inst.truncated_tail})

    if invariants:
        # failure, matched verbatim by the RL orchestrator (INVARIANT_FAILURE): "N invariant violations: {code: n, ...}"
        failures.append(f"不變量違反 {sum(invariants.values())} 處：{dict(invariants)}")
    if status["open"]:
        # warning: "N segments without seg_close (crash or still running)"
        warnings.append(f"{status['open']} 段沒有 seg_close（crash 或還在跑）")
    n_overlap = sum(s.routing_overlap_mismatch for inst in instances for s in inst.segments.values())
    if n_overlap:
        # warning: "N rewind-overlap rows whose two routing records differ (that row was recomputed; the later one is used)"
        warnings.append(f"{n_overlap} 處回捲重疊列的 routing 兩筆不同（那一列被重算；以後一筆為準）")
    n_shifted = sum(s.routing_start_shifted for inst in instances for s in inst.segments.values())
    if n_shifted:
        # warning: "N appends whose routing start is not max(prefix_len-1, 0) (late split; RL marks the missing head M)"
        warnings.append(f"{n_shifted} 筆 append 的 routing 起點不是 max(prefix_len-1, 0)（晚分岔；RL 把缺的段頭當 M）")

    lossy = {}
    for inst in instances:
        st = None if ledger_status is None else ledger_status.get(inst.name)
        if st is None:
            st = inst.last_status   # the ledger's own status/end events (every 300 s; a killed server writes no end)
        if st is None:
            lossy[inst.name] = None
            # failure: "<instance>: lossy unknown (no status snapshot, and the ledger has no status/end event)"
            failures.append(f"{inst.name}: lossy 不知道（沒有 status 快照，帳本也沒有 status/end 事件）")
            continue
        n_lossy = max(int(st["lossy"]), inst.lossy_dropped)
        lossy[inst.name] = n_lossy
        if n_lossy != 0:
            # failure: "<instance>: lossy=N (the writer queue was full and dropped events)"
            failures.append(f"{inst.name}: lossy={n_lossy}（writer 佇列滿丟了事件）")
        if st["writer_error"] is not None:
            failures.append(f"{inst.name}: writer_error={st['writer_error']}")

    unexpected = {k: v for k, v in splits.items() if k.split(":", 1)[1] not in expected}
    if unexpected:
        # failure, matched verbatim by the RL orchestrator (UNEXPECTED_SPLITS_FAILURE): "unexpected split reasons: {...}"
        failures.append(f"預期外的分岔原因：{unexpected}")

    for name, c in (("logprobs", lp), ("routing", rt), ("top_logprobs", tk)):
        if c["with"] and (c["error"] or c["none"]):
            # failure: "<name> present on only some appends: {...}"
            failures.append(f"{name} 只有部分 append 有：{dict(c)}")

    if media_bad:
        # failure: "N media blobs whose content does not match the sha256: [...]"
        failures.append(f"{len(media_bad)} 個 media blob 內容與 sha256 不符：{media_bad[:3]}")

    if contract:
        # failure: "<RESUME_CONTRACT_FAILURE> N records (fork bug, contract v2.1: ledger.resume_problems): <first 5>"
        failures.append(f"{RESUME_CONTRACT_FAILURE} {len(contract)} 筆（fork bug，契約 v2.1：ledger.resume_problems）："
                        + "；".join(contract[:5]))
    if sum(window_unknown.values()):
        # warning: "N extension appends have neither resume.pos nor cached_tokens: recompute window unknown {...}"
        warnings.append(f"{sum(window_unknown.values())} 筆延伸 append 沒有 resume.pos 也沒有 cached_tokens：重算窗不知道 "
                        f"{dict(window_unknown)}")
    all_windows = [w for ws in windows.values() for w in ws]
    window_report = {
        **window_summary(all_windows), "unknown": sum(window_unknown.values()), "new_tokens": new_tokens,
        "reforward_over_new": (sum(max(w, 0) for w in all_windows) / new_tokens) if new_tokens else None,
        "by_mode": {mode: {**window_summary(ws), "unknown": window_unknown[mode]}
                    for mode, ws in sorted(windows.items())},
        # recomputed positions (summed over windows; a position recomputed by two windows counts twice): whether the
        # first forward was the previous record's gen or prefill, and the weight versions of the two forwards
        "positions": {k: reforward[k] for k in ("positions", "from_gen", "from_prefill", "D_same", "D_cross", "D_unknown")},
        "appends_cross": reforward["appends_cross"], "segments_cross": reforward["segments_cross"],
    }
    for mode, n in window_unknown.items():                         # unknown-only modes still get a row
        window_report["by_mode"].setdefault(mode, {"n": 0, "unknown": n})
    # attempts are listed apart: their positions are not in any segment's S (prepare does not classify them)
    window_report["attempts"] = {**{k: window_summary(v) for k, v in attempt_windows.items()},
                                 "not_admitted": attempt_other["not_admitted"], "unknown": attempt_other["unknown"]}

    join = None
    tasks_without = []
    if not manifests:
        # failure: "no manifest: neither rid join nor at-least-one-segment-per-task can be checked"
        failures.append("沒有 manifest：rid join 與每題至少一段都無從檢查")
    else:
        join = rid_join(instances, manifests)
        if not join["complete"]:
            # failure, matched verbatim by the RL orchestrator (RID_JOIN_FAILURE):
            # "rid join incomplete: N missing, N extra, N with the wrong window"
            failures.append(f"rid join 不完整：缺 {len(join['missing'])}、多 {len(join['extra'])}、"
                            f"window 對錯 {len(join['wrong_window'])}")
        have = Counter()
        for inst in instances:
            for seg in inst.segments.values():
                if seg.sid is not None and seg.appends:
                    have[split_sid(seg.sid)[0]] += 1
        loaded = {inst.name for inst in instances}
        for m in manifests:
            for srv in m.servers:
                if srv["instance"] not in loaded:
                    # failure, starts with the manifest path (the RL orchestrator matches that prefix):
                    # "<manifest>: ledger instance <instance> used by the run was not loaded (ledger at <ledger_dir>)"
                    failures.append(f"{m.path}: run 用的帳本 instance {srv['instance']} 沒有讀進來"
                                    f"（帳本在 {srv['ledger_dir']}）")
            for info in m.threads.values():
                if info.in_sessions != info.in_digest:
                    warnings.append(f"{m.run}/{info.item}: thread {info.thread} in_sessions={info.in_sessions} "
                                    f"in_digest={info.in_digest}")
            for item in sorted(m.tasks()):
                if any(have[t] for t, info in m.threads.items() if info.item == item):
                    continue
                if m.items[item]["outcome"]["status"] == "failed" and not m.items[item]["threads"]:
                    # warning: "<run>/<item>: the harness recorded failed, no thread (no segment to collect)"
                    warnings.append(f"{m.run}/{item}: harness 記為 failed、沒有 thread（沒有可收的段）")
                else:
                    tasks_without.append(f"{m.run}/{item}")
        if tasks_without:
            # failure, matched verbatim by the RL orchestrator (NO_SEGMENT_FAILURE): "N tasks / games have no segment at all"
            failures.append(f"{len(tasks_without)} 題／局沒有任何段：{tasks_without[:5]}")
        if join["compaction_other_window"]:
            # warning: "N compaction requests fall in a window outside the manifest"
            warnings.append(f"{len(join['compaction_other_window'])} 個 compaction 請求落在 manifest 以外的 window")

    return {
        "collection_passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "instances": per_instance,
        "segments": dict(status),
        "invariants": dict(invariants),
        "violations": violations[:50],
        "split_reasons": dict(splits),
        "expected_splits": sorted(expected),
        "logprobs": dict(lp),
        "routing": dict(rt),
        "top_logprobs": dict(tk),
        "usage": dict(usage),
        # number of messages swapped back to the sampled ids, by kind (text = re-tokenisation, empty_think = empty
        # thinking, own_turn = the previous turn's generation rewritten by the client, own_turn_sticky = a message
        # taken over before, swapped back again)
        "replay_spans": dict(replayed),
        "replay_own_turn_closure_tokens": closure_tokens,   # tokens own_turn filled in after a truncated generation (prompt)
        "cached_over_prefix_mean": (sum(cached_ratio) / len(cached_ratio)) if cached_ratio else None,
        "window": window_report,
        "resume": {"modes": dict(resume_modes), "fallback_reasons": dict(fallback_reasons),
                   "attempt_modes": dict(attempt_modes), "attempt_fallback_reasons": dict(attempt_reasons),
                   "contract_violations": len(contract),
                   # exact with rewind > 0: window off {0, -1} is not a violation; window > 0 positions are classified
                   # as forwarded twice like any other window. By ledger.rewind_cause: concurrent (S changed by another
                   # commit while it ran) vs the single-request rewinds (stop_substituted, truncated)
                   "exact_rewinds": {c: {k: exact_rewinds[c][k] for k in ("appends", "window_gt0", "window_lt_minus1",
                                                                          "positions")} for c in REWIND_CAUSES}},
        "media_bad": media_bad,
        "lossy": lossy,
        "rid_join": join,
        "tasks_without_segments": tasks_without,
    }
