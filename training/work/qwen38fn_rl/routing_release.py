"""Decide which live-ledger routing shards may leave local disk (NFS offloader, routing_gc.py): only those no trainable
episode can read.

prepare_rl reads routing for every episode it trains. A shard moved to NFS while its episodes still wait in the pool
makes the learner read it back over NFS (10-10: 23 MB/s, prepare 75 s -> 720 s). Shard n is released only when
  * covered: a complete append line referencing a later shard has been read. The ledger writer is one thread that
    writes an append's routing (rolling to the next shard first when it does not fit) and then its event line, so the
    shard index never decreases along the event stream: once a line names shard m > n, every append into n has been
    read. A shard file that merely exists (rolled, its first event not yet written) proves nothing; and
  * consumed: every session id appended into it belongs to a group already in batches.jsonl or abandoned in
    groups.jsonl (sid null appends are probes, never joined) -- a failed episode of such a group (groups.jsonl
    failed_episodes: in the group manifest without threads, never trained) by thread, from its proxy log and run_arm
    manifest (failed_names; v6b 10-27: such episodes pinned every shard they touched), or to an in-flight group the orchestrator set aside
    under groups/abandoned-<stamp>/ (--abandon-incomplete-round) or quarantined under groups/quarantined/ (its harness
    failed or timed out: never in groups.jsonl, never trained; its manifests name its ids -- a failed episode's
    sessions by thread, aside_manifests), or to a held-out group of the stream loop (heldout/vNNNNNN/<task>) with a
    group row in heldout.jsonl -- completed or failed, it is never trained; the orchestrator writes the row only after
    the harness exited. Its sessions are named by thread (heldout_names): the group manifest, the episodes' run_arm
    manifests, and the proxy logs (one row per request, written as it ends: they still name the sessions of a harness
    SIGKILLed before any manifest). An id nothing names keeps it local (a group still running, an aside group without
    any manifest, a held-out episode that left no manifest and no proxy row): such finished groups are listed in the
    report's ``unnamed`` -- their shards stay until someone looks, never silently.
There is no time-based release: shard mtime is not a weight version (review R3, 10-10).
Events are read incrementally (byte offsets in --state), in file order from one listing, complete lines only; the
compact append layout takes a fast path, any other line is parsed as JSON. groups.jsonl / batches.jsonl are read up
to their last newline (a writer may be mid-line); a complete line that does not parse raises.

  routing_release.py --run-root R --out R/offload/routing_release.json [--state ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

FORMAT = "qwen38fn-rl-routing-release-v2"
STATE_FORMAT = "qwen38fn-rl-routing-release-state-v3"
#: v2 states (before held-out groups): the same offsets / sids, no held-out cache -- carried over, not rescanned
STATE_FORMAT_V2 = "qwen38fn-rl-routing-release-state-v2"
HELDOUT_FILE = "heldout.jsonl"
SHARD = re.compile(r"routing\.(\d+)\.bin")
EV = re.compile(rb'\{"ev":"([a-z_]+)"')
APPEND_SID = re.compile(rb'\{"ev":"append","sid":(null|"([^"]+)"),')
APPEND_ROUTING = re.compile(rb',"routing":(null|\{"file":"routing\.(\d+)\.bin")')
GROUP_GLOB = "g[0-9][0-9][0-9][0-9][0-9][0-9]"


def complete_lines(path: Path) -> list[bytes]:
    """Lines up to the last newline: the writer may be mid-line."""
    if not path.exists():
        return []
    data = path.read_bytes()
    return data[:data.rfind(b"\n") + 1].splitlines()


def parse_append(line: bytes) -> tuple[bool, str | None, int | None]:
    """(is_append, sid, routing shard) of one complete event line."""
    ev = EV.match(line)
    if ev is not None and ev.group(1) != b"append":
        return False, None, None
    m = APPEND_SID.match(line) if ev is not None else None
    r = APPEND_ROUTING.findall(line) if m is not None else []
    if m is not None and len(r) == 1:
        sid = None if m.group(1) == b"null" else m.group(2).decode()
        return True, sid, None if r[0][0] == b"null" else int(r[0][1])
    doc = json.loads(line)                                  # any other layout: decide on the parsed fields
    if doc["ev"] != "append":
        return False, None, None
    routing = doc["routing"]
    if routing is None:
        return True, doc["sid"], None
    shard = SHARD.fullmatch(routing["file"])
    if shard is None:
        raise RuntimeError(f"append routing file {routing['file']!r} is not routing.<n>.bin")
    return True, doc["sid"], int(shard.group(1))


def scan_events(inst: Path, offsets: dict, shard_sids: dict, covered: int) -> int:
    """Add (shard -> sid) pairs from complete lines appended since the stored byte offsets; return the highest shard
    any append has referenced so far."""
    for events in sorted(inst.glob("events.*.jsonl"), key=lambda p: int(p.name.split(".")[1])):
        start = offsets.get(events.name, 0)
        with events.open("rb") as f:
            f.seek(start)
            data = f.read()
        end = data.rfind(b"\n") + 1
        for line in data[:end].splitlines():
            is_append, sid, shard = parse_append(line)
            if not is_append or shard is None:
                continue
            covered = max(covered, shard)
            if sid is not None:
                shard_sids.setdefault(str(shard), set()).add(sid)
        offsets[events.name] = start + end
    return covered


def manifest_sids(group_dir: Path) -> list[str]:
    manifest = json.loads((group_dir / "tito_manifest.json").read_text())
    return [w["sglang_session_id"] for item in manifest["items"] for t in item["threads"] for w in t["windows"]]


def aside_manifests(group_dir: Path) -> list[Path]:
    """A set-aside / quarantined group's manifests: the group's own (if the harness got to write it) and its episodes'
    run_arm manifests (run_arc3_level: episodes/<ep>/arm/, outside the agent's workspace). A failed episode is in the
    group manifest without threads, but run_arm lists its thread (windows empty: no digest), so its sessions -- sid
    ``<thread>:<window>`` -- are matched by thread."""
    own = group_dir / "tito_manifest.json"
    return ([own] if own.is_file() else []) + sorted(group_dir.glob("episodes/*/arm/tito_manifest.json"))


def manifest_threads(path: Path) -> list[str]:
    return [t["thread"] for item in json.loads(path.read_text())["items"] for t in item["threads"]]


def heldout_names(group_dir: Path, task: str, samples: int) -> tuple[list[str], list[str]]:
    """A finished held-out group's threads, and its episodes (``<task>__s<k>``) that nothing names. Sources, per episode:
    the group manifest's item (threads of every item; a failed one has none), the episode's run_arm manifest
    (run_arc3_level episodes/<ep>/arm/: written on SIGTERM too) and its proxy log proxy/<ep>.jsonl (both harnesses: one
    row per request with the codex session id ``<thread>:<window>``, written as the request ends -- it names the sessions
    of a harness SIGKILLed before any manifest). An unnamed episode either never reached the model or pins the shards of
    its sessions for good: listed, never silent."""
    if not group_dir.is_dir():
        return [], []                                     # the harness never created its run dir: no episode ever ran
    named: dict[str, set[str]] = {f"{task}__s{k}": set() for k in range(samples)}
    own = group_dir / "tito_manifest.json"
    for item in json.loads(own.read_text())["items"] if own.is_file() else ():
        if item["id"] not in named:
            raise RuntimeError(f"{own}: item {item['id']!r} is not one of the {samples} episodes of {task}")
        named[item["id"]].update(t["thread"] for t in item["threads"])
    for episode, threads in named.items():
        arm = group_dir / "episodes" / episode / "arm" / "tito_manifest.json"
        if arm.is_file():
            threads.update(manifest_threads(arm))
        for line in complete_lines(group_dir / "proxy" / f"{episode}.jsonl"):
            sid = json.loads(line).get("session_id")             # request rows; forbidden / close rows carry none
            if sid is not None:
                threads.add(sid.rsplit(":", 1)[0])
    return sorted(set().union(*named.values())), sorted(e for e, threads in named.items() if not threads)


def failed_names(group_dir: Path, episodes: list[str]) -> dict:
    """A completed training group's failed episodes (groups.jsonl ``failed_episodes``): their threads, from the proxy log
    proxy/<ep>.jsonl (one row per request with the codex session id ``<thread>:<window>``) and the run_arm manifest
    episodes/<ep>/arm/tito_manifest.json (group_pack moves episodes/ away once the group is consumed; proxy/ stays), and
    the ones nothing names (they never reached the model, or pin their shards: listed in ``unnamed``)."""
    named: dict[str, set[str]] = {}
    for episode in episodes:
        threads = named.setdefault(episode, set())
        arm = group_dir / "episodes" / episode / "arm" / "tito_manifest.json"
        if arm.is_file():
            threads.update(manifest_threads(arm))
        for line in complete_lines(group_dir / "proxy" / f"{episode}.jsonl"):
            sid = json.loads(line).get("session_id")             # request rows; forbidden / close rows carry none
            if sid is not None:
                threads.add(sid.rsplit(":", 1)[0])
    return {"threads": sorted(set().union(*named.values())), "unnamed": sorted(e for e, t in named.items() if not t)}


def load_state(state_path: Path) -> dict:
    fresh = {"format": STATE_FORMAT, "offsets": {}, "shard_sids": {}, "covered": {}, "gid_sids": {}, "heldout": {}}
    if not state_path.exists():
        return fresh
    state = json.loads(state_path.read_text())
    if state.get("format") == STATE_FORMAT_V2:                # written before held-out groups existed: nothing cached
        return {**state, "format": STATE_FORMAT, "heldout": {}}
    return state if state.get("format") == STATE_FORMAT else fresh      # v1 state lacks coverage: rescan from zero


def release(run_root: Path, state_path: Path) -> dict:
    state = load_state(state_path)
    offsets, gid_sids, covered = state["offsets"], state["gid_sids"], state["covered"]
    shard_sids = {inst: {n: set(s) for n, s in shards.items()} for inst, shards in state["shard_sids"].items()}

    completed, abandoned = {}, set()
    groups_path = run_root / "groups.jsonl"
    for line in complete_lines(groups_path):
        doc = json.loads(line)
        if doc["event"] == "completed":
            completed[doc["gid"]] = doc
        elif doc["event"] == "abandoned":
            abandoned.add(doc["gid"])
        elif doc["event"] == "set_aside":       # never completed; its dir is under groups/abandoned-<stamp>/ (below)
            pass
        else:
            raise RuntimeError(f"{groups_path}: unknown event {doc['event']!r}")
    consumed = set(abandoned)
    for line in complete_lines(run_root / "batches.jsonl"):
        consumed.update(g["gid"] for g in json.loads(line)["groups"])
    if consumed - set(completed):
        raise RuntimeError(f"consumed group(s) never completed: {sorted(consumed - set(completed))[:10]}")
    sid_gid = {}
    for gid, doc in completed.items():
        key = str(gid)
        if key not in gid_sids:
            gid_sids[key] = manifest_sids(run_root / "groups" / doc["group"])
        for sid in gid_sids[key]:
            sid_gid[sid] = gid

    unnamed: dict[str, object] = {}                       # finished groups whose sessions nothing (fully) names: the report
    named_threads = {sid.rsplit(":", 1)[0]: gid for sid, gid in sid_gid.items()}
    failed_cache = state.setdefault("failed", {})         # gid -> failed_names: written once, the group is complete
    thread_failed: dict[str, int] = {}                    # thread -> completed group of a failed episode
    for gid, doc in completed.items():
        episodes = doc.get("failed_episodes") or []        # absent / empty: no episode of the group failed
        if not episodes:
            continue
        key = str(gid)
        if key not in failed_cache:
            failed_cache[key] = failed_names(run_root / "groups" / doc["group"], episodes)
        for thread in failed_cache[key]["threads"]:
            if named_threads.get(thread, gid) != gid or thread_failed.get(thread, gid) != gid:
                raise RuntimeError(f"failed episode thread {thread} of group {gid} also belongs to "
                                   f"{named_threads.get(thread, thread_failed.get(thread))}")
            thread_failed[thread] = gid
        if failed_cache[key]["unnamed"]:
            unnamed[f"failed:{doc['group']}"] = failed_cache[key]["unnamed"]

    aside = 0
    thread_aside: dict[str, str] = {}                     # thread -> aside / held-out group: sessions matched by thread
    for group_dir in sorted([*run_root.glob(f"groups/abandoned-*/{GROUP_GLOB}"),
                             *run_root.glob(f"groups/quarantined/{GROUP_GLOB}")]):
        manifests = aside_manifests(group_dir)
        key = f"aside:{group_dir.parent.name}/{group_dir.name}"
        if not manifests:
            unnamed[key] = "no manifest"                  # nothing names its ids: they stay unknown (kept local)
            continue
        if key not in gid_sids:
            gid_sids[key] = manifest_sids(group_dir) if (group_dir / "tito_manifest.json").is_file() else []
        consumed.add(key)
        aside += 1
        for sid in gid_sids[key]:
            if sid in sid_gid or sid.rsplit(":", 1)[0] in thread_failed:
                raise RuntimeError(f"{group_dir}: session {sid} also belongs to completed group "
                                   f"{sid_gid.get(sid, thread_failed.get(sid.rsplit(':', 1)[0]))}")
            sid_gid[sid] = key
        for manifest in manifests:                        # re-read every pass (a few small files): not in the state
            for thread in manifest_threads(manifest):
                thread_aside[thread] = key

    heldout, heldout_cache = 0, state["heldout"]
    sid_threads = {sid.rsplit(":", 1)[0]: gid for sid, gid in sid_gid.items()}
    heldout_path = run_root / HELDOUT_FILE
    for line in complete_lines(heldout_path):
        doc = json.loads(line)
        if doc["kind"] == "round":
            continue                                      # a round summary, no group of its own
        if doc["kind"] != "group":
            raise RuntimeError(f"{heldout_path}: unknown kind {doc['kind']!r}")
        key = f"heldout:{doc['group']}"
        if key not in heldout_cache:                      # the row is written after the harness exited: files are final
            threads, missing = heldout_names(run_root / "heldout" / doc["group"], doc["task"], doc["samples"])
            heldout_cache[key] = {"threads": threads, "unnamed": missing}
        if key in consumed:
            raise RuntimeError(f"{heldout_path}: held-out group {doc['group']} has two group rows")
        consumed.add(key)
        heldout += 1
        if heldout_cache[key]["unnamed"]:
            unnamed[key] = heldout_cache[key]["unnamed"]
        for thread in heldout_cache[key]["threads"]:
            owner = sid_threads.get(thread, thread_aside.get(thread, thread_failed.get(thread)))
            if owner is not None:
                raise RuntimeError(f"held-out {doc['group']}: thread {thread} also belongs to {owner}")
            thread_aside[thread] = key

    instances, pending, dirs = {}, {}, {}
    for server in sorted(run_root.glob("replicas/r*/ledger/*/server.json")):
        inst = server.parent
        offs = offsets.setdefault(inst.name, {})
        shards = shard_sids.setdefault(inst.name, {})
        covered[inst.name] = scan_events(inst, offs, shards, covered.get(inst.name, -1))
        files = sorted(int(m.group(1)) for f in inst.iterdir() if (m := SHARD.fullmatch(f.name)))
        for n in [n for n in shards if int(n) < covered[inst.name] and int(n) not in files]:
            del shards[n]                                   # a removed shard no append can reach again
        dirs[inst.name] = str(inst)
        ok, why = [], {}
        for n in files:
            if n >= covered[inst.name]:
                why[str(n)] = "uncovered"
                continue
            sids = shards.get(str(n), set())
            waiting = {sid_gid[s] if s in sid_gid else
                       thread_aside.get(s.rsplit(":", 1)[0], thread_failed.get(s.rsplit(":", 1)[0])) for s in sids} - consumed
            if not waiting:
                ok.append(n)
            else:
                why[str(n)] = {"waiting_gids": sorted(g for g in waiting if g is not None)[:8],
                               "unknown_sids": None in waiting}
        instances[inst.name], pending[inst.name] = ok, why
    state["shard_sids"] = {inst: {n: sorted(s) for n, s in shards.items()} for inst, shards in shard_sids.items()}
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state) + "\n")
    os.replace(tmp, state_path)
    return {"format": FORMAT, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "consumed_groups": len(consumed) - aside - heldout, "aside_groups": aside, "heldout_groups": heldout,
            "unnamed": unnamed, "covered": dict(covered), "instances": instances, "pending": pending, "dirs": dirs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state", type=Path)
    args = ap.parse_args()
    report = release(args.run_root, args.state or args.out.with_name("routing_release_state.json"))
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, default=str) + "\n")
    os.replace(tmp, args.out)
    print(json.dumps({k: (len(v) if isinstance(v, dict) and k != "covered" else v)
                      for k, v in report.items() if k not in ("pending", "dirs")}), flush=True)


if __name__ == "__main__":
    main()
