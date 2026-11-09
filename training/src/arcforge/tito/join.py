"""Map ledger segments back to the harness's tasks / games (``tito_manifest.json``) and fetch the outcome.

The manifest is written by the harness (``build()`` in ``manifest.py``; arc2 ``run_arc2.py:tito_items``, arc3
``run_arm.py:tito_items``)::

    {"schema_version": 1, "harness": "arc2" | "arc3", "written": …, "config": {…}, "server": {"instance", "ledger_dir", …},
     ["meta": {…},]
     "items": [{"id": "<task | game | episode>", ["task": "<task>", "kind": "…",] "outcome": {…}, "digests": […],
                "threads": [{"thread": "…", "in_sessions": bool, "in_digest": bool,
                             "windows": [{"window_number": n, "sglang_session_id": "<thread>:<n>",
                                          "response_ids": […], "compaction_response_id": "…" | null}]}]}],
     "counts": {…}}

This module only reads and checks shapes (raises on mismatch); it does not recompute digests.

- segment ↔ thread: ``sid = <thread>:<n>`` (codex patch 0002), ``n`` is the window (+1 per compaction).
- rid ↔ response id: the ``rid`` of sglang Responses is the response id (``rid=request.request_id`` in
  ``serving_responses.py``).
- group (RL grouping / DFlash split key) = the item's ``task`` (optional key; without it the item id, i.e. the task or
  game); episode = ``<run dir name>/<item id>`` (run dir = the directory holding the manifest). ``run_arc2_seq.py`` has
  several episodes per task: the item id is ``<task>__s<k>`` and ``task`` is the task id, so a task's s0 / s2 / refiner
  share a group while the episodes stay separate; outcome still goes by item id.

The outcome is the manifest item's ``outcome`` (written by the harness at the end of the run, the only source):

- arc2: ``status == "finished"`` → ``reward = score`` (partial credit pass@1 over the task's test pairs); ``failed`` →
  reward None. When the run dir also has ``results/<task>.json``, the score is cross-checked, raising on a difference
  (the two records disagree).
- arc3 full game (run_arm): **reward undefined (None)**, ``detail`` carries levels_completed / won / actions. See
  ``ARC3_TODO``.
- arc3 single-level episode (``work/arc_code/run_arc3_level.py``): outcome carries ``score`` (level cleared within the
  quota = 1), same path as arc2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .ledger import Instance, Segment, split_sid

SCHEMA_VERSION = 1
HARNESSES = ("arc2", "arc3")
ARC3_TODO = (
    "arc3 reward is not defined yet: the manifest outcome has levels_completed / won / actions, but RHAE needs each "
    "level's baseline_actions (metadata.json of ARC_ARC3_CORPUS, src/arcforge/arc_code/score.py) and the actions used "
    "per level; per-window scoring also lacks the time each level was completed (the arc_code game ledger has it, the "
    "manifest carries no path to it).")


@dataclass(frozen=True)
class Window:
    n: int
    response_ids: tuple[str, ...]
    compaction_response_id: str | None


@dataclass(frozen=True)
class ThreadInfo:
    thread: str
    item: str                 # task id (arc2) or game id (arc3); for seq the episode (<task>__s<k>)
    task: str                 # grouping key: the item's task field, or the item without one
    in_sessions: bool
    in_digest: bool
    windows: tuple[Window, ...]
    manifest: "Manifest"

    @property
    def episode(self) -> str:
        return f"{self.manifest.run}/{self.item}"


@dataclass(frozen=True)
class Outcome:
    status: str               # the harness's status (arc2: finished / failed; arc3: play status)
    reward: float | None
    detail: dict


class ManifestError(RuntimeError):
    """The manifest's shape, outcome or rid join does not hold. ``manifests``: which manifests the problem is in (when
    it can be pinned down; prepare_rl --error-json maps it back to groups, and the orchestrator isolates only those
    groups); empty = cannot be pinned to any one (e.g. no manifest given at all)."""

    def __init__(self, message: str, *, manifests: Iterable[str | Path] = ()):
        super().__init__(message)
        self.manifests: tuple[Path, ...] = tuple(dict.fromkeys(Path(p) for p in manifests))


def _keys(obj, keys: Iterable[str], where: str) -> None:
    if not isinstance(obj, dict):
        raise ManifestError(f"{where}: must be an object")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ManifestError(f"{where}: missing {missing}")


def _str(value, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: must be a non-empty string, got {value!r}")
    return value


class Manifest:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self._load()
        except ManifestError as error:            # every shape mismatch lies in this one: attach its path
            raise ManifestError(str(error), manifests=(self.path,)) from error

    def _load(self) -> None:
        doc = json.loads(self.path.read_text())
        where = str(self.path)
        _keys(doc, ("schema_version", "harness", "server", "items"), where)
        if doc["schema_version"] != SCHEMA_VERSION:
            raise ManifestError(f"{where}: schema_version={doc['schema_version']!r}, expected {SCHEMA_VERSION}")
        if doc["harness"] not in HARNESSES:
            raise ManifestError(f"{where}: harness={doc['harness']!r} not in {HARNESSES}")
        _keys(doc["server"], ("instance", "ledger_dir"), f"{where} server")
        self.harness: str = doc["harness"]
        self.server: dict = doc["server"]
        #: every backend this run used (the manifest's ``servers``; absent = only ``server``)
        self.servers: list[dict] = doc["servers"] if "servers" in doc else [doc["server"]]
        for i, srv in enumerate(self.servers):
            _keys(srv, ("instance", "ledger_dir"), f"{where} servers[{i}]")
        if self.servers[0] != self.server:
            raise ManifestError(f"{where}: servers[0] is not server")
        self.run_dir = self.path.parent
        self.run = self.run_dir.name
        #: the harness's run-level description (optional key; seq has the task pool, pool)
        self.meta: dict | None = doc["meta"] if "meta" in doc else None
        self.items: dict[str, dict] = {}
        self.threads: dict[str, ThreadInfo] = {}
        for i, item in enumerate(doc["items"]):
            w = f"{where} items[{i}]"
            _keys(item, ("id", "outcome", "threads"), w)
            iid = _str(item["id"], f"{w}.id")
            if iid in self.items:
                raise ManifestError(f"{w}: item id {iid} duplicated")
            _keys(item["outcome"], ("status",), f"{w}.outcome")
            task = _str(item["task"], f"{w}.task") if "task" in item else iid
            self.items[iid] = item
            for j, t in enumerate(item["threads"]):
                tw = f"{w}.threads[{j}]"
                _keys(t, ("thread", "in_sessions", "in_digest", "windows"), tw)
                thread = _str(t["thread"], f"{tw}.thread")
                if not isinstance(t["in_sessions"], bool) or not isinstance(t["in_digest"], bool):
                    raise ManifestError(f"{tw}: in_sessions / in_digest must be bool")
                if thread in self.threads:
                    raise ManifestError(f"{tw}: thread {thread} is also in item {self.threads[thread].item}")
                windows = []
                for k, win in enumerate(t["windows"]):
                    ww = f"{tw}.windows[{k}]"
                    _keys(win, ("window_number", "sglang_session_id", "response_ids", "compaction_response_id"), ww)
                    n = win["window_number"]
                    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                        raise ManifestError(f"{ww}: window_number={n!r}")
                    if win["sglang_session_id"] != f"{thread}:{n}":
                        raise ManifestError(f"{ww}: sglang_session_id={win['sglang_session_id']!r} ≠ {thread}:{n}")
                    comp = win["compaction_response_id"]
                    windows.append(Window(n=n, response_ids=tuple(_str(r, ww) for r in win["response_ids"]),
                                          compaction_response_id=None if comp is None else _str(comp, ww)))
                if [x.n for x in windows] != list(range(len(windows))):
                    raise ManifestError(f"{tw}: window_number must be consecutive from 0, got {[x.n for x in windows]}")
                self.threads[thread] = ThreadInfo(thread=thread, item=iid, task=task, in_sessions=t["in_sessions"],
                                                  in_digest=t["in_digest"], windows=tuple(windows), manifest=self)

    def tasks(self) -> set[str]:
        """Item ids (for seq, episodes; the grouping key is separately in ``ThreadInfo.task``)."""
        return set(self.items)

    def outcome(self, item: str) -> Outcome:
        raw = self.items[item]["outcome"]
        status = raw["status"]
        if self.harness == "arc3" and "score" not in raw:
            # manifest of a full-game collection (run_arm): reward undefined (ARC3_TODO). Single-level episodes
            # (run_arc3_level.py) carry score and take the same path as arc2 below.
            return Outcome(status=status, reward=None, detail=raw)
        if status == "failed":
            return Outcome(status=status, reward=None, detail=raw)
        if status != "finished":
            raise ManifestError(f"{self.path}: {self.harness} item {item} has status={status!r} "
                                f"(only finished / failed accepted)",
                                manifests=(self.path,))
        try:
            _keys(raw, ("score", "correct", "stop"), f"{self.path} items[{item}].outcome")
            results = self.run_dir / "results" / f"{item}.json"
            if results.is_file():                 # absent if the manifest was moved out of the run dir; else reconcile
                r = arc2_results(results)
                if r["score"] != raw["score"] or r["correct"] != raw["correct"]:
                    raise ManifestError(f"{item}: manifest has score={raw['score']} correct={raw['correct']}, "
                                        f"{results} has {r['score']} {r['correct']}")
        except ManifestError as error:            # a problem of this manifest (and its results)
            raise ManifestError(str(error), manifests=(self.path,)) from error
        return Outcome(status=status, reward=float(raw["score"]), detail=raw)


def arc2_results(path: Path) -> dict:
    """Result file of ``solve()`` in ``work/arc_code/arc2/run_arc2.py``."""
    r = json.loads(Path(path).read_text())
    _keys(r, ("task", "score", "correct", "stop"), str(path))
    if r["task"] != Path(path).stem:
        raise ManifestError(f"{path}: task field {r['task']!r} differs from the file name")
    return r


def load_manifests(paths: Iterable[str | Path]) -> list[Manifest]:
    out = [Manifest(p) for p in paths]
    owner: dict[str, Path] = {}
    for m in out:
        for thread in m.threads:
            if thread in owner:
                raise ManifestError(f"thread {thread} is in both {owner[thread]} and {m.path}",
                                    manifests=(owner[thread], m.path))
            owner[thread] = m.path
    return out


class Joiner:
    """sid → manifest thread. Segments in no manifest (other traffic on the same server, single segments) give None."""

    def __init__(self, manifests: Iterable[Manifest]):
        self.manifests = list(manifests)
        self.by_thread = {t: info for m in self.manifests for t, info in m.threads.items()}

    def lookup(self, seg: Segment) -> ThreadInfo | None:
        if seg.sid is None:
            return None
        return self.by_thread.get(seg.thread)       # not in the manifest = not this run's harness traffic

    def outcome(self, info: ThreadInfo) -> Outcome:
        return info.manifest.outcome(info.item)


def rid_join(instances: Iterable[Instance], manifests: Iterable[Manifest]) -> dict:
    """Manifest response ids ↔ ledger rids (appends and attempts).

    ``complete`` (the spec's "rid join 100%"): the ledger has every manifest id, the manifest has every ledger rid that
    belongs to a manifest thread, and ordinary requests' rids fall in the window the manifest names. Which window id a
    compaction request carries is listed as unverified in the spec (manifest.py says it belongs to the old window), so
    for it only the same thread is required, and a different window is counted separately as
    ``compaction_other_window``."""
    manifests = list(manifests)
    seen: dict[str, dict[str, set[int]]] = {}
    # rids the server never committed (abort, orphaned): the client usually never saw a response for
    # them (codex killed at the wall deadline mid-stream), so they are not "extra"
    attempted: dict[str, set[str]] = {}
    n_single = 0
    for inst in instances:
        for seg in inst.segments.values():
            if seg.sid is None:
                n_single += 1
                continue
            thread, n = split_sid(seg.sid)
            for rid in seg.rids():
                seen.setdefault(thread, {}).setdefault(rid, set()).add(n)
        for att in inst.attempts:
            if att["sid"] is None:
                continue
            thread, n = split_sid(att["sid"])
            seen.setdefault(thread, {}).setdefault(att["rid"], set()).add(n)
            attempted.setdefault(thread, set()).add(att["rid"])
    missing, extra, wrong_window, compaction_other, aborted = [], [], [], [], []
    n_expected = 0
    manifest_threads = set()
    for m in manifests:
        for thread, info in m.threads.items():
            manifest_threads.add(thread)
            got = seen.get(thread, {})
            exp = set()
            for w in info.windows:
                for r in w.response_ids:
                    exp.add(r)
                    if r in got and got[r] != {w.n}:
                        wrong_window.append((thread, r, w.n, sorted(got[r])))
                c = w.compaction_response_id
                if c is not None:
                    exp.add(c)
                    if c in got and got[c] != {w.n}:
                        compaction_other.append((thread, c, w.n, sorted(got[c])))
            n_expected += len(exp)
            missing += [(thread, r) for r in sorted(exp - set(got))]
            extra += [(thread, r) for r in sorted(set(got) - exp - attempted.get(thread, set()))]
            aborted += [(thread, r) for r in sorted(attempted.get(thread, set()) - exp)]
    return {"complete": bool(manifests) and not missing and not extra and not wrong_window,
            "expected": n_expected, "missing": missing, "extra": extra, "wrong_window": wrong_window,
            "compaction_other_window": compaction_other, "aborted_not_in_manifest": aborted,
            "threads_not_in_manifest": sorted(set(seen) - manifest_threads), "single_segments": n_single}


#: ``stop`` of an arc2 outcome: the harness stopped this task on the output budget (the proxy killed codex)
ARC2_BUDGET_STOP = "budget"
#: stops meaning "the harness ended the episode" (the ledger may have one extra last turn). arc2 only has the budget hit;
#: arc3 single-level episodes (run_arc3_level.py) also end codex on level clear, action cap and wall -- the response in
#: flight at that moment may likewise already be committed to the ledger without codex recording it; the same goes for
#: codex dying on its own (``died``, run_arm does not reattach). Full-game episodes (kind arc3_game) record ``won`` on
#: beating the game even if they are ended later by budget, wall or died (run_arc3_level.finish_game_episode's stop
#: checks the win first), so ``won`` counts too. RL v8's ``window`` (stop_after_first_window: the harness ends codex
#: after the proxy blocks the first request of window 1) is likewise a harness kill.
#: arc2 also kills codex on wall / deadline / cancelled (task_wall_seconds, run deadline, ended by the orchestrator); the
#: turn in flight may likewise already be in the ledger (2026-10-24: routine since the wall was cut to 2 h); prepare
#: drops these episodes as infra failures with --drop-stops, not trained on.
RL_KILL_STOPS = {"arc2": frozenset({ARC2_BUDGET_STOP, "wall", "deadline", "cancelled"}),
                 "arc3": frozenset({"budget", "context", "cleared", "won", "action_cap", "wall", "died", "window"})}
#: these harnesses' outcomes carry ``harness_killed`` (the harness / proxy decision to end the episode itself,
#: run_arc2.harness_killed): rl_rid_join lets the extra last turn through on it, not on the stop string -- one value
#: missing from a stop whitelist gets the whole group isolated (``won`` in arc3 v4), and "stop is wall" does not mean
#: codex was generating at that moment. RL_KILL_STOPS remains only for harnesses that do not write this field yet.
HARNESS_KILLED = frozenset({"arc2"})


def rl_rid_join(instances: list[Instance], manifests: list[Manifest], *, drop_episodes: bool = False) -> dict:
    """rid join for RL: ``rid_join`` must be complete, with a single named exception -- **the last turn of a budget hit**.

    When arc2's output budget runs out, the proxy kills codex; the server has already committed that turn (usually a
    length stop truncated by max_new_tokens) to the ledger, but codex had no time to write its response id into the
    rollout, so the ledger has one rid more than the manifest, and that segment never gets seg_close. That turn is an
    on-policy sample that should be trained together with the episode's reward, so it is kept; the conditions are
    checked one by one:

    - ``missing``, ``wrong_window`` and ``compaction_other_window`` are all 0;
    - the task of every extra rid's thread was ended by the harness: ``HARNESS_KILLED`` harnesses look at the outcome's
      ``harness_killed`` (required), the others at stop being in ``RL_KILL_STOPS``;
    - at most one per thread, and it is that thread's **last** append in the ledger (the last record of the last
      segment).

    Another named exception: **the turn truncated at the wrap-up line** (arc2 ``wrapup_cut``). The proxy truncates it
    at the soft line and codex, receiving incomplete, does not record its id, but it is an on-policy sample and later
    turns follow it; the one rid named by the outcome's ``wrapup_cut_rid`` is let through (any stop, need not be the
    last record).

    Any other incompleteness raises ``ManifestError`` with ``manifests`` = every manifest owning a problematic thread
    (all extra rids are judged before reporting them all at once, so the caller can isolate every broken group in one
    go). Returns the ``rid_join`` report plus ``budget_extra`` ({thread: rid}) and ``wrapup_cut_extra``
    ({thread: rid}).

    ``drop_episodes`` (prepare_rl turns it on for arc3): incompleteness that can be pinned to a thread (the three above,
    unexplained extra rids, a thread spanning two instances = a replica restarted midway) does not raise but is recorded
    in the returned ``dropped_episodes`` ({episode: [reasons]}); the caller drops only those episodes and keeps the rest
    (one mismatched game out of a group of 32 should not sink the whole group). Anything that cannot be pinned to a
    manifest thread still raises."""
    rj = rid_join(instances, manifests)
    if not manifests:
        raise ManifestError("rl_rid_join needs manifests")
    by_thread = {t: info for m in manifests for t, info in m.threads.items()}
    dropped: dict[str, list[str]] = {}

    def fail(message: str, threads: list[str]) -> None:
        if drop_episodes and threads and all(t in by_thread for t in threads):
            for t in threads:
                dropped.setdefault(by_thread[t].episode, []).append(message)
            return
        raise ManifestError(message, manifests=sorted({by_thread[t].manifest.path for t in threads if t in by_thread}))
    for key in ("missing", "wrong_window", "compaction_other_window"):
        if rj[key]:                               # the threads of these three kinds all come from the manifest
            if not drop_episodes:
                raise ManifestError(f"rid join: {key} {len(rj[key])} rows, e.g. {rj[key][:3]}",
                                    manifests=sorted({by_thread[row[0]].manifest.path for row in rj[key]}))
            for row in rj[key]:
                fail(f"rid join: {key} {row}", [row[0]])
    last: dict[str, tuple[int, str]] = {}
    owner: dict[str, str] = {}
    for inst in instances:
        for seg in inst.segments.values():
            if seg.sid is None or not seg.appends:
                continue
            thread = seg.thread
            if owner.setdefault(thread, inst.name) != inst.name:
                fail(f"thread {thread} spans two instances ({owner[thread]}, {inst.name}): cannot tell which record is "
                     f"the last", [thread])
                continue
            mark = (seg.seg, seg.appends[-1].rid)       # segment ids grow in open order within an instance
            if thread not in last or mark[0] > last[thread][0]:
                last[thread] = mark
    extra: dict[str, str] = {}
    cut_extra: dict[str, str] = {}
    bad: list[tuple[str, Path, str]] = []               # (message, manifest, thread): raise only after judging all
    for thread, rid in rj["extra"]:
        info = by_thread[thread]
        if info.episode in dropped:                     # the whole episode is already dropped (drop_episodes)
            continue
        outcome = info.manifest.items[info.item]["outcome"]
        if rid == outcome.get("wrapup_cut_rid"):
            cut_extra[thread] = rid
            continue
        stop = outcome.get("stop")
        harness = info.manifest.harness
        if harness in HARNESS_KILLED and "harness_killed" not in outcome:
            bad.append((f"rid join: thread {thread} ({info.episode}) has extra rid {rid}, but {harness}'s outcome has no "
                        f"harness_killed (manifest written by an older harness?)", info.manifest.path, thread))
        elif harness in HARNESS_KILLED and outcome["harness_killed"] is not True:
            bad.append((f"rid join: thread {thread} ({info.episode}) has extra rid {rid}, but the harness did not end this "
                        f"task (harness_killed {outcome['harness_killed']!r}, stop={stop!r})", info.manifest.path, thread))
        elif harness not in HARNESS_KILLED and stop not in RL_KILL_STOPS[harness]:
            bad.append((f"rid join: thread {thread} ({info.episode}) has extra rid {rid}, but this task has stop={stop!r}, "
                        f"not a budget hit or a harness kill ({harness}: {sorted(RL_KILL_STOPS[harness])})",
                        info.manifest.path, thread))
        elif thread in extra:
            bad.append((f"rid join: thread {thread} ({info.episode}) has more than one extra rid: {extra[thread]}, {rid}",
                        info.manifest.path, thread))
        elif last[thread][1] != rid:
            bad.append((f"rid join: thread {thread} ({info.episode}): extra rid {rid} is not the last append in the "
                        f"ledger ({last[thread][1]})", info.manifest.path, thread))
        else:
            extra[thread] = rid
    if bad and drop_episodes:
        for message, _, thread in bad:
            fail(message, [thread])
        extra = {t: r for t, r in extra.items() if by_thread[t].episode not in dropped}
    elif bad:
        more = f"\n({len(bad) - 1} more:\n  " + "\n  ".join(m for m, _, _ in bad[1:]) + ")" if len(bad) > 1 else ""
        raise ManifestError(bad[0][0] + more, manifests=sorted({p for _, p, _ in bad}))
    out = dict(rj, budget_extra=extra, wrapup_cut_extra=cut_extra)
    if drop_episodes:
        out["dropped_episodes"] = dropped
    return out
