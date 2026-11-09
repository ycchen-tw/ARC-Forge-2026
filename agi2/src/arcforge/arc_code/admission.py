"""FIFO admission for whole game lifetimes, shared by independent arm processes."""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Admission:
    def __init__(self, directory: Path, owner: str):
        self.directory = directory.resolve()
        self.owner = owner

    @classmethod
    def create(cls, directory, owner, agents, active_cap, campaign_seconds):
        if (not agents or len(set(agents)) != len(agents)
                or any(re.fullmatch(r"replica-[0-9]{2}/[A-Za-z0-9]+", agent) is None for agent in agents)
                or type(active_cap) is not int or not 1 <= active_cap <= len(agents)
                or type(campaign_seconds) is not int or campaign_seconds <= 0):
            raise ValueError("invalid bounded whole-game admission plan")
        directory.mkdir(parents=True, exist_ok=False)
        admission = cls(directory, owner)
        queued, monotonic = time.time(), time.monotonic()
        admission._write({"schema_version": 1, "owner": owner, "active_cap": active_cap,
                          "order": agents, "active": [], "cancelled": False,
                          "deadline_monotonic": monotonic + campaign_seconds,
                          "jobs": {agent: {"status": "queued", "queued_at": queued,
                                           "queued_monotonic": monotonic} for agent in agents}})
        return admission

    def _write(self, state):
        temporary = self.directory / f".state.{uuid.uuid4().hex}.tmp"
        with temporary.open("x") as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.directory / "state.json")
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @contextmanager
    def _locked(self):
        with (self.directory / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads((self.directory / "state.json").read_text())
            if state["owner"] != self.owner:
                raise ValueError("admission ownership differs")
            yield state

    def acquire(self, agent, *, required_seconds=0):
        if type(required_seconds) is not int or required_seconds < 0:
            raise ValueError("required whole-game budget must be a nonnegative integer")
        while True:
            with self._locked() as state:
                now = time.monotonic()
                if state["cancelled"] or now >= state["deadline_monotonic"]:
                    raise TimeoutError("whole-game admission cancelled or campaign deadline reached")
                if now + required_seconds > state["deadline_monotonic"]:
                    raise TimeoutError("insufficient campaign time for the required whole-game budget")
                job = state["jobs"][agent]
                if job["status"] != "queued":
                    raise RuntimeError("agent was already admitted or stopped")
                available = state["active_cap"] - len(state["active"])
                if available < 0:
                    raise RuntimeError("admission active count exceeds its cap")
                pending = [name for name in state["order"] if state["jobs"][name]["status"] == "queued"]
                if agent in pending[:available]:
                    admitted, monotonic = time.time(), now
                    job.update(status="active", admitted_at=admitted, admitted_monotonic=monotonic,
                               queue_seconds=monotonic - job["queued_monotonic"])
                    state["active"].append(agent)
                    self._write(state)
                    return dict(job)
            time.sleep(0.1)

    def finish(self, agent, *, outcome, deadline_censored, game_won, release=True):
        with self._locked() as state:
            job = state["jobs"][agent]
            if agent not in state["active"]:
                raise RuntimeError("cannot finish an agent without its active admission")
            ended, monotonic = time.time(), time.monotonic()
            job.update(status="finished" if release else "cleanup_failed", ended_at=ended,
                       ended_monotonic=monotonic, active_seconds=monotonic - job["admitted_monotonic"],
                       outcome=outcome, deadline_censored=deadline_censored, game_won=game_won,
                       fixed_work_completed=game_won)
            if release:
                state["active"].remove(agent)
            self._write(state)
            return dict(job)

    def close_after_cleanup(self):
        """Call only after all owned arm services/scopes have been proven stopped."""
        with self._locked() as state:
            state["cancelled"] = True
            for agent, job in state["jobs"].items():
                if job["status"] in {"queued", "active", "cleanup_failed"}:
                    ended, monotonic = time.time(), time.monotonic()
                    job.update(status="cancelled", outcome="owned_cleanup", ended_at=ended,
                               ended_monotonic=monotonic,
                               deadline_censored=True, game_won=False, fixed_work_completed=False)
                    if "admitted_monotonic" in job:
                        job["active_seconds"] = monotonic - job["admitted_monotonic"]
            state["active"] = []
            self._write(state)
