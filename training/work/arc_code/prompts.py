"""Prompt sets: ``prompts/<set>/{brief,resume,nudge}.md``.

The prompts used to be three Python strings in ``run_arm.py``. There is only one reason they were
moved out: **they are experimental variables**, and like the corpus they need an identity. The config
names one with ``"prompts": "<set>"`` and ARM_STATE records the fingerprint; once a set has been used
by any round it is never edited in place -- a change means a new directory, and the fingerprint
catches anyone who breaks this rule.

Roles of the three files:

- ``brief.md``: the task brief. It is both the first user prompt **and** written as the file the
  driver auto-re-injects (``AGENTS.md`` for both codex and pi) -- the latter is the only mechanism
  that keeps the brief in context after compaction. Variables: ``{game}``.
- ``resume.md``: the handoff paragraph for ``resume_mode=fresh``, appended after the brief. Variables:
  ``{levels}``, ``{actions}``.
- ``nudge.md``: the nudge sent when ``resume_mode=thread`` reattaches to the same thread. Same variables.

render is strict: a missing or extra variable raises. On 2026-09-17 a template gained a variable
while only one of its two call sites was updated, and the KeyError surfaced three hours into the run;
here it must blow up at import time.

Prompt content **does not vary by driver**. Structural differences between drivers (e.g. one agent
running every command in a new subshell) are appended after the brief by
``arms/<driver>.brief_addendum()`` and counted into the fingerprint, so the difference is visible.

**Feature-owned fragments** (``prompts/_features/<feature>/``): when the config turns on a feature,
its brief / resume / nudge fragment is appended after the set's file of the same name
(``load(name, features=...)``). Text about a feature travels with the feature: any set gets it by
adding one config key, with no need to copy out v14g, v14rg... for every base. Fragments have no
variables, are counted into the fingerprint too, and are never edited in place once used.
"""

from __future__ import annotations

import hashlib
import string
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT_DIR = UNIT / "prompts"

FILES = ("brief.md", "resume.md", "nudge.md")
#: Optional files: absent means empty string (v3 has no oom.md; behavior and fingerprint unchanged).
#: ``oom.md``: the paragraph put before nudge / resume when the previous session was killed by the memory cap.
OPTIONAL = ("oom.md", "compact_reminder.md", "developer.md", "kickoff.md",
            "instructions.md", "compact_prompt.md", "level.md", "level_seed.md",
            "window_full.md", "notes_skeleton.md", "packet_tail.md", "stuck.md", "idle.md")
#: Variables each file must use.
VARIABLES = {
    "brief.md": frozenset({"game"}),
    "resume.md": frozenset({"levels", "actions"}),
    "nudge.md": frozenset({"levels", "actions"}),
    "oom.md": frozenset({"limit"}),
    # The message codex patch 0005 sends before automatic compaction (config compact_reminder_tokens).
    "compact_reminder.md": frozenset(),
    # Behavior rules (from v11): codex's developer_instructions, the developer message at the start of
    # every session. The old arm's RULES in run_arm.py sat in this position and were lost in the migration.
    "developer.md": frozenset(),
    # From v12: the first user message. With it, the brief only goes through AGENTS.md (the driver
    # injects it every time, after compaction too) and is no longer also the first message -- up to v11
    # the same brief was always in context twice.
    "kickoff.md": frozenset(),
    # From v12: codex's base instructions and compaction prompt travel with the prompt set; switching
    # set switches all of them. Previously there was one copy each under driver_files/codex/, and
    # editing it meant editing every round that had used it.
    "instructions.md": frozenset(),
    "compact_prompt.md": frozenset(),
    # Single-level episode (run_arm --level-episode, RL): appended after the first message, says which level
    # this session starts at and that it ends when the level is cleared. A set with this file can only be
    # used for single-level episodes, and a single-level episode must have it (run_arm checks both directions).
    "level.md": frozenset({"level"}),
    # Single-level episode with a prior-state seed (run_arm --level-seed, RL round 2 v14seed): the real workspace
    # as of the start of level K is already there, and the opening is kickoff plus this paragraph -- the handoff a
    # new session gets when taking over an old workspace in a full game (resume.md) plus "ends when the level is
    # cleared". In a set that has it, level 1 (no seed) uses level.md and every later level must have a seed (run_arm checks).
    "level_seed.md": frozenset({"level", "levels", "actions"}),
    # Text sent by the gen-2 hooks (run_arm's gen2_texts writes it into gen2_budget.json); absent means run_arm's
    # g2b originals (GEN2_FULL etc.).
    # window_full.md: the "window is full" appended after tool output at the window-change point (also the
    # reminder template for codex token_budget).
    "window_full.md": frozenset(),
    # notes_skeleton.md: the notes.md SessionStart puts into the workspace for a brand-new game.
    "notes_skeleton.md": frozenset(),
    # packet_tail.md: the last paragraph of a new window's status packet.
    "packet_tail.md": frozenset(),
    # stuck.md: from the GEN2_STUCK_WINDOWS-th window on the same level, the status packet's last paragraph becomes
    # this (not sent if the file is absent).
    "stuck.md": frozenset({"level", "windows", "actions"}),
    # idle.md: for every GEN2_IDLE_TOKENS output tokens of a long think without acting within a window, PostToolUse
    # appends this after tool output (not sent if the file is absent).
    "idle.md": frozenset({"tokens"}),
}

#: Directory of feature fragments (``available()`` doesn't treat it as a set) and the files a fragment may have.
FEATURES = "_features"
FEATURE_FILES = ("brief.md", "resume.md", "nudge.md")

#: Driver files in a prompt set -> codex setting key (the value is a file path).
DRIVER_FILES = {"instructions.md": "model_instructions_file",
                "compact_prompt.md": "experimental_compact_prompt_file"}


#: Variables that may or may not be used. If used, the launcher fills them from config: with the
#: memory cap hardcoded in the brief (v4's 16 GB / 24 GB are the B200 nodes' numbers) it can't run on a host with
#: different caps -- from v5 they are variables, so the brief always states the cap this round really enforces.
OPTIONAL_VARIABLES = {
    "brief.md": frozenset({"process_limit", "workspace_limit", "cpu_limit"}),
}


def placeholders(text: str) -> frozenset[str]:
    return frozenset(f[1] for f in string.Formatter().parse(text) if f[1])


class PromptSet:
    def __init__(self, name: str, root: Path = ROOT_DIR, features: tuple[str, ...] = ()) -> None:
        self.name = name
        self.dir = root / name
        if not self.dir.is_dir():
            have = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
            raise FileNotFoundError(f"prompt set {name!r} does not exist ({self.dir}); available: {have}")
        self.texts: dict[str, str] = {}
        for f in FILES:
            path = self.dir / f
            if not path.is_file():
                raise FileNotFoundError(f"prompt set {name!r} is missing {f}")
            text = path.read_text(encoding="utf-8")
            found = placeholders(text)
            optional = OPTIONAL_VARIABLES.get(f, frozenset())
            if not VARIABLES[f] <= found or not found <= VARIABLES[f] | optional:
                raise ValueError(f"{path} has variables {sorted(found)}, expected {sorted(VARIABLES[f])}"
                                 + (f" (optionally {sorted(optional)})" if optional else ""))
            self.texts[f] = text
        for f in OPTIONAL:
            path = self.dir / f
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            found = placeholders(text)
            if found != VARIABLES[f]:
                raise ValueError(f"{path} has variables {sorted(found)}, expected {sorted(VARIABLES[f])}")
            self.texts[f] = text
        self.features = tuple(features)
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"duplicate features: {self.features}")
        #: ``_features/<feature>/<file>`` -> text; render appends them in ``features`` order.
        self.fragments: dict[str, str] = {}
        for feature in self.features:
            fdir = root / FEATURES / feature
            if not fdir.is_dir():
                have = sorted(p.name for p in (root / FEATURES).iterdir()) if (root / FEATURES).is_dir() else []
                raise FileNotFoundError(f"feature fragment {feature!r} does not exist ({fdir}); available: {have}")
            names = sorted(p.name for p in fdir.iterdir())
            if not names or not set(names) <= set(FEATURE_FILES):
                raise ValueError(f"{fdir} has {names}; fragments must be files from {FEATURE_FILES}, at least one")
            for f in names:
                path = fdir / f
                text = path.read_text(encoding="utf-8")
                if placeholders(text):
                    raise ValueError(f"{path} must not have variables, has {sorted(placeholders(text))}")
                self.fragments[f"{FEATURES}/{feature}/{f}"] = text

    def has(self, which: str) -> bool:
        return which in self.texts

    def render(self, which: str, **variables) -> str:
        if which not in self.texts:
            raise KeyError(f"prompt set {self.name!r} has no {which}")
        given = frozenset(variables)
        wanted = placeholders(self.texts[which])
        if given != wanted:
            raise ValueError(f"{self.name}/{which} wants {sorted(wanted)}, given {sorted(given)}")
        text = self.texts[which].format(**variables)
        for feature in self.features:
            fragment = self.fragments.get(f"{FEATURES}/{feature}/{which}")
            if fragment is not None:
                text = text.rstrip("\n") + "\n\n" + fragment.strip("\n") + "\n"
        return text

    def templates_limits(self) -> bool:
        """Whether the brief's memory caps are variables (from v5) or hardcoded (v3/v4 = the B200 nodes' numbers)."""
        return {"process_limit", "workspace_limit"} <= placeholders(self.texts["brief.md"])

    def templates_cpu_limit(self) -> bool:
        """Whether the brief states the watchdog's per-process CPU cap (from v12)."""
        return "cpu_limit" in placeholders(self.texts["brief.md"])

    def optional(self, which: str, **variables) -> str:
        """Optional file: returns an empty string if this set doesn't have it."""
        if which not in OPTIONAL:
            raise KeyError(f"{which} is not an optional file ({OPTIONAL})")
        return self.render(which, **variables) if which in self.texts else ""

    def brief(self, game: str, addendum: str = "", *, limits: dict[str, str] | None = None) -> str:
        """``limits``: display strings for ``process_limit`` / ``workspace_limit``. Must be given if the brief
        has those two variables and must not be given if it doesn't -- given with nowhere to go means the
        config's cap was never told to the agent."""
        variables = {"game": game, **(limits or {})}
        text = self.render("brief.md", **variables)
        if addendum:
            text = text.rstrip("\n") + "\n\n" + addendum.strip("\n") + "\n"
        return text

    def driver_files(self) -> dict[str, Path]:
        """Driver files this set carries: codex setting key -> file."""
        return {key: self.dir / f for f, key in DRIVER_FILES.items() if f in self.texts}

    def resume(self, *, levels: int, actions: int) -> str:
        return self.render("resume.md", levels=levels, actions=actions)

    def nudge(self, *, levels: int, actions: int) -> str:
        return self.render("nudge.md", levels=levels, actions=actions)

    def fingerprint(self, extra: dict[str, str] | None = None) -> dict:
        """Fingerprint of all the text the agent reads in this round.

        ``extra`` holds act.py's source (the brief tells the agent to read it, so it is part of the
        prompt) and the driver's addendum. The returned ``parts`` let one see afterwards **which part**
        changed.
        """
        parts = {f"{self.name}/{f}": text for f, text in self.texts.items()}
        parts.update(self.fragments)
        parts.update(extra or {})
        h = hashlib.sha256()
        for key in sorted(parts):
            h.update(key.encode()); h.update(b"\0")
            h.update(parts[key].encode()); h.update(b"\0")
        return {"set": self.name,
                **({"features": list(self.features)} if self.features else {}),
                "fingerprint": h.hexdigest()[:16],
                "parts": {k: hashlib.sha256(v.encode()).hexdigest()[:12] for k, v in sorted(parts.items())}}


def load(name: str, root: Path = ROOT_DIR, features: tuple[str, ...] = ()) -> PromptSet:
    return PromptSet(name, root, features)


def available(root: Path = ROOT_DIR) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
