"""The single source of truth for external dependencies, plus pre-run verification.

The situation was: the upstream commit and the four CLI versions were pinned only in README prose,
there was no ``git apply`` anywhere in the repo for the three arm patches, and ``run_arm.py:696`` only
checked that ``run.py``/``act.py`` exist. **A stale, unpatched or wrong-commit checkout would silently
pass every check and then run for six hours.**

`verify_prep()` deliberately **returns a report rather than SystemExit**: on Kaggle versions can only be
recorded, not pinned (packages come from the competition-mounted wheelhouse), so whether to fail fast is the
launcher's decision.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: Pin of upstream jerber/arc-code. Used to be a literal constant written separately in two build scripts.
ARC_CODE_URL = "https://github.com/jerber/arc-code"
ARC_CODE_COMMIT = "6b33c1f"

#: CLI version of each driver. Previously lived only in README prose, with no runtime assert.
CLI_PINS = {
    "codex": "0.154.0",
    "qwen-code": "0.24.0",
    "pi": "0.85.1",
    "mini-swe-agent": "2.4.6",
}

#: agent runtime. **Excludes arcengine / arc_agi** -- those belong to the harness venv.
AGENT_PACKAGES = ("numpy", "requests")

EXACT = "exact"       # our own wheelhouse / b200 / Colab: a version mismatch is an error
RECORD = "record"     # competition mount: record whatever resolves; a mismatch is just a named difference


@dataclass
class Runtime:
    """A runtime is (package set, resolution source, pinning policy)."""

    packages: tuple[str, ...] = AGENT_PACKAGES
    source: str = "index"          # "index" or "wheelhouse:<path>"
    pin_policy: str = EXACT
    versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pin_policy not in (EXACT, RECORD):
            raise ValueError(f"unknown pin_policy {self.pin_policy!r}")


@dataclass
class Finding:
    ok: bool
    what: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, ok: bool, what: str, detail: str = "") -> None:
        self.findings.append(Finding(ok, what, detail))

    @property
    def ok(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if not f.ok]

    def render(self) -> str:
        return "\n".join(
            f"{'PASS' if f.ok else 'FAIL'}  {f.what}"
            + (f" — {f.detail}" if f.detail else "")
            for f in self.findings)


def _git(prep: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(prep), *args],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def verify_prep(prep: Path, *, expect_commit: str = ARC_CODE_COMMIT,
                markers: dict[str, str] | None = None) -> Report:
    """Check that the upstream checkout is the one we think it is.

    ``markers``: file name -> a string that must appear in it, to confirm the patch was really applied
    (a poor man's reverse-check, but much stronger than "the file exists").
    """
    report = Report()
    for name in ("run.py", "act.py"):
        report.add((prep / name).is_file(), f"{name} exists", str(prep / name))

    head = _git(prep, "rev-parse", "HEAD")
    if head is None:
        report.add(False, "cannot read the checkout's commit",
                   "not a git repo, or git unavailable -- cannot prove which version this is")
    else:
        report.add(head.startswith(expect_commit),
                   f"commit is {expect_commit}", f"actual {head[:12]}")

    dirty = _git(prep, "status", "--porcelain")
    if dirty is not None:
        report.add(True, "work tree status readable",
                   "uncommitted changes (expected once patches are applied)" if dirty else "clean")

    for name, needle in (markers or {}).items():
        path = prep / name
        if not path.is_file():
            report.add(False, f"{name}: patch marker", "file does not exist")
            continue
        found = needle in path.read_text(encoding="utf-8", errors="replace")
        report.add(found, f"{name} contains {needle!r}",
                   "" if found else "patch not applied")
    return report


def verify_runtime(runtime: Runtime, resolved: dict[str, str]) -> Report:
    """Match the actually resolved versions against the policy. ``resolved``: package name -> version."""
    report = Report()
    for pkg in runtime.packages:
        got = resolved.get(pkg)
        if got is None:
            report.add(False, f"agent runtime has {pkg}", "not installed")
            continue
        want = runtime.versions.get(pkg)
        if runtime.pin_policy == RECORD or want is None:
            # The competition mount gives what it gives. Record it; not a failure.
            report.add(True, f"{pkg} version recorded", got)
        else:
            report.add(got == want, f"{pkg} == {want}", f"actual {got}")
    for banned in ("arcengine", "arc_agi"):
        report.add(banned not in resolved,
                   f"agent runtime excludes {banned}",
                   "engine source must not appear on the agent side")
    return report
