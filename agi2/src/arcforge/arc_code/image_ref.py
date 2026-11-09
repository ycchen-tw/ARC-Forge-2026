"""``act_image``: signed references for attaching images to ``./act do``.

The middleware gives ``<game dir name>/<step>/<sig>`` in every step response (``record.image_ref``)
and in ``/api/status`` (``image_ref``); act.py prints it as ``[[arc-image:board:<ref>]]`` or
``[[arc-image:diff:<ref>:<ref>]]``, and the model proxy (``work/arc_code/act_images.py``) verifies the
signature and swaps in that step's image. The secret key sits where the agent cannot read it (the
launch output directory, not mounted in the jail), so the agent cannot print a reference to another
game's step or to a step that does not exist. step is decimal without zero padding: one spelling per step.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path

SIG_HEX = 16
GAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
REF = re.compile(r"(?P<game>[A-Za-z0-9_][A-Za-z0-9_.-]*)/(?P<step>0|[1-9][0-9]{0,8})/(?P<sig>[0-9a-f]{16})")
MIN_KEY_BYTES = 32


def load_key(path: Path) -> bytes:
    key = path.read_bytes().strip()
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"{path}: image-ref key has {len(key)} bytes, need at least {MIN_KEY_BYTES}")
    return key


def sign(key: bytes, game: str, step: int) -> str:
    if not GAME.fullmatch(game):
        raise ValueError(f"game directory name {game!r} cannot go into an image ref")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative int, got {step!r}")
    sig = hmac.new(key, f"{game}/{step}".encode(), hashlib.sha256).hexdigest()[:SIG_HEX]
    return f"{game}/{step}/{sig}"


def verify(key: bytes, ref: str) -> tuple[str, int] | None:
    """Return ``(game, step)`` if the signature checks out, else None (bad format, tampered, signed with another key)."""
    m = REF.fullmatch(ref)
    if m is None:
        return None
    game, step = m["game"], int(m["step"])
    return (game, step) if hmac.compare_digest(sign(key, game, step), ref) else None
