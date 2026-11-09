"""A game's record: the middleware writes one step for every action response it receives.

    <workspace>/record/
      README.md          format and palette (reading this tells the agent how to use it)
      history.jsonl      one line per step, append-only
      current.txt/.png   the current board (the last step's final), atomically replaced every step
      steps/000012/
        final.txt        the last frame, 64 lines of hex
        final.png        the last frame, png_size² (default 512×512, v13 uses 640×640; drawn the VISTA way, see ``render``)
        frame_000.png …  one per frame when F ≥ 2, drawn the same way
        frames.npz       every frame this step returned, ``frames``: uint8 (F, H, W)
        animation.mp4    only when F ≥ 2; one video frame per frame (with video_repeat > 1, each frame is held that many). Beyond
                         video_max_frames, consecutive duplicate frames are dropped first, then sampled evenly down to the cap (first
                         and last always kept); animation.json records which frames it used

The record is written by the middleware, not taken from the agent's own log: actions sent by a self-written
client are in it too. step is the middleware's ``actions_used`` (opening = 0), the same number as the
official action count.

The agent can read it but not write it: bwrap mounts ``record/`` read-only into the jail; the posix fence
makes it root-owned, and the workspace is 700 so other games cannot get in. The directory is created by the
launcher; here it is opened once with ``O_NOFOLLOW`` and its owner checked, and all later writes are
relative to that fd -- if the agent replaces ``record`` with a symlink pointing elsewhere, the middleware,
running as root, does not follow it.
"""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

#: COLOR_MAP from the ARC-AGI toolkit's ``arc_agi/rendering.py`` (the same set as the official web page).
PALETTE = (
    "#ffffff", "#cccccc", "#999999", "#666666", "#333333", "#000000",
    "#e53aa3", "#ff7bcc", "#f93c31", "#1e93ff", "#88d8f1", "#ffdc00",
    "#ff851b", "#921231", "#4fcc30", "#a356d6",
)
RGB = np.frombuffer(b"".join(bytes.fromhex(c[1:]) for c in PALETTE), dtype=np.uint8).reshape(16, 3)
#: Colour of grid-line pixels: VISTA's ``rgb * 3 // 4`` (computed in uint16, then back to uint8).
DARK = (RGB.astype(np.uint16) * 3 // 4).astype(np.uint8)
PNG_SIZE = 512
FFMPEG_TIMEOUT = 120.0

README = """# Game record

Written by the game server after every action, including actions sent by your
own scripts. Read-only; step N is the N-th action counted against the budget
(step 0 is the initial board).

current.txt, current.png -- the board right now (the final board of the last step
in history.jsonl), replaced after every action.

In every PNG here a cell (x, y) -- column x, row y, both 0-63 -- covers image pixels
x*{px} .. x*{px}+{px_1} across and y*{px} .. y*{px}+{px_1} down. So a point at pixel
(u, v) of the image is cell (u // {px}, v // {px}), and the cell's center is at
pixel (x*{px}+{half}, y*{px}+{half}).

history.jsonl -- one JSON line per step, appended as you play:
  step, level (the level the action was played on), action, x, y (ACTION6 only),
  plan (the --plan of the batch, on its first action), state, levels_completed,
  win_levels, available_actions, frames (how many frames the action returned),
  changed_cells (cells that differ between the previous final board and this one).

steps/000012/ -- everything step 12 returned:
  final.txt      the board after the action: 64 lines of 64 hex digits, one per cell
  final.png      the same board as a {png_size}x{png_size} image, {px} px per cell with grid lines
  frame_000.png ... one image per frame, only when the action returned 2 or more frames
  frames.npz     every frame in order: numpy.load(path)["frames"], uint8 (frames, 64, 64);
                 frames[-1] is the final board
  animation.mp4  only when the action returned 2 or more frames: every frame, in order,
                 up to {max_frames} frames. A longer animation is shortened for the video only:
                 frames identical to the one before are dropped, then frames are taken evenly,
                 always keeping the first and the last. animation.json then lists which
                 frames the video shows ("source_frames", indices into frames.npz and
                 frame_NNN.png); history.jsonl's "video_frames" is how many it has.
                 frames.npz and frame_NNN.png always hold every frame.{repeat}

Cell values: 0=White 1=Off-White 2=Light-Gray 3=Gray 4=Off-Black 5=Black
  6=Magenta 7=Light-Magenta 8=Red 9=Blue a=Light-Blue b=Yellow c=Orange
  d=Maroon e=Green f=Purple
"""


def video_frames(frames: np.ndarray, max_frames: int | None) -> list[int]:
    """Which frames go into animation.mp4 (indices into frames). Within the cap, all of them, duplicates kept.

    The cap is for the model side: sglang's ViT runs a whole video at once inside the scheduler; 372 frames at
    384² is about 3 GiB, nearly all of the 4.2 GB left after CUDA graphs; an OOM there ends the whole server."""
    count = len(frames)
    if max_frames is None or count <= max_frames:
        return list(range(count))
    keep = [0] + [i for i in range(1, count) if not np.array_equal(frames[i], frames[i - 1])]
    if len(keep) > max_frames:
        picks = np.linspace(0, len(keep) - 1, max_frames).round().astype(int)
        keep = [keep[i] for i in sorted(set(picks.tolist()))]
    if keep[-1] != count - 1:
        keep[-1] = count - 1          # the last frame is the final board; if it duplicates the previous one, swap it in
    return keep


class RecordError(RuntimeError):
    pass


def frames_array(frame: Any) -> np.ndarray:
    """The official ``frame`` (list of 2D lists) -> uint8 (F, H, W). A wrong shape or value blows up; no clamping."""
    if not isinstance(frame, list):
        raise RecordError(f"frame is {type(frame).__name__}, not a list of grids")
    if not frame:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    try:
        arr = np.array(frame, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise RecordError(f"frame is not a stack of equal-sized integer grids: {error}") from None
    if arr.ndim != 3 or 0 in arr.shape:
        raise RecordError(f"frame has shape {arr.shape}, expected (frames, height, width)")
    if arr.min() < 0 or arr.max() > 15:
        raise RecordError(f"cell values {arr.min()}..{arr.max()} are outside 0..15")
    return arr.astype(np.uint8)


def board_text(grid: np.ndarray) -> str:
    return "".join("".join("%x" % v for v in row) + "\n" for row in grid)


def render(frames: np.ndarray, size: int, rgb: np.ndarray = RGB, dark: np.ndarray = DARK) -> np.ndarray:
    """(…, H, W) cell values -> (…, H·s, W·s, 3) RGB. Follows VISTA ``shared/render.py``: official palette,
    nearest upscaling, and the pixels of each cell's top / left border row / column multiplied by 3/4 as grid
    lines (``render_grid=True``, ``RENDER_SCALE=8``, its settings on Kaggle).

    Colours are looked up at cell level and then upscaled, and grid-line rows / columns are written straight
    from the dark palette (``DARK`` = the same ×3//4): bit-identical to the per-pixel multiplication
    (test_render_matches_reference); at c110 most of the middleware's GIL time went here.
    ``rgb`` / ``dark`` swap the palette (act_images' diff image has one extra off-board colour); the default is the official palette."""
    scale = max(1, size // max(frames.shape[-2:]))
    img = np.repeat(np.repeat(rgb[frames], scale, axis=-3), scale, axis=-2)
    edge = dark[frames]
    img[..., ::scale, :, :] = np.repeat(edge, scale, axis=-2)   # each cell's top pixel row
    img[..., :, ::scale, :] = np.repeat(edge, scale, axis=-3)   # each cell's leftmost pixel column
    return img


def png(grid: np.ndarray, size: int = PNG_SIZE) -> bytes:
    """RGB PNG (VISTA's drawing). No PIL: the gateway's venv does not have it."""
    return encode_png(render(grid, size))


def encode_png(img: np.ndarray) -> bytes:
    """(H, W, 3) uint8 -> PNG bytes. The same image always gives the same bytes (fixed zlib level)."""
    height, width, _ = img.shape
    # each scanline is filter byte 0 + its RGB bytes
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), img.reshape(height, width * 3)], axis=1).tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def mp4(frames: np.ndarray, *, ffmpeg: str, fps: float, size: int, tmp_dir: Path) -> bytes:
    """Every frame is one video frame (``fps`` is only playback speed, no sampling). Lossless RGB H.264, drawn like the PNG."""
    rgb = render(frames, size)
    _, height, width, _ = rgb.shape
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        out = Path(tmp) / "animation.mp4"
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
             "-i", "-", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
             "-threads", "1", "-an", "-movflags", "+faststart", str(out)],
            input=rgb.tobytes(), capture_output=True, timeout=FFMPEG_TIMEOUT, check=False)
        if proc.returncode != 0 or not out.is_file():
            raise RecordError(f"ffmpeg exited {proc.returncode}: "
                              f"{proc.stderr.decode(errors='replace').strip()[-400:]}")
        return out.read_bytes()


class GameRecord:
    """One per game. The caller (middleware) serialises: a game's actions are already under the per-game lock."""

    def __init__(self, root: Path, *, ffmpeg: str | None, video_fps: float = 2.0,
                 video_size: int = 384, video_max_frames: int | None = None,
                 png_size: int = PNG_SIZE, tmp_dir: Path | None = None, video_repeat: int = 1) -> None:
        self.root = root
        if png_size < 64 or png_size % 64:
            raise RecordError(f"png_size {png_size} must be a multiple of 64 (whole pixels per cell)")
        self.png_size = png_size
        self.ffmpeg, self.video_fps, self.video_size = ffmpeg, video_fps, video_size
        if video_max_frames is not None and video_max_frames < 2:
            raise RecordError(f"video_max_frames {video_max_frames} must be at least 2")
        self.video_max_frames = video_max_frames
        # How many video frames each frame is held for in the mp4. Qwen3.x merges every 2 video frames into one temporal patch: holding
        # for 2 gives each patch a single picture (without it a 2-3 frame animation gets merged into one group; mp4dup in
        # docs/research/arc3_animation_perception_20261015.md)
        if not isinstance(video_repeat, int) or video_repeat < 1:
            raise RecordError(f"video_repeat {video_repeat!r} must be an int >= 1")
        self.video_repeat = video_repeat
        self.tmp_dir = tmp_dir
        try:
            self.fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as error:
            raise RecordError(f"cannot open record dir {root}: {error}") from None
        st = os.fstat(self.fd)
        if st.st_uid != os.geteuid() or st.st_mode & 0o022:
            raise RecordError(f"{root} is owned by uid {st.st_uid} with mode {st.st_mode & 0o777:o}; "
                              f"it must belong to the recorder (uid {os.geteuid()}) and not be "
                              "group/world-writable")
        names = set(os.listdir(self.fd))
        if "steps" not in names:
            os.mkdir("steps", 0o755, dir_fd=self.fd)
        self.steps_fd = os.open("steps", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
        if "README.md" not in names:
            cap = self.video_max_frames if self.video_max_frames is not None else "any number of"
            px = self.png_size // 64
            repeat = ("" if self.video_repeat == 1 else
                      f"\n                 In animation.mp4 every frame is shown {self.video_repeat} times in a row, so"
                      f"\n                 the video holds {self.video_repeat}x as many pictures as frames: a repeat"
                      f"\n                 there is not a repeat in the game.")
            text = (README.replace("{max_frames}", str(cap)).replace("{png_size}", str(self.png_size))
                    .replace("{repeat}", repeat)
                    .replace("{px_1}", str(px - 1)).replace("{px}", str(px)).replace("{half}", str(px // 2)))
            self._write(self.fd, "README.md", text.encode())
        self.last_step: int | None = None
        self.last_final: np.ndarray | None = None
        if "history.jsonl" in names:
            self._recover()

    def _recover(self) -> None:
        """Middleware restart: continue from the last step written, and recover the last final so changed_cells can be computed."""
        with os.fdopen(os.open("history.jsonl", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)) as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
        if not lines:
            return
        self.last_step = int(json.loads(lines[-1])["step"])
        fd = os.open(f"{self.last_step:06d}/frames.npz", os.O_RDONLY | os.O_NOFOLLOW,
                     dir_fd=self.steps_fd)
        with os.fdopen(fd, "rb") as f, np.load(io.BytesIO(f.read()), allow_pickle=False) as data:
            frames = data["frames"]
        self.last_final = frames[-1] if len(frames) else None

    @staticmethod
    def _write(dir_fd: int, name: str, data: bytes) -> None:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
        with os.fdopen(fd, "wb") as f:
            f.write(data)

    def _replace(self, name: str, data: bytes) -> None:
        """Write a temp file, then rename: the agent always reads a complete copy."""
        tmp = f".{name}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644, dir_fd=self.fd)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)

    def write(self, step: int, *, action: str, obs: dict[str, Any], level: int | None,
              x: int | None = None, y: int | None = None, plan: str | None = None) -> dict[str, Any]:
        """Write one step; return the line written to history (plus ``dir``). A video failure does not block the other files:
        that line carries ``video_error``, and the caller must report it."""
        if self.last_step is not None and step <= self.last_step:
            raise RecordError(f"step {step} after step {self.last_step}: the counter went backwards")
        frames = frames_array(obs.get("frame"))
        name = f"{step:06d}"
        os.mkdir(name, 0o755, dir_fd=self.steps_fd)
        step_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.steps_fd)
        row: dict[str, Any] = {"step": step, "level": level, "action": action}
        if x is not None:
            row["x"], row["y"] = x, y
        if plan:
            row["plan"] = plan
        row.update({k: obs.get(k) for k in ("state", "levels_completed", "win_levels",
                                             "available_actions")})
        row["frames"] = len(frames)
        # Identical frames within a step (animations often hold on one picture) are drawn and compressed once; final.png / current.png are the last frame's copy
        pngs: dict[bytes, bytes] = {}

        def png_of(grid: np.ndarray) -> bytes:
            key = grid.tobytes()
            if key not in pngs:
                pngs[key] = png(grid, self.png_size)
            return pngs[key]

        try:
            self._write(step_fd, "frames.npz", _npz(frames))
            if len(frames):
                final = frames[-1]
                self._write(step_fd, "final.txt", board_text(final).encode())
                self._write(step_fd, "final.png", png_of(final))
                if len(frames) >= 2:
                    for i, frame in enumerate(frames):
                        self._write(step_fd, f"frame_{i:03d}.png", png_of(frame))
                row["changed_cells"] = (int((final != self.last_final).sum())
                                        if self.last_final is not None
                                        and self.last_final.shape == final.shape else None)
                self.last_final = final
            else:
                row["changed_cells"] = None
            if len(frames) >= 2:
                try:
                    if self.ffmpeg is None:
                        raise RecordError("no ffmpeg configured")
                    shown = video_frames(frames, self.video_max_frames)
                    self._write(step_fd, "animation.mp4",
                                mp4(np.repeat(frames[shown], self.video_repeat, axis=0),
                                    ffmpeg=self.ffmpeg, fps=self.video_fps,
                                    size=self.video_size, tmp_dir=self.tmp_dir))
                    row["video_frames"] = len(shown)
                    if len(shown) < len(frames):
                        self._write(step_fd, "animation.json", json.dumps(
                            {"frames": len(frames), "source_frames": shown,
                             **({"repeat": self.video_repeat} if self.video_repeat != 1 else {})}).encode())
                except (RecordError, OSError, subprocess.TimeoutExpired) as error:
                    row["video_error"] = f"{type(error).__name__}: {error}"[:300]
        finally:
            os.close(step_fd)
        row["time"] = round(time.time(), 3)
        line = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        fd = os.open("history.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                     0o644, dir_fd=self.fd)
        with os.fdopen(fd, "wb") as f:
            f.write(line)
        if len(frames):
            self._replace("current.txt", board_text(frames[-1]).encode())
            self._replace("current.png", png_of(frames[-1]))
        self.last_step = step
        return {**row, "dir": f"record/steps/{name}"}

    def close(self) -> None:
        os.close(self.steps_fd)
        os.close(self.fd)


def _npz(frames: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=frames)
    return buf.getvalue()


def check_ffmpeg(ffmpeg: str, tmp_dir: Path) -> None:
    """Before the run, encode a three-frame clip, decode it and compare frame by frame. Missing ffmpeg, missing libx264rgb, or not lossless all blow up here."""
    frames = np.stack([np.full((64, 64), v, dtype=np.uint8) for v in (0, 8, 15)])
    frames[1, 10:20, 30:40] = 9
    data = mp4(frames, ffmpeg=ffmpeg, fps=2.0, size=384, tmp_dir=tmp_dir)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        src = Path(tmp) / "check.mp4"
        src.write_bytes(data)
        proc = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(src),
                               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                              capture_output=True, timeout=FFMPEG_TIMEOUT, check=False)
    if proc.returncode != 0:
        raise RecordError(f"{ffmpeg} cannot decode its own output: {proc.stderr.decode(errors='replace')[-400:]}")
    decoded = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 384, 384, 3)
    expected = render(frames, 384)
    if decoded.shape != expected.shape or not np.array_equal(decoded, expected):
        raise RecordError(f"{ffmpeg} round trip is not lossless ({decoded.shape} vs {expected.shape})")
