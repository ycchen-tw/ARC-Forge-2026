"""TITO ledger → on-disk format and reader for main-model LoRA training samples (images + video). The producer is
``prepare_tito_mm.py``.

Directory (``DATA/``)::

    manifest.json                       format, processor digest, splits, parity subset, vision constants
    report.json                         pass counts of every assertion made at build time, elapsed time, sizes
    media.jsonl                         one row per unique media: sha256, modality, grid_thw, frames shape, expanded block ids
    media/<sha[:2]>/<sha>.npz           ``frames``: uint8 (T, 3, H, W), after the processor's resize, before normalize
    <split>/index.jsonl                 one row per segment: key, metadata, offset/length, media (in order of appearance in ids)
    <split>/input_ids.i32               all samples concatenated, little-endian int32
    <split>/loss_mask.u8                same, 0/1
    <split>/mrope.i32                   each sample's (3, T) concatenated in C order (offset is 3×token offset)
    <split>/routing.i16                 optional (manifest ``routing``): MoE expert ids (layers, top_k) per position,
                                        concatenated per token (offset is layers×top_k×token offset); the last
                                        position of each segment is -1

**pixels are stored as uint8, not float**: the deployed processor (transformers Qwen2VLImageProcessor /
Qwen3VLVideoProcessor, fused rescale+normalize, mean=std=0.5) outputs ``pixel_values`` that are exactly
``(u8 - 127.5) / 127.5`` (float32). For every media item the producer asserts "``patchify(frames)`` here is
``torch.equal`` to the processor's float32 output", so uint8 is a lossless representation at 1/4 the size of bf16 (images
save another half from the temporal duplication), and it compresses to very little.
The ``pixels`` read back are float32 = the ``item.feature`` sglang hands to the ViT; the ViT casts to its own dtype (bf16
in deployment).

Lightweight assertions in ``__getitem__`` (raise on violation, no repair): pad positions = union of the media ranges,
each range's length = h·w/merge², a video's range count = grid t, pixel row count = t·h·w, no loss on pads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

FORMAT = "tito-mm-v1"
SPLITS = ("train", "validation")
#: the deployed processor's normalize (mean=std=0.5 fused with rescale 1/255 into 127.5)
PIXEL_CENTER = 127.5
PIXEL_SCALE = 127.5


def media_path(data_dir: str | Path, sha256: str) -> Path:
    return Path(data_dir) / "media" / sha256[:2] / f"{sha256}.npz"


def patchify(frames: np.ndarray, patch: int, temporal_patch: int, merge: int) -> torch.Tensor:
    """uint8 (T, C, H, W) → ViT patch_embed input (gt·gh·gw, C·temporal_patch·patch²) float32.

    Same normalize and view/permute as transformers Qwen3VLVideoProcessor._preprocess (images have T=1 and the last
    frame is repeated up to temporal_patch, equivalent to Qwen2VLImageProcessor's temporal expand). The producer asserts
    ``torch.equal`` against the processor output item by item."""
    if frames.dtype != np.uint8 or frames.ndim != 4:
        raise ValueError(f"frames must be uint8 (T, C, H, W), got {frames.dtype} {frames.shape}")
    t, c, h, w = frames.shape
    if h % (patch * merge) or w % (patch * merge):
        raise ValueError(f"frames {h}×{w} is not a multiple of {patch * merge} px")
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.float32)
    x = (x - PIXEL_CENTER) / PIXEL_SCALE
    if pad := -t % temporal_patch:
        x = torch.cat((x, x[-1:].expand(pad, -1, -1, -1)), dim=0)
    gt, gh, gw = x.shape[0] // temporal_patch, h // patch, w // patch
    x = x.view(gt, temporal_patch, c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gt * gh * gw, c * temporal_patch * patch * patch)


def frame_count(modality: str, grid_thw, temporal_patch: int) -> int:
    """Number of stored frames: images 1 (the processor duplicates the image temporal_patch times; only one copy is
    stored), video t·temporal_patch."""
    if modality == "image":
        if grid_thw[0] != 1:
            raise ValueError(f"image with grid t={grid_thw[0]}")
        return 1
    if modality == "video":
        return grid_thw[0] * temporal_patch
    raise ValueError(f"unknown modality {modality!r}")


def pad_ranges(ids: np.ndarray, pad: int) -> list[tuple[int, int]]:
    """Closed-interval runs of pad in ids."""
    hit = (ids == pad).astype(np.int8)
    if not hit.any():
        return []
    d = np.diff(np.concatenate(([0], hit, [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1)]


def check_media_layout(ids: np.ndarray, media: list[dict], vision: dict, where: str) -> None:
    """Pad positions = union of the media ranges; each range is one whole run of that modality's pad, of length
    h·w/merge²; a video's range count = t; media in order of appearance in ids."""
    merge2 = vision["merge"] ** 2
    pads = {"image": vision["image_pad"], "video": vision["video_pad"]}
    want = {"image": set(), "video": set()}
    last = -1
    for i, m in enumerate(media):
        t, h, w = m["grid_thw"]
        ranges = [tuple(r) for r in m["ranges"]]
        if m["modality"] == "image" and (t != 1 or len(ranges) != 1):
            raise RuntimeError(f"{where} media[{i}]: image grid {m['grid_thw']}, {len(ranges)} ranges")
        if m["modality"] == "video" and len(ranges) != t:
            raise RuntimeError(f"{where} media[{i}]: video grid t={t}, {len(ranges)} ranges")
        for a, b in ranges:
            if b - a + 1 != h * w // merge2:
                raise RuntimeError(f"{where} media[{i}]: range [{a}, {b}] has length {b - a + 1}, h·w/{merge2}={h * w // merge2}")
            if a <= last:
                raise RuntimeError(f"{where} media[{i}]: range [{a}, {b}] does not come after the previous one ({last})")
            last = b
            want[m["modality"]].add((a, b))
    for modality, pad in pads.items():
        got = set(pad_ranges(ids, pad))
        if got != want[modality]:
            raise RuntimeError(f"{where}: {modality} pad runs {sorted(got - want[modality])[:3]} have no media, "
                               f"media ranges {sorted(want[modality] - got)[:3]} are not whole pad runs")


class TitoMMDataset:
    """``TitoMMDataset(data_dir, split)``: each item is one TITO ledger segment (≤120k tokens).

    - ``lengths``: np.int64, token count per item (for bin packing); ``loss_tokens``: loss-token count per item;
      ``keys``: list[str] (instance:seg, aligned with index); ``tokenizer_sha256``: sha256 of tokenizer.json at build
      time.
    - ``__getitem__(i)`` → dict: ``input_ids`` np.int64 (verbatim from the ledger), ``loss_mask`` np.bool_ (True at p =
      ``input_ids[p]`` is a token the model is trained to predict; the shift is done by the trainer),
      ``mrope_positions`` np.int64 (3, T) (sglang's algorithm, from 0 within the segment), ``media`` (in order of
      appearance in ids: modality, sha256, ranges, grid_thw, ``pixels`` float32 (t·h·w, C·temporal_patch·patch²)),
      ``key`` (instance:seg).

    ``routing=True``: also reads ``routing.i16``; ``tokens`` / ``__getitem__`` additionally carry ``routing`` np.int16
    (T, layers, top_k) (the ledger routing, ``prepare_tito_mm.py --routing``; raises if the data has none).

    token / mask / position are memmaps and media is one npz per sha256: several ranks each do random access with no
    shared state.

    Subclasses (``rl_batch.RLBatch``) change ``FORMAT`` to read extended formats with the same directory layout."""

    FORMAT = FORMAT

    def __init__(self, data_dir: str | Path, split: str, routing: bool = False):
        if split not in SPLITS:
            raise ValueError(f"split={split!r}, must be one of {SPLITS}")
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text())
        if self.manifest["format"] != self.FORMAT:
            raise RuntimeError(f"{self.data_dir}: format={self.manifest['format']!r}, the reader is {self.FORMAT}")
        self.split = split
        self.vision = self.manifest["vision"]
        d = self.data_dir / split
        with open(d / "index.jsonl") as f:
            self.index = [json.loads(line) for line in f]
        self.lengths = np.array([r["length"] for r in self.index], dtype=np.int64)
        self.loss_tokens = np.array([r["n_loss"] for r in self.index], dtype=np.int64)
        self.keys = [r["key"] for r in self.index]
        if len(set(self.keys)) != len(self.keys):
            raise RuntimeError(f"{d}: duplicate keys in index")
        #: sha256 of tokenizer.json when the data was built (the trainer checks it against the checkpoint)
        self.tokenizer_sha256 = self.manifest["tokenizer_sha256"]
        n = int(self.lengths.sum())
        want = self.manifest["splits"][split]
        if len(self.index) != want["samples"] or n != want["tokens"]:
            raise RuntimeError(f"{d}: index has {len(self.index)} rows / {n} tokens, manifest says {want}")
        files = {"input_ids.i32": "<i4", "loss_mask.u8": np.uint8, "mrope.i32": "<i4"}
        if n == 0:  # empty split (common with a small --subset): np.memmap cannot map an empty file, but the files must still be empty
            nonempty = [name for name in files if (d / name).stat().st_size]
            if nonempty:
                raise RuntimeError(f"{d}: index is empty but {nonempty} have content")
            self._ids, self._loss, self._pos = (np.zeros(0, dtype=t) for t in files.values())
        else:
            self._ids, self._loss, self._pos = (np.memmap(d / name, dtype=t, mode="r") for name, t in files.items())
        if self._ids.shape[0] != n or self._loss.shape[0] != n or self._pos.shape[0] != 3 * n:
            raise RuntimeError(f"{d}: bin sizes {self._ids.shape[0]}/{self._loss.shape[0]}/{self._pos.shape[0]}, "
                               f"index total length {n}")
        self.routing_shape = self.routing_spec = None
        self._routing = None
        if routing:
            # manifests written before --routing have no such field: they carry no routing
            spec = self.manifest.get("routing")
            if spec is None:
                raise RuntimeError(f"{self.data_dir}: no routing (rebuild with prepare_tito_mm.py --routing)")
            if spec["dtype"] != "int16":
                raise RuntimeError(f"{self.data_dir}: routing dtype {spec['dtype']}")
            self.routing_spec = spec
            self.routing_shape = (int(spec["layers"]), int(spec["top_k"]))
            width = self.routing_shape[0] * self.routing_shape[1]
            path = d / "routing.i16"
            if n == 0:
                if path.stat().st_size:
                    raise RuntimeError(f"{d}: index is empty but routing.i16 has content")
                self._routing = np.zeros(0, dtype="<i2")
            else:
                self._routing = np.memmap(path, dtype="<i2", mode="r")
            if self._routing.shape[0] != width * n:
                raise RuntimeError(f"{d}: routing.i16 has {self._routing.shape[0]} values, expected {width}×{n}")

    def __len__(self) -> int:
        return len(self.index)

    def load_pixels(self, m: dict) -> torch.Tensor:
        v = self.vision
        with np.load(media_path(self.data_dir, m["sha256"])) as z:
            frames = z["frames"]
        t, h, w = m["grid_thw"]
        want = (frame_count(m["modality"], m["grid_thw"], v["temporal_patch"]), 3, h * v["patch"], w * v["patch"])
        if frames.shape != want:
            raise RuntimeError(f"media {m['sha256']}: frames {frames.shape}, grid {m['grid_thw']} needs {want}")
        pixels = patchify(frames, v["patch"], v["temporal_patch"], v["merge"])
        if pixels.shape[0] != t * h * w:
            raise RuntimeError(f"media {m['sha256']}: pixels has {pixels.shape[0]} rows, t·h·w={t * h * w}")
        return pixels

    def tokens(self, i: int) -> dict:
        """``__getitem__`` without pixels (media carry metadata only): for checks that need no pixels. Same assertions."""
        r = self.index[i]
        o, n = r["offset"], r["length"]
        ids = np.array(self._ids[o:o + n], dtype=np.int64)
        loss = np.array(self._loss[o:o + n], dtype=np.bool_)
        pos = np.array(self._pos[3 * o:3 * (o + n)], dtype=np.int64).reshape(3, n)
        check_media_layout(ids, r["media"], self.vision, r["key"])
        if loss[(ids == self.vision["image_pad"]) | (ids == self.vision["video_pad"])].any():
            raise RuntimeError(f"{r['key']}: loss falls on a pad")
        if int(loss.sum()) != r["n_loss"]:
            raise RuntimeError(f"{r['key']}: loss tokens {int(loss.sum())}, index says {r['n_loss']}")
        media = [{"modality": m["modality"], "sha256": m["sha256"], "ranges": [tuple(x) for x in m["ranges"]],
                  "grid_thw": tuple(m["grid_thw"])} for m in r["media"]]
        item = {"key": r["key"], "input_ids": ids, "loss_mask": loss, "mrope_positions": pos, "media": media}
        if self._routing is not None:
            width = self.routing_shape[0] * self.routing_shape[1]
            routing = np.array(self._routing[width * o:width * (o + n)], dtype=np.int16).reshape(n, *self.routing_shape)
            if (routing[:n - 1] < 0).any() or (routing[n - 1] != -1).any():
                raise RuntimeError(f"{r['key']}: routing has -1 outside the last position")
            item["routing"] = routing
        return item

    def __getitem__(self, i: int) -> dict:
        item = self.tokens(i)
        for m in item["media"]:
            m["pixels"] = self.load_pixels(m)
        return item
