"""Teacher top-K targets aligned to a tito-mm dataset's supervised positions (format ``qad-topk-v1``).

One format for both producers, so the trainer never knows where the targets came from:

  trainer-prefill  ``train_qad.py --mode teacher`` (the trainer's own FP8 512E forward over the packed rows)
  ledger-decode    ``topk_from_ledger.py`` (sglang ``--session-ledger-top-logprobs`` rows of the served teacher,
                   read with ``arcforge.tito.ledger.read_topk``)

Element layout is the ledger's (``ids_i4,lp_f2,tail_f4,flags_u1``) plus the label, stored per split as flat arrays
whose rows are the supervised positions (``loss_mask``) of the dataset's rows in index order::

    <root>/topk.json              format, k, source, teacher, distribution, flag bits, data dir and per-split hashes
    <root>/<split>/ids.i32        [N, k] token ids, best first
    <root>/<split>/lp.f16         [N, k] log-probs normalized over the full vocabulary, T=1
    <root>/<split>/tail.f32       [N]    log(1 - sum exp(lp)); -inf = no mass left in fp32
    <root>/<split>/flags.u8       [N]    ledger TOPK_* bits (logits altered before the log-prob); 0 = natural
    <root>/<split>/labels.i32     [N]    the supervised token (= input_ids at the position), for the alignment check
    <root>/<split>/DONE           summary JSON, written last; readers refuse a split without it

Row i of the dataset owns positions [offsets[i], offsets[i+1]) with offsets = cumsum(loss_tokens). Positions with
flags != 0 are left out of the loss (``row`` returns ``keep``). Reading renormalizes the 33-class teacher
(top-k + tail) in fp32: f16 log-probs plus the tail miss 1 by ~1e-3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "qad-topk-v1"
FILES = {"ids": ("ids.i32", "<i4", True), "lp": ("lp.f16", "<f2", True), "tail": ("tail.f32", "<f4", False),
         "flags": ("flags.u8", "u1", False), "labels": ("labels.i32", "<i4", False)}
FLAG_BITS = {"1": "hard mask (forced token / grammar cut)", "2": "penalty or min_new_tokens", "4": "logit_bias",
             "8": "grammar", "16": "think-budget partial mask", "32": "custom logit processor"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def data_identity(data) -> dict:
    """What a target split is bound to: the dataset's index and loss mask bytes."""
    d = Path(data.data_dir) / data.split
    return {"rows": len(data), "positions": int(data.loss_tokens.sum()),
            "index_sha256": _sha(d / "index.jsonl"), "loss_mask_sha256": _sha(d / "loss_mask.u8")}


def create(root, datasets: dict, k: int, meta: dict) -> None:
    """Write topk.json and allocate every split's arrays (call on one rank, then barrier)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data_dirs = {str(Path(d.data_dir).resolve()) for d in datasets.values()}
    if len(data_dirs) != 1:
        raise ValueError(f"splits from different datasets: {data_dirs}")
    spec = {"format": FORMAT, "k": k, "distribution": "raw_t1", "tail": "log(1 - sum exp(top-k lp))",
            "flag_bits": FLAG_BITS, "data_dir": data_dirs.pop(),
            "splits": {s: data_identity(d) for s, d in datasets.items()}, **meta}
    for key in ("source", "teacher"):
        if key not in spec:
            raise ValueError(f"meta needs {key!r}")
    for split, ident in spec["splits"].items():
        (root / split).mkdir()
        n = ident["positions"]
        for name, dtype, wide in FILES.values():
            np.memmap(root / split / name, dtype=dtype, mode="w+", shape=((n * k if wide else n) or 1,)).flush()
    (root / "topk.json").write_text(json.dumps(spec, indent=1) + "\n")


class TopkTargets:
    """One split of a ``qad-topk-v1`` directory, bound to ``data`` (a TitoMMDataset of the same split)."""

    def __init__(self, root, data, writable: bool = False):
        self.root = Path(root)
        self.data = data
        self.split = data.split
        self.spec = json.loads((self.root / "topk.json").read_text())
        if self.spec["format"] != FORMAT:
            raise RuntimeError(f"{self.root}: format {self.spec['format']!r}, expected {FORMAT}")
        if self.split not in self.spec["splits"]:
            raise RuntimeError(f"{self.root}: no {self.split} split")
        if str(Path(data.data_dir).resolve()) != self.spec["data_dir"]:
            raise RuntimeError(f"{self.root}: built for {self.spec['data_dir']}, not {data.data_dir}")
        ident = self.spec["splits"][self.split]
        if ident["rows"] != len(data) or ident["positions"] != int(data.loss_tokens.sum()):
            raise RuntimeError(f"{self.root}/{self.split}: {ident['rows']} rows / {ident['positions']} positions, "
                               f"data has {len(data)} / {int(data.loss_tokens.sum())}")
        if not writable and not (self.root / self.split / "DONE").exists():
            raise RuntimeError(f"{self.root}/{self.split}: no DONE marker (capture incomplete)")
        self.k = int(self.spec["k"])
        self.offsets = np.concatenate([[0], np.cumsum(data.loss_tokens)]).astype(np.int64)
        n = int(self.offsets[-1])
        mode = "r+" if writable else "r"
        self.arrays, self._maps = {}, []
        for key, (name, dtype, wide) in FILES.items():
            m = np.memmap(self.root / self.split / name, dtype=dtype, mode=mode)
            self._maps.append(m)
            want = (n * self.k if wide else n) if n else 1
            if m.shape[0] != want:
                raise RuntimeError(f"{self.root}/{self.split}/{name}: {m.shape[0]} values, expected {want}")
            self.arrays[key] = m.reshape(-1, self.k) if wide else m
        self._kept = None

    def write_row(self, i: int, ids, lp, tail, flags, labels) -> None:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        n = hi - lo
        for key, v in (("ids", ids), ("lp", lp), ("tail", tail), ("flags", flags), ("labels", labels)):
            v = np.asarray(v)
            if v.shape[0] != n or (FILES[key][2] and v.shape[1] != self.k):
                raise ValueError(f"row {i} {key}: shape {v.shape}, expected ({n}{', ' + str(self.k) if FILES[key][2] else ''})")
        if np.isnan(np.asarray(lp, dtype=np.float32)).any() or np.isnan(np.asarray(tail, dtype=np.float32)).any():
            raise ValueError(f"row {i}: NaN in the teacher targets")
        self.arrays["ids"][lo:hi] = np.asarray(ids, dtype=np.int32)
        self.arrays["lp"][lo:hi] = np.asarray(lp, dtype=np.float32).astype(np.float16)
        self.arrays["tail"][lo:hi] = np.asarray(tail, dtype=np.float32)
        self.arrays["flags"][lo:hi] = np.asarray(flags, dtype=np.uint8)
        self.arrays["labels"][lo:hi] = np.asarray(labels, dtype=np.int32)

    def flush(self) -> None:
        for m in self._maps:
            m.flush()

    def finish(self, summary: dict) -> None:
        self.flush()
        (self.root / self.split / "DONE").write_text(json.dumps(summary, indent=1) + "\n")

    def row(self, i: int) -> dict:
        """Targets of row i: ids int32 [n, k], lp / tail fp32 (33-class renormalized), flags, labels, keep bool."""
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        lp = np.asarray(self.arrays["lp"][lo:hi], dtype=np.float32)
        tail = np.asarray(self.arrays["tail"][lo:hi], dtype=np.float32)
        if np.isnan(lp).any() or np.isnan(tail).any():
            raise RuntimeError(f"{self.root}/{self.split} row {i}: NaN")
        full = np.concatenate([lp, tail[:, None]], axis=1).astype(np.float64)
        top = full.max(1, keepdims=True)
        log_z = (top + np.log(np.exp(full - top).sum(1, keepdims=True)))[:, 0]
        flags = np.asarray(self.arrays["flags"][lo:hi])
        return {"ids": np.asarray(self.arrays["ids"][lo:hi]), "lp": (lp - log_z[:, None]).astype(np.float32),
                "tail": (tail - log_z).astype(np.float32), "flags": flags, "labels": np.asarray(self.arrays["labels"][lo:hi]),
                "keep": flags == 0}

    def kept_counts(self) -> np.ndarray:
        """Per dataset row: supervised positions that enter the loss (flags == 0)."""
        if self._kept is None:
            keep = (np.asarray(self.arrays["flags"]) == 0).astype(np.int64)
            c = np.concatenate([[0], np.cumsum(keep)])
            self._kept = c[self.offsets[1:]] - c[self.offsets[:-1]]
        return self._kept
