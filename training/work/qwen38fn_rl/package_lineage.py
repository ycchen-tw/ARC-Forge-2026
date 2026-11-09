"""The served package and the trainer's base must be one model.

An int3 RL run serves a deployment package (dense projections FP8) and trains LoRA on its BF16-dense variant
(``rollout.trainer_base_path``); nothing else ties the two directories together, and the launchers default both
(launch_b200_stream.sh: the gEAN320 pair). One of RL_SERVE_MODEL / RL_TRAIN_MODEL set and the other left at its
default trains the LoRA on another base and syncs it into the servers with no error anywhere (10-24 review P0-3).

The check: every tensor both directories hold under the same name, dtype and shape has the same bytes. The BF16-dense
rewrite changes the dense projections' dtype (and drops their scales), so those are not compared; everything else is
byte-identical in a true pair (gEAN320 and QAD-R pairs: 1580 tensors, 0 differ) and a crossed pair differs in hundreds
(QAD-R package with the gEAN320 base: 420). Tensors up to FULL_BYTES are compared whole, larger ones in WINDOWS evenly
spaced windows of WINDOW_BYTES: a different checkpoint differs throughout a tensor, and the whole-package compare reads
~87 GB (60 s from tmpfs) for the same answer. Standard library only (the orchestrator imports it).
"""
from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

FULL_BYTES = 64 << 20
WINDOWS = 16
WINDOW_BYTES = 1 << 20


class LineageError(RuntimeError):
    pass


def tensor_index(model_dir: Path) -> dict[str, tuple[str, int, int, str, tuple[int, ...]]]:
    """name -> (shard file, start, end, dtype, shape) from the safetensors headers (absolute byte offsets)."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (model_dir / "model.safetensors").is_file():
        shards = ["model.safetensors"]
    else:
        raise LineageError(f"{model_dir}: neither model.safetensors.index.json nor model.safetensors")
    out = {}
    for shard in shards:
        with open(model_dir / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in out:
                raise LineageError(f"{model_dir}: tensor {name} in both {out[name][0]} and {shard}")
            begin, end = entry["data_offsets"]
            out[name] = (shard, 8 + n + begin, 8 + n + end, entry["dtype"], tuple(entry["shape"]))
    return out


def windows(size: int) -> list[tuple[int, int]]:
    if size <= FULL_BYTES:
        return [(0, size)]
    return [(i * (size - WINDOW_BYTES) // (WINDOWS - 1), i * (size - WINDOW_BYTES) // (WINDOWS - 1) + WINDOW_BYTES)
            for i in range(WINDOWS)]


def check_lineage(serving: Path, trainer: Path) -> str:
    """Raises LineageError unless every shared (name, dtype, shape) tensor has the same bytes; returns a summary.
    A served merged export (merged_export.py: ``merged_manifest.json``) differs from its base by design; its manifest
    names the trainer base it was merged from, which must be ``trainer``. (The launchers start every replica from the
    serve package and sync with keep_identity, so the served model_path stays the package; this is for a replica
    started from a merged export by hand.)"""
    merged = serving / "merged_manifest.json"
    if merged.is_file():
        manifest = json.loads(merged.read_text())
        base = Path(manifest["base_model"])
        if base.resolve() != trainer.resolve():
            raise LineageError(f"served merged export {serving} (step {manifest['step']}) was merged onto {base}, "
                               f"not onto the trainer base {trainer}")
        return f"lineage: {serving.name} is a merged export (step {manifest['step']}) of the trainer base {trainer.name}"
    a, b = tensor_index(serving), tensor_index(trainer)
    shared = sorted(k for k in a if k in b and a[k][3:] == b[k][3:])
    if not shared:
        raise LineageError(f"{serving} and {trainer} share no tensor of the same name, dtype and shape")
    maps: dict[tuple[Path, str], mmap.mmap] = {}

    def view(root: Path, shard: str) -> mmap.mmap:
        if (root, shard) not in maps:
            with open(root / shard, "rb") as f:
                maps[root, shard] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return maps[root, shard]

    differ, read = [], 0
    try:
        for name in shared:
            sa, oa, ea = a[name][:3]
            sb, ob, eb = b[name][:3]
            for begin, end in windows(ea - oa):
                read += end - begin
                if view(serving, sa)[oa + begin:oa + end] != view(trainer, sb)[ob + begin:ob + end]:
                    differ.append(name)
                    break
    finally:
        for m in maps.values():
            m.close()
    if differ:
        raise LineageError(f"serving package {serving} and trainer base {trainer} are not one model: {len(differ)} of "
                           f"{len(shared)} shared tensors differ (e.g. {differ[:3]}); RL_SERVE_MODEL / RL_TRAIN_MODEL "
                           f"(configure --tokenizer-dir / --trainer-base-path) come from different checkpoints")
    return (f"lineage: {serving.name} / {trainer.name}: {len(shared)} shared tensors identical "
            f"({read / 2**30:.1f} GiB compared; {len(set(a) - set(b))} only served, {len(set(b) - set(a))} only trained)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("usage: package_lineage.py <served package> <trainer base>")
    try:
        print(check_lineage(Path(sys.argv[1]), Path(sys.argv[2])))
    except LineageError as e:
        raise SystemExit(f"✗ {e}") from None
