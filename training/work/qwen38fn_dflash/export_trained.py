"""Export a trainer checkpoint as a BF16 drafter directory in the init layout.

The output loads with `init_mtp3.load_checkpoint`, so serving can use it as
`INIT` without `MTP3_CHECKPOINT`. Its values are exactly what the serving
loader builds from init + checkpoint: frozen tensors from the init, trained
FP32 masters rounded to BF16. Optimizer state is not exported.

    python work/qwen38fn_dflash/export_trained.py \
        --checkpoint $ARC_RUNS/qwen38fn_dflash/<run>/trainer/step-NNNNNNNNN --out <new dir>
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from init_mtp3 import load_checkpoint
from load_trained import load_trained_parameters
from safetensors.torch import save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def export(checkpoint, out):
    checkpoint, out = Path(checkpoint), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    init = Path(manifest["source_init"])
    model = load_checkpoint(init)
    loaded = load_trained_parameters(model, checkpoint, init)
    state = model.state_dict()
    if set(state) != set(dict(model.named_parameters())):
        raise RuntimeError("drafter state has non-parameter entries the init layout cannot hold")

    # Publish atomically: a half-written directory never carries the final name.
    partial = out.with_name(out.name + ".partial")
    partial.mkdir(parents=True)
    files = []
    for i in range(model.config.num_layers):
        filename = f"layer-{i}.safetensors"
        save_file({k: v for k, v in state.items() if k.startswith(f"layers.{i}.")}, str(partial / filename))
        files.append(filename)
    save_file({k: v for k, v in state.items() if not k.startswith("layers.")}, str(partial / "interface.safetensors"))
    files.append("interface.safetensors")
    shutil.copyfile(init / "config.json", partial / "config.json")
    record = {
        "complete": True,
        "files": files,
        "kind": "trained-bf16-export",
        "dtype": "bfloat16",
        "step": loaded["step"],
        "loaded_parameters": loaded["loaded_parameters"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_world_size": manifest["world_size"],
        "dataset_manifest": manifest["dataset_manifest"],
        "source_init": str(init.resolve()),
        "source_init_manifest_sha256": sha256(init / "initialization_manifest.json"),
    }
    (partial / "initialization_manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    # Round trip through the serving loader, not just the files we wrote.
    reloaded = load_checkpoint(partial)
    for name, value in reloaded.named_parameters():
        if not torch.equal(value, state[name]):
            raise RuntimeError(f"exported tensor does not round-trip: {name}")
    del reloaded

    names = sorted(p.name for p in partial.iterdir())
    (partial / "SHA256SUMS").write_text("".join(f"{sha256(partial / n)}  {n}\n" for n in names))
    os.rename(partial, out)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path, help="trainer/step-NNNNNNNNN directory")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(16)
    print(json.dumps(export(args.checkpoint, args.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
