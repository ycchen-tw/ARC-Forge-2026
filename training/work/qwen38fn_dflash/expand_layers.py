"""A deeper drafter from a trained one by repeating its decoder layers: layer j of the output is a copy of source
layer order[j], e.g. --order 0,1,2,3,1,2,3,4 turns plan B's 5 layers into 8.

    python expand_layers.py --source <drafter in the init layout> --order 0,1,2,3,1,2,3,4 --zero-repeats --out <dir>

--zero-repeats (LLaMA-Pro's block expansion): every repeat -- a position whose source layer already appeared earlier
in the order -- gets self_attn.o_proj and mlp.down_proj zeroed. A plain-residual layer adds conv(attention output) and
conv(FFN output) to its stream; GroupedConv.convolve is linear in the operator output and in its boundary, which is
that same output one row earlier (context "recompute") or zero ("inject"). So a zeroed repeat adds exactly 0 and the
expanded drafter computes the source's function; training moves the zeroed projections first (their gradients are the
only nonzero ones in that layer at step 0).
--no-zero-repeats copies the repeats as they are (the function changes). Plain residual only: an HC layer's
hyper-connections also mix the streams, so zeroing its outputs is not an identity there.

Output: init_mtp3's layout (layer-<j>.safetensors, interface.safetensors byte-for-byte the source's, config.json with
num_layers = len(order), initialization_manifest.json kind "layer-expansion", SHA256SUMS), published atomically and
reloaded with init_mtp3.load_checkpoint before it is published.
"""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch
from init_mtp3 import load_checkpoint
from model import Config
from safetensors.torch import load_file, save_file

ZEROED = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def parse_order(text, num_layers):
    order = [int(x) for x in text.split(",")]
    if not order or any(not 0 <= i < num_layers for i in order):
        raise ValueError(f"--order {text}: every entry must be a source layer in 0..{num_layers - 1}")
    if sorted(set(order)) != list(range(num_layers)):
        raise ValueError(f"--order {text} drops source layers {sorted(set(range(num_layers)) - set(order))}")
    return order


def repeats(order):
    """Positions whose source layer already appeared earlier in the order."""
    return [j for j, i in enumerate(order) if i in order[:j]]


def expand(source, order, zero_repeats, out):
    """Write the expanded drafter to out (atomically); returns the manifest's record."""
    c = Config(**json.loads((source / "config.json").read_text()))
    manifest = json.loads((source / "initialization_manifest.json").read_text())
    if not manifest["complete"] or manifest["dtype"] != "bfloat16":
        raise ValueError(f"{source}: want a complete bfloat16 drafter, the manifest says {manifest}")
    if c.residual != "plain":
        raise ValueError(f"{source}: residual {c.residual!r} -- a zeroed repeat is an identity only on the plain stream")
    want = [f"layer-{i}.safetensors" for i in range(c.num_layers)] + ["interface.safetensors"]
    if manifest["files"] != want:
        raise ValueError(f"{source}: files {manifest['files']}, want {want}")
    zeroed = []
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(out)
    partial.mkdir(parents=True)
    expanded = replace(c, num_layers=len(order))
    files = []
    for j, i in enumerate(order):
        state = load_file(str(source / f"layer-{i}.safetensors"))
        prefix = f"layers.{i}."
        if any(not k.startswith(prefix) for k in state):
            raise ValueError(f"{source}/layer-{i}.safetensors holds tensors of another layer")
        layer = {f"layers.{j}.{k[len(prefix):]}": v.contiguous() for k, v in state.items()}
        if zero_repeats and j in repeats(order):
            for name in ZEROED:
                key = f"layers.{j}.{name}"
                layer[key] = torch.zeros_like(layer[key])  # KeyError: the layer has no such projection
                zeroed.append(key)
        save_file(layer, str(partial / f"layer-{j}.safetensors"))
        files.append(f"layer-{j}.safetensors")
    shutil.copyfile(source / "interface.safetensors", partial / "interface.safetensors")
    files.append("interface.safetensors")
    record = {"complete": True, "files": files, "kind": "layer-expansion", "dtype": "bfloat16",
              "source": str(source.resolve()), "source_manifest_sha256": sha256(source / "initialization_manifest.json"),
              "source_config_sha256": sha256(source / "config.json"), "order": order, "zero_repeats": zero_repeats,
              "repeats": repeats(order), "zeroed": zeroed, "converter_sha256": sha256(Path(__file__))}
    (partial / "config.json").write_text(json.dumps(asdict(expanded), indent=2) + "\n")
    (partial / "initialization_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    model = load_checkpoint(partial)  # coverage: every parameter of the expanded Config, nothing else
    sources = {i: load_file(str(source / f"layer-{i}.safetensors")) for i in sorted(set(order))}
    for name, value in model.named_parameters():
        if not name.startswith("layers."):
            continue  # interface.safetensors is the source's file, byte for byte
        _, j, rest = name.split(".", 2)
        if name in zeroed:
            if value.count_nonzero():
                raise RuntimeError(f"{name} is not zero after the round trip")
        elif not torch.equal(value, sources[order[int(j)]][f"layers.{order[int(j)]}.{rest}"]):
            raise RuntimeError(f"{name} is not source layer {order[int(j)]}'s tensor")
    del model, sources
    names = sorted(p.name for p in partial.iterdir())
    (partial / "SHA256SUMS").write_text("".join(f"{sha256(partial / n)}  {n}\n" for n in names))
    os.rename(partial, out)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path, help="a plain-residual drafter in the init layout")
    p.add_argument("--order", required=True, help="comma-separated source layer of each output layer")
    # Required, not defaulted: the two give different step-0 drafters (the source's function, or not).
    p.add_argument("--zero-repeats", required=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    c = Config(**json.loads((args.source / "config.json").read_text()))
    order = parse_order(args.order, c.num_layers)
    print(json.dumps(expand(args.source, order, args.zero_repeats, args.out), indent=1))


if __name__ == "__main__":
    main()
