"""QAD v2 export (train_qad.export_delta) + the source gEAN320 INT3 package -> a complete servable package.

Every delta tensor must already exist in the source under the same name, dtype and shape (scales -> w13_scales /
w2_scales BF16, the 144 FP8 projections -> e4m3 weight + fp32 weight_scale_inv, the rest BF16): nothing is added,
renamed or re-quantized. Source shards holding a replaced tensor are rewritten tensor by tensor (other tensors' bytes
copied verbatim, shard metadata kept); untouched shards and the non-weight files are hardlinked (same filesystem) or
copied. Afterwards every replaced tensor is read back and its sha256 checked against the delta manifest, and every
untouched tensor of a rewritten shard against the source. The package's deploy_manifest.json (per-file sha256 /
bytes that the Kaggle pins and runtime verify) is rewritten as a new file -- never through the hardlink to the source --
with the rewritten shards' hashes and a "qad" block; the other files' entries are the source's (unchanged bytes).
Writes <out>.partial, renames to <out>, then QAD_DONE.

    python materialize_package.py --source <package> --delta <export/vNNNNNN> --out <new package dir>
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, delta, out = Path(a.source), Path(a.delta), Path(a.out)
    if not (delta / "DONE").exists():
        raise RuntimeError(f"{delta}: no DONE marker")
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(out)
    dman = json.loads((delta / "delta_manifest.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    new = {}
    for f in dman["files"]:
        with safe_open(str(delta / f["file"]), "pt") as h:
            for k in h.keys():
                new[k] = h.get_tensor(k)
    if set(new) != set(dman["tensors"]):
        raise RuntimeError("delta files and manifest disagree")
    for k, v in new.items():
        if sha(v) != dman["tensors"][k]["sha256"]:
            raise RuntimeError(f"{k}: delta tensor does not match its manifest hash")
        if k not in index:
            raise KeyError(f"{k}: not in the source package (the export would add a tensor)")
    by_shard = {}
    for k in new:
        by_shard.setdefault(index[k], []).append(k)
    partial.mkdir(parents=True)
    same_fs = os.stat(src).st_dev == os.stat(partial).st_dev
    report = {"rewritten_shards": [], "linked_files": 0, "replaced": 0, "changed_values": 0}
    for entry in sorted(src.iterdir()):
        if entry.is_dir():
            raise RuntimeError(f"unexpected directory in the package: {entry}")
        target = partial / entry.name
        if entry.name in by_shard:
            tensors, meta = {}, None
            with safe_open(str(entry), "pt") as h:
                meta = h.metadata()
                for k in h.keys():
                    t = h.get_tensor(k)
                    if k in new:
                        if new[k].dtype != t.dtype or new[k].shape != t.shape:
                            raise TypeError(f"{k}: delta {new[k].dtype} {tuple(new[k].shape)}, source {t.dtype} "
                                            f"{tuple(t.shape)}")
                        report["changed_values"] += int((new[k].view(torch.uint8) != t.view(torch.uint8)).sum())
                        t = new[k]
                        report["replaced"] += 1
                    tensors[k] = t
            save_file(tensors, str(target), metadata=meta)
            report["rewritten_shards"].append(entry.name)
        elif same_fs:
            os.link(entry, target)
            report["linked_files"] += 1
        else:
            shutil.copy2(entry, target)
            report["linked_files"] += 1
    # read back
    for shard, keys in by_shard.items():
        with safe_open(str(partial / shard), "pt") as h, safe_open(str(src / shard), "pt") as s:
            if set(h.keys()) != set(s.keys()):
                raise RuntimeError(f"{shard}: tensor set changed")
            for k in h.keys():
                want = dman["tensors"][k]["sha256"] if k in new else sha(s.get_tensor(k))
                if sha(h.get_tensor(k)) != want:
                    raise RuntimeError(f"{shard}/{k}: read-back mismatch")
    if report["replaced"] != len(new):
        raise RuntimeError(f"replaced {report['replaced']} of {len(new)} delta tensors")
    deploy = partial / "deploy_manifest.json"
    if deploy.exists():
        manifest = json.loads(deploy.read_text())
        files = manifest["files"]
        for shard in report["rewritten_shards"]:
            if shard not in files:
                raise KeyError(f"deploy_manifest.json has no entry for {shard}")
            h = hashlib.sha256()
            with open(partial / shard, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            files[shard] = {"sha256": h.hexdigest(), "bytes": (partial / shard).stat().st_size}
        manifest["qad"] = {"source_package": str(src), "delta": str(delta),
                           "delta_meta": {k: v for k, v in dman.items() if k not in ("files", "tensors")},
                           "rewritten_shards": report["rewritten_shards"]}
        deploy.unlink()                       # the hardlink to the source package's manifest: replace, never edit
        deploy.write_text(json.dumps(manifest, indent=2) + "\n")
    (partial / "qad_manifest.json").write_text(json.dumps({
        "source": str(src), "delta": str(delta), "delta_meta": {k: v for k, v in dman.items() if k not in ("files", "tensors")},
        **report}, indent=1) + "\n")
    partial.rename(out)
    (out / "QAD_DONE").write_text("ok\n")
    print(json.dumps({k: v if k != "rewritten_shards" else len(v) for k, v in report.items()}))


if __name__ == "__main__":
    main()
