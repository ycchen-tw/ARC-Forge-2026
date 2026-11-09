"""MTP draft directories for the INT3 gEAN320 target (which ships no mtp.*), from the official FP8 checkpoint's MTP.

    python build_drafts.py moe320 --out <dir> [--layers N | --spec M,T43,T47 [--target-final-mixer]] [--swa-window W]
    python build_drafts.py dense --fits <fit.json>[,...] --alpha least_squares|uniform --out <dir> [--spec M[,...]]
                                 [--swa-window W]

Both write mtp.safetensors (only mtp.* tensors), model.safetensors.index.json, config.json and the tokenizer files.
config.json is the INT3 package's (verified byte-identical to the 10-10 manifest) with mtp_num_hidden_layers=1 and the
official FP8 checkpoint's quantization_config (block FP8 128x128; every non-expert MTP tensor stays BF16), so the
draft is served with --speculative-draft-model-path <dir> --speculative-draft-model-quantization fp8.

Layers. M is the MTP decoder layer (mtp.layers.0.*), its routed MoE pruned to the 320 experts kept by the gEAN320
run's MTP mask (inputs/mtp320.pt, keep = not drop, ascending ids renumbered 0..319; router rows sliced the same way).
T<L> is the target's decoder layer L (must be full attention) from the official FP8 checkpoint
(model.language_model.layers.L.*), its routed experts pruned to the 320 the INT3 deployment keeps for layer L
(deploy_manifest keep_per_layer, renumbered in that order, router rows sliced alike). The target's expert-location map
is built from the target's num_experts (320), so every draft layer must have 320 experts. Expert FP8 weights and
scales are copied unchanged. The input fusion (fc_*, pre_fc_norm_*) and the final hyper_connection_mixer are the
MTP's, once, around the layer stack.

moe320: one M layer by default. --layers N (>= 2): depth stacking, mtp.layers.1..N-1 are bit copies of mtp.layers.0
(stack source "copy"). --spec: the listed layers in order as mtp.layers.0.. (e.g. M,T43,T47; an entry may repeat;
stack source = the spec). --target-final-mixer replaces mtp.hyper_connection_mixer.* with the target's final
model.language_model.hyper_connection_mixer.* (the mixer that follows layer 47 in the target). Stacked drafts get
text_config mtpd_stack_layers=N / mtpd_stack_source (read by ../patch/mtpd_runtime.py) and modules_to_not_convert
entries for every layer (without them SGLang builds layers 1..N-1's BF16 modules as FP8 linears).

dense: every layer of --spec (default M) with its routed MoE replaced by one SwiGLU of width E*640 (16 -> 10240) built from
the 16 experts of that layer's fit.json (fit_dense.py --layer i --source <entry>, or with --fits-from-other-layers a
fit of the same source made at another position; indices in the original 512):
gate/up rows and down columns concatenated in that order. Block FP8 blocks never straddle two experts (640 = 5*128),
so the dense matrices are the experts' own FP8 bytes; alpha_e is applied to the fp32 block scales only, sqrt|alpha_e|
on expert e's up-row scales and sign(alpha_e) sqrt|alpha_e| on its down-column scales (convert_dense.split_alpha's
placement). Router and routed experts are dropped; shared expert, shared gate and every other tensor are copied.
Every layer of a draft has the same E (from its fits). text_config gains mtp_dense_intermediate_size=E*640 (read by
../patch/mtpd_runtime.py).
--swa-window W (either kind): text_config mtpd_swa_window=W, the draft layers attend to the last W positions instead
of the QSA selection (../patch/mtpd_runtime.py); the tensors are unchanged.
Every written tensor is re-read and checked against its source.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_SRC = Path(os.environ["ARC_MODELS"]) / "Qwen3.8-Flash-Next-FP8"
INT3_CONFIG = Path(os.environ["ARC_RUNS"]) / "qwen38fn_rl/20261010-gean320-w3a8/trainer-int3/ref/config.json"
INT3_CONFIG_SHA = "8b2a29dbe7f7922238b447d689bc574de9b34ef7d3c1e16adae7ea92d4b5c27a"
MASK = Path(os.environ["MTPD_RUN"]) / "inputs" / "mtp320.pt"
MASK_SHA = "76db528968dc90e56be4212630d09ab88b36185633013a032df155a6905ecd7a"
DEPLOY_MANIFEST = Path(os.environ["ARC_RUNS"]) / "qwen38fn_rl/20261010-gean320-w3a8/trainer-int3/ref/deploy_manifest.json"
DEPLOY_MANIFEST_SHA = "9476520a3a24477451b65d5c5bce719ec853874a93b0147d20b4d04d2265571f"
SMALL_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")
E, I, H, BLOCK = 512, 640, 2560, 128
PROJ = ("gate_proj", "up_proj", "down_proj")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def read_tensors(prefix, keep=None):
    """FP8 checkpoint tensors whose names start with prefix (routed experts only for ids in keep, if given)."""
    index = json.loads((FP8_SRC / "model.safetensors.index.json").read_text())["weight_map"]
    names = []
    for k in index:
        if not k.startswith(prefix):
            continue
        if keep is not None and ".mlp.experts." in k and int(k.split(".mlp.experts.")[1].split(".")[0]) not in keep:
            continue
        names.append(k)
    out = {}
    for fn in sorted({index[k] for k in names}):
        with safe_open(FP8_SRC / fn, "pt") as f:
            for k in names:
                if index[k] == fn:
                    out[k] = f.get_tensor(k)
    return out


def read_mtp():
    """All mtp.* tensors of the FP8 checkpoint, grouped by shard."""
    index = json.loads((FP8_SRC / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(k for k in index if k.startswith("mtp."))
    by_file = {}
    for k in names:
        by_file.setdefault(index[k], []).append(k)
    out = {}
    for fn, keys in sorted(by_file.items()):
        with safe_open(FP8_SRC / fn, "pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    if len(out) != 3101:
        raise RuntimeError(f"expected 3101 mtp tensors, got {len(out)}")
    return out


def parse_spec(text):
    spec = text.split(",")
    if not all(re.fullmatch(r"M|T\d+", x) for x in spec):
        raise ValueError(f"layer spec {text!r}: entries are M or T<L>")
    if len(spec) == 1 and spec != ["M"]:
        raise ValueError("a one-layer draft is the MTP layer (spec M)")
    return spec


def base_config(dense_width=None, stack_layers=None, stack_source="copy", spec=None, final_mixer="mtp", swa_window=None):
    if sha256(INT3_CONFIG) != INT3_CONFIG_SHA:
        raise RuntimeError("INT3 config copy differs from the package manifest")
    cfg = json.loads(INT3_CONFIG.read_text())
    src = json.loads((FP8_SRC / "config.json").read_text())
    t = cfg["text_config"]
    if t["num_experts"] != 320 or t["mtp_num_hidden_layers"] != 0:
        raise RuntimeError(f"unexpected INT3 config: num_experts {t['num_experts']} mtp {t['mtp_num_hidden_layers']}")
    st = src["text_config"]
    same = ("hidden_size", "moe_intermediate_size", "shared_expert_intermediate_size", "hc_count", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "mtp")
    diff = [k for k in same if t.get(k) != st.get(k)]
    if diff:
        raise RuntimeError(f"INT3 and FP8 configs differ on {diff}")
    t["mtp_num_hidden_layers"] = 1
    if dense_width is not None:
        t["mtp_dense_intermediate_size"] = dense_width
    if swa_window is not None:
        if not 1 <= swa_window <= t["indexer_budget"]:
            raise RuntimeError(f"SWA window {swa_window} outside 1..indexer_budget {t['indexer_budget']}")
        t["mtpd_swa_window"] = swa_window
    if stack_layers is not None:
        t["mtpd_stack_layers"] = stack_layers
        t["mtpd_stack_source"] = stack_source
        t["mtpd_final_mixer"] = final_mixer
        t["mtp_num_hidden_layers"] = stack_layers
        t["mtp"]["num_hidden_layers"] = stack_layers
        t["mtp"]["layer_types"] = ["full_attention"] * stack_layers
    cfg["quantization_config"] = src["quantization_config"]
    if stack_layers is not None:
        # Every layer's BF16 modules must stay unquantized like layer 0's; without these entries SGLang builds layers
        # 1..N-1's attention / indexer / shared expert as FP8 linears and loads BF16 bytes into them.
        skip = cfg["quantization_config"]["modules_to_not_convert"]
        layer0 = [m for m in skip if m.startswith("mtp.layers.0.")]
        if len(layer0) != 16:
            raise RuntimeError(f"expected 16 mtp.layers.0 entries in modules_to_not_convert, got {len(layer0)}")
        for i, item in enumerate(spec or ["M"] * stack_layers):
            if item != "M":
                src_prefix = f"model.language_model.layers.{int(item[1:])}."
                tl = sorted(m.replace(src_prefix, "mtp.layers.0.", 1) for m in skip if m.startswith(src_prefix))
                if tl != sorted(layer0):
                    raise RuntimeError(f"target layer {item} keeps a different BF16 set than the MTP layer: {tl}")
            if i:
                skip += [m.replace("mtp.layers.0.", f"mtp.layers.{i}.", 1) for m in layer0]
    return cfg


def write_dir(out, tensors, cfg, manifest):
    out.mkdir(parents=True, exist_ok=False)
    save_file(tensors, out / "mtp.safetensors", metadata={"format": "pt"})
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(v.numel() * v.element_size() for v in tensors.values())},
         "weight_map": {k: "mtp.safetensors" for k in sorted(tensors)}}, indent=1) + "\n")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in SMALL_FILES:
        shutil.copyfile(FP8_SRC / name, out / name)
    with safe_open(out / "mtp.safetensors", "pt") as f:
        if set(f.keys()) != set(tensors):
            raise RuntimeError("written key set differs")
        for k, v in tensors.items():
            got = f.get_tensor(k)
            a, b = (got.view(torch.uint8), v.view(torch.uint8)) if v.dtype == torch.float8_e4m3fn else (got, v)
            if got.dtype != v.dtype or not torch.equal(a, b):
                raise RuntimeError(f"{k}: re-read differs")
    manifest["files_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir())}
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"}, indent=1))


def split_mtp(src):
    """mtp.* -> (tensors outside the decoder layer, the decoder layer's tensors with names relative to the layer)."""
    pre = "mtp.layers.0."
    if any(k.startswith("mtp.layers.") and not k.startswith(pre) for k in src):
        raise RuntimeError("the checkpoint's MTP has more than one layer")
    return ({k: v for k, v in src.items() if not k.startswith("mtp.layers.")},
            {k[len(pre):]: v for k, v in src.items() if k.startswith(pre)})


def mtp320_keep():
    if sha256(MASK) != MASK_SHA:
        raise RuntimeError("mtp320 mask hash differs")
    mask = torch.load(MASK, map_location="cpu", weights_only=False)
    if mask["n_experts"] != E or mask["keep"] != 320 or mask["top_k"] != 10:
        raise RuntimeError(f"unexpected mask {mask['tag']}")
    keep = (~mask["drop"][0]).nonzero().flatten().tolist()
    if len(keep) != 320:
        raise RuntimeError(f"mask keeps {len(keep)}")
    return keep


def target_keep(L):
    if json.loads((FP8_SRC / "config.json").read_text())["text_config"]["layer_types"][L] != "full_attention":
        raise RuntimeError(f"target layer {L} is not full attention")
    if sha256(DEPLOY_MANIFEST) != DEPLOY_MANIFEST_SHA:
        raise RuntimeError("INT3 deploy manifest copy differs from the package")
    keep = json.loads(DEPLOY_MANIFEST.read_text())["keep_per_layer"][str(L)]
    if len(keep) != 320 or len(set(keep)) != 320:
        raise RuntimeError(f"layer {L} keep list is not 320 distinct experts")
    return keep


def raw_layer(item, mtp_layer, experts=None):
    """Decoder layer tensors (names relative to the layer, routed experts by original id): the MTP layer for M, the
    target's layer L for T<L> (only the experts in `experts`, if given)."""
    if item == "M":
        return mtp_layer
    L = int(item[1:])
    pre = f"model.language_model.layers.{L}."
    return {k[len(pre):]: v for k, v in read_tensors(pre, keep=None if experts is None else set(experts)).items()}


def routed(k):
    return k.startswith("mlp.experts.") or k == "mlp.gate.weight"


def prune(lay, keep):
    """Routed experts restricted to keep (renumbered in that order), router rows alike; everything else unchanged."""
    out = {k: v for k, v in lay.items() if not routed(k)}
    out["mlp.gate.weight"] = lay["mlp.gate.weight"][keep].contiguous()
    for j, e in enumerate(keep):
        for proj in PROJ:
            for suffix in ("weight", "weight_scale_inv"):
                out[f"mlp.experts.{j}.{proj}.{suffix}"] = lay[f"mlp.experts.{e}.{proj}.{suffix}"]
    return out


def assemble(outer, layers, spec):
    """outer tensors + layers[i] as mtp.layers.<i>.*; all layers must have one tensor set (names, shapes, dtypes)."""
    ref = {k: (tuple(v.shape), v.dtype) for k, v in layers[0].items()}
    for item, lay in zip(spec, layers):
        got = {k: (tuple(v.shape), v.dtype) for k, v in lay.items()}
        if got != ref:
            bad = sorted(k for k in set(got) | set(ref) if got.get(k) != ref.get(k))
            raise RuntimeError(f"layer {item} tensor set differs from layer {spec[0]}: {bad[:5]}")
    out, storages = dict(outer), {v.untyped_storage().data_ptr() for v in outer.values()}
    for i, lay in enumerate(layers):
        for k, v in lay.items():
            ptr = v.untyped_storage().data_ptr()
            if ptr in storages:  # a repeated layer: safetensors refuses shared storage
                v = v.clone()
                ptr = v.untyped_storage().data_ptr()
            storages.add(ptr)
            out[f"mtp.layers.{i}.{k}"] = v
    return out


def moe320(a):
    if a.layers > 1 and a.spec:
        raise RuntimeError("--layers and --spec are exclusive")
    spec = parse_spec(a.spec) if a.spec else ["M"] * a.layers
    if a.target_final_mixer and all(x == "M" for x in spec):
        raise RuntimeError("--target-final-mixer goes with a spec containing a target layer")
    keep = mtp320_keep()
    outer, mtp_layer = split_mtp(read_mtp())
    layers, kept = [], {}
    for item in spec:
        if item == "M":
            layers.append(prune(mtp_layer, keep))
        else:
            kept[item] = kept.get(item) or target_keep(int(item[1:]))
            layers.append(prune(raw_layer(item, mtp_layer, kept[item]), kept[item]))
    if a.target_final_mixer:
        mix = read_tensors("model.language_model.hyper_connection_mixer.")
        mix = {k.replace("model.language_model.", "mtp.", 1): v for k, v in mix.items()}
        if sorted(mix) != sorted(k for k in outer if k.startswith("mtp.hyper_connection_mixer.")):
            raise RuntimeError("target final mixer tensor set differs from the MTP mixer")
        outer.update(mix)
    out = assemble(outer, layers, spec)
    n = len(spec)
    source = "copy" if not a.spec else ",".join(spec)
    suffix = "" if n == 1 else f"-L{n}" if source == "copy" else "-" + "_".join(spec) + ("-mixT" if a.target_final_mixer else "")
    suffix += f"-swa{a.swa_window}" if a.swa_window else ""
    extra = {}
    if kept:
        extra = {"target_layers_kept_original_ids": kept, "deploy_manifest_sha256": DEPLOY_MANIFEST_SHA}
    write_dir(Path(a.out), out, base_config(stack_layers=n if n > 1 else None, stack_source=source, spec=spec,
                                            final_mixer="target" if a.target_final_mixer else "mtp",
                                            swa_window=a.swa_window), {
        "kind": "mtp320-fp8" + suffix, "source": str(FP8_SRC), "mask": str(MASK),
        "mask_sha256": MASK_SHA, "kept_original_ids": keep, "spec": spec, "stack_source": source,
        "final_mixer": "target" if a.target_final_mixer else "mtp", "swa_window": a.swa_window,
        "int3_config_sha256": INT3_CONFIG_SHA, **extra})


def densify(lay, experts, alpha, label):
    """The layer with its routed MoE replaced by the alpha-weighted dense SwiGLU of `experts` (see the docstring)."""
    if not experts or len(set(experts)) != len(experts) or not torch.isfinite(alpha).all() \
            or (alpha == 0).any():
        raise RuntimeError(f"{label}: bad fit: experts {experts} alpha {alpha}")
    root = alpha.abs().sqrt()
    up_s, down_s = root, alpha.sign() * root
    out = {k: v for k, v in lay.items() if not routed(k)}
    nb = I // BLOCK  # scale blocks per expert along the intermediate axis
    g, gs, u, us, d, ds = [], [], [], [], [], []
    for j, e in enumerate(experts):
        w = {proj: lay[f"mlp.experts.{e}.{proj}.weight"] for proj in PROJ}
        s = {proj: lay[f"mlp.experts.{e}.{proj}.weight_scale_inv"].float() for proj in PROJ}
        if w["gate_proj"].shape != (I, H) or s["gate_proj"].shape != (nb, H // BLOCK) \
                or w["down_proj"].shape != (H, I) or s["down_proj"].shape != (H // BLOCK, nb):
            raise RuntimeError(f"{label} expert {e}: unexpected FP8 block layout")
        g.append(w["gate_proj"]), gs.append(s["gate_proj"])
        u.append(w["up_proj"]), us.append(s["up_proj"] * up_s[j])
        d.append(w["down_proj"]), ds.append(s["down_proj"] * down_s[j])
    out["mlp.gate_proj.weight"] = torch.cat(g).contiguous()
    out["mlp.gate_proj.weight_scale_inv"] = torch.cat(gs).contiguous()
    out["mlp.up_proj.weight"] = torch.cat(u).contiguous()
    out["mlp.up_proj.weight_scale_inv"] = torch.cat(us).contiguous()
    out["mlp.down_proj.weight"] = torch.cat(d, dim=1).contiguous()
    out["mlp.down_proj.weight_scale_inv"] = torch.cat(ds, dim=1).contiguous()
    check_dense_function(lay, out, experts, alpha, label)
    return out


def dense(a):
    spec = parse_spec(a.spec)
    fits = a.fits.split(",")
    if len(fits) != len(spec):
        raise RuntimeError(f"{len(spec)} layers, {len(fits)} fits")
    outer, mtp_layer = split_mtp(read_mtp())
    layers, info = [], []
    for i, (item, path) in enumerate(zip(spec, fits)):
        fit = json.loads(Path(path).read_text())
        legacy = "source" not in fit and "layer" not in fit and spec == ["M"]  # 10-11 dense-run fit (MTP only)
        if not legacy and fit.get("source") != item:
            raise RuntimeError(f"{path} fits source {fit.get('source')}, layer {i} is {item}")
        if not legacy and fit.get("layer") != i and not a.fits_from_other_layers:
            raise RuntimeError(f"{path} was fitted at layer {fit.get('layer')}, used at layer {i} "
                               "(--fits-from-other-layers to allow)")
        experts = fit["experts"]
        alpha = torch.tensor(fit["alpha_" + a.alpha] if a.alpha != "uniform" else [1.0 / len(experts)] * len(experts),
                             dtype=torch.float32)
        layers.append(densify(raw_layer(item, mtp_layer, experts), experts, alpha, f"layer {i} ({item})"))
        info.append({"layer": i, "source": item, "fit_layer": fit.get("layer", 0), "fit": str(Path(path).resolve()),
                     "fit_sha256": sha256(path),
                     "experts_original_ids": experts, "alpha": alpha.tolist()})
    widths = {len(x["experts_original_ids"]) * I for x in info}
    if len(widths) != 1:
        raise RuntimeError(f"layers of one draft need one dense width, got {sorted(widths)}")
    width = widths.pop()
    out = assemble(outer, layers, spec)
    n = len(spec)
    manifest = {"kind": f"dense{width}-{a.alpha}-fp8" + ("" if n == 1 else "-" + "_".join(spec))
                + (f"-swa{a.swa_window}" if a.swa_window else ""), "source": str(FP8_SRC),
                "spec": spec, "alpha_placement": "sqrt_split on fp32 block scales", "int3_config_sha256": INT3_CONFIG_SHA}
    if a.swa_window:
        manifest["swa_window"] = a.swa_window
    if n == 1:  # the 10-11 one-layer manifest fields
        manifest.update({"fit": info[0]["fit"], "fit_sha256": info[0]["fit_sha256"],
                         "experts_original_ids": info[0]["experts_original_ids"], "alpha": info[0]["alpha"]})
    else:
        manifest["layers"] = info
    write_dir(Path(a.out), out, base_config(dense_width=width, stack_layers=n if n > 1 else None,
                                            stack_source=",".join(spec), spec=spec, swa_window=a.swa_window), manifest)


def dequant(w, s):
    return w.float() * s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)


def check_dense_function(lay, out, experts, alpha, label):
    """The dense FFN (dequantized) equals sum_e alpha_e f_e(x) of the dequantized FP8 experts on random inputs."""
    x = torch.randn(64, H, generator=torch.Generator().manual_seed(0))

    def ffn(gw, uw, dw):
        return torch.nn.functional.linear(torch.nn.functional.silu(x @ gw.T) * (x @ uw.T), dw)

    want = torch.zeros(64, H)
    for j, e in enumerate(experts):
        f = {p: dequant(lay[f"mlp.experts.{e}.{p}.weight"], lay[f"mlp.experts.{e}.{p}.weight_scale_inv"]) for p in PROJ}
        want += alpha[j] * ffn(f["gate_proj"], f["up_proj"], f["down_proj"])
    dq = {p: dequant(out[f"mlp.{p}.weight"], out[f"mlp.{p}.weight_scale_inv"]) for p in PROJ}
    got = ffn(dq["gate_proj"], dq["up_proj"], dq["down_proj"])
    swapped = ffn(dq["up_proj"], dq["gate_proj"], dq["down_proj"])
    err = ((got - want).norm() / want.norm()).item()
    err_sw = ((swapped - want).norm() / want.norm()).item()
    print(f"{label}: dense vs sum alpha_e f_e: rel err {err:.2e} (gate/up swapped {err_sw:.2e})")
    if not err < 1e-5 or not err_sw > 0.1:
        raise RuntimeError(f"{label}: dense FFN does not reproduce the fixed alpha-weighted expert sum")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("moe320")
    m.add_argument("--out", required=True)
    m.add_argument("--layers", type=int, default=1)
    m.add_argument("--spec")
    m.add_argument("--target-final-mixer", action="store_true")
    m.add_argument("--swa-window", type=int)
    d = sub.add_parser("dense")
    d.add_argument("--fits", required=True)
    d.add_argument("--spec", default="M")
    d.add_argument("--alpha", choices=["least_squares", "uniform"], required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--swa-window", type=int)
    # a source's fit reused at another position (e.g. the M_T43_T47 collection's T43 fit for every T43 of a longer
    # stack); the source must still match
    d.add_argument("--fits-from-other-layers", action="store_true")
    a = ap.parse_args()
    {"moe320": moe320, "dense": dense}[a.cmd](a)


if __name__ == "__main__":
    main()
