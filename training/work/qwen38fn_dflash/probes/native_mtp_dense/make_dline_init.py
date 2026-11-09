"""A D-line (MTP3/MTP5-DFlash, ../../model.py) drafter built from one of our dense native-MTP drafts.

    python make_dline_init.py --draft <build_drafts.py dense dir, e.g. dense10240-ls-M_T35_T39_T43_T47> \\
        --init <out: training init dir> --deploy <out: deploy-bf16 dir> [--mask-gr none]

The draft's layers become the D-line decoder layers one to one, so the drafter's block slot 0 (anchor token + the
target GR one position back, through the same input fusion and per-layer clean caches) starts as that NEXTN draft's
first draft step, up to QSA -> causal SWA 1024 and BF16 rounding:
  layers.i.mlp.routed.{gate,up,down}_proj   <- the draft's block-FP8 dense FFN (alpha already in its scales),
                                               dequantized in FP32 and rounded once to BF16 (init_mtp3.Source's math)
  layers.i.* (attention, q/k norms, hyper-connections, shared expert and gate) <- the draft's BF16 tensors
  layers.i.{attention,mlp}_conv             identity (base tap 0 = 1, projection 0), as init_mtp3.py
  fc_*, pre_fc_norm_*, hyper_connection_mixer <- the draft's mtp.* (the MTP's, or the target's with -mixT)
  embed_tokens, lm_head, target_reducer      <- the official FP8 checkpoint (target embedding / head / final mixer)
  candidate_selector                         init_mtp3.py's rules (successor 0, predecessor N(0, .02))
  mask_embedding                             embed[248070] (<|audio_start|>; init_mtp3.py uses 248077)
The draft's QSA indexer tensors are the only ones left unused. Every drafter parameter is filled exactly once.
--init is a training starting point (all BF16, with embedding and head); --deploy is what arc_dflash serves (the same
layer files hard-linked, the interface without embedding/head, config quantization with no FP8 modules), the layout of
make_dense_deploy.py --fp8 (none).
"""
import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DLINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DLINE))
from init_mtp3 import Source, config_from_source  # noqa: E402
from model import MTP3  # noqa: E402

FP8_SRC = Path(os.environ["ARC_MODELS"]) / "Qwen3.8-Flash-Next-FP8"
# arc_dflash's Config (the serving fork) has no attention_sink field; the trainer's defaults it to False.
SERVING_DROPS = ("attention_sink",)
DEPLOY_QUANT = {"method": "block_fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
                "scale": "weight_scale_inv fp32 [N/128, K/128], w = q * s", "activation_scheme": "dynamic",
                "modules": [], "others": "bf16 (unchanged)"}
SHARED_WITH_TARGET = ("embed_tokens.weight", "lm_head.weight")
MASK_TOKEN = 248070  # <|audio_start|>


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class Draft:
    """Tensors of a build_drafts.py draft dir; FP8 block weights dequantized like init_mtp3.Source."""

    def __init__(self, path):
        self.path = Path(path)
        index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        if set(index.values()) != {"mtp.safetensors"}:
            raise RuntimeError(f"{path}: expected one mtp.safetensors")
        self.f = safe_open(str(self.path / "mtp.safetensors"), "pt")
        self.keys = set(self.f.keys())
        self.used = set()

    def bf16(self, key):
        value = self.f.get_tensor(key)
        if value.dtype != torch.bfloat16:
            raise RuntimeError(f"{key}: {value.dtype}, expected a BF16 draft tensor")
        self.used.add(key)
        return value

    def dequant(self, key):
        value, scale = self.f.get_tensor(key), self.f.get_tensor(key.removesuffix(".weight") + ".weight_scale_inv")
        if value.dtype != torch.float8_e4m3fn or value.ndim != 2 or \
                list(scale.shape) != [(value.shape[0] + 127) // 128, (value.shape[1] + 127) // 128]:
            raise RuntimeError(f"{key}: not a block-FP8 weight")
        self.used |= {key, key.removesuffix(".weight") + ".weight_scale_inv"}
        return (value.float() * scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
                [: value.shape[0], : value.shape[1]]).bfloat16()


@torch.no_grad()
def build(draft, c):
    with torch.device("meta"):
        model = MTP3(c).to(dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    source = Source(FP8_SRC)
    filled = {}
    for name, param in model.named_parameters():
        if name.startswith("layers."):
            i, suffix = name.split(".", 2)[1], name.split(".", 2)[2]
            if "conv." in suffix:
                continue
            if suffix.startswith("mlp.routed."):
                value = draft.dequant(f"mtp.layers.{i}.mlp.{suffix.removeprefix('mlp.routed.')}")
            else:
                value = draft.bf16(f"mtp.layers.{i}.{suffix}")
        elif name.startswith("candidate_selector.") or name == "mask_embedding":
            continue
        elif name.startswith("target_reducer."):
            value = source.read(name.replace("target_reducer.", "model.language_model.hyper_connection_mixer."))
        elif name == "embed_tokens.weight":
            value = source.read("model.language_model.embed_tokens.weight")
        elif name == "lm_head.weight":
            value = source.read("lm_head.weight")
        else:
            value = draft.bf16("mtp." + name)
        if tuple(value.shape) != tuple(param.shape):
            raise RuntimeError(f"{name}: source {tuple(value.shape)} != drafter {tuple(param.shape)}")
        param.copy_(value)
        filled[name] = "copied"
    for li, layer in enumerate(model.layers):
        for conv_name in ("attention_conv", "mlp_conv"):
            conv = getattr(layer, conv_name)
            conv.base_kernel.zero_()
            conv.base_kernel[:, 0] = 1
            conv.kernel_projection.weight.zero_()
            filled[f"layers.{li}.{conv_name}.base_kernel"] = "identity"
            filled[f"layers.{li}.{conv_name}.kernel_projection.weight"] = "zero"
    selector = model.candidate_selector
    torch.manual_seed(20260909)  # init_mtp3.py's seed
    torch.nn.init.normal_(selector.predecessor_codebook, std=0.02)
    selector.successor_codebook.zero_()
    selector.hidden_projection.reset_parameters()
    model.mask_embedding.copy_(model.embed_tokens.weight[MASK_TOKEN])
    for name in ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook",
                 "candidate_selector.hidden_projection.weight"):
        filled[name] = "init_mtp3 rule"
    filled["mask_embedding"] = f"embed[{MASK_TOKEN}]"
    params = {n for n, _ in model.named_parameters()}
    if set(filled) != params:
        raise RuntimeError(f"unfilled {sorted(params - set(filled))[:8]}, unknown {sorted(set(filled) - params)[:8]}")
    unused = draft.keys - draft.used
    if any(".self_attn.indexer." not in k for k in unused):
        raise RuntimeError(f"draft tensors not used: {sorted(k for k in unused if '.self_attn.indexer.' not in k)[:8]}")
    return model, source.audit, sorted(unused)


def write(out, files, config, manifest):
    """files: {name: state dict or existing path to hard-link}; writes config, manifest and SHA256SUMS."""
    out.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        if isinstance(content, Path):
            os.link(content, out / name)
        else:
            save_file(content, str(out / name), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "initialization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sums = {p.name: sha256(p) for p in sorted(out.iterdir())}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    return sums


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft", required=True, type=Path)
    p.add_argument("--init", required=True, type=Path)
    p.add_argument("--deploy", required=True, type=Path)
    p.add_argument("--mask-gr", choices=("copy", "none"), default="copy",
                   help="Config.mask_gr: the mask slots' GR, the anchor's copied (copy) or none (mask embedding alone)")
    a = p.parse_args()
    torch.set_num_threads(16)
    text = json.loads((a.draft / "config.json").read_text())["text_config"]
    width = text.get("mtp_dense_intermediate_size")
    if not width or width % 640:
        raise RuntimeError(f"{a.draft}: not a dense draft (mtp_dense_intermediate_size {width})")
    if "mtpd_swa_window" in text:
        raise RuntimeError(f"{a.draft}: SWA drafts are not converted (the D-line window is its own config)")
    n = int(text.get("mtpd_stack_layers") or 1)
    build_manifest = json.loads((a.draft / "build_manifest.json").read_text())
    c = config_from_source(FP8_SRC, n)
    from dataclasses import replace
    c = replace(c, ffn_type="dense_shared", dense_intermediate_size=width, mask_gr=a.mask_gr)
    draft = Draft(a.draft)
    model, audit, unused = build(draft, c)
    state = model.state_dict()
    layer_states = {f"layer-{i}.safetensors": {k: v for k, v in state.items() if k.startswith(f"layers.{i}.")}
                    for i in range(n)}
    interface = {k: v for k, v in state.items() if not k.startswith("layers.")}
    config = {k: v for k, v in asdict(c).items() if k not in SERVING_DROPS}
    if c.mask_gr == "copy":  # the default: left out, so serving trees from before mask_gr load the export
        del config["mask_gr"]
    if any(asdict(c)[k] for k in SERVING_DROPS):
        raise RuntimeError(f"{SERVING_DROPS} must stay off for the serving fork")
    files = list(layer_states) + ["interface.safetensors"]
    common = {"draft": str(a.draft.resolve()), "draft_kind": build_manifest["kind"],
              "draft_spec": build_manifest.get("spec", ["M"]),
              "draft_files_sha256": build_manifest["files_sha256"],
              "attention": "causal SWA: 0 <= query_position - key_position < 1024; no QSA indexer",
              "slot0_reference": "the draft's NEXTN first draft step, up to QSA -> SWA 1024 and BF16 rounding",
              "mask_token_embedding_source_id": MASK_TOKEN, "mask_gr": c.mask_gr, "extra_final_norm": False,
              "production_gates_passed": False}
    init_sums = write(a.init, {**layer_states, "interface.safetensors": interface}, config, {
        "complete": True, "kind": "mtp-target-dense-init", "files": files, **common,
        "source": str(FP8_SRC), "source_tensors": audit, "draft_tensors_unused": unused,
        "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "frozen_parameters": sum(x.numel() for x in model.parameters() if not x.requires_grad)})
    deploy_files = {name: a.init / name for name in layer_states}
    deploy_files["interface.safetensors"] = {k: v for k, v in interface.items() if k not in SHARED_WITH_TARGET}
    deploy_sums = write(a.deploy, deploy_files, {**config, "quantization": DEPLOY_QUANT}, {
        "complete": True, "files": files, "kind": "dense-routed-deploy", "dtype": "bfloat16",
        "shared_with_target": list(SHARED_WITH_TARGET)})
    (a.deploy / "conversion_manifest.json").write_text(json.dumps({
        "complete": True, "kind": "dense-routed-deploy", "source": str(a.init.resolve()),
        "source_kind": "mtp-target-dense-init", "source_sha256": init_sums, "fp8_groups": [],
        "fp8_weight_rel_err": {}, "dropped": list(SHARED_WITH_TARGET)}, indent=1) + "\n")
    print(json.dumps({"layers": n, "width": width, "mask_gr": c.mask_gr, "spec": common["draft_spec"], "init": str(a.init), "deploy": str(a.deploy),
                      "draft_tensors_unused": len(unused), "deploy_files": sorted(deploy_sums)}, indent=1))


if __name__ == "__main__":
    main()
