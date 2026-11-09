"""Convert the trained MoE drafter export into a dense-routed one, plus FP8 / INT4 copies of the dense FFN.

Per layer, the routed MoE (router + 512 experts, top-10) becomes one SwiGLU of width M*640 built from M fixed
experts: gate/up rows and down columns concatenated, alpha (default 1/M) split between up rows and down columns
(split_alpha). That is exactly the fixed alpha-weighted sum of the chosen experts, not the input-dependent top-10 MoE
-- an initialization for recovery training and a deployment-experiment checkpoint, not a drop-in quality
equivalent. The shared expert, its gate and every other tensor are copied unchanged.

    python work/qwen38fn_dflash/convert_dense.py --source <trained bf16 export> \
        --router-stats <router_*.rank0.json> <router_*.rank2.json> --out-prefix <dir>/<name>-dense6400

writes <prefix>-bf16, <prefix>-fp8 (routed dense in block FP8: e4m3, one fp32 dequant scale per 128x128, the
layout of patch 0032 / sglang Fp8LinearMethod `weight_scale_inv`) and <prefix>-int4 (routed dense in GPTQ v1
format: sym, 4 bit, group 128, desc_act false, round-to-nearest without calibration).

Dense keys: layers.i.mlp.routed.{gate,up,down}_proj.weight ([M*I, H], [M*I, H], [H, M*I]).
"""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Config, Experts  # noqa: E402

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_BLOCK = 128
INT4_GROUP = 128
MOE_KEYS = ("mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj")
MOE_CONFIG = ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "norm_topk_prob")
PROJ = ("gate_proj", "up_proj", "down_proj")
# Where alpha goes, recorded in every manifest. Manifests without the field (before 2026-10-07) put the whole alpha_e
# on expert e's down columns.
ALPHA_PLACEMENT = "sqrt_split"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source):
    sums = dict(reversed(line.split("  ")) for line in (source / "SHA256SUMS").read_text().splitlines())
    for name, digest in sorted(sums.items()):
        if sha256(source / name) != digest:
            raise RuntimeError(f"source checksum mismatch: {name}")
    manifest = json.loads((source / "initialization_manifest.json").read_text())
    if manifest["kind"] != "trained-bf16-export" or not manifest["complete"]:
        raise RuntimeError(f"source is not a complete trained BF16 export: {manifest['kind']}")
    raw = json.loads((source / "config.json").read_text())
    c = Config(**raw)
    if c.num_layers != 5 or c.num_experts != 512 or c.moe_intermediate_size != 640 or not c.norm_topk_prob:
        raise RuntimeError(f"unexpected source config: {raw}")
    return raw, c, manifest, sums


def select_experts(stats_paths, c, m):
    """Top-m experts per layer by router mass summed over the given ranks (one per DP group)."""
    mass = None
    records = []
    for path in stats_paths:
        d = json.loads(path.read_text())
        layer_mass = [torch.tensor(d["router_mass"][str(i)], dtype=torch.float64) for i in range(c.num_layers)]
        if any(x.shape != (c.num_experts,) for x in layer_mass):
            raise RuntimeError(f"{path}: router_mass is not [{c.num_experts}] per layer")
        mass = layer_mass if mass is None else [a + b for a, b in zip(mass, layer_mass)]
        records.append({"path": str(path.resolve()), "sha256": sha256(path), "cycle": d["cycle"]})
    keep, kept_share = {}, {}
    for i, x in enumerate(mass):
        order = x.argsort(descending=True, stable=True)
        if x[order[m - 1]] <= 0:
            raise RuntimeError(f"layer {i}: fewer than {m} experts with router mass")
        keep[i] = order[:m].tolist()
        kept_share[i] = float(x[order[:m]].sum() / x.sum())
    return keep, kept_share, records


def split_alpha(alpha):
    """(up, down) per-expert scales with up * down = alpha: sqrt|alpha_e| on expert e's up rows, sign(alpha_e)
    sqrt|alpha_e| on its down columns. silu(gate x) * up x is linear in up, so this is the same FFN as alpha on down
    alone, but down is not left 1/alpha times smaller than every other matrix (Adam's per-element step would then move
    it that much faster relative to its own scale). A zero alpha would zero both the rows and the columns, a unit that
    never gets a gradient again: such an expert does not belong in the selection, so it is an error."""
    if alpha.dtype != torch.float32 or not torch.isfinite(alpha).all() or (alpha == 0).any():
        raise ValueError(f"alpha must be finite, nonzero FP32: {alpha}")
    root = alpha.abs().sqrt()
    return root, alpha.sign() * root


def densify(gate_up, down, keep, alpha, c):
    i_size = c.moe_intermediate_size
    idx = torch.tensor(keep)
    up_scale, down_scale = split_alpha(alpha)
    chosen = gate_up.index_select(0, idx)  # [M, 2I, H], each expert [gate; up]
    g = chosen[:, :i_size, :].reshape(len(keep) * i_size, c.hidden_size)
    u = (chosen[:, i_size:, :].float() * up_scale[:, None, None]).reshape(len(keep) * i_size, c.hidden_size)
    d = (down.index_select(0, idx).float() * down_scale[:, None, None]).permute(1, 0, 2).reshape(
        c.hidden_size, len(keep) * i_size
    )
    return g.contiguous(), u.to(torch.bfloat16).contiguous(), d.to(torch.bfloat16).contiguous()


def check_dense(gate_up, down, keep, alpha, g, u, d, c, layer):
    """Exact layout checks, then the dense FFN against the repo's own Experts on the chosen experts."""
    i_size = c.moe_intermediate_size
    up_scale, down_scale = split_alpha(alpha)
    for j, e in enumerate(keep):
        rows = slice(j * i_size, (j + 1) * i_size)
        if not torch.equal(g[rows], gate_up[e, :i_size]):
            raise RuntimeError(f"layer {layer}: gate rows of expert {e} are not an exact copy")
        if not torch.equal(u[rows], (gate_up[e, i_size:].float() * up_scale[j]).to(torch.bfloat16)):
            raise RuntimeError(f"layer {layer}: up rows of expert {e} are not sqrt|alpha| * the expert")
        if not torch.equal(d[:, rows], (down[e].float() * down_scale[j]).to(torch.bfloat16)):
            raise RuntimeError(f"layer {layer}: down columns of expert {e} are not sign(alpha) sqrt|alpha| * the expert")
    sub = replace(c, num_experts=len(keep), num_experts_per_tok=len(keep))
    ref = Experts(sub)
    with torch.no_grad():
        ref.gate_up_proj.copy_(gate_up.index_select(0, torch.tensor(keep)).float())
        ref.down_proj.copy_(down.index_select(0, torch.tensor(keep)).float())
        gen = torch.Generator().manual_seed(layer)
        x = torch.randn(256, c.hidden_size, generator=gen)
        indices = torch.arange(len(keep)).expand(x.shape[0], -1)
        weights = alpha.float().expand(x.shape[0], -1)
        want = ref(x, indices, weights)
        got = dense_ffn(x, g.float(), u.float(), d.float())
        swapped = dense_ffn(x, u.float(), g.float(), d.float())
    err, err_swapped = rel(got, want), rel(swapped, want)
    # BF16 rounding of the scaled up rows and down columns is the only difference; a gate/up swap must stand far
    # outside it.
    if not err < 5e-3 or not err_swapped > 0.1:
        raise RuntimeError(f"layer {layer}: dense vs Experts rel err {err:.3e}, gate/up swapped {err_swapped:.3e}")
    return err, err_swapped


def dense_ffn(x, g, u, d):
    return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def quantize_fp8(w):
    """[N, K] bf16 -> ([N, K] e4m3, [N/128, K/128] fp32 dequant scale); same arithmetic as patch 0032."""
    n, k = w.shape
    if n % FP8_BLOCK or k % FP8_BLOCK:
        raise ValueError(f"block FP8 needs dims divisible by {FP8_BLOCK}: {tuple(w.shape)}")
    blocks = w.float().view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))
    s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (blocks / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).view(n, k).to(torch.float8_e4m3fn)
    return q, s


def dequant_fp8(q, s):
    n, k = q.shape
    return (q.float().view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK) * s[:, None, :, None]).view(n, k)


def quantize_int4_gptq(w):
    """[N, K] bf16 -> GPTQ v1 tensors (sym, 4 bit, group 128, desc_act false), round to nearest.

    GPTQ's symmetric quantizer: scale = 2 * amax / 15 per (group, output), zero = 8, q = clamp(round(w/s) + 8, 0, 15).
    qweight [K/8, N] int32 packs 8 consecutive inputs (input 8r+j at bits 4j); qzeros [K/128, N/8] int32 packs 8
    outputs and stores zero - 1 (= 7, the v1 convention); scales [K/128, N] fp16; g_idx [K] = k // 128.
    """
    n, k = w.shape
    if k % INT4_GROUP or n % 8:
        raise ValueError(f"GPTQ int4 needs K % {INT4_GROUP} == 0 and N % 8 == 0: {tuple(w.shape)}")
    groups = w.float().view(n, k // INT4_GROUP, INT4_GROUP)
    amax = groups.abs().amax(-1)
    scale = torch.where(amax > 0, 2 * amax / 15, torch.ones_like(amax)).half()  # [N, G], stored fp16
    q = (groups / scale.float()[..., None]).round().add(8).clamp(0, 15).to(torch.int64).view(n, k)
    qt = q.t().contiguous().view(k // 8, 8, n)  # [K/8, 8, N]
    packed = torch.zeros(k // 8, n, dtype=torch.int64)
    for j in range(8):
        packed |= qt[:, j, :] << (4 * j)
    zeros = torch.full((k // INT4_GROUP, n // 8, 8), 7, dtype=torch.int64)
    qzeros = torch.zeros(k // INT4_GROUP, n // 8, dtype=torch.int64)
    for j in range(8):
        qzeros |= zeros[:, :, j] << (4 * j)
    return {
        "qweight": to_int32(packed),
        "qzeros": to_int32(qzeros),
        "scales": scale.t().contiguous(),
        "g_idx": (torch.arange(k, dtype=torch.int32) // INT4_GROUP),
    }, q


def to_int32(x):
    """uint32 bit patterns held in int64 -> the same bits as int32."""
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def dequant_int4_gptq(t):
    qweight, qzeros, scales, g_idx = t["qweight"], t["qzeros"], t["scales"], t["g_idx"]
    kp, n = qweight.shape
    w = qweight.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(w >> (4 * j)) & 0xF for j in range(8)], 1).view(kp * 8, n)  # [K, N]
    z = qzeros.to(torch.int64) & 0xFFFFFFFF
    z = torch.stack([(z >> (4 * j)) & 0xF for j in range(8)], -1).view(z.shape[0], n) + 1  # v1: stored zero - 1
    return ((q - z[g_idx.long()]).float() * scales.float()[g_idx.long()]).t().contiguous(), q.t()


def publish(partial, out, record):
    (partial / "conversion_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    names = sorted(p.name for p in partial.iterdir())
    (partial / "SHA256SUMS").write_text("".join(f"{sha256(partial / n)}  {n}\n" for n in names))
    partial.rename(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--router-stats", type=Path, nargs="+", required=True,
                   help="router stats JSON, one rank per DP group (ranks of one pair hold identical mass)")
    p.add_argument("--experts", type=int, default=10, help="experts concatenated per layer; width = experts * 640")
    p.add_argument("--out-prefix", type=Path, required=True)
    args = p.parse_args()

    raw, c, source_manifest, source_sums = verify_source(args.source)
    m = args.experts
    width = m * c.moe_intermediate_size
    outs = {v: args.out_prefix.with_name(f"{args.out_prefix.name}-{v}") for v in ("bf16", "fp8", "int4")}
    for out in outs.values():
        if out.exists() or out.with_name(out.name + ".partial").exists():
            raise FileExistsError(out)
    keep, kept_share, stats_records = select_experts(args.router_stats, c, m)
    alpha = torch.full((m,), 1.0 / m, dtype=torch.float32)

    dense_raw = {k: v for k, v in raw.items() if k not in MOE_CONFIG}
    dense_raw |= {"ffn_type": "dense_shared", "dense_intermediate_size": width}
    quant_common = {"modules": [f"layers.*.mlp.routed.{p}" for p in PROJ], "others": "bf16 (unchanged)"}
    configs = {
        "bf16": dense_raw,
        "fp8": dense_raw | {"quantization": quant_common | {
            "method": "block_fp8", "fmt": "e4m3", "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
            "scale": "weight_scale_inv fp32 [N/128, K/128], w = q * s", "activation_scheme": "dynamic"}},
        "int4": dense_raw | {"quantization": quant_common | {
            "method": "gptq", "checkpoint_format": "gptq", "bits": 4, "group_size": INT4_GROUP, "sym": True,
            "desc_act": False, "calibration": "none (round to nearest)"}},
    }
    partials = {}
    for v, out in outs.items():
        partials[v] = out.with_name(out.name + ".partial")
        partials[v].mkdir(parents=True)
        (partials[v] / "config.json").write_text(json.dumps(configs[v], indent=2) + "\n")
        shutil.copyfile(args.source / "interface.safetensors", partials[v] / "interface.safetensors")

    report = {}
    for layer in range(c.num_layers):
        name = f"layer-{layer}.safetensors"
        state = load_file(str(args.source / name))
        pre = f"layers.{layer}."
        gate_up, down = state[pre + "mlp.experts.gate_up_proj"], state[pre + "mlp.experts.down_proj"]
        if gate_up.shape != (c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size) or gate_up.dtype != torch.bfloat16:
            raise RuntimeError(f"layer {layer}: gate_up_proj {tuple(gate_up.shape)} {gate_up.dtype}")
        if down.shape != (c.num_experts, c.hidden_size, c.moe_intermediate_size) or down.dtype != torch.bfloat16:
            raise RuntimeError(f"layer {layer}: down_proj {tuple(down.shape)} {down.dtype}")
        g, u, d = densify(gate_up, down, keep[layer], alpha, c)
        err, err_swapped = check_dense(gate_up, down, keep[layer], alpha, g, u, d, c, layer)
        kept = {k: v for k, v in state.items() if not any(k == pre + s for s in MOE_KEYS)}
        if len(state) - len(kept) != len(MOE_KEYS):
            raise RuntimeError(f"layer {layer}: expected {len(MOE_KEYS)} MoE tensors, removed {len(state) - len(kept)}")
        dense = dict(zip(PROJ, (g, u, d)))
        save_file(kept | {f"{pre}mlp.routed.{p}.weight": w for p, w in dense.items()}, str(partials["bf16"] / name))

        fp8, int4, layer_report = {}, {}, {"router_mass_kept": kept_share[layer], "experts": keep[layer],
                                          "dense_vs_experts_rel_err": err, "gate_up_swapped_rel_err": err_swapped}
        with torch.no_grad():
            x = torch.randn(256, c.hidden_size, generator=torch.Generator().manual_seed(100 + layer))
            want = dense_ffn(x, g.float(), u.float(), d.float())
            deq8, deq4 = {}, {}
            for p, w in dense.items():
                q, s = quantize_fp8(w)
                fp8[f"{pre}mlp.routed.{p}.weight"], fp8[f"{pre}mlp.routed.{p}.weight_scale_inv"] = q, s
                deq8[p] = dequant_fp8(q, s)
                t, q4 = quantize_int4_gptq(w)
                back, q4_back = dequant_int4_gptq(t)
                if not torch.equal(q4_back, q4):
                    raise RuntimeError(f"layer {layer} {p}: int4 pack/unpack does not round-trip")
                int4.update({f"{pre}mlp.routed.{p}.{k}": v for k, v in t.items()})
                deq4[p] = back
                layer_report[f"{p}_weight_rel_err"] = {"fp8": rel(deq8[p], w.float()), "int4": rel(back, w.float())}
            layer_report["ffn_output_rel_err"] = {
                "fp8": rel(dense_ffn(x, *(deq8[p] for p in PROJ)), want),
                "int4": rel(dense_ffn(x, *(deq4[p] for p in PROJ)), want),
            }
        save_file(kept | fp8, str(partials["fp8"] / name))
        save_file(kept | int4, str(partials["int4"] / name))
        report[layer] = layer_report
        print(f"layer {layer}: mass kept {kept_share[layer]:.4f}, dense vs Experts {err:.2e} (swap {err_swapped:.2f}), "
              f"FFN out rel err fp8 {layer_report['ffn_output_rel_err']['fp8']:.4f} "
              f"int4 {layer_report['ffn_output_rel_err']['int4']:.4f}", flush=True)
        del state, gate_up, down

    # Reload what was written: every untouched tensor bit-identical to the source, every new key present.
    for layer in range(c.num_layers):
        name = f"layer-{layer}.safetensors"
        with safe_open(str(args.source / name), "pt") as src:
            for v, part in partials.items():
                with safe_open(str(part / name), "pt") as f:
                    for k in src.keys():
                        if any(k == f"layers.{layer}.{s}" for s in MOE_KEYS):
                            if k in f.keys():
                                raise RuntimeError(f"{v} {name}: MoE tensor {k} survived")
                        elif not torch.equal(f.get_tensor(k), src.get_tensor(k)):
                            raise RuntimeError(f"{v} {name}: {k} differs from the source")
    for part in partials.values():
        if sha256(part / "interface.safetensors") != source_sums["interface.safetensors"]:
            raise RuntimeError("interface copy differs from the source")

    for v, out in outs.items():
        publish(partials[v], out, {
            "complete": True,
            "kind": f"dense-routed-{v}",
            "source": str(args.source.resolve()),
            "source_step": source_manifest["step"],
            "source_sha256": source_sums,
            "router_stats": stats_records,
            "selection": f"per layer, top {m} experts by router mass summed over the router-stats files",
            "alpha": f"uniform 1/{m}",
            "alpha_placement": ALPHA_PLACEMENT,
            "dense_intermediate_size": width,
            "layers": report,
        })
        print("wrote", out)


if __name__ == "__main__":
    main()
