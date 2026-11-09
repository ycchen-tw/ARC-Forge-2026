"""An HC drafter (Config.residual "hc", ffn_type "dense_shared": the D line's exports) as the plain-residual drafter
(residual "plain", ffn_type "dense", context "inject" or "recompute"): the init of plan B
(docs/plans/dflash_plain_inject_drafter_20261018.md).

    python convert_plain.py --source <trained-bf16-export> --features <job dir> [<job dir> ...] --out <dir> \
        [--context inject|recompute] [--max-tokens 24576] [--anchors-per-job 256]

Weights, by model.py name:

- bit for bit: embed_tokens, lm_head, target_reducer, mask_embedding, pre_fc_norm_*, fc_embedding, candidate_selector
  and every layer's self_attn (but o_proj), attention_conv and mlp_conv.
- fc_hidden [H, hc * H] = [W, W, W, W] / hc for the source's W [H, H]: each HC stream starts as W gr_b plus the shared
  embedding term, so the plain stream starts as their mean, exactly.
- mlp.{gate,up}_proj = [routed; shared] rows, mlp.down_proj = [routed, s * shared] columns, s = E[sigmoid(shared
  gate)] (the gate itself is dropped).
- o_proj and mlp.down_proj times the layer's E[mean over branches of the HC residual gate] (2 sigmoid(...) / hc of the
  attention's and the MLP's hyper-connection): the mean stream's update.
- the RMSNorm gains, fitted one sublayer at a time on calibration rows, so each fit sees the converted model's own
  upstream. The targets come from the source run through the reference path (train.clean_trace ->
  CleanCache.before -> MTP3.propose) under forward hooks:
  input_layernorm / post_attention_layernorm: per channel, least squares of the plain stream's normalized state onto
  the HC mixer output it replaces (raw_a / raw_c), on the rows the trunk runs -- inject: the proposal blocks;
  recompute: the clean rows, whose K/V and conv boundaries the blocks read;
  norm: the same onto the HC hyper_connection_mixer output of the proposal blocks;
  context_layernorm (inject): least squares in the layer's [k_proj; v_proj] output space of the injected rows onto the
  HC clean pass's (k before k_norm; each part scaled by its RMS so both count alike).
s and the gate means come from the same rows as the fits. Everything runs in FP32 on one device.

The conversion is exact when the HC model is a single stream in disguise (identical branches, constant mixes, unit
gates, a constant shared gate): test_convert_plain.py checks it, and that injection's layer-0 gain is then recovered.

Output: init_mtp3's layout (layer-<i>.safetensors, interface.safetensors, config.json, initialization_manifest.json
kind "plain-conversion", SHA256SUMS) plus calibration.json (every fit's relative residual, s and the gate statistics).
"""

import argparse
import hashlib
import json
import os
import re
import time
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from data import eligible_anchors
from init_mtp3 import load_checkpoint
from model import MTP3, CleanCache, LayerCache
from safetensors.torch import load_file, save_file
from train import clean_trace

# Source parameters the plain drafter has no place for (their effect is folded into the derived weights or fitted).
DROPPED = re.compile(r"^(hyper_connection_mixer\..*|layers\.\d+\.(attn_hyper_connection|mlp_hyper_connection)\..*"
                     r"|layers\.\d+\.mlp\.(routed|shared_expert|shared_expert_gate)\..*)$")
RIDGE = 1e-6


def plain_config(c, context):
    if c.residual != "hc" or c.ffn_type != "dense_shared" or c.attention_sink:
        raise ValueError("the conversion takes an HC dense_shared drafter without attention sinks")
    return replace(c, residual="plain", context=context, ffn_type="dense",
                   dense_intermediate_size=c.dense_intermediate_size + c.shared_expert_intermediate_size)


def rms_unit(x, eps):
    """RMSNorm without its gain, in FP32."""
    z = x.float()
    return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)


@dataclass
class Folds:
    """Per layer: s = E[sigmoid(shared gate)] and the mean HC residual gates of the attention and the MLP."""

    shared: list
    attention_gate: list
    mlp_gate: list


@torch.no_grad()
def derive(source, c, folds):
    """The plain drafter (config c) from the HC source; every RMSNorm gain of the new residual is 0 until fitted."""
    device, dtype = source.fc_hidden.weight.device, source.fc_hidden.weight.dtype
    with torch.device("meta"):
        model = MTP3(c)
    model.to_empty(device=device)
    model.to(dtype)
    src = dict(source.named_parameters())
    used = set()

    def take(name):
        used.add(name)
        return src[name]

    for name, p in model.named_parameters():
        m = re.match(r"^layers\.(\d+)\.(.*)$", name)
        i, rest = (int(m.group(1)), m.group(2)) if m else (None, name)
        if name == "fc_hidden.weight":
            value = take(name).repeat(1, c.hc_count) / c.hc_count
        elif rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            part = rest.split(".")[1]
            value = torch.cat((take(f"layers.{i}.mlp.routed.{part}.weight"),
                               take(f"layers.{i}.mlp.shared_expert.{part}.weight")), 0)
        elif rest == "mlp.down_proj.weight":
            value = torch.cat((take(f"layers.{i}.mlp.routed.down_proj.weight"),
                               take(f"layers.{i}.mlp.shared_expert.down_proj.weight") * folds.shared[i]), 1)
            value = value * folds.mlp_gate[i]
        elif rest == "self_attn.o_proj.weight":
            value = take(name) * folds.attention_gate[i]
        elif rest in ("input_layernorm.weight", "post_attention_layernorm.weight", "context_layernorm.weight") \
                or name == "norm.weight":
            value = torch.zeros_like(p)
        else:
            value = take(name)
        if value.shape != p.shape:
            raise ValueError(f"{name}: derived {tuple(value.shape)}, model {tuple(p.shape)}")
        p.copy_(value)
    unused = {n for n in src if n not in used and not DROPPED.match(n)}
    if unused:
        raise ValueError(f"source parameters with no place in the plain drafter: {sorted(unused)}")
    return model


class Recorder:
    """Forward hooks on the HC source. Rows land in self.rows[phase][key], one entry per call, in call order: per layer
    the mixers' outputs (attn<i> / mlp<i>: raw, gate), the shared gate's logits (shared<i>), k_proj / v_proj outputs
    (k<i> / v<i>), and the output mixer's (out)."""

    def __init__(self, model, keep):
        self.phase, self.keep, self.rows, self.handles = None, keep, {}, []
        for i, layer in enumerate(model.layers):
            for key, module in ((f"attn{i}", layer.attn_hyper_connection), (f"mlp{i}", layer.mlp_hyper_connection),
                                (f"shared{i}", layer.mlp.shared_expert_gate), (f"k{i}", layer.self_attn.k_proj),
                                (f"v{i}", layer.self_attn.v_proj)):
                self.handles.append(module.register_forward_hook(self._hook(key)))
        self.handles.append(model.hyper_connection_mixer.register_forward_hook(self._hook("out")))

    def _hook(self, key):
        def hook(_module, _inputs, output):
            if self.phase is None:
                raise RuntimeError("the HC source ran outside a recorded phase")
            if key.rstrip("0123456789") in self.keep[self.phase]:
                self.rows.setdefault(self.phase, {}).setdefault(key, []).append(output)
        return hook

    def take(self, phase, key, flat_dims):
        """The phase's rows of key concatenated in call order, flattened to [rows, ...] (flat_dims leading axes)."""
        parts = self.rows[phase].pop(key)
        if isinstance(parts[0], tuple):
            return tuple(torch.cat([p[j].flatten(0, flat_dims - 1) for p in parts]) for j in range(len(parts[0])))
        return torch.cat([p.flatten(0, flat_dims - 1) for p in parts])

    def close(self):
        for h in self.handles:
            h.remove()


@dataclass
class Sequence:
    ids: torch.Tensor     # [T]
    gr: torch.Tensor      # [T, hc, H]
    anchors: torch.Tensor  # sorted, unique, in [1, T)
    name: str


def load_sequences(paths, max_tokens, anchors_per_job, device):
    sequences = []
    for path in paths:
        f = load_file(str(Path(path) / "features.safetensors"))
        ids, gr, mask = (f[k][:max_tokens].to(device) for k in ("input_ids", "target_gr", "loss_mask"))
        if not torch.equal(f["positions"][:max_tokens], torch.arange(len(ids))):
            raise ValueError(f"{path}: capture positions are not the index")
        anchors = eligible_anchors(mask)
        if len(anchors) < anchors_per_job:
            raise ValueError(f"{path}: {len(anchors)} eligible anchors in the first {max_tokens} tokens")
        g = torch.Generator().manual_seed(zlib.crc32(Path(path).name.encode()))
        anchors = anchors[torch.randperm(len(anchors), generator=g)[:anchors_per_job].to(device)].sort().values
        sequences.append(Sequence(ids, gr.float(), anchors, Path(path).name))
    return sequences


def block_inputs(model, seq):
    """MTP3.propose's slot embeddings, slot GR and positions for seq's anchors."""
    b = model.config.block_size
    a = seq.anchors
    embeds = torch.cat((model.embed_tokens(seq.ids[a])[:, None].float(),
                        model.mask_embedding.float().expand(len(a), b - 1, -1)), 1)
    grs = model.proposal_gr(seq.gr[a - 1][:, None].expand(-1, b, -1, -1),
                            (torch.arange(b, device=a.device) == 0)[None])
    return embeds, grs, a[:, None] + torch.arange(b, device=a.device)


@torch.no_grad()
def record_source(source, sequences, context):
    """The HC source's clean pass and proposals over every sequence, with the rows each fit needs."""
    keep = {"clean": {"k", "v"} if context == "inject" else {"attn", "mlp", "shared"},
            "block": {"attn", "mlp", "shared", "out"} if context == "inject" else {"out"}}
    recorder = Recorder(source, keep)
    n = source.config.num_layers
    per_sequence = []
    for seq in sequences:
        recorder.phase = "clean"
        trace = clean_trace(source, seq.ids, seq.gr, chunk_size=min(1024, source.config.window))
        recorder.phase = "block"
        embeds, grs, positions = block_inputs(source, seq)
        source(embeds, grs, positions, trace.before(seq.anchors, source.config.window))
        recorder.phase = None
        del trace
        rows = {phase: {} for phase in keep}
        for phase in keep:
            dims = 2  # [1, chunk, ...] clean, [anchors, b, ...] block
            for key in sorted(recorder.rows[phase]):
                rows[phase][key] = recorder.take(phase, key, dims)
        per_sequence.append(rows)
    recorder.close()
    # concatenate over sequences
    merged = {phase: {key: (tuple(torch.cat([r[phase][key][j] for r in per_sequence])
                                  for j in range(len(per_sequence[0][phase][key])))
                            if isinstance(per_sequence[0][phase][key], tuple)
                            else torch.cat([r[phase][key] for r in per_sequence]))
                      for key in per_sequence[0][phase]}
              for phase in keep}
    trunk = "block" if context == "inject" else "clean"
    folds = Folds([], [], [])
    stats = []
    for i in range(n):
        _, attention_gate = merged[trunk][f"attn{i}"]
        _, mlp_gate = merged[trunk][f"mlp{i}"]
        sigma = torch.sigmoid(merged[trunk][f"shared{i}"].float())
        folds.shared.append(sigma.mean().item())
        folds.attention_gate.append(attention_gate.float().mean().item())
        folds.mlp_gate.append(mlp_gate.float().mean().item())
        stats.append({
            "layer": i,
            "shared_sigmoid_mean": folds.shared[-1], "shared_sigmoid_std": sigma.std().item(),
            "attention_gate_mean": folds.attention_gate[-1],
            "attention_gate_std_over_rows": attention_gate.float().mean(-1).std().item(),
            "attention_gate_std_over_branches": attention_gate.float().std(-1).mean().item(),
            "mlp_gate_mean": folds.mlp_gate[-1],
            "mlp_gate_std_over_rows": mlp_gate.float().mean(-1).std().item(),
            "mlp_gate_std_over_branches": mlp_gate.float().std(-1).mean().item(),
        })
    return merged, folds, stats


def channel_gain(n, target):
    """Per channel argmin_g ||g n - target||^2 over the rows, and the fit's relative residual (and that of g = 1)."""
    n, target = n.reshape(-1, n.shape[-1]).double(), target.reshape(-1, target.shape[-1]).double()
    gain = (n * target).sum(0) / (n.square().sum(0) + RIDGE * n.square().sum(0).mean())
    norm = target.norm()
    return gain.float(), {"relative_residual": ((gain * n - target).norm() / norm).item(),
                          "relative_residual_unit_gain": ((n - target).norm() / norm).item()}


def context_gain(nf, keys, values, k_proj, v_proj):
    """argmin_g of ||k_proj(g nf) - keys||^2 / s_k^2 + ||v_proj(g nf) - values||^2 / s_v^2 over the rows: a 2-D gain
    is linear in g, so the normal equations are ((W^T W) * (nf^T nf)) g = sum_rows nf * (y W)."""
    nf = nf.double()
    system = torch.zeros(nf.shape[1], nf.shape[1], dtype=torch.float64, device=nf.device)
    rhs = torch.zeros(nf.shape[1], dtype=torch.float64, device=nf.device)
    gram = nf.T @ nf
    parts = []
    for y, w in ((keys, k_proj), (values, v_proj)):
        y, w = y.double(), w.double()
        s2 = y.square().mean()
        system += (w.T @ w) * gram / s2
        rhs += (nf * (y @ w)).sum(0) / s2
        parts.append((y, w))
    system += RIDGE * system.diagonal().mean() * torch.eye(len(rhs), dtype=system.dtype, device=system.device)
    gain = torch.linalg.solve(system, rhs)
    report = {}
    for label, (y, w) in zip(("k", "v"), parts, strict=True):
        report[f"{label}_relative_residual"] = (((gain * nf) @ w.T - y).norm() / y.norm()).item()
        report[f"{label}_relative_residual_unit_gain"] = ((nf @ w.T - y).norm() / y.norm()).item()
    return gain.float(), report


class Capture:
    """A module's inputs over the calls while open (the stream entering post_attention_layernorm), concatenated on
    `dim` (the calls' chunk axis: the sequence for a clean pass, the anchors for proposal blocks)."""

    def __init__(self, module):
        self.values = []
        self.handle = module.register_forward_pre_hook(lambda _module, inputs: self.values.append(inputs[0]))

    def close(self, dim):
        self.handle.remove()
        return torch.cat(self.values, dim)


def block_layer(layer, h, positions, prefix, i, anchors_per_call=128):
    """Layer i over proposal blocks h [anchors, b, H] with their stacked clean prefix, a few anchors at a time (SDPA
    under a mask expands GQA in FP32: [anchors, heads, window, head_dim] per call)."""
    out = []
    for start in range(0, len(h), anchors_per_call):
        rows = slice(start, start + anchors_per_call)
        layer_prefix = prefix.layers[i]
        part = LayerCache(layer_prefix.key[rows], layer_prefix.value[rows], tuple(b[rows] for b in layer_prefix.boundaries))
        out.append(layer(h[rows], positions[rows], part, prefix.positions[rows])[0])
    return torch.cat(out)


def layer_pass(layer, h, positions, window):
    """One PlainDecoder over a clean sequence h [1, T, H] in chunks with its sliding K/V window and conv boundary, as
    train.clean_trace runs it layer by layer: the outputs and the layer's full clean cache."""
    chunk = min(1024, window)
    prefix = prefix_positions = None
    outs, keys, values, bounds = [], [], [], []
    for start in range(0, h.shape[1], chunk):
        x, p = h[:, start : start + chunk], positions[:, start : start + chunk]
        y, cache = layer(x, p, prefix, prefix_positions)
        outs.append(y)
        keys.append(cache.key)
        values.append(cache.value)
        bounds.append(cache.boundaries)
        k = torch.cat((prefix.key, cache.key), 2) if prefix is not None else cache.key
        v = torch.cat((prefix.value, cache.value), 2) if prefix is not None else cache.value
        pp = torch.cat((prefix_positions, p), 1) if prefix is not None else p
        prefix = LayerCache(k[:, :, -window:], v[:, :, -window:], tuple(b[:, -1:] for b in cache.boundaries))
        prefix_positions = pp[:, -window:]
    full = LayerCache(torch.cat(keys, 2), torch.cat(values, 2), tuple(torch.cat(b, 1) for b in zip(*bounds)))
    return torch.cat(outs, 1), full


@torch.no_grad()
def fit(model, sequences, merged):
    """Fit every RMSNorm gain of the plain drafter in place, upstream first; returns the fits' reports."""
    c = model.config
    eps = c.rms_norm_eps
    report = {"layers": []}
    inject = c.context == "inject"
    clean_inputs = [(model.embed_tokens(s.ids[1:])[None].float(), s.gr[:-1][None],
                     torch.arange(1, len(s.ids), device=s.ids.device)[None]) for s in sequences]
    blocks = [block_inputs(model, s) for s in sequences]
    layer_reports = [{"layer": i} for i in range(c.num_layers)]
    if inject:
        nf = torch.cat([rms_unit(model.fuse(e, g)[0], eps) for e, g, _ in clean_inputs])
        for i, layer in enumerate(model.layers):
            gain, layer_reports[i]["context_layernorm"] = context_gain(
                nf, merged["clean"][f"k{i}"], merged["clean"][f"v{i}"], layer.self_attn.k_proj.weight.float(),
                layer.self_attn.v_proj.weight.float())
            layer.context_layernorm.weight.copy_(gain - 1)
        del nf
        views = []
        for s, (e, g, p) in zip(sequences, clean_inputs, strict=True):
            views.append(model.context(e, g, p).before(s.anchors, c.window))
        prefix = CleanCache.stack(views)
        del views
        embeds = torch.cat([e for e, _, _ in blocks])
        grs = torch.cat([g for _, g, _ in blocks])
        positions = torch.cat([p for _, _, p in blocks])
        h = model.fuse(embeds, grs)
        for i, layer in enumerate(model.layers):
            raw_a, _ = merged["block"][f"attn{i}"]
            raw_c, _ = merged["block"][f"mlp{i}"]
            gain, layer_reports[i]["input_layernorm"] = channel_gain(rms_unit(h, eps), raw_a)
            layer.input_layernorm.weight.copy_(gain - 1)
            capture = Capture(layer.post_attention_layernorm)
            block_layer(layer, h, positions, prefix, i)
            gain, layer_reports[i]["post_attention_layernorm"] = channel_gain(rms_unit(capture.close(0), eps), raw_c)
            layer.post_attention_layernorm.weight.copy_(gain - 1)
            h = block_layer(layer, h, positions, prefix, i)
        del prefix
    else:
        streams = [model.fuse(e, g) for e, g, _ in clean_inputs]
        prefixes = [[] for _ in sequences]
        prefix_positions = [None] * len(sequences)
        for i, layer in enumerate(model.layers):
            raw_a, _ = merged["clean"][f"attn{i}"]
            raw_c, _ = merged["clean"][f"mlp{i}"]
            gain, layer_reports[i]["input_layernorm"] = channel_gain(
                torch.cat([rms_unit(x[0], eps) for x in streams]), raw_a)
            layer.input_layernorm.weight.copy_(gain - 1)
            mids = []
            for x, (_, _, p) in zip(streams, clean_inputs, strict=True):
                capture = Capture(layer.post_attention_layernorm)
                layer_pass(layer, x, p, c.window)
                mids.append(rms_unit(capture.close(1)[0], eps))
            gain, layer_reports[i]["post_attention_layernorm"] = channel_gain(torch.cat(mids), raw_c)
            layer.post_attention_layernorm.weight.copy_(gain - 1)
            del mids
            for j, (x, (_, _, p)) in enumerate(zip(streams, clean_inputs, strict=True)):
                streams[j], cache = layer_pass(layer, x, p, c.window)
                view = CleanCache(p, (cache,)).before(sequences[j].anchors, c.window)
                prefixes[j].append(view.layers[0])
                prefix_positions[j] = view.positions
        del streams
        prefix = CleanCache.stack([CleanCache(q, tuple(layers)) for q, layers in zip(prefix_positions, prefixes,
                                                                                       strict=True)])
        del prefixes
        embeds = torch.cat([e for e, _, _ in blocks])
        grs = torch.cat([g for _, g, _ in blocks])
        positions = torch.cat([p for _, _, p in blocks])
        h = model.fuse(embeds, grs)
        for i, layer in enumerate(model.layers):
            h = block_layer(layer, h, positions, prefix, i)
        del prefix
    hidden = merged["block"]["out"]
    gain, report["norm"] = channel_gain(rms_unit(h, eps), hidden)
    model.norm.weight.copy_(gain - 1)
    report["layers"] = layer_reports
    return report


@torch.no_grad()
def convert(source, sequences, context):
    """The plain drafter from the HC source and the calibration sequences, and the calibration report."""
    c = plain_config(source.config, context)
    merged, folds, statistics = record_source(source, sequences, context)
    model = derive(source, c, folds)
    model.requires_grad_(False)
    report = fit(model, sequences, merged)
    return model, {"context": context, "folds": statistics, "fits": report}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def write_init(model, out, record, calibration):
    """init_mtp3's layout, BF16, published atomically; returns the reloaded model's max deviation (0 expected)."""
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(out)
    partial.mkdir(parents=True)
    state = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in model.state_dict().items()}
    if set(state) != set(dict(model.named_parameters())):
        raise RuntimeError("the drafter holds state besides its parameters; the layout below would drop it")
    files = []
    for i in range(model.config.num_layers):
        save_file({k: v for k, v in state.items() if k.startswith(f"layers.{i}.")}, str(partial / f"layer-{i}.safetensors"))
        files.append(f"layer-{i}.safetensors")
    save_file({k: v for k, v in state.items() if not k.startswith("layers.")}, str(partial / "interface.safetensors"))
    files.append("interface.safetensors")
    (partial / "config.json").write_text(json.dumps(asdict(model.config), indent=2) + "\n")
    (partial / "initialization_manifest.json").write_text(
        json.dumps({"complete": True, "files": files, "kind": "plain-conversion", "dtype": "bfloat16", **record},
                   indent=2) + "\n")
    (partial / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    reloaded = load_checkpoint(partial)
    for name, value in reloaded.named_parameters():
        if not torch.equal(value, state[name]):
            raise RuntimeError(f"written tensor does not round-trip: {name}")
    names = sorted(p.name for p in partial.iterdir())
    (partial / "SHA256SUMS").write_text("".join(f"{sha256(partial / n)}  {n}\n" for n in names))
    os.rename(partial, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path, help="an HC dense_shared drafter in the init layout")
    p.add_argument("--features", required=True, type=Path, nargs="+", help="capture job dirs (calibration rows)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--context", choices=("inject", "recompute"), default="inject")
    p.add_argument("--max-tokens", type=int, default=24576, help="each job's first tokens (its clean pass)")
    p.add_argument("--anchors-per-job", type=int, default=256)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(16)
    began = time.monotonic()
    source = load_checkpoint(args.source, dtype=torch.float32, device="cuda")
    source.requires_grad_(False)
    sequences = load_sequences(args.features, args.max_tokens, args.anchors_per_job, "cuda")
    model, calibration = convert(source, sequences, args.context)
    calibration |= {"features": [str(f.resolve()) for f in args.features], "max_tokens": args.max_tokens,
                    "anchors_per_job": args.anchors_per_job, "seconds": time.monotonic() - began}
    record = {"source": str(args.source.resolve()),
              "source_manifest_sha256": sha256(args.source / "initialization_manifest.json"),
              "source_config_sha256": sha256(args.source / "config.json"),
              "converter_sha256": sha256(Path(__file__))}
    write_init(model, args.out, record, calibration)
    print(json.dumps(calibration, indent=1))


if __name__ == "__main__":
    main()
