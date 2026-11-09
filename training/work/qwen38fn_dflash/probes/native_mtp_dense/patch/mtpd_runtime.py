"""Opt-in runtime patch for the MTP dense10240 experiment (Qwen4-Exp native MTP drafter, sglang-arc-2-rl-int3, TP1).

Installed in every SGLang interpreter by ./sitecustomize.py when ARC_MTPD=1 (after the INT3 plugin, which it chains).

1. Dense MTP FFN. When the draft config's text_config carries `mtp_dense_intermediate_size` W, the MTP layer's MLP
   is built as DenseSharedMLP instead of the routed MoE: one SwiGLU of width W (gate_up_proj / down_proj, the
   checkpoint's mtp.layers.0.mlp.{gate,up,down}_proj, block FP8 like the experts it came from) plus the unchanged
   shared expert and its sigmoid gate, exactly Qwen2MoeSparseMoeBlock's combine with the router and routed experts
   replaced by the fixed dense FFN. Only the draft (is_nextn) is affected; target layers are built as before. In a
   stacked draft every layer's MLP is dense. After the draft's load_weights each dense MLP subtree is compared bit for
   bit against the draft checkpoint (ARC_MTPD_DRAFT_DIR): a parameter that the loader silently skipped is an error,
   not a random FFN.
2. Collection (ARC_MTPD_COLLECT_DIR set, eager server only): the draft MoE block's input rows are sampled per
   forward mode (ARC_MTPD_RATE_<MODE>, default ARC_MTPD_RATE_DEFAULT) and saved with their mode, cloned before the
   forward (FusedMoE runs in place), tagged with the draft layer (mtp.layers.<i>). The first ARC_MTPD_CHECK_CALLS
   calls per layer and mode also keep the block's output for up to 64 rows, so the offline router/expert
   reimplementation can be checked against the served MoE. Each draft decoder layer's change of the hyper-connection
   residual stream is summed per layer and mode (||out - in||^2, ||in||^2, <in, out>, per-row ||out - in|| / ||in||;
   the input cloned before the layer runs) to layer-delta-<pid>.json.
3. Depth stacking. When the draft config's text_config carries `mtpd_stack_layers` N (>= 2), the draft is built with
   N MTP decoder layers run in sequence inside every draft step (input fusion once before them, the final HC mixer and
   lm_head once after): ModelConfig._config_draft_model's forced num_hidden_layers=1 / layer_types=[full_attention]
   become N (KV pool, layer ids), and the draft Qwen4ExpModel is constructed with N layers. The checkpoint carries
   mtp.layers.0..N-1. After load_weights every draft parameter is rebuilt from the draft checkpoint with the MTP
   loader's own name mapping (q/k/v -> qkv_proj, gate/up -> gate_up_proj, experts -> w13/w2) and compared bit for bit:
   a parameter no checkpoint tensor reaches, or a checkpoint tensor no parameter takes, is an error. With
   mtpd_stack_source "copy" layers 1..N-1 must also equal layer 0; otherwise it is build_drafts.py's layer spec
   (e.g. "M,T43,T47": the MTP layer, then the target's layers 43 and 47; "target<L>" is the 10-11 name of "M,T<L>")
   and must have N entries. num_nextn_predict_layers stays 1: the draft steps reuse this one module.
   The draft's QSA KV pool is built by KVCacheConfigurator._build_hybrid_linear_kv_pool with full_attention_layer_ids
   hard-coded to [0] for a draft worker; that literal is replaced (source rewrite, asserted to match exactly once) by
   the draft config's range(N). Not changed: the target's memory budget still prices one draft layer
   (spec_aux_config.eagle_draft_num_layers comes from num_nextn_predict_layers), so the extra layers' KV comes out of
   the headroom left after the target pool.
4. Sliding-window draft attention. When the draft config's text_config carries `mtpd_swa_window` W (<= the QSA
   token budget), every draft QSAIndexer still runs (its key compression and pool writes unchanged) but its returned
   selection is replaced by the last W logical positions of each query row (p, p-1, .., p-W+1, valid entries first,
   -1 after), the same [rows, token_topk + ratio - 1] int32 layout. The draft decode steps reuse the draft-extend
   selection (QSAMTPSharedSparseIndices keeps its first token_topk columns and appends the positions drafted since),
   so they attend to that window plus the new draft tokens. Only draft (is_nextn) layers are wrapped.
"""
from __future__ import annotations

import json
import os
import re
import threading

_INSTALLED = False


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None else float(v)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    from torch import nn

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.models import qwen2_moe, qwen3_5
    from sglang.srt.models.qwen4_exp import Qwen4ExpModel
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
    from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import add_prefix

    if qwen3_5.Qwen2MoeSparseMoeBlock is not Qwen2MoeSparseMoeBlock:
        raise RuntimeError("qwen3_5.Qwen2MoeSparseMoeBlock is already replaced; refusing to patch twice")

    class DenseSharedMLP(Qwen2MoeMLP):
        """Fixed dense SwiGLU (the densified routed experts) + the gated shared expert, TP1 only."""

        def __init__(self, config, quant_config, prefix):
            if get_parallel().tp_size != 1:
                raise RuntimeError(f"DenseSharedMLP is TP1 only (tp_size={get_parallel().tp_size})")
            width = int(config.mtp_dense_intermediate_size)
            super().__init__(hidden_size=config.hidden_size, intermediate_size=width, hidden_act=config.hidden_act,
                             quant_config=quant_config, reduce_results=False, prefix=prefix)
            if not config.shared_expert_intermediate_size > 0:
                raise RuntimeError("dense MTP expects the shared expert")
            self.shared_expert = Qwen2MoeMLP(hidden_size=config.hidden_size,
                                             intermediate_size=config.shared_expert_intermediate_size,
                                             hidden_act=config.hidden_act, quant_config=quant_config,
                                             reduce_results=False, prefix=add_prefix("shared_expert", prefix))
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
            self.mtpd_width = width

        def forward(self, hidden_states, forward_batch=None, defer_finalize=False):
            if defer_finalize:
                raise RuntimeError("DenseSharedMLP does not support deferred finalize")
            dense = super().forward(hidden_states)
            shared = self.shared_expert(hidden_states)
            return dense + torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared

    collect_dir = os.environ.get("ARC_MTPD_COLLECT_DIR")
    collector = _Collector(collect_dir) if collect_dir else None

    class MaybeDenseMoeBlock(Qwen2MoeSparseMoeBlock):
        """Qwen2MoeSparseMoeBlock, except the MTP draft with a dense config becomes DenseSharedMLP."""

        def __new__(cls, *args, **kwargs):
            config = kwargs["config"]
            if kwargs.get("is_nextn") and getattr(config, "mtp_dense_intermediate_size", None):
                if args:
                    raise RuntimeError("expected keyword construction of the MoE block")
                module = DenseSharedMLP(config, kwargs.get("quant_config"), kwargs.get("prefix", ""))
                print(f"ARC_MTPD_DENSE_BUILT prefix={kwargs.get('prefix')} width={module.mtpd_width}", flush=True)
                return module
            return super().__new__(cls)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mtpd_nextn = bool(kwargs.get("is_nextn"))
            if self._mtpd_nextn:
                m = re.fullmatch(r"mtp\.layers\.(\d+)\.mlp", kwargs.get("prefix", ""))
                if m is None:
                    raise RuntimeError(f"draft MoE block with unexpected prefix {kwargs.get('prefix')!r}")
                self._mtpd_layer = int(m.group(1))

        def forward(self, hidden_states, forward_batch=None, defer_finalize=False):
            if collector is None or not self._mtpd_nextn or torch.cuda.is_current_stream_capturing():
                return super().forward(hidden_states, forward_batch, defer_finalize)
            return collector.wrap(self._mtpd_layer, super().forward, hidden_states, forward_batch, defer_finalize)

    qwen3_5.Qwen2MoeSparseMoeBlock = MaybeDenseMoeBlock

    config_draft_model = ModelConfig._config_draft_model

    def _config_draft_model(self):
        config_draft_model(self)
        n = getattr(self.hf_text_config, "mtpd_stack_layers", None)
        if self.is_draft_model and n is not None:
            if self.hf_config.architectures[0] != "Qwen4ExpForCausalLMMTP" or int(n) < 2:
                raise RuntimeError(f"mtpd_stack_layers={n} on {self.hf_config.architectures}")
            self.hf_text_config.num_hidden_layers = int(n)
            self.hf_text_config.layer_types = ["full_attention"] * int(n)
            print(f"ARC_MTPD_STACK_CONFIG layers={n}", flush=True)

    ModelConfig._config_draft_model = _config_draft_model

    model_init = Qwen4ExpModel.__init__

    def qwen4_model_init(self, config, quant_config=None, prefix="", is_nextn=False):
        n = getattr(config, "mtpd_stack_layers", None)
        if is_nextn and n is not None:
            config.num_hidden_layers = int(n)
            config.layer_types = ["full_attention"] * int(n)
        model_init(self, config, quant_config, prefix, is_nextn)
        if is_nextn and n is not None:
            print(f"ARC_MTPD_STACK_BUILT layers={len(self.layers)}", flush=True)
        w = getattr(config, "mtpd_swa_window", None)
        if w is not None:
            if not is_nextn:
                raise RuntimeError("mtpd_swa_window on a non-draft model")
            _wrap_swa_indexers(self, int(w))

    Qwen4ExpModel.__init__ = qwen4_model_init
    _patch_draft_kv_layers()

    def load_weights(self, weights):
        result = Qwen3_5ForCausalLMMTP.load_weights(self, weights)
        n = getattr(self.config, "mtpd_stack_layers", None)
        if n is not None:
            draft_dir = os.environ.get("ARC_MTPD_DRAFT_DIR")
            if not draft_dir:
                raise RuntimeError("stacked MTP needs ARC_MTPD_DRAFT_DIR to verify the loaded layers")
            print("ARC_MTPD_CKPT_VERIFIED " + json.dumps(_verify_against_checkpoint(self, draft_dir)), flush=True)
            source = getattr(self.config, "mtpd_stack_source", "copy")
            if source == "copy":
                print("ARC_MTPD_STACK_VERIFIED " + json.dumps(_verify_stack(self.model.layers, int(n))), flush=True)
            else:
                spec = ["M", "T" + source[len("target"):]] if re.fullmatch(r"target\d+", source) else source.split(",")
                if len(spec) != int(n) or not all(re.fullmatch(r"M|T\d+", x) for x in spec):
                    raise RuntimeError(f"mtpd_stack_source {source!r} is not a {n}-layer spec")
                print(f"ARC_MTPD_STACK_SPEC {','.join(spec)}", flush=True)
        mlps = [layer.mlp for layer in self.model.layers]
        dense = [isinstance(m, DenseSharedMLP) for m in mlps]
        if any(dense):
            if not all(dense):
                raise RuntimeError(f"dense draft with some MoE layers: {dense}")
            draft_dir = os.environ.get("ARC_MTPD_DRAFT_DIR")
            if not draft_dir:
                raise RuntimeError("dense MTP needs ARC_MTPD_DRAFT_DIR to verify the loaded FFN")
            for i, m in enumerate(mlps):
                print(f"ARC_MTPD_DENSE_VERIFIED layer={i} " + json.dumps(_verify_dense(m, draft_dir, i)), flush=True)
        elif collector is not None:
            delta = _LayerDelta(collect_dir)
            for i, layer in enumerate(self.model.layers):
                layer.register_forward_pre_hook(delta.pre(i), with_kwargs=True)
                layer.register_forward_hook(delta.post(i), with_kwargs=True)
            print(f"ARC_MTPD_COLLECT_ON dir={collect_dir} layers={len(mlps)} mlp={type(mlps[0]).__name__}", flush=True)
        return result

    Qwen4ExpForCausalLMMTP.load_weights = load_weights
    _INSTALLED = True
    print(f"ARC_MTPD_INSTALLED pid={os.getpid()} collect={bool(collector)}", flush=True)


def _wrap_swa_indexers(model, window):
    """Replace every QSAIndexer's selection under `model` by the last `window` logical positions (module docstring)."""
    import torch
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    indexers = [(name, m) for name, m in model.named_modules() if isinstance(m, QSAIndexer)]
    if not indexers:
        raise RuntimeError("mtpd_swa_window set but the draft has no QSAIndexer")
    for name, indexer in indexers:
        if window < 1 or window > indexer.token_topk:
            raise RuntimeError(f"{name}: SWA window {window} outside 1..token_topk {indexer.token_topk}")
        original = indexer.forward
        seen = set()

        def forward(hidden_states, positions, forward_batch, indexer_metadata, _original=original, _name=name,
                    _seen=seen):
            selected = _original(hidden_states, positions, forward_batch, indexer_metadata)
            mode = forward_batch.forward_mode
            decode_like = (mode.is_decode() or getattr(mode, "is_target_verify", lambda: False)()
                           or getattr(mode, "is_draft_extend_v2", lambda: False)())
            if decode_like:
                if indexer_metadata.hisparse is not None:
                    raise RuntimeError("SWA draft attention does not support QSA HiSparse")
                logical = indexer_metadata.decode_logical_positions
                if logical is None:
                    logical = indexer_metadata.get_seqlens_expanded() - 1
            else:
                logical = getattr(forward_batch, "positions", None)
                if logical is None:
                    logical = positions[0] if positions.ndim == 2 else positions
                logical = logical.flatten()
            rows, width = selected.shape
            if logical.numel() < rows:
                raise RuntimeError(f"{_name}: {logical.numel()} logical positions for {rows} selection rows")
            back = torch.arange(window, device=selected.device, dtype=torch.int64)
            idx = logical[:rows].to(torch.int64).unsqueeze(1) - back.unsqueeze(0)
            out = torch.full((rows, width), -1, dtype=selected.dtype, device=selected.device)
            out[:, :window] = torch.where(idx >= 0, idx, -1).to(selected.dtype)
            if mode.name not in _seen:
                _seen.add(mode.name)
                print(f"ARC_MTPD_SWA_ACTIVE {_name} window={window} mode={mode.name} rows={rows} width={width}",
                      flush=True)
            return out

        indexer.forward = forward
    print(f"ARC_MTPD_SWA_WRAPPED window={window} indexers={[n for n, _ in indexers]}", flush=True)


def _patch_draft_kv_layers():
    """KVCacheConfigurator._build_hybrid_linear_kv_pool: a draft worker's full_attention_layer_ids [0] -> range(N)
    when the draft config carries mtpd_stack_layers (unchanged otherwise)."""
    import inspect
    import re
    import textwrap

    from sglang.srt.mem_cache import kv_cache_configurator as kvcm

    def draft_full_ids(configurator):
        n = getattr(configurator.model_config.hf_text_config, "mtpd_stack_layers", None)
        if n is None:
            return [0]
        print(f"ARC_MTPD_STACK_KV_POOL full_attention_layer_ids={list(range(int(n)))}", flush=True)
        return list(range(int(n)))

    src = textwrap.dedent(inspect.getsource(kvcm.KVCacheConfigurator._build_hybrid_linear_kv_pool))
    pattern = re.compile(r"\[0\]\n(\s+)if self\.is_draft_worker")
    if len(pattern.findall(src)) != 1:
        raise RuntimeError("_build_hybrid_linear_kv_pool no longer has exactly one '[0] if self.is_draft_worker'")
    src = pattern.sub(r"_arc_mtpd_draft_full_ids(self)\n\1if self.is_draft_worker", src)
    kvcm._arc_mtpd_draft_full_ids = draft_full_ids
    exec(compile(src, kvcm.__file__, "exec"), kvcm.__dict__)
    kvcm.KVCacheConfigurator._build_hybrid_linear_kv_pool = kvcm.__dict__.pop("_build_hybrid_linear_kv_pool")


def _verify_against_checkpoint(mtp, draft_dir):
    """Every draft parameter (except the target-shared embed_tokens / lm_head and attention k/v scale scalars) equals
    the draft checkpoint, assembled the way Qwen3_5ForCausalLMMTP.load_weights maps names; and every checkpoint tensor
    is used."""
    import re

    import torch
    from safetensors import safe_open
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

    index = json.load(open(os.path.join(draft_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def t(k):
        fn = index[k]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(draft_dir, fn), "pt")
        return handles[fn].get_tensor(k)

    plain, stacked, experts = {}, {}, {}
    for k in index:
        n = k.replace("mtp.", "model.", 1).replace("model.fc", "fc").replace("model.pre_fc", "pre_fc")
        n = n.replace(".self_attn", "")
        m = re.fullmatch(r"(.*\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)", n)
        if m:
            base, e, proj, suf = m.groups()
            key = f"{base}.experts.{'w2' if proj == 'down_proj' else 'w13'}_{suf}"
            experts.setdefault(key, {}).setdefault(int(e), {})[proj] = k
            continue
        m = re.fullmatch(r"(.*)\.(q_proj|k_proj|v_proj)\.(.*)", n)
        if m:
            stacked.setdefault(f"{m.group(1)}.qkv_proj.{m.group(3)}", {})[m.group(2)] = k
            continue
        m = re.fullmatch(r"(.*)\.(gate_proj|up_proj)\.(.*)", n)
        if m:
            stacked.setdefault(f"{m.group(1)}.gate_up_proj.{m.group(3)}", {})[m.group(2)] = k
            continue
        plain[n] = k

    def same(got, want):
        if got.shape != want.shape:
            return False
        if got.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return want.dtype == got.dtype and torch.equal(got.view(torch.uint8), want.to(got.device).view(torch.uint8))
        return torch.equal(got, want.to(device=got.device, dtype=got.dtype))

    problems, checked, skipped, w13_orders = [], 0, [], set()
    params = dict(mtp.named_parameters())
    for name, p in params.items():
        if name.startswith(("model.embed_tokens.", "lm_head.")):
            continue
        if re.search(r"\.(k_scale|v_scale|k_scale_float|v_scale_float)$", name):
            skipped.append(name)
            continue
        got = p.detach()
        if name in plain:
            ok = same(got, t(plain[name]))
        elif name in stacked:
            parts = stacked[name]
            order = ("q_proj", "k_proj", "v_proj") if ".qkv_proj." in name else ("gate_proj", "up_proj")
            if set(parts) != set(order):
                problems.append(f"{name}: checkpoint parts {sorted(parts)}")
                continue
            ok = same(got, torch.cat([t(parts[o]) for o in order]))
        elif name in experts:
            per = experts[name]
            if sorted(per) != list(range(got.shape[0])):
                problems.append(f"{name}: {got.shape[0]} experts in the parameter, checkpoint has {len(per)}")
                continue
            # FusedMoE._weight_loader_impl swaps w1/w3 for Fp8MoEMethod under flashinfer trtllm ("w31" layout).
            moe = mtp.get_submodule(name.rsplit(".", 1)[0])
            if not isinstance(moe.quant_method, Fp8MoEMethod):
                raise RuntimeError(f"{name}: expected an FP8 draft MoE, got {type(moe.quant_method).__name__}")
            w13_order = ("up_proj", "gate_proj") if moe.use_flashinfer_trtllm_moe else ("gate_proj", "up_proj")
            w13_orders.add(w13_order)
            ok = all(
                same(got[e], torch.cat([t(per[e][o]) for o in w13_order]) if "w13" in name
                     else t(per[e]["down_proj"]))
                for e in range(got.shape[0]))
        else:
            problems.append(f"parameter without checkpoint tensor: {name} {tuple(got.shape)}")
            continue
        checked += 1
        if not ok:
            problems.append(f"{name}: loaded tensor differs from the checkpoint")
    unused = (set(plain) | set(stacked) | set(experts)) - set(params)
    problems += [f"checkpoint tensor unused: {u}" for u in sorted(unused)]
    if problems:
        raise RuntimeError("draft does not match its checkpoint:\n  " + "\n  ".join(problems[:40]))
    return {"checked_params": checked, "checkpoint_tensors": len(index), "skipped_scale_params": len(skipped),
            "w13_order": sorted("+".join(o) for o in w13_orders),
            "layers": sorted({int(n.split(".")[2]) for n in params if n.startswith("model.layers.")})}


def _verify_stack(layers, n):
    """Layers 1..n-1 are bit copies of layer 0 (parameters and persistent buffers)."""
    import torch

    if len(layers) != n:
        raise RuntimeError(f"draft has {len(layers)} layers, config says {n}")
    ref = layers[0].state_dict()
    for i in range(1, n):
        sd = layers[i].state_dict()
        if sd.keys() != ref.keys():
            raise RuntimeError(f"layer {i} tensor names differ from layer 0: {sorted(set(sd) ^ set(ref))[:5]}")
        for k, v in ref.items():
            a, b = sd[k], v
            if a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError(f"layer {i} {k}: {tuple(a.shape)} {a.dtype} != layer 0 {tuple(b.shape)} {b.dtype}")
            if a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                a, b = a.view(torch.uint8), b.view(torch.uint8)
            if not torch.equal(a, b):
                raise RuntimeError(f"layer {i} {k} differs from layer 0")
    return {"layers": n, "tensors_per_layer": len(ref)}


def _verify_dense(mlp, draft_dir, layer):
    """Every parameter under draft layer `layer`'s MLP equals the draft checkpoint's tensors (FP8 compared as bytes)."""
    import torch
    from safetensors import safe_open

    index = json.load(open(os.path.join(draft_dir, "model.safetensors.index.json")))["weight_map"]
    p = f"mtp.layers.{layer}.mlp."

    def t(name):
        with safe_open(os.path.join(draft_dir, index[name]), "pt") as f:
            return f.get_tensor(name)

    expected = {
        "gate_up_proj.weight": torch.cat([t(p + "gate_proj.weight"), t(p + "up_proj.weight")]),
        "gate_up_proj.weight_scale_inv": torch.cat([t(p + "gate_proj.weight_scale_inv"),
                                                    t(p + "up_proj.weight_scale_inv")]),
        "down_proj.weight": t(p + "down_proj.weight"),
        "down_proj.weight_scale_inv": t(p + "down_proj.weight_scale_inv"),
        "shared_expert.gate_up_proj.weight": torch.cat([t(p + "shared_expert.gate_proj.weight"),
                                                        t(p + "shared_expert.up_proj.weight")]),
        "shared_expert.down_proj.weight": t(p + "shared_expert.down_proj.weight"),
        "shared_expert_gate.weight": t(p + "shared_expert_gate.weight"),
    }
    params = dict(mlp.named_parameters())
    if set(params) != set(expected):
        raise RuntimeError(f"dense MTP parameter set {sorted(params)} != expected {sorted(expected)}")
    report = {}
    for name, want in expected.items():
        got = params[name].detach().cpu()
        if got.shape != want.shape or got.dtype != want.dtype:
            raise RuntimeError(f"{name}: loaded {tuple(got.shape)} {got.dtype} != checkpoint {tuple(want.shape)} {want.dtype}")
        a, b = (got.view(torch.uint8), want.view(torch.uint8)) if got.dtype == torch.float8_e4m3fn else (got, want)
        if not torch.equal(a, b):
            raise RuntimeError(f"{name}: loaded tensor differs from the draft checkpoint")
        report[name] = [list(got.shape), str(got.dtype).replace("torch.", "")]
    return report


class _LayerDelta:
    """Per draft layer and forward mode, how much the layer changes the HC residual stream (collection servers only)."""

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, f"layer-delta-{os.getpid()}.json")
        self.sums, self.inputs, self.calls = {}, {}, 0

    def pre(self, layer):
        def hook(module, args, kwargs):
            import torch

            if torch.cuda.is_current_stream_capturing():
                return
            self.inputs[layer] = kwargs["hidden_states"].clone()
        return hook

    def post(self, layer):
        def hook(module, args, kwargs, output):
            import torch

            if torch.cuda.is_current_stream_capturing():
                return
            x = self.inputs.pop(layer).float()
            y = output[0].float()
            if x.shape != y.shape:
                raise RuntimeError(f"layer {layer}: HC stream {tuple(x.shape)} -> {tuple(y.shape)}")
            if not x.shape[0]:
                return
            d = y - x
            key = f"L{layer}-{kwargs['forward_batch'].forward_mode.name}"
            s = self.sums.setdefault(key, {"rows": 0, "d2": 0.0, "x2": 0.0, "y2": 0.0, "xy": 0.0, "row_rel_sum": 0.0})
            s["rows"] += x.shape[0]
            s["d2"] += d.square().sum().item()
            s["x2"] += x.square().sum().item()
            s["y2"] += y.square().sum().item()
            s["xy"] += (x * y).sum().item()
            s["row_rel_sum"] += (d.norm(dim=-1) / x.norm(dim=-1)).sum().item()
            self.calls += 1
            if self.calls % 100 == 0:
                self.flush()
        return hook

    def flush(self):
        out = {}
        for k, s in self.sums.items():
            out[k] = dict(s, rel=(s["d2"] / s["x2"]) ** 0.5, cos=s["xy"] / (s["x2"] * s["y2"]) ** 0.5,
                          row_rel_mean=s["row_rel_sum"] / s["rows"])
        with open(self.path + ".tmp", "w") as f:
            json.dump(out, f, indent=1)
        os.replace(self.path + ".tmp", self.path)


class _Collector:
    """Samples the draft MoE block's input rows per forward mode; see the module docstring."""

    def __init__(self, out_dir):
        import torch

        self.torch = torch
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.default_rate = _env_float("ARC_MTPD_RATE_DEFAULT", 1.0)
        self.cap = int(os.environ.get("ARC_MTPD_MAX_ROWS", "400000"))
        self.flush_rows = int(os.environ.get("ARC_MTPD_FLUSH_ROWS", "20000"))
        self.check_calls = int(os.environ.get("ARC_MTPD_CHECK_CALLS", "24"))
        self.gen = torch.Generator().manual_seed(int(os.environ.get("ARC_MTPD_SEED", "0")))
        self.lock = threading.Lock()
        self.buf_x, self.buf_mode, self.buf_layer, self.kept, self.files = [], [], [], 0, 0
        self.checks = {}
        self.seen = {}
        self.modes = {}

    def rate(self, mode):
        return _env_float(f"ARC_MTPD_RATE_{mode}", self.default_rate)

    def wrap(self, layer, forward, hidden_states, forward_batch, defer_finalize):
        torch = self.torch
        mode = forward_batch.forward_mode.name if forward_batch is not None else "NONE"
        code = self.modes.setdefault(mode, len(self.modes))
        n = hidden_states.shape[0]
        seen = f"L{layer}-{mode}"
        self.seen[seen] = self.seen.get(seen, 0) + n
        x_sel = None
        if self.kept < self.cap and n:
            keep = (torch.rand(n, generator=self.gen) < self.rate(mode)).nonzero().flatten()
            if keep.numel():
                x_sel = hidden_states[keep.to(hidden_states.device)].clone()
        calls = self.checks.setdefault((layer, mode), [])
        x_chk = hidden_states[:64].clone() if len(calls) < self.check_calls and n else None
        out = forward(hidden_states, forward_batch, defer_finalize)
        if x_chk is not None:
            calls.append({"x": x_chk.cpu(), "out": out[: x_chk.shape[0]].clone().cpu(), "mode": mode, "layer": layer})
            if len(calls) == self.check_calls:
                torch.save(calls, os.path.join(self.dir, f"check-{os.getpid()}-L{layer}-{mode}.pt"))
        if x_sel is not None:
            self.buf_x.append(x_sel.cpu())
            self.buf_mode.append(torch.full((x_sel.shape[0],), code, dtype=torch.int8))
            self.buf_layer.append(torch.full((x_sel.shape[0],), layer, dtype=torch.int8))
            self.kept += x_sel.shape[0]
            if sum(b.shape[0] for b in self.buf_x) >= self.flush_rows or self.kept >= self.cap:
                self.flush()
        return out

    def flush(self):
        torch = self.torch
        if not self.buf_x:
            return
        path = os.path.join(self.dir, f"rows-{os.getpid()}-{self.files:04d}.pt")
        torch.save({"x": torch.cat(self.buf_x), "mode": torch.cat(self.buf_mode), "layer": torch.cat(self.buf_layer),
                    "modes": dict(self.modes)}, path)
        self.files += 1
        self.buf_x, self.buf_mode, self.buf_layer = [], [], []
        stats = {"kept": self.kept, "cap": self.cap, "files": self.files, "seen_rows_by_mode": self.seen,
                 "modes": self.modes, "rates": {m: self.rate(m) for m in self.modes}}
        with open(os.path.join(self.dir, f"stats-{os.getpid()}.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print(f"ARC_MTPD_COLLECT_FLUSH {json.dumps(stats)}", flush=True)
