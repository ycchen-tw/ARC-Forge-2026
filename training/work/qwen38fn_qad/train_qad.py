"""gEAN320 W3A8 Joint QAD v2 trainer (docs/plans/qwen38fn_gean320_w3a8_joint_qad_v2_20261015.md).

Modes (torchrun, FSDP2 over 2 or 4 ranks, tito-mm-v1 data from work/qwen38fn_lora/prepare_tito_mm.py):

  teacher  The full 512E FP8 model (``--model`` = official Qwen3.8-Flash-Next-FP8, its own FP8 PLE) runs the same packed
           multimodal forward as the student and writes, for every supervised position of every row, its top-K ids,
           their full-vocabulary log-probs and the tail log-prob (topk_tail_kl.teacher_topk_tail), plus the label
           token, to ``<out>/targets/<split>/<index>.npz``. Same build_batch, packing, media and PLE ids as training,
           so the targets are aligned by construction; the label column proves it at train time.
  train    Student = gEAN320 INT3 package with BF16 dense (``--model``, build_int3_bf16dense.py), PLE INT5, routing
           natural. Trainable, all from step 0: every INT3 group scale (int3_scale_qat, codes fixed), every router,
           shared experts + shared gate, the 228 QSA/GDN projections (144 through dense_full_qat = deployed block FP8,
           84 BF16). FP32 masters, AdamW per group, loss KL32_tail (topk_tail_kl; ``--kl reverse``: the reverse KL of the
           same 33-class coarsening, for the student's own trajectories) over supervised positions,
           normalized by the window's global supervised count, gradient accumulation over ``--accum`` microsteps.
           Checkpoints: torch.distributed.checkpoint of the trainable params + optimizer. Exports: the trained
           tensors in the package's deployed format (export_delta).
  score    Label log-probs of chosen rows under the student (``--resume`` a checkpoint or the step-0 base), for
           parity against a served export.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen38fn_lora"))
sys.path.insert(0, str(HERE))
import dense_full_qat  # noqa: E402
import fp8_dense_qat  # noqa: E402
import int3_moe  # noqa: E402
import int3_scale_qat  # noqa: E402
import qad_timing  # noqa: E402
import topk_targets  # noqa: E402
import trainer_kernels  # noqa: E402
import topk_tail_kl  # noqa: E402
from topk_tail_kl import fused_topk_tail_kl, teacher_topk_tail, topk_tail_kl_both  # noqa: E402
from train_tito_mm import IGNORE, build_batch, host_rss_gib, qsa_backend, supervised_hidden  # noqa: E402

TOPK = 32
# trainer_kernels minus lora (there is no LoRA); qsa_select / gdn_gva since 2026-10-15 (--fast-kernels)
FAST_KERNELS = "hc,hc_inject,gdn_norm,act_quant,qsa_select,qsa_dkdv,gdn_gva"
SCALE_NAMES = ("gate_up_scale", "down_scale")
PACKAGE_SCALE_KEY = {"gate_up_scale": "w13_scales", "down_scale": "w2_scales"}


# --------------------------------------------------------------------------------------------------- model forward
KL = {"direction": None}   # --kl, set once by main before any 'train' forward


def qad_forward(self, input_ids=None, labels=None, position_ids=None, packed_meta=None, media=None,
                attention_mask=None, inputs_embeds=None, ple_input_ids=None, kd=None, mode=None, **kwargs):
    """'teacher': (ids, lp, tail, labels) of every supervised position; 'train': (summed KL32_tail in the --kl
    direction, labels); 'eval': (summed [forward, reverse] KL32_tail fp64, labels); 'score': (label log-probs, labels)."""
    shifted, wanted, _ = supervised_hidden(self, input_ids, labels, position_ids, packed_meta, media, attention_mask,
                                           inputs_embeds, ple_input_ids, kwargs)
    if mode == "teacher":
        ids, lp, tail = teacher_topk_tail(shifted, self.lm_head.weight, TOPK)
        return ids, lp, tail, wanted
    if mode == "score":
        parts = []
        for lo in range(0, len(wanted), 4096):
            z = torch.nn.functional.linear(shifted[lo:lo + 4096], self.lm_head.weight).float()
            parts.append(z.log_softmax(-1).gather(1, wanted[lo:lo + 4096, None])[:, 0])
        return torch.cat(parts), wanted
    if mode not in ("train", "eval"):
        raise ValueError(mode)
    if kd["ids"].shape[0] != shifted.shape[0]:
        raise RuntimeError(f"{kd['ids'].shape[0]} teacher rows for {shifted.shape[0]} supervised positions")
    if not torch.equal(kd["labels"], wanted):
        raise RuntimeError("teacher targets are not aligned with this batch's labels")
    keep = kd["keep"]
    if not bool(keep.all()):   # ledger rows whose logits were altered before the log-prob (flags != 0)
        shifted = shifted[keep]
        kd = {k: kd[k][keep] for k in ("ids", "lp", "tail")}
    if mode == "eval":
        return topk_tail_kl_both(shifted, self.lm_head.weight, kd["ids"], kd["lp"], kd["tail"]), wanted
    if KL["direction"] is None:
        raise RuntimeError("KL direction not set")
    return fused_topk_tail_kl(shifted, self.lm_head.weight, kd["ids"], kd["lp"], kd["tail"],
                              direction=KL["direction"]), wanted


# ------------------------------------------------------------------------------------------------------ model build
LOSS_KEY = {"forward": "kl32_tail", "reverse": "rkl32_tail"}
TRAIN_GROUPS = ("scale", "router", "shared", "dense_fp8", "dense_bf16")


def trainable_names(model, train_groups=TRAIN_GROUPS):
    """Names (in model.named_parameters order) of the QAD v2 trainable set, restricted to ``train_groups`` (the full
    inventory is checked either way); raises on anything unexpected."""
    from dense_full_qat import INSTALLED

    dense_keys = {k + ".weight" for k in INSTALLED}
    bf16_leaves = (".self_attn.o_proj.weight", ".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")
    names, groups = [], {"scale": 0, "router": 0, "shared": 0, "dense_fp8": 0, "dense_bf16": 0}
    for name, _ in model.named_parameters():
        if not name.startswith("model.language_model.layers."):
            continue
        if name.rsplit(".", 1)[-1] in SCALE_NAMES and ".mlp.experts." in name:
            kind = "scale"
        elif name.endswith(".mlp.gate.weight"):
            kind = "router"
        elif ".mlp.shared_expert." in name or name.endswith(".mlp.shared_expert_gate.weight"):
            kind = "shared"
        elif name in dense_keys:
            kind = "dense_fp8"
        elif name.endswith(bf16_leaves):
            kind = "dense_bf16"
        else:
            continue
        groups[kind] += 1
        if kind in train_groups:
            names.append(name)
    want = {"scale": 96, "router": 48, "shared": 48 * 4, "dense_fp8": 144, "dense_bf16": 84}
    if groups != want:
        raise RuntimeError(f"trainable inventory {groups}, expected {want}")
    return names


def group_of(name):
    if name.rsplit(".", 1)[-1] in SCALE_NAMES:
        return "scale"
    if name.endswith(".mlp.gate.weight"):
        return "router"
    return "dense_shared"


def shard(model, mesh, device, train):
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    from fp8_base import RowShardedPLE

    text = model.model.language_model
    if not isinstance(text.layers[1].ple.ple_embedding.ngram_embedding, RowShardedPLE):
        raise RuntimeError("PLE must come from the direct loader (row-sharded)")
    # trainable FP32 masters compute in BF16 (= what deploys); gradients reduce in FP32
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32 if train else torch.bfloat16)
    for layer in text.layers:
        if layer.mlp.experts.gate_up_proj.dtype == torch.float8_e4m3fn:
            fully_shard(layer.mlp.experts, mesh=mesh, mp_policy=MixedPrecisionPolicy(), reshard_after_forward=True)
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    visual = set(model.model.visual.parameters())
    model.model.visual.to(device=device)
    # the root must not cast its float inputs: the teacher targets (fp32 log-probs) go in through forward
    root = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=policy.reduce_dtype, cast_forward_inputs=False)
    fully_shard(model, mesh=mesh, mp_policy=root, reshard_after_forward=True, ignored_params=visual)
    for buffer in model.buffers():
        if buffer.device.type != "cuda":
            buffer.data = buffer.data.to(device)
    for module in model.modules():  # the loss is already a global mean: sum, never average, over ranks
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def load(args, rank, device, world, role):
    from multimodal import load_multimodal
    from torch.distributed.device_mesh import init_device_mesh

    from distributed_utils import observe_flash_qla

    if role == "teacher":
        model, loading, kernels = load_multimodal(args.model, "fp8", device, "fp8")
        if model.config.text_config.num_experts != 512:
            raise RuntimeError(f"teacher has {model.config.text_config.num_experts} experts, expected the full 512")
        model.requires_grad_(False)
        names = []
    else:
        model, loading, kernels = load_multimodal(args.model, "int3", device, args.ple_codec)
        model.requires_grad_(False)
        scale_modules = int3_scale_qat.install(model.model.language_model.layers)
        int3_moe.configure(args.int3_act, len(scale_modules))
        modules, verified = dense_full_qat.install(model, args.dense_qat_reference, device)
        fp8_dense_qat.set_gemm(args.dense_qat_gemm, device)
        print(f"[rank{rank}] INT3 experts W3A{16 if args.int3_act == 'bf16' else 8}, dense GEMM {args.dense_qat_gemm}",
              flush=True)
        print(f"[rank{rank}] dense full QAT: {len(modules)} block-FP8 projections verified bit for bit: {verified}",
              flush=True)
        names = trainable_names(model, args.train_groups)
        params = dict(model.named_parameters())
        for name in trainable_names(model):   # install() made every scale a trainable Parameter: freeze unselected
            if name not in names:
                params[name].requires_grad_(False)
        for name in names:
            p = params[name]
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
    bound = trainer_kernels.install(model, trainer_kernels.parse(args.fast_kernels))
    print(f"[rank{rank}] fast kernels: {bound}", flush=True)
    if role != "teacher":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    shard(model, init_device_mesh("cuda", (world,)), device, role == "train")
    model.forward = MethodType(qad_forward, model)
    model.vit_lora = False  # read by train_tito_mm.supervised_hidden: QAD never trains the ViT (runs under no_grad)
    gdn = observe_flash_qla()
    text_config = model.config.text_config
    vision_config = model.config.vision_config
    vision_config.pad_ids = (model.config.image_token_id, model.config.video_token_id)
    if role != "teacher":
        set_experts_per_tok(model, args.experts_per_tok or text_config.num_experts_per_tok)
    params = dict(model.named_parameters())
    trainable = [params[n] for n in names]
    scale_modules = [layer.mlp.experts for layer in model.model.language_model.layers] if role != "teacher" else []
    return model, names, trainable, scale_modules, loading, kernels, gdn, text_config, vision_config


def set_experts_per_tok(model, k):
    """Every text-layer router picks ``k`` experts (the INT3 expert paths take any [T, k] routing)."""
    layers = model.model.language_model.layers
    routers = [m for layer in layers for m in layer.mlp.modules() if type(m).__name__.endswith("TopKRouter")]
    if len(routers) != len(layers):
        raise RuntimeError(f"{len(routers)} routers for {len(layers)} layers")
    if not 1 <= k <= routers[0].num_experts:
        raise ValueError(f"experts per token {k} outside 1..{routers[0].num_experts}")
    for r in routers:
        r.top_k = k
    return len(routers)


# ------------------------------------------------------------------------------------------------------ targets I/O
def load_targets(targets, rows_and_indices, device):
    """Teacher targets (qad-topk-v1) of the given dataset rows, concatenated in row order, on ``device``."""
    parts = []
    for row, index in rows_and_indices:
        t = targets.row(index)
        if not np.array_equal(t["labels"], row["input_ids"][row["loss_mask"]]):
            raise RuntimeError(f"{row['key']}: targets row {index} labels differ from the dataset row")
        parts.append(t)
    cat = lambda k, dt: torch.from_numpy(np.concatenate([t[k] for t in parts])).to(device=device, dtype=dt)  # noqa: E731
    return {"ids": cat("ids", torch.int64), "lp": cat("lp", torch.float32), "tail": cat("tail", torch.float32),
            "labels": cat("labels", torch.int64), "keep": cat("keep", torch.bool)}


# ---------------------------------------------------------------------------------------------------- export (delta)
@torch.no_grad()
def export_delta(model, names, out_dir, rank, meta):
    """Every trainable tensor in the deployed package format (rank 0 writes; every rank joins the gathers):
    scales -> <layer>.mlp.experts.w13_scales / w2_scales BF16; the 144 FP8 projections -> e4m3 weight +
    weight_scale_inv (quantize_block_fp8 of the BF16 cast, = the forward); the rest -> BF16. Atomic: .partial -> DONE."""
    from safetensors.torch import save_file

    import fp8_dense_qat as F

    params = dict(model.named_parameters())
    fp8 = {k + ".weight" for k in dense_full_qat.INSTALLED}
    tensors, manifest = {}, {}
    for name in names:
        p = params[name]
        full = p.full_tensor() if hasattr(p, "full_tensor") else p.detach()
        if rank != 0:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in SCALE_NAMES:
            out = {name.rsplit(".", 1)[0] + "." + PACKAGE_SCALE_KEY[leaf]: full.to(torch.bfloat16)}
        elif name in fp8:
            q, sw = F.quantize_block_fp8(full.to(torch.bfloat16).float())
            out = {name: q, name.removesuffix(".weight") + ".weight_scale_inv": sw}
        else:
            out = {name: full.to(torch.bfloat16)}
        for k, v in out.items():
            v = v.cpu().contiguous()
            tensors[k] = v
            manifest[k] = {"dtype": str(v.dtype), "shape": list(v.shape),
                           "sha256": hashlib.sha256(v.view(torch.uint8).numpy()).hexdigest()}
    if rank == 0:
        target = Path(out_dir)
        partial = target.with_name(target.name + ".partial")
        if target.exists() or partial.exists():
            raise FileExistsError(target)
        partial.mkdir(parents=True)
        shard, size, files = {}, 0, []

        def flush():
            nonlocal shard, size
            if shard:
                fname = f"delta-{len(files):05d}.safetensors"
                save_file(shard, str(partial / fname))
                files.append({"file": fname, "keys": sorted(shard)})
            shard, size = {}, 0

        for k in sorted(tensors):
            shard[k] = tensors[k]
            size += tensors[k].numel() * tensors[k].element_size()
            if size >= 4 * 2**30:
                flush()
        flush()
        (partial / "delta_manifest.json").write_text(json.dumps({"format": "gean320-qad-v2-delta", **meta,
                                                                 "files": files, "tensors": manifest}, indent=1) + "\n")
        (partial / "DONE").write_text("ok\n")
        partial.rename(target)
    dist.barrier()


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["teacher", "train", "score"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,validation", help="teacher: which splits to capture")
    ap.add_argument("--limit-rows", type=int, default=0, help="teacher/train: only the first N rows of each split (smoke)")
    ap.add_argument("--targets", help="train: <teacher out>/targets")
    ap.add_argument("--dense-qat-reference", help="train/score: the deployed INT3 package (FP8 dense names + bits)")
    ap.add_argument("--ple-codec", default="int5",
                    help="fp8 / int5 (the deployed table, from --model's FP8 rows), or nvfp4:<checkpoint dir> (another "
                         "checkpoint's NVFP4 PLE table, e.g. nvidia/Qwen3.8-Flash-Next-FP4-Local)")
    ap.add_argument("--bin-tokens", type=int, default=120_000)
    ap.add_argument("--accum", type=int, default=1, help="microsteps per optimizer step")
    ap.add_argument("--steps", type=int, default=0, help="train: optimizer steps (0 = one epoch)")
    ap.add_argument("--kl", choices=list(topk_tail_kl.DIRECTIONS), default="forward",
                    help="train: KL32_tail direction (reverse = mode seeking, for on-policy student trajectories); "
                         "eval logs both")
    ap.add_argument("--lr-scale", type=float, default=1e-5)
    ap.add_argument("--lr-router", type=float, default=3e-6)
    ap.add_argument("--lr-dense", type=float, default=2e-6)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine", "linear"], default="constant",
                    help="after the linear warmup: constant (run1), or cosine / linear decay to --lr-min-ratio x peak "
                         "at --lr-decay-steps")
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--lr-decay-steps", type=int, default=0, help="step where the decay ends (0 = --steps)")
    ap.add_argument("--train-groups", default=",".join(TRAIN_GROUPS),
                    help="train: comma list of " + "/".join(TRAIN_GROUPS) + " to train (default all = run1; the rest stay "
                         "frozen at the package values)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="train: KL32_tail on the validation split every N steps")
    ap.add_argument("--export-every", type=int, default=0)
    ap.add_argument("--store", help="train: directory for checkpoint-<step>/ and export/ (default --out; e.g. tmpfs "
                                    "when /data is short); logs, plan and progress stay in --out")
    ap.add_argument("--experts-per-tok", type=int, default=0,
                    help="train/score: student routers pick this many experts per token (0 = the package's config); "
                         "norm_topk_prob as configured; the teacher always keeps its own")
    ap.add_argument("--sweep-experts-per-tok", default="",
                    help="train: comma list; evaluate the step-0 student on the validation split at each value, then exit")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"],
                    help="train: Weights & Biases logging from rank 0 (train records, evals, config); offline writes "
                         "<out>/wandb for a later `wandb sync`")
    ap.add_argument("--wandb-project", default="arc-qwen-qad")
    ap.add_argument("--wandb-name", default="", help="run name (default: the --out directory name)")
    ap.add_argument("--grad-kernels", choices=["quack", "fused", "fp32"], default="quack",
                    help="train: scale / dense weight-gradient kernels and the KL loss's dH GEMM: fp32 = per-expert / "
                         "chunked FP32 GEMMs, TF32 off (the reference; what run1 / k12 used, 10.3k tok/s); fused = "
                         "scale_grad_triton (one launch per projection, --grad-precision operands) and a TF32 dH "
                         "(19.9k tok/s with bf16; step-1 gradients within 1.3%% of fp32, the TF32 dH's share); quack = the "
                         "same rounding with the operand formed in a pre-pass and quack's SM100 tcgen05 GEMM (~4x "
                         "faster weight-gradient GEMMs)")
    ap.add_argument("--fast-kernels", default=FAST_KERNELS,
                    help="trainer_kernels.py kernels: hc, gdn_norm, act_quant (bitwise or BF16-rounding replacements "
                         "of frozen eager regions); qsa_select (fused QSA selection, bitwise); gdn_gva (FlashQLA "
                         "grouped-value GDN, dq / dk summation order only)")
    ap.add_argument("--int3-act", choices=["fp8", "bf16"], default="fp8",
                    help="student: routed experts' forward activation format (int3_moe.py): fp8 = W3A8 (the Humming "
                         "W3A8 deployment, run1 / r7), bf16 = W3A16 like the B200 rollout's Humming A16 "
                         "(ARC_INT3_ACT=bf16); the scale gradient follows the forward's operands (int3_scale_qat.py)")
    ap.add_argument("--dense-qat-gemm", choices=list(fp8_dense_qat.GEMM_KERNELS), default="triton",
                    help="student: forward GEMM of the 144 block-FP8 projections: triton = SGLang fused_moe_kernel; "
                         "cutlass = flashinfer's SM100 CUTLASS groupwise kernel, bit for bit equal (needs the "
                         "flashinfer overlay; train_qad.sh sets it up)")
    ap.add_argument("--grad-precision", choices=["bf16x2", "bf16"], default="bf16",
                    help="train, --grad-kernels fused: operand precision of the scale / dense weight-gradient kernels "
                         "(FP32 accumulation either way): bf16x2 = hi + lo split, ~1e-5 relative; bf16 = one dot, "
                         "~2e-3 relative (the dense gradient is BF16 anyway), 25-35%% faster")
    ap.add_argument("--time-sections", action="store_true",
                    help="train: per-step wall time of data / forward+backward / optimizer (synchronized) and GPU time "
                         "of the QAD backward sections (scale gradients, dense dW, MoE dX) in each record")
    ap.add_argument("--profile-step", type=int, default=0,
                    help="train: torch.profiler over this step; rank 0 writes <out>/profile-step<N>.txt (by CUDA total), -self.txt (by self CUDA time), -shapes.txt (by op and input shapes) and .trace.json.gz "
                         "(Chrome trace)")
    ap.add_argument("--keep-checkpoints", type=int, default=0,
                    help="train: after each save keep only the newest N checkpoint-<step> of --store (0 = keep all)")
    ap.add_argument("--resume", help="train/score: a checkpoint-<step> directory")
    ap.add_argument("--score-keys", help="score: JSON list of dataset keys")
    ap.add_argument("--qsa-backend", default="auto")
    ap.add_argument("--offload-prefetch", type=int, default=1,
                    help="train: offloaded checkpoint inputs copied back ahead of backward (depth). 10-16: the "
                         "backward still idles ~30 ms per layer on the H2D copy issued ~13 ms before use; depth 3 "
                         "changed nothing (bitwise, 20.95 vs 21.04 s/step), so the cause is elsewhere (open)")
    ap.add_argument("--offload-keep-gib", type=float, default=0.0,
                    help="train: the newest checkpointed layer inputs stay on the GPU while they fit in this many GiB; "
                         "only older ones go to host memory (checkpoint_offload.py keep_bytes)")
    ap.add_argument("--no-offload", action="store_true",
                    help="train: keep the checkpointed layer inputs on the GPU (default: pinned-host offload, async)")
    ap.add_argument("--plan", help="train: JSON list of explicit steps [[bins of rank 0..]...] (tests)")
    ap.add_argument("--dump-grad-norms", action="store_true", help="train: per-parameter grad norms of step 1 (tests)")
    ap.add_argument("--dump-grads", default="", help="train: regex; full gradients of matching trainable tensors of the "
                                                    "first step to <out>/grads-step<N>.pt (rank 0; kernel A/B tests)")
    args = ap.parse_args()
    if args.ple_codec not in ("fp8", "int5") and not args.ple_codec.startswith("nvfp4:"):
        raise ValueError(f"--ple-codec {args.ple_codec!r}: fp8, int5 or nvfp4:<dir>")
    args.train_groups = tuple(g for g in args.train_groups.split(",") if g)
    if not args.train_groups or set(args.train_groups) - set(TRAIN_GROUPS):
        raise ValueError(f"--train-groups {args.train_groups}: a non-empty subset of {TRAIN_GROUPS}")
    if args.mode != "teacher" and not args.dense_qat_reference:
        ap.error("--dense-qat-reference is required for the student")
    if args.mode == "train" and not args.targets:
        ap.error("--targets is required for train")

    from tito_bins import cyclic_pad, plan_steps
    from tito_mm_data import TitoMMDataset

    from arcforge.train.preflight import assert_flash_qla_stack, assert_gdn_backend, assert_liger_moe_patch

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    if world not in (2, 4):
        raise RuntimeError("FSDP over 2 or 4 ranks only")
    assert_flash_qla_stack()
    assert_gdn_backend("flash_qla")
    assert_liger_moe_patch()
    out = Path(args.out)
    store = Path(args.store) if args.store else out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        if store != out:
            store.mkdir(parents=True, exist_ok=False)
        (out / "code").mkdir()
        for path in list(HERE.glob("*.py")) + list((HERE.parent / "qwen38fn_lora").glob("*.py")):
            shutil.copy2(path, out / "code" / f"{path.parent.name}__{path.name}")
        (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    dist.barrier()
    splits = {s: TitoMMDataset(args.data, s) for s in ("train", "validation")}
    tok_sha = hashlib.sha256((Path(args.model) / "tokenizer.json").read_bytes()).hexdigest()
    if splits["train"].tokenizer_sha256 != tok_sha:
        raise RuntimeError("dataset tokenizer does not match the model")
    # cpu-exact reproduces the deployed CUDA bicubic bit for bit (prepare_tito_mm.py); cpu-emulated does not
    if splits["train"].manifest["processor"].get("device") not in ("cuda", "cpu-exact"):
        raise RuntimeError("media were not processed as deployed (prepare_tito_mm --device cuda or cpu-exact)")
    role = "teacher" if args.mode == "teacher" else ("train" if args.mode == "train" else "score")
    model, names, trainable, scale_modules, loading, kernels, gdn, text_config, vision_config = load(
        args, rank, device, world, role)
    print(f"[rank{rank}] ready: {sum(p.numel() for p in trainable)} trainable elements in {len(trainable)} tensors",
          flush=True)
    from checkpoint_offload import CheckpointInputOffload, PinnedSlots

    width = text_config.hidden_size * text_config.hc_count
    slots = PinnedSlots(args.bin_tokens * width) if args.mode == "train" and not args.no_offload else None

    def batch_of(rows, audit=False, score=False):
        for row in rows:
            if row["loss_mask"][0]:
                raise RuntimeError(f"{row['key']}: supervised first token (packing would leak across segments)")
        batch, media = build_batch(rows, 0, device, vision_config, text_config.vocab_size, score=score, audit=audit)
        batch["packed_meta"]["qsa_mask_backend"] = qsa_backend(args.qsa_backend, rows)
        return batch, media

    def gdn_check(before, update):
        calls = gdn["calls"] - before
        if calls != 36 * (2 if update else 1):
            raise RuntimeError(f"{calls} FlashQLA calls this pass; another GDN backend ran")

    if args.mode == "teacher":
        model.eval()
        wanted_splits = args.splits.split(",")
        if rank == 0:
            topk_targets.create(out / "targets", {s: splits[s] for s in wanted_splits}, TOPK, {
                "source": "trainer-prefill", "teacher": args.model, "limit_rows": args.limit_rows,
                "loading": json.loads(json.dumps(loading, default=str)), "created_unix": time.time()})
        dist.barrier()
        for split in wanted_splits:
            data = splits[split]
            writer = topk_targets.TopkTargets(out / "targets", data, writable=True)
            todo = list(range(len(data)))[: args.limit_rows or None]
            # one bin of whole rows per rank and forward; padded positions forward but do not write
            bins = []
            from arcforge.train.packed_text import plan_bins
            for b in plan_bins([int(data.lengths[i]) for i in todo], args.bin_tokens):
                bins.append([todo[i] for i in b])
            chosen, own = cyclic_pad(bins, world)
            stats = {"rows": 0, "positions": 0, "label_in_topk": 0, "mass_topk": 0.0}
            for start in range(0, len(chosen), world):
                indices = chosen[start + rank]
                rows = [data[i] for i in indices]
                batch, media = batch_of(rows, audit=start == 0)
                before = gdn["calls"]
                with torch.no_grad():
                    ids, lp, tail, wanted = model(**batch, media=media, mode="teacher")
                gdn_check(before, False)
                trainer_kernels.assert_step_calls(False)
                if own[start + rank]:
                    ids_c, lp_c, tail_c, lab_c = ids.cpu().numpy(), lp.cpu().numpy(), tail.cpu().numpy(), wanted.cpu().numpy()
                    lo = 0
                    for row, index in zip(rows, indices):
                        n = int(row["loss_mask"].sum())
                        sl = slice(lo, lo + n)
                        expect = row["input_ids"][row["loss_mask"]]
                        if not np.array_equal(lab_c[sl], expect):
                            raise RuntimeError(f"{row['key']}: supervised labels out of order")
                        writer.write_row(index, ids_c[sl], lp_c[sl], tail_c[sl], np.zeros(n, np.uint8), lab_c[sl])
                        lo += n
                    if lo != len(lab_c):
                        raise RuntimeError("supervised positions left over after the rows")
                    stats["rows"] += len(rows)
                    stats["positions"] += len(lab_c)
                    stats["label_in_topk"] += int((ids_c == lab_c[:, None]).any(1).sum())
                    stats["mass_topk"] += float(np.exp(lp_c.astype(np.float64)).sum())
                if rank == 0 and (start // world) % 10 == 0:
                    print(json.dumps({"phase": "teacher", "split": split, "step": start // world,
                                      "of": len(chosen) // world, "rows_done_rank0": stats["rows"]}), flush=True)
            writer.flush()
            totals = [None] * world
            dist.all_gather_object(totals, stats)
            if rank == 0:
                s = {k: sum(t[k] for t in totals) for k in stats}
                s["label_in_topk_rate"] = s["label_in_topk"] / max(1, s["positions"])
                s["mean_topk_mass"] = s["mass_topk"] / max(1, s["positions"])
                print(json.dumps({"phase": "teacher_done", "split": split, **s}), flush=True)
                if s["label_in_topk_rate"] < 0.9:
                    raise RuntimeError(f"label-in-top-{TOPK} {s['label_in_topk_rate']:.3f}: targets misaligned?")
                writer.finish(s)
            dist.barrier()
        dist.destroy_process_group()
        return

    # ------------------------------------------------------------------------------------------- student modes
    # the sweep only evaluates: it needs no training targets
    targets = ({s: topk_targets.TopkTargets(args.targets, splits[s])
                for s in (("validation",) if args.sweep_experts_per_tok else ("train", "validation"))}
               if args.mode == "train" else {})
    groups = {"scale": [], "router": [], "dense_shared": []}
    for n, p in zip(names, trainable):
        groups[group_of(n)].append(p)
    lrs = {"scale": args.lr_scale, "router": args.lr_router, "dense_shared": args.lr_dense}
    optimizer = torch.optim.AdamW([{"params": groups[g], "lr": lrs[g], "name": g} for g in groups if groups[g]],
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, foreach=True)
    data = splits["train"]

    def ckpt_state():
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        msd, osd = get_state_dict(model, optimizer, options=StateDictOptions(ignore_frozen_params=True))
        extra = set(msd) - set(names)
        buffers = {n for n, _ in model.named_buffers()}
        if set(names) - set(msd) or extra - buffers:
            raise RuntimeError(f"checkpoint misses {sorted(set(names) - set(msd))[:3]} or holds non-buffer extras "
                               f"{sorted(extra - buffers)[:3]}")
        msd = {n: msd[n] for n in names}  # buffers are rebuilt by the loader, never trained
        return {"model": msd, "optim": osd}

    def save(step, plan_sha):
        import torch.distributed.checkpoint as dcp

        target = store / f"checkpoint-{step}"
        dcp.save(ckpt_state(), checkpoint_id=str(target))
        if rank == 0:
            (target / "trainer.json").write_text(json.dumps({"step": step, "plan_sha256": plan_sha,
                                                             "args": vars(args)}, indent=1) + "\n")
            if args.keep_checkpoints:
                done = sorted((int(d.name.split("-")[1]), d) for d in store.glob("checkpoint-*")
                              if (d / "trainer.json").exists())
                if done[-1][1] != target:
                    raise RuntimeError(f"newest checkpoint in {store} is {done[-1][1]}, not the one just saved")
                for _, old in done[:-args.keep_checkpoints]:
                    shutil.rmtree(old)
                    print(f"[rank0] removed {old}", flush=True)
        dist.barrier()

    def restore(path):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_state_dict

        state = ckpt_state()
        dcp.load(state, checkpoint_id=str(path))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"],
                       options=StateDictOptions(ignore_frozen_params=True, strict=False))
        return json.loads((Path(path) / "trainer.json").read_text())

    if args.mode == "score":
        if args.resume:
            restore(args.resume)
        model.eval()
        wanted_keys = json.loads(Path(args.score_keys).read_text())
        where = {k: (s, i) for s, d in splits.items() for i, k in enumerate(d.keys)}
        chosen, own = cyclic_pad(wanted_keys, world)
        (out / "scores").mkdir(exist_ok=True) if rank == 0 else None
        dist.barrier()
        for start in range(0, len(chosen), world):
            s, i = where[chosen[start + rank]]
            row = splits[s][i]
            batch, media = batch_of([row], score=True)
            with torch.no_grad():
                logp, _ = model(**batch, media=media, mode="score")
            trainer_kernels.assert_step_calls(False)
            if own[start + rank]:
                np.save(out / "scores" / (row["key"].replace("/", "_").replace(":", "__") + ".npy"), logp.cpu().numpy())
            dist.barrier()
        dist.destroy_process_group()
        return

    # train
    if args.plan:
        steps = json.loads(Path(args.plan).read_text())
        if any(len(s) != world for s in steps):
            raise ValueError("--plan steps need one bin per rank")
    else:
        lengths = data.lengths[: args.limit_rows or None]  # --limit-rows: the first N rows (those a smoke captured)
        steps, _ = plan_steps(lengths, args.bin_tokens, world, args.seed)
    windows = [steps[i:i + args.accum] for i in range(0, len(steps) - len(steps) % args.accum, args.accum)]
    limit = args.steps or len(windows)
    if limit > len(windows):
        raise ValueError(f"--steps {limit} exceeds the {len(windows)} windows of the plan")
    plan_sha = hashlib.sha256(json.dumps([steps, args.accum, data.keys]).encode()).hexdigest()
    decay_end = args.lr_decay_steps or limit
    if args.lr_schedule != "constant" and decay_end <= args.warmup_steps:
        raise ValueError(f"--lr-decay-steps {decay_end} must exceed --warmup-steps {args.warmup_steps}")
    if not 0.0 <= args.lr_min_ratio <= 1.0:
        raise ValueError("--lr-min-ratio must be in [0, 1]")

    def lr_factor(done):
        """LR multiplier of optimizer step done + 1."""
        step_no = done + 1
        if step_no <= args.warmup_steps:
            return step_no / args.warmup_steps
        if args.lr_schedule == "constant":
            return 1.0
        p = min(1.0, (step_no - args.warmup_steps) / (decay_end - args.warmup_steps))
        shape = 0.5 * (1.0 + math.cos(math.pi * p)) if args.lr_schedule == "cosine" else 1.0 - p
        return args.lr_min_ratio + (1.0 - args.lr_min_ratio) * shape

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    first = 1
    if args.resume:
        info = restore(args.resume)
        if info["plan_sha256"] != plan_sha:
            raise RuntimeError("resume needs the same data, plan, bins and accumulation")
        first = info["step"] + 1
        for _ in range(info["step"]):
            sched.step()
        print(f"[rank{rank}] resumed {args.resume} at step {info['step']}", flush=True)
    if rank == 0:
        (out / "plan.json").write_text(json.dumps({"windows": len(windows), "accum": args.accum,
                                                    "sha256": plan_sha}) + "\n")
    logs = []
    wb = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        wb = wandb.init(project=args.wandb_project, name=args.wandb_name or out.name, mode=args.wandb_mode,
                        dir=str(out), config={**vars(args), "world_size": world, "windows": len(windows),
                                              "plan_sha256": plan_sha, "first_step": first, "trainable": len(names),
                                              "torch": torch.__version__})
        print(f"[rank0] wandb {args.wandb_mode}: project={args.wandb_project} name={wb.name} url={wb.url}", flush=True)

    def report(rec, step, prefix):
        """progress.json + stdout + wandb (rank 0)."""
        logs.append(rec)
        if rank != 0:
            return
        print(json.dumps(rec), flush=True)
        (out / "progress.json").write_text(json.dumps(logs, indent=1) + "\n")
        if wb is not None:
            flat = {}
            for key, v in rec.items():
                if key in ("phase", "step"):
                    continue
                if isinstance(v, dict):
                    flat.update({f"{prefix}/{key}/{k}": x for k, x in v.items()})
                elif isinstance(v, (int, float)):
                    flat[f"{prefix}/{key}"] = v
            wb.log(flat, step=step)

    from tito_bins import plan_eval_steps

    vdata = splits["validation"]
    kept_train = targets["train"].kept_counts() if "train" in targets else None
    kept_val = targets["validation"].kept_counts()
    eval_plan = (plan_eval_steps(vdata.lengths, args.bin_tokens, world, args.seed)
                 if args.eval_every or args.sweep_experts_per_tok else [])

    def evaluate(step):
        model.eval()
        total = torch.zeros(3, device=device, dtype=torch.float64)
        for bins, scored in eval_plan:
            indices = bins[rank]
            rows = [vdata[i] for i in indices]
            kd = load_targets(targets["validation"], list(zip(rows, indices)), device)
            batch, media = batch_of(rows)
            before = gdn["calls"]
            with torch.no_grad():
                sums, _ = model(**batch, media=media, kd=kd, mode="eval")
            gdn_check(before, False)
            trainer_kernels.assert_step_calls(False)
            dense_full_qat.assert_step_calls(False)
            int3_moe.assert_step_calls(False)
            if scored[rank]:
                total += torch.cat([sums, torch.tensor([sum(int(kept_val[i]) for i in indices)], device=device,
                                                       dtype=torch.float64)])
        dist.all_reduce(total)
        k = {r.top_k for layer in model.model.language_model.layers for r in layer.mlp.modules()
             if type(r).__name__.endswith("TopKRouter")}
        if len(k) != 1:
            raise RuntimeError(f"routers disagree on experts per token: {k}")
        k = k.pop()
        rec = {"phase": "eval", "step": step, "kl32_tail_dev": float(total[0] / total[2]),
               "rkl32_tail_dev": float(total[1] / total[2]), "positions": int(total[2]),
               "experts_per_tok": k}
        report(rec, step, "eval" if not args.sweep_experts_per_tok else f"sweep_k{k}")

    if args.sweep_experts_per_tok:
        if args.resume:
            raise ValueError("--sweep-experts-per-tok evaluates the step-0 student")
        for k in [int(v) for v in args.sweep_experts_per_tok.split(",")]:
            set_experts_per_tok(model, k)
            t0 = time.perf_counter()
            evaluate(0)
            if rank == 0:
                print(json.dumps({"phase": "sweep", "experts_per_tok": k, "eval_seconds": time.perf_counter() - t0}),
                      flush=True)
        if wb is not None:
            wb.finish()
        if slots is not None:
            slots.close()
        dist.destroy_process_group()
        return
    if args.eval_every and first == 1:
        evaluate(0)
    qad_timing.TIMING["on"] = args.time_sections
    if args.time_sections:   # the frozen ViT runs once per micro-step (no_grad, not recomputed): time it per rank
        visual_forward = model.model.visual.forward

        def timed_visual(*a, **k):
            with qad_timing.section("vit"):
                return visual_forward(*a, **k)

        model.model.visual.forward = timed_visual
    int3_scale_qat.SCALE_GRAD["mode"] = dense_full_qat.DENSE_GRAD["mode"] = args.grad_kernels
    topk_tail_kl.DH["mode"] = "fp32" if args.grad_kernels == "fp32" else "tf32"
    KL["direction"] = args.kl
    int3_scale_qat.SCALE_GRAD["precision"] = dense_full_qat.DENSE_GRAD["precision"] = args.grad_precision

    def mark():
        if args.time_sections:
            torch.cuda.synchronize()
        return time.perf_counter()

    for step in range(first, limit + 1):
        prof = None
        if args.profile_step == step:
            from torch.profiler import ProfilerActivity, profile

            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
            prof.__enter__()
        wall = {"data": 0.0, "fwd_bwd": 0.0}
        window = windows[step - 1]
        sup_total = sum(int(kept_train[i]) for micro in window for b in micro for i in b)
        model.train()
        model.model.visual.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss_local = 0.0
        tokens_local = 0
        rank_load = {"tokens": 0, "supervised": 0, "images": 0, "patches": 0}
        for m, micro in enumerate(window):
            t_data = mark()
            indices = micro[rank]
            rows = [data[i] for i in indices]
            kd = load_targets(targets["train"], list(zip(rows, indices)), device)
            batch, media = batch_of(rows, audit=step == first and m == 0)
            tokens_local += batch["input_ids"].shape[1]
            rank_load["tokens"] += batch["input_ids"].shape[1]
            rank_load["supervised"] += sum(int(kept_train[i]) for i in indices)
            if media is not None and args.time_sections:   # grid_thw is on the GPU: read it only when timing
                rank_load["images"] += int(media["grid_thw"].shape[0])
                rank_load["patches"] += int(media["grid_thw"].prod(-1).sum())
            before = gdn["calls"]
            offload = None if slots is None else CheckpointInputOffload(batch["input_ids"].shape[1], width, slots,
                                                                         asynchronous=True,
                                                                         prefetch=args.offload_prefetch,
                                                                         keep_bytes=int(args.offload_keep_gib * 2**30))
            t_fb = mark()
            wall["data"] += t_fb - t_data
            with (offload.context() if offload is not None else torch.enable_grad()):
                loss_sum, _ = model(**batch, media=media, kd=kd, mode="train")
                if not torch.isfinite(loss_sum):
                    raise RuntimeError("nonfinite loss")
                (loss_sum / sup_total).backward()
            wall["fwd_bwd"] += mark() - t_fb
            if offload is not None and offload.tensors < 48:
                raise RuntimeError(f"checkpoint residual offload did not execute: {offload.tensors}")
            gdn_check(before, True)
            trainer_kernels.assert_step_calls(True)
            dense_full_qat.assert_step_calls(True)
            int3_moe.assert_step_calls(True)
            loss_local += float(loss_sum.detach())
        missing = [n for n, p in zip(names, trainable) if p.grad is None]
        if missing:
            raise RuntimeError(f"{len(missing)} trainable tensors got no gradient: {missing[:4]}")
        t_opt = mark()
        int3_scale_qat.mask_zero_scale_grads(scale_modules)
        if args.dump_grad_norms and step == first:
            norms = {n: float(p.grad.full_tensor().double().norm()) if hasattr(p.grad, "full_tensor")
                     else float(p.grad.double().norm()) for n, p in zip(names, trainable)}
            if rank == 0:
                (out / f"grad_norms-step{step}.json").write_text(json.dumps(norms, indent=0) + "\n")
        if args.dump_grads and step == first:
            import re

            dumped = {}
            for n, p in zip(names, trainable):
                if re.search(args.dump_grads, n):
                    full = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                    if rank == 0:
                        dumped[n] = full.detach().float().cpu()
            if rank == 0:
                if not dumped:
                    raise RuntimeError(f"--dump-grads {args.dump_grads!r} matched nothing")
                torch.save(dumped, out / f"grads-step{step}.pt")
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.clip))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"gradient norm {norm}")
        optimizer.step()
        sched.step()
        sc = int3_scale_qat.check_scales(scale_modules)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        wall["optim"] = time.perf_counter() - t_opt
        if prof is not None:
            prof.__exit__(None, None, None)
            if rank == 0:
                (out / f"profile-step{step}.txt").write_text(
                    prof.key_averages().table(sort_by="cuda_time_total", row_limit=60) + "\n")
                (out / f"profile-step{step}-self.txt").write_text(
                    prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=120, max_name_column_width=90)
                    + "\n")
                (out / f"profile-step{step}-shapes.txt").write_text(
                    prof.key_averages(group_by_input_shape=True).table(
                        sort_by="self_cuda_time_total", row_limit=100, max_name_column_width=40,
                        max_shapes_column_width=120) + "\n")
                prof.export_chrome_trace(str(out / f"profile-step{step}.trace.json.gz"))
        stats = torch.tensor([loss_local, tokens_local, sc["elements"], sc["changed_bf16"], sc["rel_update_sum"]],
                             device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        rec = {"phase": "train", "step": step, LOSS_KEY[args.kl]: float(stats[0]) / sup_total, "supervised": sup_total,
               "tokens": int(stats[1]), "grad_norm_preclip": norm, "seconds": seconds,
               "tok_s": float(stats[1]) / seconds, "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
               "host_rss_gib": host_rss_gib(), "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
               "scale_bf16_changed_frac": float(stats[3] / stats[2]), "scale_rel_update_mean": float(stats[4] / stats[2])}
        peaks = [None] * world
        dist.all_gather_object(peaks, rec["peak_gib"])
        rec["peak_gib_max"] = max(peaks)
        if args.time_sections:
            rec["wall_s"] = wall
            rec["gpu_s"] = qad_timing.collect()
            ranks = [None] * world   # per-rank load and section times: what the slowest rank was doing
            dist.all_gather_object(ranks, {**rank_load, **{f"gpu_{k}": round(v, 4) for k, v in rec["gpu_s"].items()}})
            rec["ranks"] = ranks
        report(rec, step, "train")
        if args.eval_every and (step % args.eval_every == 0 or step == limit):
            evaluate(step)
        if args.save_every and (step % args.save_every == 0 or step == limit):
            save(step, plan_sha)
        if args.export_every and (step % args.export_every == 0 or step == limit):
            export_delta(model, names, store / "export" / f"v{step:06d}", rank,
                         {"step": step, "source_package": args.dense_qat_reference, "trainer_model": args.model,
                          "run": str(out)})
    if wb is not None:
        wb.finish()
    if slots is not None:
        slots.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
