"""Qwen3.8-Flash-Next forward/backward and frozen-base LoRA feasibility probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path


def import_stack():
    # The training venv intentionally has broken FA2 beside working FA4. Qwen4Exp
    # does not use FA2 here, but importing PEFT reaches the parent package.
    sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model

    # PEFT 0.18.1 eagerly probes TorchAO. The shared venv has TorchAO 0.18,
    # where LinearActivationQuantizedTensor was removed; plain nn.Linear LoRA
    # should fall through to PEFTs default dispatcher instead of failing here.
    from peft.tuners.lora import torchao as peft_torchao
    try:
        from torchao.quantization import LinearActivationQuantizedTensor  # noqa: F401
    except ImportError:
        peft_torchao.is_torchao_available = lambda: False

    if tuple(map(int, transformers.__version__.split(".")[:2])) < (5, 16):
        raise RuntimeError(f"Qwen4Exp needs transformers>=5.16, got {transformers.__version__}")
    return torch, transformers, LoraConfig, TaskType, get_peft_model


def memory_gib(torch) -> list[dict[str, float]]:
    rows = []
    for device in range(torch.cuda.device_count()):
        rows.append(
            {
                "device": device,
                "allocated": torch.cuda.memory_allocated(device) / 2**30,
                "reserved": torch.cuda.memory_reserved(device) / 2**30,
                "peak_allocated": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
    return rows


def parameter_counts(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def gradient_report(torch, model) -> dict[str, object]:
    finite = nonzero = tensors = elements = 0
    missing = []
    norms = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        tensors += 1
        elements += param.grad.numel()
        is_finite = bool(torch.isfinite(param.grad).all())
        finite += int(is_finite)
        nz = bool(torch.count_nonzero(param.grad))
        nonzero += int(nz)
        norms[name] = float(param.grad.float().norm())
    return {
        "tensors": tensors,
        "elements": elements,
        "finite_tensors": finite,
        "nonzero_tensors": nonzero,
        "missing": missing,
        "norms": norms,
    }


def assert_gradients(
    report: dict[str, object], *, allowed_missing_substrings: tuple[str, ...] = ()
) -> None:
    tensors = int(report["tensors"])
    if tensors == 0:
        raise RuntimeError("no trainable parameter received a gradient")
    if int(report["finite_tensors"]) != tensors:
        raise RuntimeError(f"non-finite gradients: {report}")
    if int(report["nonzero_tensors"]) == 0:
        raise RuntimeError(f"all gradients are zero: {report}")
    unexpected_missing = [
        name for name in report["missing"]
        if not any(part in name for part in allowed_missing_substrings)
    ]
    if unexpected_missing:
        raise RuntimeError(f"trainable parameters missing gradients: {unexpected_missing}")


def attach_lora(torch, model, lora_config_cls, task_type, get_peft_model, targets, rank):
    model.requires_grad_(False)
    config = lora_config_cls(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        task_type=task_type.CAUSAL_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, config)
    counts = parameter_counts(model)
    if counts["trainable"] <= 0:
        raise RuntimeError("PEFT inserted no trainable parameters")
    bad_base = [
        name for name, param in model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]
    if bad_base:
        raise RuntimeError(f"base parameters unexpectedly trainable: {bad_base[:20]}")
    return model, counts


def random_batch(torch, vocab_size, device, batch_size, sequence_length):
    generator = torch.Generator(device=device).manual_seed(20260907)
    input_ids = torch.randint(
        8, vocab_size - 1, (batch_size, sequence_length), generator=generator, device=device
    )
    labels = input_ids.clone()
    labels[:, :2] = -100
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, labels


def shard_batch(torch, shard, device, sequence_length):
    import pyarrow.parquet as pq

    table = pq.read_table(shard, columns=["input_ids", "labels"])
    input_ids = table["input_ids"][0].as_py()
    labels = table["labels"][0].as_py()
    supervised = [index for index, label in enumerate(labels) if label != -100]
    if not supervised:
        raise RuntimeError(f"first row has no supervised labels: {shard}")
    start = max(0, supervised[0] - 2)
    stop = min(len(input_ids), start + sequence_length)
    start = max(0, stop - sequence_length)
    ids_tensor = torch.tensor(input_ids[start:stop], device=device, dtype=torch.long).unsqueeze(0)
    labels_tensor = torch.tensor(labels[start:stop], device=device, dtype=torch.long).unsqueeze(0)
    if int((labels_tensor != -100).sum()) < 2:
        raise RuntimeError("selected shard window has fewer than two supervised labels")
    return (
        (ids_tensor, torch.ones_like(ids_tensor), labels_tensor),
        {"shard": shard, "row": 0, "start": start, "stop": stop, "supervised": int((labels_tensor != -100).sum())},
    )


def apply_qwen4_liger(model) -> dict[str, int]:
    """Reuse only Qwen3.5 Liger pieces whose Qwen4 math and parameter ABI match."""
    from types import MethodType

    from liger_kernel.transformers.model.qwen3_5_moe import lce_forward
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerExperts, LigerQwen3MoeSwiGLUMLP
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    model.forward = MethodType(lce_forward, model)
    counts = {"flce": 1, "experts": 0, "shared_expert": 0, "rms_norm": 0}
    for layer in model.model.layers:
        layer.mlp.experts.forward = MethodType(LigerExperts.forward, layer.mlp.experts)
        layer.mlp.shared_expert.forward = MethodType(
            LigerQwen3MoeSwiGLUMLP.forward, layer.mlp.shared_expert
        )
        counts["experts"] += 1
        counts["shared_expert"] += 1
    for module in model.modules():
        if isinstance(module, Qwen4ExpTextRMSNorm) and module.group_size is None:
            module.offset = 1.0
            module.casting_mode = "gemma"
            module.variance_epsilon = module.eps
            module.in_place = False
            module.row_mode = None
            module.forward = MethodType(LigerRMSNorm.forward, module)
            counts["rms_norm"] += 1
    model._arc_liger_counts = counts
    return counts


def one_step(
    torch,
    model,
    batch,
    *,
    optimizer=None,
    allowed_missing_substrings: tuple[str, ...] = (),
) -> dict[str, object]:
    input_ids, attention_mask, labels = batch
    model.train()
    trainable_before = (
        {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        if optimizer is not None else {}
    )
    started = time.perf_counter()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        output_router_logits=False,
    )
    forward_s = time.perf_counter() - started
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise RuntimeError(f"invalid loss: {loss}")
    started = time.perf_counter()
    loss.backward()
    backward_s = time.perf_counter() - started
    report = gradient_report(torch, model)
    assert_gradients(report, allowed_missing_substrings=allowed_missing_substrings)
    optimizer_changed = 0
    if optimizer is not None:
        optimizer.step()
        optimizer_changed = sum(
            int(not torch.equal(trainable_before[name], param.detach()))
            for name, param in model.named_parameters() if param.requires_grad
        )
        if optimizer_changed == 0:
            raise RuntimeError("optimizer step changed no trainable tensor")
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "forward_s": forward_s,
        "backward_s": backward_s,
        "gradients": report,
        "optimizer_changed_tensors": optimizer_changed,
        "memory_gib": memory_gib(torch),
    }


def run_tiny(torch, transformers, lora_config_cls, task_type, get_peft_model, use_liger=False):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    config = Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        hc_count=2,
        hc_lowrank=16,
        ple_layer_ids=[],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=32,
        indexer_budget=8,
        indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000, "partial_rotary_factor": 1.0},
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
        use_cache=False,
    )
    device = torch.device("cuda:0")
    model = Qwen4ExpForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    if use_liger:
        apply_qwen4_liger(model)
    batch = random_batch(torch, config.vocab_size, device, 2, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # QSA chooses blocks with a discrete top-k. Its indexer has no gradient from
    # the LM loss in the upstream reference implementation; everything else must.
    full = one_step(
        torch,
        model,
        batch,
        optimizer=optimizer,
        allowed_missing_substrings=(".self_attn.indexer.",),
    )

    del optimizer, model
    torch.cuda.empty_cache()
    model = Qwen4ExpForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    if use_liger:
        apply_qwen4_liger(model)
    model, counts = attach_lora(
        torch,
        model,
        lora_config_cls,
        task_type,
        get_peft_model,
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        8,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0.0)
    lora = one_step(torch, model, batch, optimizer=optimizer)
    return {
        "mode": "tiny",
        "liger": getattr(model.base_model.model, "_arc_liger_counts", None),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "full_backward": full,
        "lora_counts": counts,
        "lora_step": lora,
    }


def load_real_model(torch, model_path: str):
    from transformers import AutoConfig, FineGrainedFP8Config
    from transformers.integrations.finegrained_fp8 import FP8Experts
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    # Transformers 5.16.1 assumes every expert implementation has a TP-plan
    # override and dereferences None for grouped_mm. An empty rewrite is the
    # intended identity mapping; keep the workaround local and assert its scope.
    FP8Experts._impl_tp_layer_overrides.setdefault("grouped_mm", {})

    composite = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = composite.text_config
    config.use_cache = False
    quant = dict(composite.quantization_config)
    quant["dequantize"] = True
    quant_config = FineGrainedFP8Config.from_dict(quant)
    # Accelerates balanced heuristic packs this checkpoint onto only three
    # devices (159 GiB on one card, zero on the fourth). Layer 1 owns the 51B
    # PLE table, so keep layers 0-1 on GPU0 and spread regular layers explicitly.
    device_map = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.hyper_connection_mixer": 3,
        "model.norm": 3,
        "lm_head": 3,
    }
    for layer in range(2, 17):
        device_map[f"model.layers.{layer}"] = 1
    for layer in range(17, 32):
        device_map[f"model.layers.{layer}"] = 2
    for layer in range(32, 48):
        device_map[f"model.layers.{layer}"] = 3
    started = time.perf_counter()
    model = Qwen4ExpForCausalLM.from_pretrained(
        model_path,
        config=config,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        experts_implementation="grouped_mm",
        local_files_only=True,
    )
    # Transformers 5.16.1 leaves packed expert and PLE values as raw FP8 cast
    # to BF16. Apply the checkpoint scales in place, then record both hard gates.
    from fp8_fix import apply_expert_scales, apply_ple_scale

    model._arc_expert_scale_report = apply_expert_scales(torch, model, model_path)
    apply_ple_scale(torch, model, model_path)
    return model, time.perf_counter() - started


def run_real(
    torch,
    transformers,
    lora_config_cls,
    task_type,
    get_peft_model,
    args,
):
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"real probe requires exactly four visible GPUs, got {torch.cuda.device_count()}")
    if args.liger:
        raise RuntimeError(
            "Liger is certified by the tiny single-GPU gate, but its Triton kernels "
            "must run one process per GPU; do not use it with this single-process device-map probe"
        )
    model, load_s = load_real_model(torch, args.model)
    input_device = model.get_input_embeddings().weight.device
    if args.data_shard:
        batch, batch_meta = shard_batch(torch, args.data_shard, input_device, args.sequence_length)
    else:
        batch = random_batch(torch, model.config.vocab_size, input_device, 1, args.sequence_length)
        batch_meta = {"source": "random", "sequence_length": args.sequence_length}
    common = {
        "mode": "real",
        "phase": args.phase,
        "model": args.model,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "load_s": load_s,
        "device_map": getattr(model, "hf_device_map", None),
        "ple_scale_applied": model._arc_ple_scale,
        "expert_scales_applied": model._arc_expert_scale_report,
        "liger": getattr(model, "_arc_liger_counts", None),
        "base_counts": parameter_counts(model),
        "batch": batch_meta,
        "memory_after_load_gib": memory_gib(torch),
    }
    if args.phase == "forward":
        model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            output = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                labels=batch[2],
                use_cache=False,
                output_router_logits=False,
            )
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise RuntimeError(f"invalid forward loss: {output.loss}")
        common.update(
            loss=float(output.loss),
            forward_s=time.perf_counter() - started,
            memory_after_forward_gib=memory_gib(torch),
        )
        return common

    targets = [part for part in args.lora_targets.split(",") if part]
    model, counts = attach_lora(
        torch, model, lora_config_cls, task_type, get_peft_model, targets, args.lora_rank
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=0.0)
    step = one_step(torch, model, batch, optimizer=optimizer)
    common.update(lora_targets=targets, lora_counts=counts, lora_step=step)
    if args.save_adapter:
        model.save_pretrained(args.save_adapter, safe_serialization=True)
        common["adapter"] = args.save_adapter
    return common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tiny", "real"), required=True)
    parser.add_argument("--phase", choices=("forward", "lora"), default="lora")
    parser.add_argument("--model", default="")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--data-shard", default="")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--save-adapter", default="")
    parser.add_argument("--liger", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "real" and not args.model:
        parser.error("--mode real requires --model")
    if args.sequence_length < 4:
        parser.error("--sequence-length must be at least 4")
    return args


def main():
    args = parse_args()
    torch, transformers, lora_config_cls, task_type, get_peft_model = import_stack()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260907)
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    if args.mode == "tiny":
        result = run_tiny(torch, transformers, lora_config_cls, task_type, get_peft_model, args.liger)
    else:
        result = run_real(
            torch, transformers, lora_config_cls, task_type, get_peft_model, args
        )
    from arcforge.train.preflight import describe_gdn_backend
    result["gdn_backend"], result["gdn_registry"] = describe_gdn_backend()
    result["status"] = "PASS"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
