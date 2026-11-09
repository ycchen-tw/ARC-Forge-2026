"""Four-process FSDP2 Qwen3.8-Flash-Next LoRA training on ViGGO.

The 177B frozen base is sharded per decoder layer.  The 2.85M LoRA parameters
stay replicated and their gradients are explicitly averaged, which keeps PEFT's
adapter export in its ordinary (non-DTensor) format.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from distributed_utils import ForwardTimers, RowShardedEmbedding, assert_replicated, observe_flash_qla, sync_gradients
from fp8_fix import apply_expert_scales, apply_ple_scale
from frozen_moe import install_frozen_moe
from packed_qwen4 import install_packing, pack_rows
from probe_train import apply_qwen4_liger, import_stack, parameter_counts
from qsa_fastpath import install_short_qsa
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from viggo_data import load_split, render_training

QSA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def log0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8, help="examples per rank")
    parser.add_argument("--sequence-cap", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--targets", default=",".join(QSA_TARGETS))
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--liger", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-adapter", action="store_true")
    parser.add_argument("--frozen-moe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--batch-sizes", help="comma-separated per-rank batches, each runs --steps")
    parser.add_argument("--profile-step", type=int, default=0)
    parser.add_argument("--short-qsa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-at-batch", type=int, default=0)
    parser.add_argument("--ablate-after-gate", action="store_true")
    parser.add_argument("--study-epochs", type=int, default=2)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--packing-experiment", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.sequence_cap, args.rank, args.alpha) <= 0:
        parser.error("steps, batch-size, sequence-cap, rank and alpha must be positive")
    targets = [part.strip() for part in args.targets.split(",") if part.strip()]
    if not targets:
        parser.error("--targets cannot be empty")
    args.targets = targets
    if args.packing_experiment and not args.packing:
        parser.error('--packing-experiment requires --packing')
    if args.ablate_after_gate and (args.steps < 2 or args.study_epochs < 1):
        parser.error("ablation requires at least two timing steps per batch and positive study epochs")
    if args.ablate_after_gate and set(targets) != set(QSA_TARGETS + ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]):
        parser.error("one-load ablation requires all QSA+GDN target modules")
    return args


def load_cpu_model(model_path: str, base_storage='bf16', device=None, ple_codec='fp8'):
    if base_storage!='bf16':
        if base_storage not in ('fp8','fp8-ple','int3') or device is None:
            raise ValueError('FP8/INT3 load requires an explicit valid mode and rank device')
        if base_storage=='int3':
            from fp8_base import load_int3_cpu_model
            return load_int3_cpu_model(model_path,device,ple_codec=ple_codec)
        from fp8_base import load_fp8_cpu_model
        return load_fp8_cpu_model(model_path,device,keep_experts_fp8=base_storage=='fp8',ple_codec=ple_codec)
    if ple_codec!='fp8':
        raise ValueError(f'--ple-codec {ple_codec} needs a direct loader (--base-storage fp8 or int3), not bf16')
    from transformers import AutoConfig, FineGrainedFP8Config, __version__
    from transformers.integrations.finegrained_fp8 import FP8Experts
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    if __version__ != '5.16.1':
        raise RuntimeError('manual Qwen4 FP8 scale correction is audited only for transformers 5.16.1; re-audit loader before changing versions')

    FP8Experts._impl_tp_layer_overrides.setdefault("grouped_mm", {})
    composite = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = composite.text_config
    config.use_cache = False
    quant = dict(composite.quantization_config)
    quant["dequantize"] = True
    started = time.perf_counter()
    model = Qwen4ExpForCausalLM.from_pretrained(
        model_path,
        config=config,
        quantization_config=FineGrainedFP8Config.from_dict(quant),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        experts_implementation="grouped_mm",
        local_files_only=True,
    )
    expert_report = apply_expert_scales(torch, model, model_path)
    ple_scale = apply_ple_scale(torch, model, model_path)
    return model, time.perf_counter() - started, expert_report, ple_scale


def attach_replicated_lora(model, lora_config_cls, task_type, get_peft_model, args, device):
    model.requires_grad_(False)
    config = lora_config_cls(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type=task_type.CAUSAL_LM,
        target_modules=args.targets,
    )
    model = get_peft_model(model, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable_names):
        raise RuntimeError(f"unexpected trainable parameters: {trainable_names[:20]}")
    for parameter in trainable:
        parameter.data = parameter.data.to(device=device, dtype=torch.float32)
        dist.broadcast(parameter.data, 0)
    return model, trainable, trainable_names


def shard_frozen_base(model, trainable: list[torch.nn.Parameter], mesh, device):
    ignored = set(trainable)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    common = {"mesh": mesh, "mp_policy": policy, "reshard_after_forward": True}
    layers = model.base_model.model.model.layers
    if len(layers) != 48:
        raise RuntimeError(f"expected 48 decoder layers, found {len(layers)}")
    for layer in layers:
        if layer.mlp.experts.gate_up_proj.dtype==torch.float8_e4m3fn:
            fully_shard(layer.mlp.experts,mesh=mesh,mp_policy=MixedPrecisionPolicy(),reshard_after_forward=True)
        layer_ignored = {parameter for parameter in layer.parameters() if parameter in ignored}
        fully_shard(layer, ignored_params=layer_ignored, **common)
    fully_shard(model, ignored_params=ignored, **common)
    for buffer in model.buffers():
        if buffer.device.type != "cuda":
            buffer.data = buffer.data.to(device)
    ordinary = [parameter for parameter in trainable if type(parameter) is torch.nn.Parameter]
    if len(ordinary) != len(trainable):
        raise RuntimeError("LoRA parameters were unexpectedly converted to DTensors")
    return layers


def tensorize(tokenizer, rows, sequence_cap: int, device):
    rendered = [render_training(tokenizer, row, sequence_cap) for row in rows]
    width = max(len(ids) for ids, _ in rendered)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    supervised = 0
    for index, (ids, row_labels) in enumerate(rendered):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(row_labels, dtype=torch.long, device=device)
        supervised += sum(label != -100 for label in row_labels)
    return input_ids, attention_mask, labels, supervised


def batch_rows(rows, order, step: int, rank: int, world: int, batch_size: int):
    global_batch = world * batch_size
    offset = step * global_batch + rank * batch_size
    return [rows[order[(offset + index) % len(order)]] for index in range(batch_size)]


def scalar_mean(value: torch.Tensor, world: int) -> float:
    reduced = value.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float(reduced / world)


def main() -> None:
    args = parse_args()
    torch_stack, transformers, lora_config_cls, task_type, get_peft_model = import_stack()
    if torch_stack is not torch:
        raise RuntimeError("import_stack returned a different torch module")
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"this gate requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (world,))
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    warmup = torch.zeros(1, device=device)
    dist.all_reduce(warmup)

    from transformers import AutoTokenizer

    from arcforge.train.preflight import describe_gdn_backend

    gdn_backend, gdn_registry = describe_gdn_backend()
    for line in gdn_registry:
        log0(rank, f"[gdn] {line}")
    log0(rank, f"[gdn] selected={gdn_backend}")
    gdn_observed = observe_flash_qla()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train_rows = load_split(args.data, "train")
    order = list(range(len(train_rows)))
    random.Random(args.seed).shuffle(order)

    log0(rank, f"[load] four CPU replicas from {args.model}")
    model, load_seconds, expert_report, ple_scale = load_cpu_model(args.model)
    log0(rank, f"[load] rank0 complete in {load_seconds:.1f}s; applying liger={args.liger}")
    liger_report = apply_qwen4_liger(model) if args.liger else None
    qsa_report = install_short_qsa(model) if args.short_qsa else None
    packing_report = install_packing(model) if args.packing else None
    model, trainable, trainable_names = attach_replicated_lora(
        model, lora_config_cls, task_type, get_peft_model, args, device
    )
    frozen_moe_report = install_frozen_moe(model.base_model.model)[0] if args.liger and args.frozen_moe else None
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    counts = parameter_counts(model)
    ple = model.base_model.model.model.layers[1].ple.ple_embedding
    original = ple.ngram_embedding
    sharded = RowShardedEmbedding(original.weight, device)
    audit_ids = torch.tensor([0, sharded.start, sharded.end - 1, sharded.num_embeddings - 1], device=device)
    expected = original.weight.detach()[audit_ids.cpu()].to(device)
    torch.testing.assert_close(sharded(audit_ids), expected, rtol=0, atol=0)
    ple.ngram_embedding = sharded
    del original, expected
    log0(rank, "[ple] row-sharded lookup agrees bitwise with full CPU table")
    layers = shard_frozen_base(model, trainable, mesh, device)
    dist.barrier()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    static_allocated = torch.cuda.memory_allocated(device) / 2**30
    static_reserved = torch.cuda.memory_reserved(device) / 2**30
    log0(
        rank,
        f"[fsdp2] ready; LoRA replicated={counts['trainable']:,} params, "
        f"static={static_allocated:.1f}GiB/{static_reserved:.1f}GiB allocated/reserved",
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, fused=True)
    metrics_path = Path(args.out) / f"train_metrics.rank{rank}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_examples = 0
    total_tokens = 0
    calibration = []
    with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics:
        batch_sizes = [int(n) for n in args.batch_sizes.split(",")] if args.batch_sizes else [args.batch_size]
        for step in range(args.steps * len(batch_sizes)):
            args.batch_size = batch_sizes[step // args.steps]
            if step % args.steps == 0 and args.checkpoint_at_batch:
                args.gradient_checkpointing = args.batch_size >= args.checkpoint_at_batch
                if args.gradient_checkpointing:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    model.gradient_checkpointing_disable()
            rows = batch_rows(train_rows, order, step, rank, world, args.batch_size)
            input_ids, attention_mask, labels, supervised = tensorize(
                tokenizer, rows, args.sequence_cap, device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            profiler = None
            if rank == 0 and step + 1 == args.profile_step:
                profiler = ForwardTimers(layers)
            batch_kwargs = pack_rows(tokenizer, rows, args.sequence_cap, device) if args.packing else {
                'input_ids':input_ids,'attention_mask':attention_mask,'labels':labels}
            output = model(
                **batch_kwargs,
                use_cache=False,
                output_router_logits=False,
            )
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError(f"rank {rank} non-finite loss at step {step + 1}: {output.loss}")
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - step_started
            if profiler is not None:
                (Path(args.out) / "profile.json").write_text(json.dumps(profiler.finish(), indent=2))
            token_count = torch.tensor(supervised, dtype=torch.float32, device=device)
            dist.all_reduce(token_count)
            (output.loss * supervised / token_count).backward()
            sync_gradients(trainable)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f"rank {rank} non-finite gradient at step {step + 1}")
            optimizer.step()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - step_started
            if profiler is not None:
                log0(rank, f"[profile] forward={forward_seconds:.3f}s total={seconds:.3f}s; gdn executed={gdn_observed['calls']}")
            mean_loss = scalar_mean(output.loss * supervised * world / token_count, world)
            mean_grad = scalar_mean(grad_norm, world)
            width = int(input_ids.shape[1])
            global_examples = args.batch_size * world
            token_stats = torch.tensor([batch_kwargs['input_ids'].numel(), int(attention_mask.sum())], device=device)
            dist.all_reduce(token_stats)
            duration = torch.tensor(seconds, device=device)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            seconds = float(duration)
            global_tokens = int(token_stats[0])
            total_examples += global_examples
            total_tokens += global_tokens
            record = {
                "step": step + 1,
                "rank": rank,
                "local_loss": float(output.loss.detach()),
                "mean_loss": mean_loss,
                "mean_grad_norm": mean_grad,
                "seconds": seconds,
                "forward_seconds": forward_seconds,
                "local_batch": args.batch_size,
                "sequence_width": width,
                "local_supervised_tokens": supervised,
                "global_tokens_per_second": global_tokens / seconds,
                "real_tokens_per_second": int(token_stats[1]) / seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gradient_checkpointing": args.gradient_checkpointing,
            }
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            calibration.append(record)
            if (step + 1) % args.log_every == 0:
                log0(
                    rank,
                    f"step {step + 1:04d}/{args.steps * len(batch_sizes)} loss={mean_loss:.6f} "
                    f"gn={mean_grad:.3e} shape={global_examples}x{width} "
                    f"{seconds:.3f}s {global_tokens / seconds:.0f} tok/s",
                )

    elapsed = time.perf_counter() - started
    local_memory = torch.tensor(
        [
            torch.cuda.memory_allocated(device) / 2**30,
            torch.cuda.max_memory_allocated(device) / 2**30,
            torch.cuda.max_memory_reserved(device) / 2**30,
        ],
        device=device,
    )
    gathered_memory = [torch.zeros_like(local_memory) for _ in range(world)]
    dist.all_gather(gathered_memory, local_memory)
    final_loss = mean_loss
    replica_error = assert_replicated(trainable)
    if args.save_adapter:
        dist.barrier()
        if rank == 0:
            adapter = Path(args.out) / "adapter"
            adapter_state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
            model.save_pretrained(adapter, state_dict=adapter_state, safe_serialization=True, save_embedding_layers=False)
            tokenizer.save_pretrained(adapter)
        dist.barrier()
    if rank == 0:
        result = {
            "status": "NUMERICAL_PACKING_GATE_COMPLETED" if args.packing else "PASS",
            "packing_strict_equivalence_verified": False if args.packing else None,
            "model": args.model,
            "data": args.data,
            "world_size": world,
            "steps": args.steps,
            "local_batch_size": args.batch_size,
            "benchmark_batch_sizes": batch_sizes,
            "gradient_checkpointing": args.gradient_checkpointing,
            "global_batch_size": args.batch_size * world,
            "sequence_cap": args.sequence_cap,
            "examples_seen": total_examples,
            "padded_tokens_seen": total_tokens,
            "train_seconds": elapsed,
            "padded_tokens_per_second": total_tokens / elapsed,
            "final_loss": final_loss,
            "replica_max_abs_error": replica_error,
            "ple_lookup_bitwise_audit": True,
            "lr": args.lr,
            "lora_rank": args.rank,
            "lora_alpha": args.alpha,
            "targets": args.targets,
            "parameter_counts": counts,
            "trainable_tensor_count": len(trainable_names),
            "load_seconds_rank0": load_seconds,
            "expert_scales": expert_report,
            "ple_scale": ple_scale,
            "liger": liger_report,
            "frozen_moe": frozen_moe_report,
            "gdn_backend": gdn_backend,
            "gdn_registry": gdn_registry,
            "gdn_executed": gdn_observed,
            "short_qsa": qsa_report,
            "packing": packing_report,
            "memory_gib": [row.tolist() for row in gathered_memory],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        (Path(args.out) / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.ablate_after_gate:
        from ablate_viggo_fsdp import GDN, QSA, configs, dump, evaluate, reset_adapter, train_arm
        if set(args.targets) != set(QSA + GDN):
            raise RuntimeError("one-load ablation gate must attach the QSA+GDN superset")
        groups = {}
        for record in calibration:
            key = (record["local_batch"], record["gradient_checkpointing"])
            groups.setdefault(key, []).append(record["real_tokens_per_second"])
        chosen = max(groups, key=lambda key: statistics.median(groups[key][1:]))
        args.batch_size, args.gradient_checkpointing = chosen
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        args.epochs = args.study_epochs
        args.out = str(Path(args.out) / "ablation")
        arms = configs()
        if rank == 0:
            Path(args.out).mkdir()
            dump(Path(args.out) / "plan.json", {"args": vars(args), "arms": arms,
                  "throughput_calibration": {str(key): values for key, values in groups.items()},
                  "selection": "lowest final-epoch clean-validation token NLL"})
        del optimizer
        reset_adapter(model, arms[0], args.seed)
        train_mrs = {row["mr"] for row in train_rows}
        validation = [row for row in load_split(args.data, "validation") if row["mr"] not in train_mrs]
        if len(train_rows) != 5103 or len(validation) != 693 or gdn_observed["calls"] == 0:
            raise RuntimeError("full-data / actual FlashQLA preflight failed")
        baseline = evaluate(model, tokenizer, validation, args, device)
        if not 1.45 < baseline["token_nll"] < 1.52:
            raise RuntimeError(f"base NLL disagrees with audited reference: {baseline}")
        log0(rank, f"[ablation] selected batch/GC={chosen}; base={baseline}")
        if rank == 0:
            dump(Path(args.out) / "baseline.json", baseline)
        results = []
        for config in arms:
            results.append(train_arm(model, tokenizer, train_rows, validation, config, args.seed, args, device))
            if rank == 0:
                dump(Path(args.out) / "results.json", results)
        winners = sorted(results, key=lambda r: r["validation"][-1]["token_nll"])[:2]
        for winner in winners:
            config = {key: winner[key] for key in ("name", "lr", "alpha", "targets")}
            results.append(train_arm(model, tokenizer, train_rows, validation, config, args.seed + 1, args, device))
            if rank == 0:
                dump(Path(args.out) / "results.json", results)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
