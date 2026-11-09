"""Full-data LR x alpha and module ablations with one frozen-base load.

All arms start from identical per-module LoRA initialization. Inactive adapters
have zero B and no gradients. Validation never includes train MRs. Test is not
read here. Outputs include every example count, validation result and adapter.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from checkpoint_policy import install_checkpoint_policy, set_prefetch
from distributed_utils import RowShardedEmbedding, assert_replicated, observe_flash_qla, sync_gradients
from frozen_moe import install_frozen_moe
from packed_qwen4 import install_packing, pack_rows
from probe_train import apply_qwen4_liger, import_stack
from qsa_fastpath import install_short_qsa
from torch.distributed.device_mesh import init_device_mesh
from train_viggo_fsdp import attach_replicated_lora, load_cpu_model, log0, shard_frozen_base, tensorize
from viggo_data import load_split

QSA = ["q_proj", "k_proj", "v_proj", "o_proj"]
GDN = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def configs():
    grid = [{"name": f"qsa_lr{lr:g}_a{alpha}", "lr": lr, "alpha": alpha, "targets": QSA}
            for lr in (5e-5, 2e-4, 8e-4) for alpha in (8, 16, 32)]
    modules = [{"name": name, "lr": 2e-4, "alpha": 16, "targets": targets}
               for name, targets in [("qsa_qv", ["q_proj", "v_proj"]),
                                     ("qsa_o", ["o_proj"]), ("gdn", GDN), ("qsa_gdn", QSA + GDN)]]
    return grid + modules


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def distributed_batches(rows, order, rank, world, batch_size):
    """No dropping/wrapping. Empty ranks execute a zero-weight filler forward."""
    for start in range(0, len(order), world * batch_size):
        indices = order[start + rank * batch_size:start + (rank + 1) * batch_size]
        yield [rows[i] for i in indices] if indices else [rows[order[0]]], bool(indices)


@torch.no_grad()
def evaluate(model, tokenizer, rows, args, device):
    model.eval()
    rank, world = dist.get_rank(), dist.get_world_size()
    sums = torch.zeros(2, device=device, dtype=torch.float64)
    for batch, valid in distributed_batches(rows, list(range(len(rows))), rank, world, args.batch_size):
        ids, mask, labels, tokens = tensorize(tokenizer, batch, args.sequence_cap, device)
        output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False,
                       output_router_logits=False, skip_logits=True)
        if not torch.isfinite(output.loss):
            raise RuntimeError("nonfinite validation loss")
        if valid:
            sums += torch.stack([output.loss.double() * tokens, sums.new_tensor(tokens)])
    dist.all_reduce(sums)
    return {"token_nll": float(sums[0] / sums[1]), "tokens": int(sums[1]),
            "references": len(rows), "unique_mrs": len({r['mr'] for r in rows})}


@torch.no_grad()
def reset_adapter(model, config, seed):
    selected = []
    for name, module in model.named_modules():
        if ".lora_A." in name or ".lora_B." in name:
            continue
        if name.rsplit(".", 1)[-1] not in QSA + GDN:
            continue
        a = module.lora_A["default"].weight
        b = module.lora_B["default"].weight
        module_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
        generator = torch.Generator(device=a.device).manual_seed(module_seed)
        a.uniform_(-1 / math.sqrt(a.shape[1]), 1 / math.sqrt(a.shape[1]), generator=generator)
        b.zero_()
        active = name.rsplit(".", 1)[-1] in config["targets"]
        a.requires_grad_(active)
        b.requires_grad_(active)
        a.grad = b.grad = None
        module.scaling["default"] = config["alpha"] / a.shape[0]
        if active:
            selected.extend([a, b])
    assert_replicated(selected)
    return selected


def export_adapter(model, tokenizer, config, destination):
    dist.barrier()
    if dist.get_rank() == 0:
        original = model.peft_config["default"]
        exported = copy.deepcopy(original)
        exported.target_modules = set(config["targets"])
        exported.lora_alpha = config["alpha"]
        model.peft_config["default"] = exported
        state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        model.save_pretrained(destination, state_dict=state, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(destination)
        model.peft_config["default"] = original
    dist.barrier()


def train_arm(model, tokenizer, rows, validation, config, seed, args, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    out = Path(args.out) / f"{config['name']}_seed{seed}"
    if rank == 0:
        out.mkdir(exist_ok=False)
    parameters = reset_adapter(model, config, seed)
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=0.0, fused=True)
    trace, evaluations = [], []
    packing_parity = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        examples = 0
        for step, (batch, valid) in enumerate(distributed_batches(rows, order, rank, world, args.batch_size)):
            ids, mask, labels, count = tensorize(tokenizer, batch, args.sequence_cap, device)
            count *= int(valid)
            stats = torch.tensor([count, len(batch) * int(valid), int(mask.sum()) * int(valid)], device=device)
            dist.all_reduce(stats)
            tick = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_kwargs = pack_rows(tokenizer, batch, args.sequence_cap, device) if args.packing else {
                'input_ids':ids, 'attention_mask':mask, 'labels':labels}
            output = model(**batch_kwargs, use_cache=False, output_router_logits=False)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"nonfinite loss {config['name']} {epoch} {step}")
            (output.loss * count / stats[0]).backward()
            sync_gradients(parameters)
            if args.packing and epoch == 0 and step == 1:
                packed_gradient = torch.cat([p.grad.flatten() for p in parameters]).clone()
                model.zero_grad(set_to_none=True)
                reference = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False, output_router_logits=False)
                (reference.loss * count / stats[0]).backward()
                sync_gradients(parameters)
                reference_gradient = torch.cat([p.grad.flatten() for p in parameters])
                relative = float((packed_gradient-reference_gradient).norm()/reference_gradient.norm())
                cosine = float(torch.nn.functional.cosine_similarity(packed_gradient,reference_gradient,dim=0))
                losses = torch.stack([output.loss.detach(),reference.loss.detach()]).float()*count
                dist.all_reduce(losses)
                losses = losses/stats[0]
                packing_parity = {'packed_loss':float(losses[0]),'unpacked_loss':float(losses[1]),
                                  'raw_gradient_relative_l2':relative,'raw_gradient_cosine':cosine,
                                  'packed_raw_norm':float(packed_gradient.norm()),
                                  'unpacked_raw_norm':float(reference_gradient.norm())}
                within_tolerance = abs(float(losses[0]-losses[1]))<=0.005 and relative<=0.03 and cosine>=0.9995
                packing_parity['within_strict_tolerance'] = within_tolerance
                packing_parity['explicit_numerical_experiment'] = args.packing_experiment
                log0(rank, f'[packing raw-gradient parity] {packing_parity}')
                if not within_tolerance and not args.packing_experiment:
                    raise RuntimeError(f'packing raw-gradient parity failed: {packing_parity}')
                if not within_tolerance:
                    log0(rank, '[packing] STRICT EQUIVALENCE FAILED; continuing explicitly requested numerical experiment')
                offset = 0
                for p in parameters:
                    p.grad.copy_(packed_gradient[offset:offset+p.numel()].view_as(p))
                    offset += p.numel()
                del reference,reference_gradient,packed_gradient
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite gradients")
            optimizer.step()
            torch.cuda.synchronize()
            metrics = torch.tensor([float(output.loss.detach()) * count, time.perf_counter() - tick], device=device)
            dist.all_reduce(metrics[:1])
            dist.all_reduce(metrics[1:], op=dist.ReduceOp.MAX)
            examples += int(stats[1])
            record = {"epoch": epoch + 1, "step": step + 1, "nll": float(metrics[0] / stats[0]),
                      "seconds": float(metrics[1]), "real_tokens_per_second": float(stats[2] / metrics[1]),
                      "examples": int(stats[1]), "grad_norm": float(norm)}
            trace.append(record)
            if rank == 0:
                with (out / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            if (step + 1) % 10 == 0:
                log0(rank, f"[{config['name']}] epoch={epoch+1} step={step+1} nll={record['nll']:.4f} {record['real_tokens_per_second']:.0f} real tok/s")
        if examples != len(rows):
            raise RuntimeError(f"epoch coverage {examples} != {len(rows)}")
        replica_error = assert_replicated(parameters)
        validation_result = evaluate(model, tokenizer, validation, args, device)
        evaluations.append({"epoch": epoch + 1, **validation_result})
        log0(rank, f"[val] {config['name']} seed={seed} epoch={epoch+1}: {validation_result}")
        export_adapter(model, tokenizer, config, out / f"epoch{epoch+1}")
    memory = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    result = {**config, "seed": seed, "rank": args.rank, "epochs": args.epochs,
              "status":"NUMERICAL_PACKING_EXPERIMENT_COMPLETE" if args.packing_experiment else "TRAINING_COMPLETE",
              "trainable": sum(p.numel() for p in parameters), "train_examples_per_epoch": len(rows),
              "global_batch_size": world * args.batch_size, "seconds": time.perf_counter() - started,
              "validation": evaluations, "replica_max_abs_error": replica_error, "peak_allocated_gib": float(memory),
              "packing":args.packing,"packing_raw_gradient_parity":packing_parity}
    if rank == 0:
        dump(out / "result.json", result)
        dump(out / "trace.json", trace)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--replicate-top", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-policy", choices=('layer','hybrid'), default='layer')
    parser.add_argument("--prefetch-depth", type=int, choices=(0,1,2,4), default=0,
                        help='0 preserves FSDP defaults; explicit deeper prefetch uses more memory')
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--packing-experiment", action="store_true", help="record failed strict parity; explicitly study numerically different MoE routing")
    parser.add_argument("--only", help="comma-separated arm names, for recovery or an explicit subset")
    args = parser.parse_args()
    if args.packing_experiment and not args.packing:
        parser.error('--packing-experiment requires --packing')
    if args.checkpoint_policy != 'layer' and not args.gradient_checkpointing:
        parser.error('--checkpoint-policy hybrid requires --gradient-checkpointing')
    args.rank, args.alpha, args.targets = 8, 16, QSA + GDN
    _, _, lora_cls, task_type, get_peft = import_stack()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError("requires four GPUs")
    torch.manual_seed(args.seed)
    gdn_observed = observe_flash_qla()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = load_split(args.data, "train")
    train_mrs = {row["mr"] for row in rows}
    validation = [row for row in load_split(args.data, "validation") if row["mr"] not in train_mrs]
    if (len(rows), len(validation), len({r['mr'] for r in validation})) != (5103, 693, 231):
        raise RuntimeError("unexpected pinned dataset coverage")
    arms = configs()
    if args.only:
        requested = args.only.split(",")
        arms = [arm for arm in arms if arm["name"] in requested]
        if len(arms) != len(requested):
            raise ValueError("unknown or duplicate requested arm")
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        dump(out / "plan.json", {"args": vars(args), "arms": arms, "selection": "lowest final-epoch clean-validation token NLL"})
    log0(rank, "[load] CPU model, corrected FP8 scales, Liger and row-sharded PLE")
    model, load_seconds, scales, ple_scale = load_cpu_model(args.model)
    liger = apply_qwen4_liger(model)
    install_short_qsa(model)
    if args.packing:
        install_packing(model)
    model, all_trainable, names = attach_replicated_lora(model, lora_cls, task_type, get_peft, args, device)
    install_frozen_moe(model.base_model.model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ple = model.base_model.model.model.layers[1].ple.ple_embedding
    ple.ngram_embedding = RowShardedEmbedding(ple.ngram_embedding.weight, device)
    shard_frozen_base(model, all_trainable, init_device_mesh("cuda", (world,)), device)
    if args.checkpoint_policy == 'hybrid':
        select_checkpoint_policy, _ = install_checkpoint_policy(model)
        select_checkpoint_policy('hybrid')
    if args.prefetch_depth:
        set_prefetch(model,args.prefetch_depth)
    log0(rank,f'[systems] checkpoint={args.checkpoint_policy if args.gradient_checkpointing else "off"}, prefetch={args.prefetch_depth}')
    torch.cuda.empty_cache()
    baseline = evaluate(model, tokenizer, validation, args, device)
    log0(rank, f"[base] {baseline}")
    log0(rank, f"[gdn executed] {gdn_observed}")
    if gdn_observed["calls"] == 0:
        raise RuntimeError("FlashQLA did not execute; inspect the actual fallback before running ablations")
    if not 1.45 < baseline["token_nll"] < 1.52:
        raise RuntimeError("base NLL disagrees with previously audited model; abort study")
    if rank == 0:
        dump(out / "baseline.json", {**baseline, "load_seconds": load_seconds, "scales": scales, "ple_scale": ple_scale, "liger": liger})
    results = []
    for config in arms:
        results.append(train_arm(model, tokenizer, rows, validation, config, args.seed, args, device))
        if rank == 0:
            dump(out / "results.json", results)
    winners = sorted(results, key=lambda r: r["validation"][-1]["token_nll"])[:args.replicate_top]
    for winner in winners:
        config = {key: winner[key] for key in ("name", "lr", "alpha", "targets")}
        results.append(train_arm(model, tokenizer, rows, validation, config, args.seed + 1, args, device))
        if rank == 0:
            dump(out / "results.json", results)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
