#!/usr/bin/env python3
"""Compile/probe the guarded SGLang Qwen4 MNNVL+HC production kernels."""

from __future__ import annotations

import argparse
import json
import os

import torch

HIDDEN = 2560
HC = 4


def compile_only() -> None:
    from sglang.kernels.ops.communication.qwen4_mnnvl_hc import (
        compile_qwen4_mnnvl_hc,
    )
    from sglang.kernels.ops.elementwise.hc_gate import compile_hc_gate

    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Qwen4 MNNVL+HC requires SM100")
    compile_hc_gate(HC, HIDDEN, torch.bfloat16)
    compile_qwen4_mnnvl_hc()
    print("Qwen4 MNNVL+HC production JIT self-check passed")


def make_tensors(rows: int, rank: int):
    shard_gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    shared_gen = torch.Generator(device="cuda").manual_seed(42)
    shard = torch.randn((rows, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=shard_gen)
    residual = torch.randn(
        (rows, HC * HIDDEN),
        device="cuda",
        dtype=torch.bfloat16,
        generator=shared_gen,
    )
    normed = torch.randn(
        (rows, HC * HIDDEN),
        device="cuda",
        dtype=torch.bfloat16,
        generator=shared_gen,
    )
    weight = torch.randn(
        (HC, HC * HIDDEN),
        device="cuda",
        dtype=torch.bfloat16,
        generator=shared_gen,
    )
    return shard, residual, normed, weight


def distributed_probe(args) -> None:
    import torch.distributed as dist
    from flashinfer.comm import create_allreduce_fusion_workspace
    from flashinfer.comm.mnnvl import TorchDistBackend
    from flashinfer.comm.trtllm_mnnvl_ar import (
        MNNVLAllreduceFusionStrategy,
        trtllm_mnnvl_allreduce,
    )
    from sglang.kernels.ops.communication.qwen4_mnnvl_hc import (
        qwen4_mnnvl_hc_combine,
    )
    from sglang.kernels.ops.elementwise.hc_combine import hc_combine
    from sglang.kernels.ops.elementwise.hc_gate import hc_gate

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"production guard requires TP4, got TP{world}")
    if torch.cuda.get_device_name() != "NVIDIA B200":
        raise RuntimeError("production guard requires NVIDIA B200")

    compile_only()
    workspace = create_allreduce_fusion_workspace(
        backend="mnnvl",
        world_size=world,
        rank=rank,
        max_token_num=max(args.rows),
        hidden_dim=HIDDEN,
        dtype=torch.bfloat16,
        gpus_per_node=world,
        comm_backend=TorchDistBackend(),
        group=dist.group.WORLD,
    )
    dist.barrier()

    results = []
    for rows in args.rows:
        shard, residual, normed, weight = make_tensors(rows, rank)
        reduced = torch.empty_like(shard)
        baseline = torch.empty_like(residual)
        fused = torch.empty_like(residual)
        gate = hc_gate(normed, weight, HC, HIDDEN)
        strategy = MNNVLAllreduceFusionStrategy.select_strategy(world, rows, HIDDEN, torch.bfloat16)

        def baseline_call():
            trtllm_mnnvl_allreduce(
                shard,
                workspace,
                launch_with_pdl=True,
                output=reduced,
                strategy=strategy,
            )
            hc_combine(
                reduced,
                residual,
                normed,
                weight,
                HC,
                HIDDEN,
                out=baseline,
            )

        def fused_call():
            qwen4_mnnvl_hc_combine(shard, residual, gate, fused, workspace, launch_with_pdl=True)

        for _ in range(20):
            baseline_call()
            fused_call()
        torch.cuda.synchronize()
        bitwise = torch.equal(baseline, fused)
        max_abs = float((baseline.float() - fused.float()).abs().max())

        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            fused_call()
        before = fused.clone()
        for _ in range(args.graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        stable = torch.equal(before, fused)

        checks = torch.tensor([int(bitwise), int(stable)], device="cuda", dtype=torch.int32)
        dist.all_reduce(checks, op=dist.ReduceOp.MIN)
        max_abs_tensor = torch.tensor(max_abs, device="cuda")
        dist.all_reduce(max_abs_tensor, op=dist.ReduceOp.MAX)
        if rank == 0:
            results.append(
                {
                    "rows": rows,
                    "strategy": strategy.name.lower(),
                    "bitwise_all_ranks": bool(checks[0]),
                    "max_abs_all_ranks": float(max_abs_tensor),
                    "graph_stable_all_ranks": bool(checks[1]),
                    "graph_replays": args.graph_replays,
                }
            )
        dist.barrier()

    if rank == 0:
        print(json.dumps({"world_size": world, "results": results}, indent=2))
    workspace.destroy()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32, 64, 128, 256, 512])
    parser.add_argument("--graph-replays", type=int, default=1000)
    args = parser.parse_args()
    if args.compile_only:
        compile_only()
    else:
        distributed_probe(args)


if __name__ == "__main__":
    main()
