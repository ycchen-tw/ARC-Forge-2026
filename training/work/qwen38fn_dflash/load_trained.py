"""Load trained FP32 masters (FSDP2 shards or the fast engine's replicated copy) into the BF16 serving model
without a disk export."""

import json
from pathlib import Path

import torch
from checkpointing import load_compressed


@torch.no_grad()
def load_trained_parameters(model, path, initialization):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != "torch-zstd-v1":
        raise ValueError("serving loader requires the audited torch-zstd-v1 checkpoint format")
    if Path(manifest["source_init"]).resolve() != Path(initialization).resolve():
        raise ValueError("checkpoint belongs to another native initialization")
    # The rank count is whatever the run trained with (MTP3 on 2, MTP5 on 4).
    # FSDP2 shards only the decoder layers, along dim 0 in rank order. The fast
    # engine's parameters are replicated: rank 0 wrote the one complete copy
    # (train.save_state, manifest "replicated"; checkpoints from before it carry
    # no key and are all FSDP shards).
    world = manifest["world_size"]
    if manifest.get("replicated", False):
        names = sorted(p.name for p in path.glob("*.pt.zst"))
        if names != ["replicated.pt.zst"]:
            raise ValueError(f"a replicated checkpoint holds exactly replicated.pt.zst, not {names}")
        state = load_compressed(path / "replicated.pt.zst")
        if state["rank"] != 0 or state["world_size"] != world or state["step"] != manifest["step"]:
            raise ValueError("checkpoint rank/world/step mismatch")
        shards = [state.pop("parameters")]
        del state
    else:
        if len(list(path.glob("rank-*.pt.zst"))) != world:
            raise ValueError("checkpoint rank files differ from the manifest world size")
        shards = []
        for rank in range(world):
            state = load_compressed(path / f"rank-{rank}.pt.zst")
            if state["rank"] != rank or state["world_size"] != world or state["step"] != manifest["step"]:
                raise ValueError("checkpoint rank/world/step mismatch")
            shards.append(state.pop("parameters"))
            del state
    expected = {
        name
        for name, _ in model.named_parameters()
        if not name.startswith(("embed_tokens.", "lm_head.", "target_reducer."))
    }
    if any(set(shard) != expected for shard in shards):
        raise ValueError("checkpoint parameter coverage differs from the model")
    for name in sorted(expected):
        target = model.get_parameter(name)
        if name.startswith("layers."):
            value = torch.cat([shard.pop(name).to(target.dtype) for shard in shards], dim=0)
        else:
            value = shards[0].pop(name)
            if not all(torch.equal(value, shard.pop(name)) for shard in shards[1:]):
                raise ValueError(f"replicated FP32 parameter diverged: {name}")
        if value.shape != target.shape:
            raise ValueError(f"checkpoint shape mismatch: {name}")
        target.copy_(value)
    return {"step": manifest["step"], "loaded_parameters": len(expected)}
