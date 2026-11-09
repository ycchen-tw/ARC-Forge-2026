"""Pinned ViGGO CSV loading, Qwen chat rendering, and deployment checks."""

from __future__ import annotations

import csv
import re
from pathlib import Path

SYSTEM = (
    "Convert the video-game dialogue act into one natural English utterance. "
    "Preserve every slot value and output only the utterance."
)


def load_split(root: str | Path, split: str) -> list[dict[str, str]]:
    path = Path(root) / f"{split}.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"gem_id", "mr", "ref"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"unexpected ViGGO schema in {path}: {set(rows[0]) if rows else 'empty'}")
    return rows


def first_unique_mrs(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    unique = []
    seen = set()
    for row in rows:
        if row["mr"] in seen:
            continue
        unique.append(row)
        seen.add(row["mr"])
        if len(unique) == limit:
            return unique
    raise ValueError(f"requested {limit} unique MRs, found only {len(unique)}")


def messages(mr: str, reference: str | None = None) -> list[dict[str, str]]:
    result = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Dialogue act:\n{mr}"},
    ]
    if reference is not None:
        result.append({"role": "assistant", "content": reference})
    return result


def completion_payload(model: str, mr: str, max_tokens: int, lora_path: str | None = None):
    """Explicit LoRA routing: registering an alias does not select the adapter."""
    return {"model": model, "messages": messages(mr), "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "lora_path": lora_path}


def validate_model_route(model: str, lora_path: str | None, advertised: list[dict]):
    bare_adapter = any(row['id'] == model and row.get('parent') for row in advertised)
    if bare_adapter and not lora_path and ':' not in model:
        raise ValueError('bare adapter alias would route to base; use explicit lora_path or base:adapter')


def _input_ids(rendered) -> list[int]:
    """Normalize Transformers 5.16 BatchEncoding and older list returns."""
    ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError(f"expected one rendered conversation, got batch size {len(ids)}")
        ids = ids[0]
    return list(ids)


def render_training(tokenizer, row: dict[str, str], sequence_cap: int):
    prompt_ids = _input_ids(tokenizer.apply_chat_template(
        messages(row["mr"]), tokenize=True, add_generation_prompt=True, enable_thinking=False
    ))
    full_ids = _input_ids(tokenizer.apply_chat_template(
        messages(row["mr"], row["ref"]), tokenize=True, add_generation_prompt=False, enable_thinking=False
    ))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"chat template prefix mismatch for {row['gem_id']}")
    if len(full_ids) > sequence_cap:
        raise RuntimeError(f"{row['gem_id']} has {len(full_ids)} tokens > cap {sequence_cap}")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if len(labels) != len(full_ids) or sum(label != -100 for label in labels) < 2:
        raise RuntimeError(f"invalid assistant loss mask for {row['gem_id']}")
    return full_ids, labels


def render_prompt(tokenizer, mr: str):
    return _input_ids(tokenizer.apply_chat_template(
        messages(mr), tokenize=True, add_generation_prompt=True, enable_thinking=False
    ))


def slot_values(mr: str) -> list[str]:
    values = []
    for group in re.findall(r"\[([^]]+)\]", mr):
        values.extend(part.strip() for part in group.split(",") if part.strip())
    return values


def slot_coverage(mr: str, text: str) -> dict[str, object]:
    folded = " ".join(text.casefold().split())
    values = slot_values(mr)
    matched = [value for value in values if " ".join(value.casefold().split()) in folded]
    return {"matched": matched, "total": len(values), "fraction": len(matched) / max(1, len(values))}
