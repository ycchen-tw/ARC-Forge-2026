"""The TIR protocol the student model was trained on: system prompt, tool schema, user-prompt rendering.

All three must be byte-identical to what the teacher used to generate the training data. `tir_config.json` is the
frozen config from then, and `ARC_PROMPT` is copied from `render_arc_prompt(style="harness")`, identical word for word
to the copy in `work/expert_prune/eval_tir_prune.py` (that copy is the frozen provenance of the 2026-08-22 eval batch,
no longer maintained).
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("tir_config.json")

# The ARC-Prize harness user prompt, byte-identical to render_arc_prompt(style="harness").
ARC_PROMPT = (
    "You are participating in a puzzle solving competition. You are an expert at "
    "solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. "
    "Your goal is to identify the pattern or transformation in the training examples "
    "that maps the input to the output, then apply that pattern to the test input to "
    "give a final output.\n\nRespond in the format of the training output examples"
    "\n\n--Training Examples--\n{training_examples}--End of Training Examples--\n\n"
    "--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:{tail}"
)


def load_config(path: str | Path | None = None) -> dict:
    """Read the frozen TIR config. Raise on a missing key -- an incomplete protocol is more dangerous than failing to run."""
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())
    for key in ("tool", "tool_name", "system_prompt_template", "rollout"):
        if key not in cfg:
            raise RuntimeError(f"{path or CONFIG_PATH}: missing {key!r}; this is not a TIR config")
    return cfg


def system_prompt(cfg: dict, max_turns: int) -> str:
    """`max_turns` goes into the prompt deliberately -- the model was trained to know how many turns it has left."""
    return cfg["system_prompt_template"].format(
        tool_timeout=cfg["rollout"]["tool_timeout_s"],
        tool_max_output_chars=cfg["rollout"]["tool_max_output_chars"],
        max_turns=max_turns,
    )


def render_user_prompt(train_pairs: list[dict], test_input: list[list[int]]) -> str:
    ex = ""
    for i, p in enumerate(train_pairs):
        ex += "--Example %d-- \n\n INPUT: \n\n" % i
        ex += json.dumps(p["input"]) + "\n\n"
        ex += "OUTPUT: \n\n"
        ex += json.dumps(p["output"]) + "\n\n"
    return ARC_PROMPT.format(
        training_examples=ex, tail="\n", test_input=json.dumps(test_input)
    )
