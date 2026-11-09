"""學生模型受訓時的那一份 TIR 協定：system prompt、tool schema、user prompt 渲染。

三者都必須與 teacher 產生訓練資料時逐位元組相同。`tir_config.json` 是當時的凍結設定，
`ARC_PROMPT` 抄自 `render_arc_prompt(style="harness")`，與
`work/expert_prune/eval_tir_prune.py` 裡那份逐字相同（那份是 2026-08-22 那批 eval 的
凍結 provenance，不再維護）。
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("tir_config.json")

# ARC-Prize harness 的 user prompt，與 render_arc_prompt(style="harness") 逐位元組相同。
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
    """讀凍結的 TIR 設定。缺鍵就 raise——協定不完整比跑不起來更危險。"""
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())
    for key in ("tool", "tool_name", "system_prompt_template", "rollout"):
        if key not in cfg:
            raise RuntimeError(f"{path or CONFIG_PATH}: 缺 {key!r}，這不是 TIR 設定")
    return cfg


def system_prompt(cfg: dict, max_turns: int) -> str:
    """`max_turns` 進 prompt 是刻意的——模型被訓練成知道自己還剩幾輪。"""
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

