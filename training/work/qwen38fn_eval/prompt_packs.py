"""Prompt ablation combinations (prompt packs) for ARC-AGI-2 TIR.

A pack = one frozen choice of three things (system prompt, user prompt rendering, answer parsing).
`work/qwen38_eval/eval_tir_session.py` loads it by path with `--prompt-pack-file <this file> --prompt-pack <name>`
(the same convention it uses to load sandbox / parsing, not a cross-unit import).

**Without `--prompt-pack` the harness is bit-for-bit unchanged.** The `official` pack exists only to give the
ablation batch a "same server, same boot" control arm; what it renders must be bit-identical to the harness's built-in
path, enforced by the sha256 lock in `tests/test_prompt_packs.py`.

## Why answer parsing travels with the pack

The official `backscan_json_parser` **scans backward from the end of the string for the last `]` / `}`**. Two things
hit it:

1. Changing the grid representation (`tight` / `space`) → the model answers in the examples' format, and the tail is
   not JSON.
2. Asking for a rationale / code → the tail may not be the answer.

In both cases the harness would measure **format compliance, not solving ability**. So each pack carries its own
parser: arms that keep the representation use `parse_json_only` (exactly the official backscan, behavior bit-for-bit
unchanged), arms with a changed representation use `parse_any` (one candidate per format, **taking the one that ends
latest in the string**). After a run, always look at `parsed` / `answer_src` side by side.

## Images

The `image` pack returns an OpenAI multimodal content list (`[{"type":"text"...},{"type":"image_url"...}]`).
**The text grid is untouched; the image is an added redundant channel** -- `work/qwen36_grid_vision/` measured, on
Qwen3.6, real-ARC solving with text 16% > image 8%; replacing text with images is a known bad idea.
base64 is large; when the harness stores transcripts it swaps it for a placeholder (see `strip_images`).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from typing import Callable

# ---------------------------------------------------------------- official frozen values

# Bit-identical to eval_tir_session.py's ARC_PROMPT (ARC-Prize harness prompt_manager).
OFFICIAL_USER_TEMPLATE = (
    "You are participating in a puzzle solving competition. You are an expert at "
    "solving puzzles.\n\nBelow is a list of input and output pairs with a pattern. "
    "Your goal is to identify the pattern or transformation in the training examples "
    "that maps the input to the output, then apply that pattern to the test input to "
    "give a final output.\n\nRespond in the format of the training output examples"
    "\n\n--Training Examples--\n{training_examples}--End of Training Examples--\n\n"
    "--Test Input--\n{test_input}\n--End of Test Input--\n\nYour response:{tail}"
)

# tir_config.json's system_prompt_template split into paragraphs, for the packs to add / remove.
# Split: S1 opening (repeats the user prompt's first sentence verbatim) / S2 sandbox variables / S3 grid definition and
# limits / S4 "verify hypotheses with the tool" coaching / S5 turn budget and final-answer contract.
S1_INTRO = (
    "You are participating in a puzzle solving competition. You are an expert at "
    "solving puzzles."
)
S2_SANDBOX = (
    "You have a stateful Python sandbox. The puzzle's data is already loaded in it, "
    "so you never need to retype the grids:\n\n"
    '    train_pairs   list of {{"input": grid, "output": grid}} -- the same training '
    "pairs shown in the question\n"
    "    test_input    grid -- the same test input shown in the question\n"
    "    np            numpy, already imported"
)
S3_LIMITS = (
    "A grid is a list of lists of integers 0-9. Variables you define persist across "
    "calls, so you can build up your analysis over several calls. The sandbox has no "
    "network and cannot write files; it is for computation only. Each call is cut off "
    "after {tool_timeout:g} seconds and its output is truncated at "
    "{tool_max_output_chars} characters, so print summaries rather than dumping large "
    "grids."
)
S4_COACHING = (
    "Use the sandbox to check your reasoning rather than to replace it. A hypothesis "
    "is worth running when you can state it as code: write the transformation as a "
    "function, apply it to every training input, and compare against the expected "
    "training output. A mismatch tells you exactly where the hypothesis is wrong."
)
S5_CONTRACT = (
    "You have at most {max_turns} tool-calling turns. When you are ready, stop calling "
    "the tool and reply with the final output grid for the test input, in the same "
    "JSON format as the training outputs. Your final reply must contain the answer "
    "grid itself -- what the sandbox printed is not read by anyone."
)

# Arms that swap the grid representation use this version of S5: the original sentence hard-codes "in the same JSON
# format as the training outputs", but those arms' training outputs are no longer JSON; copying it would make the prompt
# contradict itself -- an extra variable unrelated to "representation". Only the one word "JSON" is removed, the rest
# is verbatim.
S5_CONTRACT_FMT = S5_CONTRACT.replace(
    "in the same JSON format as the training outputs",
    "in the same format as the training outputs",
)


# ------------------------------------------------------------- grid renderers
#
# tok/cell are **measured** values on this model's tokenizer for a 30x30 grid (not estimates):
#   json 3.000 / space 1.999 / tight 1.032 / alpha 1.032
# On this model tight is **split per cell** (['0','1','2',...]), the opposite of DSv4 merging 3 digits into 1 token
# -- the hard precondition from `work/grid_format/`, "per-cell splitting must be re-verified when changing models",
# passes here.


def grid_json(g: list[list[int]]) -> str:
    return json.dumps(g)


def grid_space(g: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def grid_tight(g: list[list[int]]) -> str:
    return "\n".join("".join(str(c) for c in row) for row in g)


GRID_RENDERERS = {
    "json": grid_json,
    "space": grid_space,
    "tight": grid_tight,
}

# ------------------------------------------------------------- answer parsing
#
# `official`-type arms use parse_json_only (= the official backscan, bit-for-bit unchanged); arms with a changed grid
# representation use parse_any (one candidate per format, take the one that ends latest; see that function's docstring).

# Any behavior change of a parser **must** bump this number: it goes into the fingerprint, so that the sha in a
# transcript maps back to "which parser version scored it". A function name cannot express a behavior change.
# **Per-parser version numbers**, not one global one. A global version would make "changed parse_any" also change the
# fingerprints of the nine arms that use parse_json_only, a fake change. +1 whenever behavior changes.
PARSER_VERSIONS = {
    "parse_json_only": 1,
    # v1 (the 2026-08-28 batch): JSON first → grabs the wrong grid on tight/space (2 rollouts each)
    # v2: switched to format first → the same error in the opposite direction, caught by the tests
    # v3: take the candidate that "ends latest in the string", no priority order any more
    "parse_any": 3,
}

# **The fingerprint computation changed once, so the sha of old transcripts does not match the current value.**
# This table keeps "transcript sha → which prompt" always resolvable,
# verified by tests/test_prompt_packs.py against actual transcripts (not prose; it is checked).
HISTORIC_FINGERPRINTS = {
    # The 2026-08-28 prompt ablation batch (`$ARC_RUNS/qwen38fn_eval/pp-*/`).
    # Back then the version field was the global `pv2`, now it is the per-parser `parse_json_onlyv1` / `parse_anyv3`;
    # the string format changed, so all 11 moved; **the prompt text itself never changed**.
    "97ccfdfb": "official", "f3b532eb": "null", "b5cc9b4a": "minimal",
    "4a037c68": "verify", "756a4def": "coach", "06549e04": "rationale",
    "c32fb0f2": "rulecode", "78281336": "tight", "bb1d8b77": "space",
    "3cd9e244": "shape", "98533f86": "image",
}
# The tight/space scores were recomputed offline with parse_any v3; only 3 rollouts changed in the whole batch
# (see docs/research/qwen38fn_prompt_ablation.md §6 item 2).

_TIGHT_RE = re.compile(r"^[0-9]+$")
_SPACE_RE = re.compile(r"^[0-9](?: [0-9])*$")


def _backscan_pos(text: str):
    """Like `_backscan`, but also returns where the answer ends in the string. `None` means not found."""
    g = _backscan(text, _want_pos=True)
    return g


def _backscan(text: str, _want_pos: bool = False):
    """Line-by-line replica of the official backscan_json_parser (keeps the pack file free of external paths)."""
    last = -1
    close = None
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("]", "}"):
            last, close = i, text[i]
            break
    if last == -1:
        return None
    opening = "[" if close == "]" else "{"
    depth, start = 1, -1
    for i in range(last - 1, -1, -1):
        if text[i] == close:
            depth += 1
        elif text[i] == opening:
            depth -= 1
            if depth == 0:
                start = i
                break
    if start == -1:
        return None
    try:
        obj = json.loads(text[start : last + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list) and obj and all(isinstance(r, list) for r in obj):
        return (obj, last) if _want_pos else obj
    return None


def _last_digit_block(text: str, pattern: re.Pattern, _want_pos: bool = False):
    """Takes the **last** block of consecutive lines matching pattern; returns None if the widths differ.

    Deliberately does not look further back for an earlier block of equal width. A last block of unequal width = the
    model gave a malformed answer, which should be recorded as malformed; falling back to some earlier block would
    **silently replace the malformed answer with another grid**, and that grid might happen to be right -- that would
    count the parser's behavior into the score.
    """
    blocks: list[tuple[list[str], int]] = []
    cur: list[str] = []
    end = 0
    pos = 0
    for raw in text.splitlines(keepends=True):
        ln = raw.strip()
        if ln and pattern.match(ln):
            cur.append(ln)
            end = pos + len(raw.rstrip("\n"))
        else:
            if cur:
                blocks.append((cur, end))
            cur = []
        pos += len(raw)
    if cur:
        blocks.append((cur, end))
    if not blocks:
        return None
    rows, end = blocks[-1]
    if len({len(x.replace(" ", "")) for x in rows}) != 1:
        return None
    grid = [[int(c) for c in ln.replace(" ", "")] for ln in rows]
    return (grid, end) if _want_pos else grid


def parse_json_only(text: str):
    return _backscan(text)


def parse_any(text: str):
    """One candidate per format, **taking the one that ends latest in the string**.

    The priority cannot be hard-coded. Both directions have been wrong:
      - **JSON first** (v1): on the `tight` arm, if the model quoted a JSON grid earlier and gave the compact answer
        last, backscan from the end picks up that JSON -- **the wrong grid**.
        In the 2026-08-28 batch tight/space each hit this on 2 rollouts (0.6%).
      - **Format first** (the v2 fix): the reverse; when the model lists a compact intermediate result first and gives
        the JSON answer last, it grabs the earlier one.
    The correct rule is position: **the contract says "the answer grid is the last thing in the reply", so take the one
    that appears last.**
    """
    cands = [
        _last_digit_block(text, _SPACE_RE, _want_pos=True),
        _last_digit_block(text, _TIGHT_RE, _want_pos=True),
        _backscan_pos(text),
    ]
    cands = [c for c in cands if c is not None]
    if not cands:
        return None
    return max(cands, key=lambda c: c[1])[0]


# ------------------------------------------------------------- image rendering

# The official ARC palette (arcprize.org's 10 colors).
ARC_PALETTE = [
    (0x00, 0x00, 0x00), (0x00, 0x74, 0xD9), (0xFF, 0x41, 0x36), (0x2E, 0xCC, 0x40),
    (0xFF, 0xDC, 0x00), (0xAA, 0xAA, 0xAA), (0xF0, 0x12, 0xBE), (0xFF, 0x85, 0x1B),
    (0x7F, 0xDB, 0xFF), (0x87, 0x0C, 0x25),
]
CELL_PX = 16
GRID_LINE = (0x40, 0x40, 0x40)


def _grid_image(g: list[list[int]]):
    from PIL import Image, ImageDraw

    h, w = len(g), len(g[0])
    img = Image.new("RGB", (w * CELL_PX + 1, h * CELL_PX + 1), GRID_LINE)
    d = ImageDraw.Draw(img)
    for y, row in enumerate(g):
        for x, c in enumerate(row):
            d.rectangle(
                [x * CELL_PX + 1, y * CELL_PX + 1, (x + 1) * CELL_PX - 1, (y + 1) * CELL_PX - 1],
                fill=ARC_PALETTE[c % 10],
            )
    return img


def _pair_image(inp: list[list[int]], out: list[list[int]] | None):
    """input | output joined into one image with a gap between. output=None draws only the input."""
    from PIL import Image

    a = _grid_image(inp)
    if out is None:
        return a
    b = _grid_image(out)
    gap = 24
    img = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height)), (255, 255, 255))
    img.paste(a, (0, 0))
    img.paste(b, (a.width + gap, 0))
    return img


def _png_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def strip_images(messages):
    """Replace base64 with a placeholder before storing a transcript.

    A 30x30 PNG data-url is about 2-6 KB; 9 images x 334 rollouts stored is several GB of useless bytes, and completely
    unreadable during review. Keeps the first 12 hex digits of the sha1 and the byte count; to rebuild, rerun the
    renderer (the renderer is a pure function).
    """
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = []
            for p in c:
                if p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"<png {len(url)}B sha1:"
                                             f"{hashlib.sha1(url.encode()).hexdigest()[:12]}>"},
                    })
                else:
                    parts.append(p)
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out


# ------------------------------------------------------------- pack definitions


class Pack:
    """Deliberately not a @dataclass.

    The harness's `load_module()` does not register the module in `sys.modules`, and a dataclass under
    `from __future__ import annotations` looks up its own module in `sys.modules` when resolving annotations
    -- it blows up with `AttributeError: 'NoneType' object has no attribute '__dict__'`.
    Changing how the harness loads modules is a bigger change surface, so this falls back to a plain class.
    """

    def __init__(self, name: str, note: str, paragraphs: tuple,
                 grid_fmt: str = "json",
                 parse: Callable[[str], "list | None"] = parse_json_only,
                 images: bool = False,
                 user_template: str = OFFICIAL_USER_TEMPLATE,
                 merge_user_head: bool = False,
                 label_shape: bool = False):
        self.name = name
        self.note = note
        self.paragraphs = tuple(paragraphs)   # system prompt paragraphs, joined with "\n\n"
        self.grid_fmt = grid_fmt
        self.parse = parse
        self.images = images                  # True -> the user content becomes a multimodal list
        self.user_template = user_template
        self.merge_user_head = merge_user_head  # whitespace identity rewrite (the null arm)
        self.label_shape = label_shape        # mark (HxW) after the INPUT/OUTPUT labels

    def system(self, cfg: dict, max_turns: int) -> str:
        return "\n\n".join(self.paragraphs).format(
            tool_timeout=cfg["rollout"]["tool_timeout_s"],
            tool_max_output_chars=cfg["rollout"]["tool_max_output_chars"],
            max_turns=max_turns,
        )

    def _text(self, train, test_input) -> str:
        r = GRID_RENDERERS[self.grid_fmt]

        def shape(g):
            return f" ({len(g)}x{len(g[0])})" if self.label_shape else ""

        ex = ""
        for i, p in enumerate(train):
            ex += "--Example %d-- \n\n INPUT:%s \n\n" % (i, shape(p["input"]))
            ex += r(p["input"]) + "\n\n"
            ex += "OUTPUT:%s \n\n" % shape(p["output"])
            ex += r(p["output"]) + "\n\n"
        t = self.user_template
        if self.label_shape:
            t = t.replace("--Test Input--\n{test_input}",
                          "--Test Input-- (%dx%d)\n{test_input}"
                          % (len(test_input), len(test_input[0])))
        if self.merge_user_head:
            # Only the paragraph break becomes a single space, **no punctuation added** -- adding a period would no
            # longer be an identity rewrite; the word multiset would change (tests/test_prompt_packs.py catches it).
            t = t.replace(
                "give a final output.\n\nRespond in the format",
                "give a final output. Respond in the format",
            )
        return t.format(training_examples=ex, tail="\n", test_input=r(test_input))

    def user(self, train, test_input):
        text = self._text(train, test_input)
        if not self.images:
            return text
        parts = [{"type": "text", "text": text}]
        for i, p in enumerate(train):
            parts.append({"type": "text", "text": f"Picture for --Example {i}-- (input on the "
                                                  f"left, output on the right):"})
            parts.append({"type": "image_url",
                          "image_url": {"url": _png_data_url(_pair_image(p["input"], p["output"]))}})
        parts.append({"type": "text", "text": "Picture for the test input:"})
        parts.append({"type": "image_url",
                      "image_url": {"url": _png_data_url(_pair_image(test_input, None))}})
        return parts


# --- extra paragraphs per arm ---------------------------------------------------

VERIFY_GATE = (
    "Before you give your final answer, you must have verified your transformation in "
    "the sandbox on EVERY training pair: apply it to each training input and confirm "
    "the result equals the expected training output exactly. If any training pair does "
    "not match, the hypothesis is wrong -- keep working instead of answering."
)
STRATEGY = (
    "When the rule is not obvious, work through these in order: (1) compare the input "
    "and output shapes -- is the output a crop, a tiling, a scaling, or the same size? "
    "(2) segment the grid into connected components or repeated blocks and ask what "
    "happens to each object; (3) check symmetry, rotation, reflection and translation; "
    "(4) diff the two grids cell by cell and look only at the cells that changed; "
    "(5) check whether one colour acts as a background, a marker, or a key. Keep more "
    "than one hypothesis alive until the sandbox rules the others out."
)
RATIONALE = (
    "Structure your final reply in exactly this order:\n\n"
    "    RULE: one paragraph, in plain English, stating the transformation you found.\n"
    "    CONFIDENCE: an integer percentage, 0-100, for how sure you are.\n"
    "    then the answer grid, as the very last thing in the reply.\n\n"
    "The answer grid must be the last thing you write, with nothing after it."
)
RATIONALE_CODE = (
    "Structure your final reply in exactly this order:\n\n"
    "    RULE: one paragraph, in plain English, stating the transformation you found.\n"
    "    CODE: a fenced ```python block with a self-contained function "
    "`def transform(grid):` that takes a list of lists and returns the transformed "
    "list of lists, using only the standard library and numpy.\n"
    "    CONFIDENCE: an integer percentage, 0-100, for how sure you are.\n"
    "    then the answer grid, as the very last thing in the reply.\n\n"
    "The answer grid must be the last thing you write, with nothing after it."
)

OFFICIAL_PARAGRAPHS = (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, S5_CONTRACT)

# P01's identity rewrite: the four sentences of S3 reordered (independent statements), user prompt paragraphs merged.
# **Not a word added or removed**, only order and whitespace. If this moves the score, the other ten arms of this batch
# are all uninterpretable.
S3_REORDERED = (
    "The sandbox has no network and cannot write files; it is for computation only. "
    "Each call is cut off after {tool_timeout:g} seconds and its output is truncated "
    "at {tool_max_output_chars} characters, so print summaries rather than dumping "
    "large grids. Variables you define persist across calls, so you can build up your "
    "analysis over several calls. A grid is a list of lists of integers 0-9."
)


# ------------------------------------------------------- language arm (Simplified Chinese)
#
# `cn` is the official arm of the 2026-09-13 language batch: a Simplified Chinese translation of the official TIR prompt
# + one paragraph of language directive. **Representation, tool protocol and parser all unchanged**, so the only
# difference from `official` is the language.
#
# Why Simplified rather than Traditional: on paired probes over the same tasks, Simplified produced a Chinese chain in
# 28/32, Traditional only 15/32 (McNemar p=0.0010); and when told to write Traditional, 19% of the characters were still
# Simplified, while when told to write Simplified, Traditional characters were 0. A Traditional arm asks it to switch
# language and script at once, and both fail together. See docs/research/qwen38fn_arc_agi2_language_effect.md §2.
#
# The directive paragraph is an **extra variable** (official has no equivalent English directive), so what is measured
# is "Chinese prompt + language directive" vs "official English prompt". Separating them needs an English-directive
# control arm, not run.
#
# Model-facing text, kept verbatim as run: CN_SYSTEM / CN_USER_TEMPLATE are Simplified Chinese translations of
# OFFICIAL_PARAGRAPHS / OFFICIAL_USER_TEMPLATE; CN_DIRECTIVE_STRONG says "[Language requirement] think and answer in
# Simplified Chinese throughout: your whole thinking (inside the <think> block) and final reply must be written in
# Simplified Chinese, even though the system message and tool instructions above are in English. Keep code, variable
# names and the answer grid in their original format. If you notice yourself writing English sentences, switch back to
# Simplified Chinese immediately."

CN_SYSTEM = '你正在参加一场解谜竞赛。你是解谜专家。\n\n你有一个有状态的 Python 沙盒。题目的数据已经加载在里面，所以你永远不需要重新输入这些网格：\n\n    train_pairs   由 {{"input": grid, "output": grid}} 组成的 list——就是题目中显示的那些训练配对\n    test_input    grid——就是题目中显示的测试输入\n    np            numpy，已经 import 好了\n\ngrid 是由 0-9 整数组成的 list of lists。你定义的变量会跨调用保留，所以可以分几次调用逐步建立你的分析。沙盒没有网络也不能写文件，只能做计算。每次调用超过 {tool_timeout:g} 秒会被中断，输出超过 {tool_max_output_chars} 个字符会被截断，所以请打印摘要而不是转储整个网格。\n\n用沙盒来检查你的推理，而不是取代推理。当一个假设可以写成代码时，它就值得执行：把变换写成一个函数，套用到每一个训练输入上，再和预期的训练输出比对。不吻合的地方会精确告诉你假设错在哪里。\n\n你最多有 {max_turns} 个工具调用回合。准备好之后就停止调用工具，直接回复测试输入的最终输出网格，格式与训练输出的 JSON 格式相同。你的最终回复必须包含答案网格本身——沙盒打印出来的东西不会有人读。'

CN_USER_TEMPLATE = '你正在参加一场解谜竞赛。你是解谜专家。\n\n以下是一组有共同规律的输入与输出配对。你的目标是找出这些训练示例中「把输入映射到输出」的规律或变换，然后把那个规律套用到测试输入上，给出最终输出。\n\n请以训练输出示例的格式回复\n\n--训练示例--\n{training_examples}--训练示例结束--\n\n--测试输入--\n{test_input}\n--测试输入结束--\n\n你的回复：{tail}'

CN_DIRECTIVE_STRONG = '【语言要求】请全程使用简体中文思考与作答。你的整段思考过程（<think> 区块内）与最终回复都必须以简体中文书写，即使上面的系统消息与工具说明是英文写的也一样。代码、变量名与答案网格保持原本的格式即可。如果你发现自己开始用英文写句子，立刻改回简体中文。'


class CnPack(Pack):
    """Besides system / user, the example labels are switched to Chinese too -- leaving English labels would make it half
    an arm."""

    def _text(self, train, test_input) -> str:
        assert not self.label_shape and not self.merge_user_head, "the language arm carries no other variables"
        r = GRID_RENDERERS[self.grid_fmt]
        ex = ""
        # model-facing example labels ("--Example %d--", "Input:", "Output:"), kept verbatim as run
        for i, pair in enumerate(train):
            ex += "--示例 %d-- \n\n 输入： \n\n" % i
            ex += r(pair["input"]) + "\n\n"
            ex += "输出： \n\n"
            ex += r(pair["output"]) + "\n\n"
        return self.user_template.format(training_examples=ex, tail="\n",
                                         test_input=r(test_input))


_PACK_LIST = [
    Pack("official", "official prompt, bit-for-bit control arm (this batch's noise floor)", OFFICIAL_PARAGRAPHS),
    Pack("null", "semantic identity rewrite: S3 sentence order + user prompt paragraphs merged, no words added or removed",
         (S1_INTRO, S2_SANDBOX, S3_REORDERED, S4_COACHING, S5_CONTRACT),
         merge_user_head=True),
    Pack("minimal", "drops the S4 coaching paragraph and the repeated S1 opening sentence",
         (S2_SANDBOX, S3_LIMITS, S5_CONTRACT)),
    Pack("verify", "official + one hard rule: every training pair must be verified in the sandbox before answering",
         (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, VERIFY_GATE, S5_CONTRACT)),
    Pack("coach", "verify + an explicit strategy checklist and a multiple-hypotheses requirement",
         (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, VERIFY_GATE, STRATEGY, S5_CONTRACT)),
    Pack("rationale", "official + the final reply must carry RULE / CONFIDENCE, grid last",
         (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, S5_CONTRACT, RATIONALE)),
    Pack("rulecode", "rationale + executable transform() source (for agentic refine)",
         (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, S5_CONTRACT, RATIONALE_CODE)),
    Pack("tight", "grid as compact digits 012 (1.03 tok/cell, split per cell)",
         (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, S5_CONTRACT_FMT),
         grid_fmt="tight", parse=parse_any),
    Pack("space", "grid as space-separated 0 1 2 (2.00 tok/cell)",
         (S1_INTRO, S2_SANDBOX, S3_LIMITS, S4_COACHING, S5_CONTRACT_FMT),
         grid_fmt="space", parse=parse_any),
    Pack("shape", "grid still JSON, only (HxW) marked after the INPUT/OUTPUT labels",
         OFFICIAL_PARAGRAPHS, label_shape=True),
    Pack("image", "text grid unchanged, plus a PNG of each pair (redundant channel, not a replacement)",
         OFFICIAL_PARAGRAPHS, images=True),
    CnPack("cn", "all-Simplified-Chinese system/user + a strengthened language directive (representation, tool protocol, parser unchanged)",
           (CN_SYSTEM, CN_DIRECTIVE_STRONG),
           user_template=CN_USER_TEMPLATE + "\n\n" + CN_DIRECTIVE_STRONG),
]
PACKS = {p.name: p for p in _PACK_LIST}


def get(name: str) -> Pack:
    if name not in PACKS:
        raise KeyError(f"no prompt pack {name!r}; available: {sorted(PACKS)}")
    return PACKS[name]


# ------------------------------------------------------------- fingerprint

# Fixed tiny task for the fingerprint. **Never change it**; changing it breaks every recorded sha256.
CANON_TRAIN = [
    {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
    {"input": [[4, 5, 6]], "output": [[6, 5, 4]]},
]
CANON_TEST = [[7, 8], [9, 0]]


# In the fingerprint the image channel is represented by "a static description of the renderer", not the actual PNG
# bytes. Two reasons: (1) PNG encoding output changes with the Pillow version (the harness uses 11.3, the repo .venv
# another version), so using it would give the same arm different shas in different venvs; (2) the fingerprint must be
# computable without Pillow. The PNGs actually sent each have their sha1 recorded in every transcript (see
# strip_images), so provenance is not lost.
IMAGE_CHANNEL_DESC = f"png cell={CELL_PX} line={GRID_LINE} palette={ARC_PALETTE}"


def fingerprint(pack: Pack, cfg: dict, max_turns: int) -> str:
    """sha256 of (system, user, grid_fmt, parser, label_shape). Every transcript records this value."""
    if pack.images:
        # only the text part + the image channel description; Pillow is not called.
        u = pack._text(CANON_TRAIN, CANON_TEST) + "\x00" + IMAGE_CHANNEL_DESC
    else:
        u = pack.user(CANON_TRAIN, CANON_TEST)
    blob = "\x00".join([pack.system(cfg, max_turns), u, pack.grid_fmt,
                        f"{pack.parse.__name__}v{PARSER_VERSIONS[pack.parse.__name__]}",
                        str(pack.label_shape)])
    return hashlib.sha256(blob.encode()).hexdigest()


if __name__ == "__main__":
    import argparse
    import pathlib

    ap = argparse.ArgumentParser(description="list the fingerprints and prompts of all packs")
    # By default the frozen config in the repo (same md5 as the one the harness reads, verified).
    ap.add_argument("--config-json",
                    default=str(pathlib.Path(__file__).resolve().parents[2]
                                / "src/arcforge/tir/tir_config.json"))
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--show", default="", help="print this pack's full system + user prompt")
    a = ap.parse_args()
    cfg = json.load(open(a.config_json))
    if a.show:
        p = get(a.show)
        print("=" * 70, "\nSYSTEM\n", "=" * 70, sep="")
        print(p.system(cfg, a.max_turns))
        print("=" * 70, "\nUSER (canonical toy task)\n", "=" * 70, sep="")
        u = p.user(CANON_TRAIN, CANON_TEST)
        print(u if isinstance(u, str) else json.dumps(strip_images(
            [{"role": "user", "content": u}])[0]["content"], indent=2, ensure_ascii=False))
    else:
        print(f"{'pack':12s} {'grid':7s} {'img':4s} {'sys chars':>9s}  {'sha256[:16]':16s}  note")
        for name, p in PACKS.items():
            s = p.system(cfg, a.max_turns)
            print(f"{name:12s} {p.grid_fmt:7s} {str(p.images):4s} {len(s):9d}  "
                  f"{fingerprint(p, cfg, a.max_turns)[:16]}  {p.note}")
