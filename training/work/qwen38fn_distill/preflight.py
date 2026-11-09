"""Probes to run before launching. Any FAIL means do not run -- these are all of the "no error, but the data is wrong" kind.

The seven offline checks need no server; the five online ones do (`--base-url`). The offline batch takes seconds and
must be rerun after any change to pack_menu or the runner.

Why these deserve a separate checker: no way of breaking append-only raises an exception. A tool segment missing one
`\\n`, `|trim` not mirrored, `output_ids` missing the trailing EOS -- each results in "trajectories look normal, but the
trained student misbehaves at that seam", found only at evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pack_menu = _load(Path(__file__).parent / "pack_menu.py", "pack_menu")
rr = _load(Path(__file__).parent / "rollout_runner.py", "rollout_runner")

RESULTS = []


def check(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


TRAIN = [{"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
         {"input": [[4, 5, 6]], "output": [[6, 5, 4]]}]
TEST = [[7, 8], [9, 0]]


# ------------------------------------------------------------------ offline

@check("1. seam markers are all atomic single tokens")
def _markers(ctx):
    bad = []
    for m in ("<|im_start|>", "<|im_end|>", "<think>", "</think>",
              "<tool_response>", "</tool_response>"):
        n = len(ctx["tok"](m, add_special_tokens=False)["input_ids"])
        if n != 1:
            bad.append(f"{m}={n}tok")
    return (not bad), ("all 1 token" if not bad else f"not atomic: {bad}")


@check("2. P0 of all 78 settings ends with the assistant/<think> cue")
def _cue(ctx):
    bad = []
    for s in pack_menu.all_settings():
        try:
            ctx["runner"].build_prompt(s, TRAIN, TEST)
        except RuntimeError as e:
            bad.append(f"{s.mode}/{s.pack.name}: {e}")
    return (not bad), ("78/78 pass" if not bad else f"{len(bad)} broken: {bad[:2]}")


@check("3. append-only ≡ apply_chat_template (single tool call)")
def _splice_single(ctx):
    ok, detail = _splice(ctx, n_calls=1)
    return ok, detail


@check("4. append-only ≡ apply_chat_template (two tool calls in one turn)")
def _splice_multi(ctx):
    ok, detail = _splice(ctx, n_calls=2)
    return ok, detail


def _splice(ctx, n_calls):
    """Take the template render as ground truth and compare the ids we splice together.

    Multiple tool calls are tested on purpose: the template puts several results of one turn into **one** user turn
    (`<|im_start|>user` appears only once); a wrong splice would add an extra set of turn markers.
    """
    tok, runner = ctx["tok"], ctx["runner"]
    setting = pack_menu.Setting("tir", "official", "json", False, "xhigh")
    sysp = setting.system(ctx["cfg"], 120)
    user = setting.user(TRAIN, TEST)
    base = [{"role": "system", "content": sysp}, {"role": "user", "content": user}]
    calls = [{"type": "function",
              "function": {"name": "python", "arguments": {"code": f"print({i})"}}}
             for i in range(n_calls)]
    outs = [f"result {i}\n" for i in range(n_calls)]      # trailing whitespace kept on purpose, to test trim
    a0 = {"role": "assistant", "reasoning_content": "thinking", "content": "", "tool_calls": calls}
    tools = [{"role": "tool", "tool_call_id": str(i), "content": o} for i, o in enumerate(outs)]
    a1 = {"role": "assistant", "reasoning_content": "done", "content": "[[1, 2]]"}

    def render(msgs, gen):
        return tok.apply_chat_template(msgs, tools=[ctx["cfg"]["tool"]],
                                       add_generation_prompt=gen, tokenize=False,
                                       reasoning_effort="xhigh")

    ref = tok(render(base + [a0] + tools + [a1], False),
              add_special_tokens=False)["input_ids"]

    p0 = render(base, True)
    seg_a0 = render(base + [a0], False)[len(p0):]
    g0 = seg_a0[: -1] if seg_a0.endswith("<|im_end|>\n") else seg_a0
    seg_t = rr.tool_segment(outs)
    seg_a1 = render(base + [a0] + tools + [a1], False)[len(render(base + [a0] + tools, True)):]

    ids = tok(p0, add_special_tokens=False)["input_ids"]
    for txt in (g0, seg_t, seg_a1):
        ids = ids + tok(txt, add_special_tokens=False)["input_ids"]
    if ids == ref:
        return True, f"{len(ids)} tokens bit-identical"
    i = next((k for k, (x, y) in enumerate(zip(ref, ids)) if x != y), min(len(ref), len(ids)))
    return False, (f"ref={len(ref)} spliced={len(ids)} first divergence at {i}: "
                   f"{tok.decode(ref[max(0,i-8):i+8])!r} vs {tok.decode(ids[max(0,i-8):i+8])!r}")


@check("5. tool output trim matches the template")
def _trim(ctx):
    seg = rr.tool_segment(["  padded  \n\n"])
    want = ("\n<|im_start|>user\n<tool_response>\npadded\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n")
    return seg == want, ("match" if seg == want else f"got {seg!r}")


@check("6. fingerprints of all 78 settings are distinct")
def _fingerprints(ctx):
    seen = {}
    for s in pack_menu.all_settings():
        fp = s.fingerprint(ctx["cfg"], 120)
        if fp in seen:
            return False, f"{s.mode}/{s.pack.name} and {seen[fp]} share a fingerprint"
        seen[fp] = f"{s.mode}/{s.pack.name}"
    return True, f"{len(seen)} all distinct"


@check("7. all three grid representations split cell by cell, and each parser recovers the grid")
def _grids(ctx):
    tok = ctx["tok"]
    grid = [[(r * 7 + c) % 10 for c in range(30)] for r in range(30)]
    msg = []
    for fmt in ("json", "space", "tight"):
        rend = pack_menu.pp.GRID_RENDERERS[fmt]
        n = len(tok(rend(grid), add_special_tokens=False)["input_ids"])
        if n / 900 < 1.0:
            return False, f"{fmt} only {n/900:.3f} tok/cell -- several cells merged into one token"
        parse = pack_menu.pp.parse_json_only if fmt == "json" else pack_menu.pp.parse_any
        got = parse("The answer is:\n" + rend(grid))
        if got != grid:
            return False, f"{fmt} parser cannot recover the original grid"
        msg.append(f"{fmt} {n/900:.3f}")
    return True, "  ".join(msg) + " tok/cell, all parsers pass"


# ------------------------------------------------------------------ online

@check("8. /generate returns output_ids, with length equal to completion_tokens")
def _gen_basic(ctx):
    if not ctx.get("gen"):
        return None, "no server, skipped"
    d = ctx["gen"]
    n, ct = len(d["output_ids"]), d["meta_info"]["completion_tokens"]
    return n == ct, f"output_ids={n} completion_tokens={ct}"


@check("9. output_ids end with <|im_end|> (the append-only turn boundary)")
def _gen_eos(ctx):
    if not ctx.get("gen"):
        return None, "no server, skipped"
    ids = ctx["gen"]["output_ids"]
    eos = ctx["tok"].convert_tokens_to_ids("<|im_end|>")
    fr = ctx["gen"]["meta_info"].get("finish_reason") or {}
    if (fr.get("type") if isinstance(fr, dict) else fr) == "length":
        return None, "stopped on length this time, EOS not testable (rerun with a larger max_new_tokens)"
    return ids[-1] == eos, f"last token = {ids[-1]} (<|im_end|>={eos})"


@check("10. decode(output_ids) equals the text the server returns (trailing EOS allowed)")
def _gen_text(ctx):
    """The `/generate` `text` field **strips** the trailing `<|im_end|>`; `output_ids` keeps it.

    This is neither a bug nor our problem: append-only uses `output_ids`, and the turn-boundary EOS must be there (check 9
    covers that). What is verified here is "apart from that one EOS, both say the same thing" -- i.e. `output_ids` carries
    nothing `text` cannot see. DSv4 has the same kind of trap the other way round: `/v1/completions` eats the EOS, so
    multi-turn must go through `/generate`.
    """
    if not ctx.get("gen"):
        return None, "no server, skipped"
    d = ctx["gen"]
    got = ctx["tok"].decode(d["output_ids"], skip_special_tokens=False)
    for eos in ("<|im_end|>", "<|endoftext|>"):
        if got.endswith(eos):
            got = got[: -len(eos)]
            break
    return got == d["text"], ("same after stripping trailing EOS" if got == d["text"] else
                              f"diverges: {got[-60:]!r} vs {d['text'][-60:]!r}")


@check("11. KV pool holds the measured usage (worst case warned separately)")
def _kv(ctx):
    """The hard criterion uses the **measured per-rollout usage**, not the context cap.

    This used to be `conc × cap`, which assumes all rollouts sit at the 40,960 cap at once. Measured, that is not what
    happens: conc 200 ran for two hours with KV usage steady at 0.21-0.39, and the peak works out to **about 23.2k tokens
    per rollout** -- rollouts that hit the cap end immediately and are replaced by new ones, so they never accumulate.

    And the consequence of a full KV is **retract** (requests paused, re-prefilled later): throughput halves, but **no
    data is lost and no error is raised**. That is a performance problem, not a correctness problem, so a hard FAIL is the
    wrong gate. Under the old criterion any conc > pool/cap could never pass, and a gate that is always red only trains
    people to ignore it.

    So: measured usage over → FAIL (it really will retract continuously); only the worst case over → warning.
    """
    info = ctx.get("info")
    if not info:
        return None, "no server, skipped"
    pool = info.get("max_total_num_tokens")
    if not pool:
        return False, f"/get_server_info has no max_total_num_tokens: {sorted(info)[:8]}"
    conc, cap, obs = ctx["conc"], ctx["cap"], ctx["obs_ctx"]
    real, worst = conc * obs / pool, conc * cap / pool
    tail = "" if worst <= 1.0 else f"; ⚠️ worst case {worst:.2f}× will retract (throughput halves, no data lost)"
    return real <= 0.85, (f"pool={pool:,}  measured usage {conc}×{obs:,}={conc * obs:,} "
                          f"({real:.2f}×){tail}")


@check("12. MTP is active and steps match the expected value")
def _mtp(ctx):
    info = ctx.get("info")
    if not info:
        return None, "no server, skipped"
    algo = info.get("speculative_algorithm") or info.get("speculative_algo")
    steps = info.get("speculative_num_steps")
    want = ctx["spec_steps"]
    if algo is None and steps is None:
        return False, "server info shows no speculative settings (MTP not enabled?)"
    return steps == want, f"algorithm={algo} num_steps={steps} (expected {want})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config-json", default=str(_ROOT / "src/arcforge/tir/tir_config.json"))
    ap.add_argument("--base-url", default="", help="when given, also run the five online checks")
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--context-cap", type=int, default=40960)
    ap.add_argument("--spec-steps", type=int, default=3)
    ap.add_argument("--obs-ctx", type=int, default=23200,
                    help="measured average KV usage per rollout (tokens). The default comes from\n                         the peak usage 0.39 of the conc 200 batch, see check 11")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cfg = json.loads(Path(a.config_json).read_text())

    class _A:
        max_turns = 120
    runner = rr.Runner.__new__(rr.Runner)
    runner.a, runner.cfg, runner.tok = _A(), cfg, tok
    runner.tool_schema, runner.tool_name = cfg["tool"], cfg["tool_name"]

    ctx = {"tok": tok, "cfg": cfg, "runner": runner, "conc": a.concurrency,
           "cap": a.context_cap, "spec_steps": a.spec_steps,
           "obs_ctx": a.obs_ctx}

    if a.base_url:
        import httpx
        with httpx.Client(timeout=600.0) as c:
            ctx["info"] = c.get(f"{a.base_url}/get_server_info").json()
            s = pack_menu.Setting("tir", "official", "json", False, "low")
            p0 = runner.build_prompt(s, TRAIN, TEST)
            ids = tok(p0, add_special_tokens=False)["input_ids"]
            ctx["gen"] = c.post(f"{a.base_url}/generate", json={
                "input_ids": ids,
                "sampling_params": {"temperature": 1.0, "top_p": 0.95, "top_k": -1,
                                    "max_new_tokens": 6000, "skip_special_tokens": False},
            }).json()

    width = max(len(n) for n, _ in RESULTS)
    fails = 0
    for name, fn in RESULTS:
        try:
            ok, detail = fn(ctx)
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        mark = "·" if ok is None else ("✓" if ok else "✗")
        print(f"{mark} {name:<{width}}  {detail}")
        fails += int(ok is False)
    print()
    if fails:
        print(f"✗ {fails} FAIL -- do not launch")
        sys.exit(1)
    print("✓ all pass")


if __name__ == "__main__":
    main()
