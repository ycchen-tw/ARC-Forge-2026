---
name: qwen38fn_distill
area: [teacher, corpus]
status: active
machines: [B200]
headline: "Rerun the NVARC distillation corpus with Qwen3.8-Flash-Next as teacher: token-level append-only, CoT:TIR=1:3, 40,960 hard budget"
---

# qwen38fn_distill

Run teacher rollouts on **NVARC synthetic tasks** with the `Qwen3.8-Flash-Next-FP8` from [`qwen38fn_eval`](../qwen38fn_eval/) (out of the box ARC-AGI-2
partial 88.6%) to produce a distillation corpus. The comparison is the two DSv4-Flash
batches (`nvarc_distill_rollouts.md`,
`dsv4_tool_integrated_reasoning.md`).

**Corpus, pair selection, and n_train are bit-identical to the two DSv4 batches** -- `nvarc_training.jsonl.zst`
47,337 tasks, `sha256(uid:attempt)` deterministically picks 4 train + 1 test. So the variable in this unit is
**the teacher plus the sampling design**, not the tasks.

## Four differences from the DSv4 batches

| | DSv4 | Here |
|---|---|---|
| Storage | two schemas, `notool` / `tool` | **one** (flat `ids` + spans), shared by CoT/TIR |
| Representation | JSON only | **json / space / tight, all three drawn** (all split cell by cell, see below) |
| Prompt | one fixed | **78 settings drawn at random** (mode × instruction × representation × shape × effort) |
| Budget | 196k / 131k, stops only when hit | **40,960 hard cap**, prompt + generation + tool counted together, truncated beyond |

## Sampling design

Each sample is drawn deterministically from `sha256(uid:attempt)` with **an independent hash per axis** -- axes must not
be correlated, otherwise couplings like "the ones that drew rulecode all happen to be space grids" silently leak into the dataset.

- **mode** tir 75% / cot 25%
- **instruction** seven for TIR (official / rationale / rulecode / verify / coach / minimal / null),
  six for CoT (cot-official is "no system message at all", the same condition as DSv4 notool)
- **representation** json 40% / space 30% / tight 30%
- **label_shape** (HxW after INPUT/OUTPUT) 15%
- **effort** xhigh 60% / medium 30% / low 10%

`rationale` asks for RULE + CONFIDENCE, `rulecode` additionally for an executable `def transform(grid)`
-- after collection it is actually run once in a clean sandbox, recording `code_train_ok` / `code_test_match`.
The prompt ablation measured "trust the execution result of
transform()" as **+5.7pp** more accurate than "trust the answer the model reports" (27:8, p=0.0019), so this is a free
extra verifier, and since it cannot be recovered from `ids` it must be stored.

## All three representations split cell by cell

Measured on 30x30: `json` 3.000 / `space` 1.999 / `tight` 1.032 tok/cell. **The criterion is "does a cell share a
token with another cell", not the absolute tok/cell** -- `tight`'s 1.032 is one token per cell plus the amortized
newline, not compression. DSv4's tokenizer merges 3 digits into 1 token, so tight collapses there; this one does not
(the hard precondition of `grid_format` has been verified on this model).

Changing the representation drags two things along; miss either and that arm is broken: the S5 contract hardcodes
"in the same JSON format" and must become `S5_CONTRACT_FMT`, and the parser must switch from `parse_json_only` to `parse_any`.
The tight arm in `qwen38fn_eval` hit this: 0.6% parsed, 0.0% correct, which looked like a representation disaster
but was really the parser not recognizing it.

## How to run

```bash
source env/env.sh
bash work/qwen38fn_distill/serve.sh 2>&1 | tee $ARC_RUNS/qwen38fn_distill/serve.log
$HOME/work/dsv4/venv-qwen4/bin/python work/qwen38fn_distill/preflight.py \
    --tokenizer $ARC_MODELS/Qwen3.8-Flash-Next-FP8 --base-url http://127.0.0.1:30041
$HOME/work/dsv4/venv-qwen4/bin/python work/qwen38fn_distill/rollout_runner.py \
    --corpus $HOME/work/dsv4/nvarc_distill/nvarc_training.jsonl.zst \
    --out $ARC_RUNS/qwen38fn_distill/out --tokenizer $ARC_MODELS/Qwen3.8-Flash-Next-FP8 \
    --base-url http://127.0.0.1:30041 --concurrency 200
bash work/qwen38fn_distill/watch.sh &      # records only, never restarts
```

The same command resumes (it scans `(uid, attempt)` in `out/*.parquet` and skips them).

## Why `/generate` instead of the chat API

The chat API does "decode → re-render → re-tokenize" every turn, so the stored target sequence may differ from the
tokens the teacher actually emitted. **48-task smoke, measured: 9 trajectories (18.75%) differ after re-tokenising**
(28% in the DSv4 batch), and the divergence falls entirely on the assistant spans that carry the loss. Going through
`/generate` and appending `output_ids` as-is avoids this.

The twelve checks in `preflight.py` lock down the contract; four of them are of the "no error, but the data is wrong" kind:
seam-marker atomicity, append-only ≡ `apply_chat_template` (including two tool calls in one turn),
`|trim` of tool output matching the template, and `output_ids` ending with `<|im_end|>`
(the `text` field strips it, `output_ids` does not -- on the DSv4 side the same kind of trap is `/v1/completions` eating the EOS).

## When consuming this data: the chat template must be the teacher's

To reproduce the training prompt (train/serve consistency in evaluation), **`apply_chat_template` must use the
teacher's tokenizer/template**, not the student's. Measured:

| Model | Template mentions of `reasoning_effort` |
|---|---|
| `Qwen3.8-Flash-Next-FP8` (teacher) | **5** |
| `Qwen3.5-9B` | **0** |
| `Qwen3.6-35B-A3B` | **0** |

Passing `reasoning_effort="xhigh"` with the same messages, the teacher renders `Reasoning effort is set
to xhigh…`, while the student's template **lacks that sentence and raises no error** -- `apply_chat_template` silently
ignores kwargs it does not know. All three efforts collapse into the same rendering.

**Two properties that make it especially hard to notice**:

1. **`pack_sha` does not cover this layer.** It hashes `(system, user, grid_fmt, parser, label_shape)`,
   not the template rendering. So **all-correct shas with all-wrong tokens is possible**, and a downstream session
   actually hit it: 78/78 `pack_sha` matched, but 78/78 had different tokens.
2. **The assert in `rollout_runner.build_prompt()` that "P0 must end with `<|im_start|>assistant\n<think>\n`" does not
   catch it either** -- both sides end with exactly the same string.

**So consistency can only be verified at the token level**: take the first `span_role == 0` span in a shard and
compare bit for bit against your own rebuilt `tok(p0, add_special_tokens=False)["input_ids"]`, sampling one record
from each `pack_sha` group (what actually appears is exactly all 78). Downstream got 78/78 token-identical this way.

Corollary: an SFT checkpoint **must have the teacher's `chat_template.jinja` swapped in**, otherwise evaluation feeds
the model a prompt it never saw in training -- no error, no crash, the score just becomes the score of a different experiment.
(Precedent of the same kind: the `gptoss_sft` checkpoint shipped the stock template instead of the fork used in training.)

## A/B with this data: unreadable without pairing

Downstream ran an LR probe on Qwen3.5-9B (4 arms × 40 steps, same seed / same bins / same order) and measured:

| | Magnitude |
|---|---|
| **Within-arm** step-to-step ce std | **0.053** |
| Between-arm ce gap | 0.003–0.019 |
| sd of the **paired** difference | 0.0012–0.0071 |

**Step-to-step variance is an order of magnitude larger than the hyperparameter effect**, so comparing mean±sd reads as "all four arms tied", and applying "on a tie take the highest lr" picks 5e-5 -- the arm that actually came **last on 30/30 steps**. After pairing, 2e-5 is lowest on 30/30 steps, with differences in the same direction at 2.3–2.7 sd.

The variance comes from **this data's length distribution**: p50 total length is 26,609, but 34% of trajectories sit
exactly at the 40,960 cap, so the bin composition of packing varies a lot from step to step (same arm: step 32 is
0.828, step 35 is 0.736). Same seed / same bins cancels it out entirely.

**So any comparison made with a few dozen steps of ce must be paired** (same seed, same bins, same order; compare
per-step differences, not mean±sd). Precedent in the repo: the conclusion of `fp32_master_arc_eval.md`
also only holds thanks to paired arms.

(Numbers come from downstream training runs, not measured by this unit.)

## Files

| | |
|---|---|
| [`pack_menu.py`](pack_menu.py) | 78 settings, CoT paragraphs, deterministic sampling. Instruction paragraphs and grid renderers are imported from `qwen38fn_eval/prompt_packs.py`, not copied |
| [`rollout_runner.py`](rollout_runner.py) | token-level append-only runner shared by CoT/TIR; **the semaphore wraps the whole rollout**, not a single request |
| [`preflight.py`](preflight.py) | seven offline + five online checks; any FAIL means do not run |
| [`serve.sh`](serve.sh) | thin wrapper over `qwen38fn_eval/serve_qwen38fn.sh` (GPU 4-7, ctx 45056, MTP steps 3) |
| [`watch.sh`](watch.sh) | health recorder, one row every five minutes. **Records only, never restarts** |
| [`supervise.sh`](supervise.sh) | watchdog that does restart. It dares to because it goes through `serve.sh`, which before start-up checks GPU by GPU whether it belongs to someone else and refuses when it cannot tell (no fail open). Restart cap 6, stops when hit |
| [`summarize.py`](summarize.py) | yield and token statistics per mode × instruction × representation × effort |
| [`verify_spans.py`](verify_spans.py) | verifies span and loss-mask boundaries of written shards, seven checks, on a sample |
| [`relabel.py`](relabel.py) | one-off: brings `stop_reason` of the first 1,400 records back to the consistent definition (see below) |

## MTP diagnostic tools (grown while chasing the accept_len defect)

| | |
|---|---|
| `bench_mtp_kv.py` | measures accept_len of a single serving configuration. Fixed generation length (`ignore_eos`) + real corpus + parsing a byte-offset range of the server log. `--prompt-mode repeat` uses periodic input to rule out the "text is hard to predict" explanation |
| `analyze_mtp_kv_sweep.py` | solves for **per-position** acceptance rates `p₁ / p₂ / p₃` by subtracting across steps sweeps. sglang only emits a batch-level aggregate accept_len, so this is the only way to split by position |
| `run_mtp_kv_sweep.sh` | orthogonal sweep of target KV × draft KV × steps |
| `run_patch_bisect.sh` | turns off the 08-29 patches one by one via env flags, no rebuild needed |
| `run_predictability_probe.sh` | steps × {repeat, corpus} inputs, both modes run within the same server cell |

With this toolkit we found and fixed a silent defect (from the 2nd position of the draft chain onward it consumed the
wrong hidden state; steps=3 accept_len 1.998 → 3.036). Root cause and fix in
`qwen38fn_mtp_draft_hidden_bug.md`.
**That bug only affected speed, not data** -- verify is exact, and the 9,300 records already collected are clean.

## A fixed labelling defect (2026-08-29, first 1,400 records)

When generation hit `max_new_tokens`, the old version marked a record `answered_in_reasoning` as long as a grid could
be backscanned from the reasoning. That overestimates: a trajectory cut by the budget is not "finished answering"; that
grid is just something the model wrote halfway through thinking. Measured: **106 of the first 1,400 records (7.6%)**
were affected, so `fit%` was overestimated, and downstream filtering on `stop_reason` would take truncated trajectories as complete answers.

**The data itself is intact** -- `ids` / `span_*` / `parsed` / `correct` are all correct, and the truth is in the data:
in a cut-off trajectory the last assistant span does not end with `<|im_end|>` (measured: in all 316 records `len(ids)`
equals the cap exactly). So it is pure relabelling; `relabel.py` has fixed it offline, and the runner now gives
**truncation priority over "an answer can be fished out"**.

The criterion is EOS, not `len(ids)==cap`: the two are equivalent in practice, but EOS is the semantic criterion (the
turn did not end), while cap is an implementation detail that breaks if the cap changes.

## Concurrency and budget

`--concurrency 200` is **the number of live rollouts at once**, not HTTP requests. Wrapping a single call is the defect
fixed in `qwen38_eval` (prefixes evicted, hit rate 27.7% vs 94.1%). The httpx connection pool must be enlarged
accordingly -- the default is 100, beyond which requests queue silently.

Each turn's `max_new_tokens` is `40960 - len(ids)`, so the cap is clamped on the client side; the server's
`--context-length 45056` is only margin. **Lowering ctx from 262144 to 45056 grows the FP8 KV pool from 6,296,256 to
11,904,832** (the QSA index CUDA graph is allocated at a size fixed by the context cap).
Worst case 200×40,960 = 8.19M, 0.69× of the pool.

> The 14,672,000 in the docs is the number for the **NVFP4 target** (not available on this machine); do not compare against it.

## Outputs

| Location | Description |
|---|---|
| `$ARC_RUNS/qwen38fn_distill/out/` | parquet shards, one per 100 records (`.tmp` → `os.replace()`) |
| `$ARC_RUNS/qwen38fn_distill/{run,serve}.log` `health.tsv` | per-record progress, server log, health row every five minutes |

`stop_reason ∈ {answered, answered_in_reasoning, context_cap, turn_cap, malformed, prompt_too_long}`.
**Truncated trajectories are stored in full** (ids + spans intact) -- they are the only evidence of "which tasks do not
fit in 40,960"; whether to use them is decided at the render stage. `loss mask = span_role == 1`.
