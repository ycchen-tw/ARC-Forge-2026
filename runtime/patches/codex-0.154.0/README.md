# codex-0.154.0: the arc-code driver

Base tag `rust-v0.154.0` (commit `6b9826e`, see `BASE`). Upstream is [openai/codex](https://github.com/openai/codex), Apache-2.0.

Build: [`work/arc_code/build_codex_musl.sh`](../../work/arc_code/build_codex_musl.sh).
It produces one **musl static binary** (about 262 MB after strip); the B200 nodes and Kaggle share the same binary.

| patch | purpose | default |
|---|---|---|
| 0001 view-video-tool | `view_video` tool; `InputModality::Video`, `FunctionCallOutputContentItem::InputVideo`, `model_input_modalities` override, `strip_video_when_unsupported` | `features.view_video` defaults to **off** |
| 0002 sglang-session-release | every `/responses` body carries `session_id` = window id; after compaction and at shutdown it sends `POST /close_session`, so sglang `--enable-session-radix-cache` releases its protection of the old prefix | `features.sglang_session_release` defaults to **off** |
| 0003 compaction-reuse-prefix | local auto-compaction sends this turn's tools (`parallel_tool_calls=true`) so that it continues the cached prefix; optionally strips the `<analysis>` scratchpad from the reply; the summary takes only the last assistant message of **this** compaction reply | `features.compaction_reuse_prefix`, `features.compaction_strip_analysis` default **off**; the summary guard is **always on** |
| 0004 external-compact-request | the harness creates `$CODEX_HOME/compact_request` (its content may be a token threshold) → at the next turn boundary that can roll over, a normal local compaction runs; the file is always consumed; below the threshold it is consumed without compacting. The arc-code v9 level-clear gate uses it (the middleware drops the file once it receives levels/Lnn.md) | `features.external_compact_request` defaults to **off** |
| 0005 pre-compact-reminder | once per context window: when the context passes the token count on the first line of `$CODEX_HOME/compact_reminder`, the rest of the text is inserted into the running turn as a user message, so the model writes notes before auto-compaction; a compaction that shortens the context re-arms it. arc-code v10 uses it (85k, auto-compaction at 100k) | `features.pre_compact_reminder` defaults to **off** |
| 0006 compaction-drops-reminder | when compaction keeps recent user messages, it drops 0005's reminder (messages identical to the text of `$CODEX_HOME/compact_reminder`). Without it, every window's reminder is kept and they pile up in front of the summary (v11 Save & Run: 84 of 87 compactions kept 1–4 copies) | follows 0005: without a reminder file nothing is dropped |
| 0007 real-token-truncation | `tool_output_tokenizer = "<tokenizer.json>"`: the tool-output token budget uses that HF tokenizer's **real token count**; above it, the first and last halves are kept and the middle is cut (at a character boundary), and the marker states the real number of tokens removed. See below | key unset: nothing changes; set but the file is unreadable / `tool_output_token_limit` missing → start-up fails |

Tests: `work/arc_code/probes/codex_patches/` —
`run_mock.py` (fake server, 28 scenarios, no GPU), `run_vision.py` (real model looking at images/videos),
`vision_suite.py` (Kaggle version, brought into the notebook by `arc_agi3/arc_code_qwen/notebook/vision_main.py`),
`vision_matrix.py` (stress matrix: long context / many images / resolution / MCP interleaved images and text; `CFG["vision_suite"]="matrix"`).

Two things on the codex side that affect measurements (0.154 source): images are always downscaled at high detail
to ≤2048 per side and ≤2500 32px patches (square images 1600², `utils/image/src/lib.rs`); MCP tools are sent as
`namespace` tools (cannot be turned off for a custom provider), and their results are an order-preserving
`function_call_output` content list with an extra `Wall time … Output:` prefix (`core/src/tools/context.rs`).

## Why musl

This machine has glibc 2.39; the Kaggle competition image has **2.35** (`kaggle_env_specs.md`).
A binary built for the gnu target fails on Kaggle with `GLIBC_2.38 not found`;
`arc_code_qwen/probes/build_kaggle_assets.py` already hit the same problem for bubblewrap.
The official release is a musl static binary; doing the same means both sides run the bit-identical binary.

**No sudo needed.** The official release workflow installs only `binutils pkg-config libcap-dev`;
this machine has the first two, and `libcap-dev` is for `--bin bwrap` — that binary cannot create namespaces on Kaggle
(`kaggle-arc3-sandbox-capabilities`), so we don't build it.
The dependency closure of `codex-cli` (1228/1344 packages) contains **no `bindgen`**, so libclang is not needed either;
`v8` is only in `code-mode-host`, which `--bin codex` never touches.

## The four parts of 0001

**The `view_video` tool** follows the shape of `view_image` (read file → data URL → content item),
without resizing — the source comment in `view_image` is explicit: *the history insertion path owns image
preparation and resizing*.

**The wire tag is `video_url`, not `input_video`.** The Responses API has no official video content
part; OpenAI-compatible servers that accept video (sglang, vLLM) recognize the chat-style `video_url`.
On the Rust side it is still called `InputVideo` and is aligned with the wire through `#[serde(rename)]`, so **this side needs no server change**.

**`model_input_modalities`.** codex cannot read the modality metadata of an OpenAI-compatible `/v1/models`
(`arms/codex.py` already has the same comment for `model_context_window`); unknown slugs always fall back to
text+image, so the video gate can never open. The point is that it must go through `ModelInfoOverrides` —
**every** model-info resolution passes through there; setting it only on `Config` is silently ineffective.
This trap only shows up end to end: `cargo check` is clean, the tool registers, the model even calls it, and the gate still blocks.

**`strip_video_when_unsupported`.** Matches the existing image/audio strippers. Without it,
after switching to a model that does not support video, old videos in the history travel all the way to the API before failing, and the error does not point at the real cause.

## 0002: release the sglang session after compaction

sglang's session radix cache reads only the `session_id` in the **body** (it ignores codex's `session-id` header);
KV tagged with an id is evicted last, and there is **no TTL**; only a client `/close_session` releases it.
The id is codex's existing window id `<thread>:<n>` (n+1 on every compaction, the same string as `x-codex-window-id`),
so each close covers exactly one window that is already dead. `/close_session` is at the server root;
base_url must end in `/v1`, otherwise opening the session fails immediately.

A failed close emits only a Warning, not an Error — the latter makes `codex exec` exit 1, while a failed close only affects cache residency.
**Limitation**: a failure of the close at shutdown is visible only on stderr, because exec unsubscribes from the event stream before shutdown.
A process killed by SIGTERM/OOM does not close; not fixed, because the eviction key is `(session_ref>0, session_ref, LRU)`:
an unclosed dead session is the oldest among "nodes with a session" and is the first to be pushed out by new sessions.

**On the sglang side, only requests that "cross a multiple of 256 during decode" are protected** (hybrid mamba, `extra_buffer`):
for one that does not cross, `mamba_last_track_seqlen` is None → `cache_len` 0 → the node is root →
`register_session_ref` silently skips it, and close logs `indexed 0`. Requests with short outputs are therefore never protected;
xhigh's long reasoning almost always crosses. This is upstream behavior, not the client sending a wrong id.

## 0003: compaction reuses the cached prefix

**Root cause**: local compaction requests carry no tools. Qwen's chat template puts the tools at the very start of the system block,
so a compaction request diverges from the conversation at about token 41, and every compaction re-prefills the whole thing.
Measured on v4: 286 compactions ≈ 24.1M tokens ≈ 63% of prefill.

**Fix** (`compaction_reuse_prefix`): inline auto-compaction sends the same tools as the interrupted turn and
`parallel_tool_calls=true`; `tool_choice` stays auto (none would make sglang drop the tools from the template).
Tool calls in the reply are never executed and are always discarded. Manual `/compact` is unaffected.

**Old bug (guard always on)**: the summary used to be "the last assistant message in the whole history".
If the compaction reply had no assistant message, an **earlier** message (or an empty string) became the summary and replaced the whole history.
Now only the last one of this reply is taken; if there is none, it is a retryable error and the same request is resent; once retries are exhausted (`stream_max_retries`, default 5) it emits an Error and the turn fails.

**`compaction_strip_analysis`**: before anything is stored, every `<analysis>…</analysis>` is stripped from the reply
(an unclosed one is stripped to the end). If nothing remains after stripping, that also counts as a failure and is retried.

## 0007: truncate tool output by real tokens

**Root cause**: `qwen38fn` is not in codex's model list and falls back to `bytes(10_000)`; `tool_output_token_limit=2500`
becomes 2500×4 bytes in bytes mode, and tokens mode is just bytes÷4. Board hex is about 1 byte/token, so about 10k tokens actually get in.

**Fix**: when `tool_output_tokenizer` is set, the tokenizer is installed process-wide in `codex-utils-output-truncation` (loaded once),
the limit always becomes a tokens policy, and every `Tokens` budget (exec/shell/MCP/code mode output, history records,
unified exec's `Original token count` and the response-length loop) uses the real count. Upstream conventions are kept:
the truncation marker and the `Warning: truncated output…` header are **not counted** against the budget, and MCP output and history take 1.2×.
Side effect: compaction's kept user messages, realtime, etc. also go through the same `truncate_text`, so with the key set they also switch to real tokens.

The `tokenizers` crate is pinned at `=0.23.2` with only `onig` enabled (the Python version uses the same regex engine; syntect already compiles onig into this binary).
Verification: Rust vs Python `tokenizers`/`transformers` on 215 strings (boards, Chinese/English/Japanese, emoji, special-token literals, fuzz)
gives identical ids one by one; both truncated halves stay within budget and the removed count is exact. mock (`run_mock.py` t30–t33, measured): three 64×64 boards,
11,076 tokens → 1,250+1,250 kept, the whole function_call_output 2,578 tokens (8,949 without the key); without the key the wire is bit-identical to p6.
Bazel's lock was not touched (we only use cargo build). Upstream's `config_schema_matches_fixture` has failed since 0001; not fixed.

## This alone is not enough

The tools put media into `function_call_output`, and sglang **drops the media** there —
which is why `view_image` was always broken. It only works together with
[`patches/sglang-0.5.20/0018-responses-tool-output-media.patch`](../sglang-0.5.20/0018-responses-tool-output-media.patch).
Verification record:
`docs/research/codex_video_input_20260921.md`.
