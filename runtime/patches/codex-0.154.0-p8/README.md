# codex-0.154.0-p8: the arc-code driver, patch set p8

Base tag `rust-v0.154.0` (`REF=6b9826e3aa83b1a5947db50f4332cb9c65f1b340`, see `PATCHES` and `BASE`; the same base as p7).
Upstream is [openai/codex](https://github.com/openai/codex), Apache-2.0.

p8 is the p7 set in [`../codex-0.154.0/`](../codex-0.154.0/README.md) with 0001 revised and 0008 added, built on an 8×B200 node
on 2026-10-20. It ships as the Kaggle dataset `arcforge-codex-0154-p8`; which final submissions run p7 and which run p8 is listed in
[`runtime/README.md`](../../README.md).

| patch | vs p7 | purpose | default |
|---|---|---|---|
| 0001 view-video-tool | **revised** (same file name) | p7's 0001 plus a token estimate for `video_url` read from the mp4 header | `features.view_video` defaults to **off**; the estimate has no switch (upstream has none for images or audio either) |
| 0002–0007 | byte-identical | see the [p7 README](../codex-0.154.0/README.md) | unchanged |
| 0008 server-reasoning-included | **new** | declares that the usage the server reports already includes the reasoning sent back, so the context check no longer re-estimates it and adds it on top | `features.server_reasoning_included` defaults to **off**; the Kaggle and RL configs set it to `"true"` |

`PATCHES` holds the sha256 of every patch (`sha256sum -c` passes on this directory); 0002–0007 compare byte-identical with
`../codex-0.154.0/`.

## 0001: video token estimate

Upstream estimates images and audio; video items had no estimate, so `video_url` was counted as text: base64 bytes÷4,
i.e. file bytes÷3. A 450 KB lossless recording was estimated at about 150k tokens while sglang actually spends about 1.7k,
and codex compacted the moment the tool returned. All 30 compactions in the v20 eval's 560 games were caused by this;
14 of the 321 in Kaggle v25.

p8 reads the frame count and size from the mp4/mov header, without decoding, and estimates
`ceil(frames/2) × (ceil(w/32)·ceil(h/32) + 16)`. `max_frames`/`max_tokens_per_frame` are deliberately not applied, because
Kaggle's sglang charges by the container's frame count: 66 frames at max_frames=16 actually spent 33 groups. If the header
can't be read (webm, corrupt file), the raw size is used as before and a warning is printed. Against actual usage the
estimate is 1.11–2.22× on 188 clips in the v20 ledger and 1.06–1.75× on 59 Kaggle clips; it never underestimated.

## 0008: server_reasoning_included

Without it, codex adds a re-estimate of the reasoning "before the last user message" to the usage the server reports.
As soon as 0005's reminder (a user message) is inserted, about 11k extra is counted, and compaction fires about one
response after the reminder (Kaggle median 92k, not 100k); `exec resume`'s pre-turn check has the same problem. Upstream's
accounting is meant for OpenAI's own servers.

With the key on, `compact_reminder_tokens` 85000 and `auto_compact_token_limit` 100000 leave about 15k real tokens between
the reminder and compaction (at v25/v14r's growth of 1.3–2.7k per response, about 6–12 responses), and compaction fires at
a real context of about 100k. Single-level RL has no reminder and no tail resume, so there the key barely changes behavior;
it is on anyway to match Kaggle.

**Switch the binary and the key together**: under `--strict-config` p7 refuses to start when it sees this key
(`unknown configuration field features.server_reasoning_included`) instead of silently treating it as unset.

## Tests

The mock suite (`run_mock.py` in arc-forge's `work/arc_code/probes/codex_patches/`, not part of this release) gained 10
scenarios, with a `"reasoning": N` script key in `mock_responses.py` that returns a reasoning item with N characters of
encrypted_content. p7 runs the first 5 and reproduces the old behavior 5/5; p8 runs all 10 and passes 10/10.

- v1: an 823 KB recording at ctx 30k — p7 compacts, p8 doesn't. v3: at ctx 134k p8 still compacts.
- r1/r2: after 0005's reminder, with 0008 off it compacts on the next response (same as p7); with it on it waits until the
  real context passes 100k.
- p1/p2: the mid-turn check after `exec resume` (a new turn resets the flag to false), with 0008 off and on.
- q1/q2: `exec resume`'s pre-turn check when the previous turn received the reminder, with 0008 off and on.
- b0/b1: the step right after the reminder views the 823 KB recording, with 0008 off and on.

## Binary

The deployed binary has sha256 `3b4dc03403070f03c822ff776a980e9afb1e76b7a3d79cb2547ef6f14b68a882` (`SHA256SUMS`; path
`vendor/x86_64-unknown-linux-musl/bin/codex` in the dataset's install prefix), a musl static-pie. Its `--version` prints
`codex-cli 0.154.0`, the same as p7 (`62ee15a0…`), so the hash is the only way to tell them apart; the AGI-2 launcher
pins it (`CODEX_SHA256` in [`agi2/work/arc_code/arc2/kaggle_run.py`](../../../agi2/work/arc_code/arc2/kaggle_run.py)).

## Build

[`work/arc_code/build_codex_musl.sh`](../../work/arc_code/build_codex_musl.sh) is hard-wired to the p7 directory
(`PATCH_DIR=$REPO_ROOT/patches/codex-0.154.0`, no override) and applies every `[0-9][0-9][0-9][0-9]-*.patch` in it.
To build p8, run it from a copy of `runtime/` in which that directory holds the p8 set:

```bash
cp -r runtime /tmp/codex-p8-build && cd /tmp/codex-p8-build
rm -r patches/codex-0.154.0 && cp -r patches/codex-0.154.0-p8 patches/codex-0.154.0
ROOT=$HOME/work/codex-0.154.0-p8 bash work/arc_code/build_codex_musl.sh   # binary at $ROOT/codex
```

`ROOT` keeps the checkout, toolchain, target dir and output apart from a p7 build. The script's marker checks cover
0001–0007; 0008 is checked only by `git apply` succeeding. Toolchain, disk and the no-sudo setup are as in the p7 README and
the script header. Whether a rebuild reproduces the deployed hash bit for bit was not checked.

Files here: `PATCHES` (base and per-patch sha256), `SHA256SUMS` (the binary), `HANDOFF.txt` (build note).
