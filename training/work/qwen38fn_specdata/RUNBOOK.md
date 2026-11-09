# Operations and tools

The 8k batch is fully collected. New runs use a new tag and do not overwrite old manifests. `--exclude-runs` skips UIDs
already drawn.

## Rerunning

First check `nvidia-smi` per the shared-GPU rules. All outputs use a new tag and do not overwrite old manifests.

```bash
source env/env.sh
export MODEL=<Qwen3.8-Flash-Next-FP8 directory> TAG=<YYYYMMDD-tag>
export PYTHON=$HOME/work/dsv4/venv-qwen4/bin/python
export RUN=$ARC_RUNS/qwen38fn_specdata/$TAG
$PYTHON work/qwen38fn_specdata/generate.py prepare --out "$RUN" --limit-per-split 22223 \
  --exclude-runs "$ARC_RUNS/qwen38fn_specdata/20260907-nemotron-100k-v1"
GPUS=0,1,2,3 PORT=30470 bash work/qwen38fn_specdata/serve.sh
# in another shell with the same environment variables
$PYTHON work/qwen38fn_specdata/generate.py generate \
  --out "$RUN" --model "$MODEL" --server http://127.0.0.1:${PORT:-30471} \
  --max-new-tokens 8192
```

`prepare` needs HF gated access and supports a local HF token (including `~/.hf/token`) and
`--local-data <parquet directory>`. Each of the nine official splits takes the next batch of unique prompts in file order
(old batches 11,112 each; the new 8k batch 22,223 each, excluding old UIDs), not a uniform random sample of the whole
dataset. The full prefix up to each user turn is a prompt, without the answer to be generated; earlier assistant turns
are masked context, and sources and licenses are in SQLite `sources`.
Each prompt gets one effort chosen with equal weight by UID hash; only `--all-efforts` runs all three.
The seed controls only the effort assignment and does not guarantee sampling reproducibility.

## Generation contract

Official `apply_chat_template(tokenize=True, return_dict=False, reasoning_effort=...)`;
only low / medium / xhigh accepted, and the assistant prefix ends with `<think>\n`.
`/generate` is token-in/token-out; outputs are not decoded and re-tokenized.
max prompt+output 40,960, default output 4,096 (rerunning the old batch must state 32,768 explicitly);
T=0.8, top_p=1, top_k=-1. EOS per generation_config: 248046, 248044.
train/validation split about 99:1 by UID hash; the same prompt never crosses splits.
Atomic Parquet shards + a single-writer lock: resumable, but refuses to change the manifest.
`contract-errors/` keeps the rejected raw responses; it stops safely when disk falls below 5 GiB.

The server defaults to GPU4–7, changeable with `GPUS` / `PORT` (the new 8k batch: GPU0–3, :30470);
TP4/EP4, conc300, NEXTN3/4, FP8 KV, ReplaySSM fp32;
start-up verifies the PLE patch.

## Tool boundaries

| Tool | Purpose / limits |
|---|---|
| `supervise.py` | claims the server, recovers after 15 minutes without progress, at most six retries; stops the server when done |
| `prepare_continuation.py` | a stopped partial run → a new batch of uncollected prompts; the old manifest is not changed |
| `audit.py` | metadata of all shards + a 64-row token/template spot check by default; not a full token verification |
| `inventory.py --runs <batch...> --out <new dir>` | freezes batch metadata, cross-batch UID dedup, Parquet MD5; not a full quality check; only for one effort per prompt |
| `finalize.py` | full per-row verification of a single batch, atomic JSONL export; does not support merging this round's partial + continuation with different output caps |
| `prepare_backfill.py` | backfill of long prompts for a finished batch; cancelled by the user this time, not run |
| `preflight.py` / `probe_ple_commit.py` | diagnostics and PLE state regression; not data-generation entry points |

`finalize.py --supplement` is only for a long-context backfill with the same sampling / same output cap;
`--require-all` allows no missing prompts. JSONL without `complete.json` must not be used.
This time the unfinished exporter was stopped, with its scratch kept in the short batch's
`specforge-incomplete-session-close/`; Parquet is the official deliverable, and no fake complete-export marker was
created.
