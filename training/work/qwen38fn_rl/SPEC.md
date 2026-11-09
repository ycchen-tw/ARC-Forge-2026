# qwen38fn_rl spec v1 (2026-10-07)

Contracts between the components. Change this file before changing an interface. v1 scope: **synchronous rounds** (a round's rollout
finishes completely before training; the server is idle during sync), BPO, R3 force, rollout = FP8 TP4 (GPU 0–3; non-spec, or
DFlash K=4, see the end of §6), trainer = 4 ranks (GPU 4–7), toy = codex arc2 harness.

## 1. Loss: BPO (arXiv 2609.15987) or MIS

`rl.algo` (prepare_rl `--algo`, default bpo; old configs without the key = bpo) selects one; the parameter table is `rl_loss.ALGOS`.
For each generated token t (loss_mask positions), T = the sampling temperature in the batch manifest (one for the whole batch). BPO:

```
logp_t = log_softmax(z_t / T)[y_t]          trainer, fp32, with gradient
mu_t   = exp(behavior_logprob_t)            ledger (after the sampling temperature, before top-p)
pi_t   = exp(logp_t).detach()
omega  = (1 + eps - mu_t) / (1 + eps - pi_t)
M_t    = 0 if coef_t > 0 and omega > 1 + eps_high
         0 if coef_t < 0 and omega < 1 - eps_low
         1 otherwise
w_t    = M_t * min(omega, C)                all detached
loss   = - sum_t coef_t * w_t * logp_t      summed over all ranks and all bins of this optimizer step; the trainer divides by nothing else
```

Defaults eps=0.1, C=3.0, eps_low=0.2, eps_high=0.28. MIS (`--algo mis`, direct token importance sampling, two-sided mask):

```
r_t  = exp(logp_t - behavior_logprob_t)      pi / mu, logp_t as above
M_t  = 1 if mis_low <= r_t <= mis_high       judged in log space; independent of coef's sign (same bounds for positive and negative advantage)
w_t  = M_t * r_t                             all detached; 0 outside the bounds, not clamped to the bound
loss = - sum_t coef_t * w_t * logp_t         summed as in BPO
```

Defaults mis_low=0.2, mis_high=5.0. No score centering. Both share coef, advantage and reduction (below). g = the same task (all
samples of one round; one group in stream), computed before splitting into minibatches; a group whose advantages are all 0 (all
rewards equal, and no efficiency term) is dropped whole. Switchable options (prepare_rl; manifest.rl records
`adv_normalization`, `loss_reduction`, `efficiency`):

- `--adv-normalization std` (default): `A_i = (R_i - mean_g) / std_g` (population std); `none`: `A_i = R_i - mean_g` (Dr. GRPO).
- `--efficiency-beta β` (default 0 = off; only with `none`): adds an efficiency term. Compared only among episodes with R = 1 (all
  tests correct): `e_i = 1 - L_i / B` (L = the outcome's `output_tokens`, the tokens the proxy counts against the output budget; B =
  `--efficiency-budget`, the harness's `output_token_budget`; negative beyond B), `A_eff_i = β_g (e_i - mean_allcorrect e)`, 0 with fewer
  than 2 all-correct episodes. All-correct groups: `β_g = β`; mixed groups: `β_g = min(β, guard (1 - mean_g R) /
  max_allcorrect(mean e - e_i))` (`--efficiency-guard`, default 0.25): the longest all-correct episode loses at most guard times its
  task advantage and never flips sign. All-correct groups therefore also enter the batch (A is only the efficiency term).
- `--arc3-reward eff` (default `score` = R is the outcome's score; only with harness arc3 and `none`, not with `--efficiency-beta`,
  all four parameters given explicitly): `R_i = c_i (1 + w(p̂) E_i)`, c = level cleared, p̂ = in-group clear rate. With ≥ 2 cleared,
  `E_i = clip(-(2 β_a x_i + β_t y_i), ±clip)`, then re-centered among cleared games, x / y = ln actions / ln `tokens_at_clear` minus
  the mean over cleared games; otherwise E = 0. `w = 0` (p̂ ≤ w_start), else `(p̂ - w_start) / (1 - w_start)`. So
  `A_i = (c_i - p̂) + w c_i E_i`, and all-cleared groups also enter the batch. manifest.rl records `arc3_reward` (parameters, group
  counts, p̂ histogram, clipped count, the efficiency term's share of the gradient). Plan: `docs/plans/arc3_level_rl_round2_20261019.md` §2.3.
- `--arc3-reward c0` (same pairing restrictions, all six parameters given explicitly; the v3 default): E is replaced by the
  uncapped `D_i = λ_a (q_a - mean q_a) + λ_t (q_t - mean q_t)`, centered among cleared games, `q = σ(κ (mean_C ln cost - ln cost))`
  (cost = actions / `tokens_at_clear`), only with ≥ 2 cleared. `R_i = c_i (1 + s w D_i)`, with s ≤ 1 keeping each cleared R ≥
  `success_floor`. The mean R of cleared games is 1, so `A_i = (c_i - p̂) + s w c_i D_i`: cleared vs uncleared differ by a fixed 1,
  and a cleared game with poor efficiency can have a negative advantage. `arc3_reward` also records the number of groups where the
  guard kicked in (`guard_hits`) and the token term's share of the efficiency term (`token_share`).
- `--loss-reduction sequence` (default, seq-mean-token-mean): `coef_t = A_i / (E * L_i)`, i = the episode t belongs to, E = the
  number of episodes kept in this minibatch, L_i = episode i's generated tokens; `prompt`: `coef_t = A_i / (H * T_g)`, H = the
  number of groups in this minibatch, T_g = the sum of L over g's kept episodes (`minibatches[m].group_tokens`): a group's tokens
  are averaged together and every group has equal weight. Old batches without the `loss_reduction` key are sequence. The trainer
  checks `coef == A_i / denominator` row by row (`RLBatch.coef_denominator`).

Reported per optimizer step: mean / p50 / p95 / p99 of |logp − behavior_logprob|, p1 / p99 of the delta, the share with M=0 (split
by A>0 and A<0), the share of Σ|coef| the mask keeps (`coef_kept_frac`), the shares of ratio outside 0.8–1.28 / 0.5–2 / 0.2–5 and
`pi_mass_outside_*` (E_mu[r·1(outside the band)] = an estimate of the trainer's probability mass outside the band), loss, gradient
norm, token count, seconds and peak memory per rank; BPO also omega's mean / p95 and the share with omega>C, MIS also the shares
blocked at the lower / upper bound and the mean weight `is_weight_mean`; policy entropy (from 10-23): the trainer's full-vocabulary
entropy of softmax(logits / T) at each generated token (`entropy_mean` / `_p10` / `_p50` / `_p90`, and by advantage sign
`entropy_mean_adv_pos` / `_adv_neg`, computed in the same online pass as the logprob, with no extra memory) and the mean −log π of
the sampled tokens `nll_mean` (on-policy, E_mu[−log mu] is the rollout policy's entropy; the two differ only by train/serve
mismatch and lag). The same set is also given split by lag (`by_lag`) and by the token's position within its segment
(`by_position`: 0–16k / 16k–64k / 64k–128k / 128k+).

## 2. RL batch: format `tito-rl-v1` (produced by `prepare_rl`)

Reuses the `tito-mm-v1` dir of `work/qwen38fn_lora/tito_mm_data.py` (single split `train`), plus:

| File | Contents |
|---|---|
| `train/behavior_logprob.f32` | one per position; valid at loss_mask positions, NaN elsewhere |
| `train/coef.f32` | one per position; 0 outside loss_mask |
| `train/routing.i16` | as in tito-mm-v1 (required) |
| `train/routing_status.u8` | per position: 0 = R recorded (including D_same, forwarded twice at the same version; D_unknown of `unversioned`; and D_cross on servers without session_exact_resume: the first routing is used, §5), 1 = N never forwarded (only allowed at the end of a segment); kept data never has other values |
| `train/weight_version.i32` | per position: at loss_mask positions = the weight version when that token was sampled, elsewhere −1 (the whole batch −1 for `unversioned`) |

Each row of `index.jsonl` additionally has `episode` (`<run dir name>/<item id>`), `group` (classic = task id; group mode =
`<group dir name>:<tid>`, §7), `reward` (the outcome's 0/1), `advantage`, `training_reward` (the R the advantage was centered from:
shaped R for arc3 eff / c0, reward otherwise; null with `--efficiency-beta`), `gen_tokens`, `weight_version`, `weight_version_tokens`
(the episode's tokens per version), `run`, `routing_D_same` / `routing_D_unknown` / `routing_D_cross` (positions of this segment
forwarded twice and written as R, §5).
`manifest.json` additionally has `rl`: `round`, `weight_version` (a single value = string, including `unversioned`; a set = a list
sorted by version number, §8), `weight_version_tokens`, `temperature`, `algo` (bpo / mis) and its parameters (bpo: eps, cap, eps_low,
eps_high; mis: mis_low, mis_high; another algo's parameters are forbidden), `join` (classic / group), `current_version` /
`max_staleness` and the lag (V − version) histograms `staleness_loss_tokens` / `staleness_episodes` / `staleness_episodes_stale`
(streaming only, otherwise null), episode / group counts, drop counts by reason, `reforward` (positions and episodes of D_same /
D_unknown / D_cross, D_cross episodes whose rule was not enforced `D_cross_unenforced_episodes`, and kept D_same / D_unknown /
D_cross positions), `reforward_rule` (the full text of the §5 rule) and `max_d_cross_fraction` (default
`arcforge.rl_storage.MAX_D_CROSS_FRACTION`; the orchestrator uses the same definition), `minibatches` = `[{"steps": [[one bin's index
list per rank], ...]}, ...]` (one minibatch = one optimizer step, accumulating gradients micro-step by micro-step within the step;
episodes of the same group are always in the same minibatch; bin ≤ 120k tokens).

prepare must raise when: a generated token lacks a logprob; sampling parameters are not T = expected, top_p = 1, top_k off,
min_p = 0, penalty = 0; an episode whose manifest records no compaction has a segment with window ≠ 0; a token's weight_version is not
in the `--weight-version` set, or is newer than `--current-version`; classic and group-mode manifests are mixed, or a group manifest
is not one task or its sample numbers are not 0…n−1; routing states per §5; the session continuation contract is violated (an
append's `resume`, contract v2.1: `docs/plans/tito_session_ledger.md`); an extension turn has neither `resume.pos` nor
`usage.cached_tokens` (it is unknown which positions were forwarded twice); an episode's segments come from servers with
session_exact_resume on and off.
An append whose `weight_versions` is null: that episode is dropped (`weight_version_unrecorded`), and only above 0.5% per round does
it raise (§8). Under streaming, episodes with generated tokens older than V−S are dropped (`stale`, at the same gate as other
per-episode drops, before group_lt2 / std0).
Episodes where compaction happened (a thread spanning more than one window, or a compaction request) are dropped (`compacted`);
episodes whose arc2 outcome has `context_overflows` > 0 (the harness hit the context length and opened a new thread) are dropped
(`context_overflow`). An outcome without the `context_overflows` key raises. Both come before group_lt2 / std0, with no share cap.

## 3. Trainer service (`work/qwen38fn_lora/train_rl_service.py`)

- torchrun with 4 ranks. Loading, FSDP, LoRA, R3 (`routing_replay.py` force) and merged export (`merged_export.py`) are shared with
  `train_tito_mm.py`: factored into modules within the same unit, not copied.
- rank0 serves HTTP (127.0.0.1:PORT, address written to `<out>/trainer_endpoint.json`); the other ranks run a gloo command loop
  (`broadcast_object_list`). Any exception: log it and let the process die; the orchestrator sees it via health.
- Endpoints (JSON):
  - `GET /health` → `{ok, step, weight_version, busy}`
  - `POST /train {batch_dir, minibatch}` → one optimizer step (micro-step accumulation → sync → clip 1.0 → AdamW), returns the §1 metrics
  - `POST /publish {weight_version}` → `<out>/merged/v{weight_version:06d}/`, returns `{path, manifest_sha256, seconds}`
  - `POST /save` → `checkpoint-<step>` (LoRA, optimizer, step, weight_version, args)
  - `POST /stop`
- **The loss is computed outside forward**: forward returns only `logp` (fp32) at loss_mask positions, because the FSDP root module
  casts fp32 forward parameters to bf16.
- logprob kernel: chunked along rows and vocabulary, recomputed in backward, **no gradient for lm_head** (frozen). The temperature
  is applied here.
- LR constant 2e-5 (provisional, tunable), AdamW betas (0.9, 0.98), wd 0.
- Required tests: logp agrees with `train_tito_mm.py --mode score` on the same segment (fp32 tolerance); the kernel's gradient agrees
  with naive log_softmax; with omega≡1 and M≡1 the loss equals REINFORCE; MIS with r≡1 equals REINFORCE and weights outside the
  bounds are 0; after resume, LoRA / Adam state is bit-identical.

## 4. Orchestrator (`work/qwen38fn_rl/orchestrator.py`)

Each round r (synchronous):
1. Take the next 64 tasks from the shuffled training list (repeats only once it is used up).
2. Run 8 `run_arc2.py` concurrently against the rollout server (`<R>/rounds/r{r:03d}/s{k}`); any rc≠0 stops.
3. Run `python -m arcforge.tito audit` on each run; FAIL stops (known-benign cases must be whitelisted). Only two kinds pass: one extra rid from the last turn of a task that hit the budget, and splits capped by `audit.allow_splits` (toy: `text_differs` ≤ 64; the audit's `split_reasons` is cumulative over the whole server instance, not per round). Reason: at the start of an assistant turn the model occasionally samples a non-standard empty thinking (`<think>\n</think>\n\n`, or `<think>\n` followed directly by the answer; the base weights do it too), codex renders it back as the standard `<think>\n\n</think>\n\n` on resend, and the server splits the segment; each segment is consistent on its own and prepare treats them as several samples of the same episode, but each segment has to carry the whole prefix again. In the LR 1e-4 toy, rounds 0–6 added 3, 1, 0, 1, 1, 1, 1 of them, not increasing with training. The fix is the fork's `7d14dc8b7` `--session-replay-empty-think` (opt-in: swaps back to the sampled ids, the span records `kind: empty_think`); both rl/v1 and exp/rl-dflash have it, and `launch_rollout_server.sh` turns it on by default.
4. `prepare_rl` → `<R>/rounds/r{r:03d}/batch`.
5. Call `/train` for each minibatch.
6. `/publish {weight_version: r+1}`.
7. Sync: the server must be idle; `POST /update_weights_from_disk {model_path, weight_version: "r+1", flush_cache: true, keep_identity: true}`;
   verify that `/model_info` has the right weight_version and an unchanged served name; stop on failure.
8. Write one row to `<R>/rounds.jsonl`: mean reward, live-group share, output tokens, share that hit the budget, trainer metrics,
   routing state counts, time per stage.

held-out: a fixed 32 tasks × 8 samples, run at r=0 and every 5 rounds, not trained on. Before each round, check that /data has ≥120 GB
free and for the STOP file; touch only its own PIDs / units.

## 5. Routing states and thresholds (v1 synchronous)

| State | Meaning | Handling |
|---|---|---|
| R | recorded | use |
| N | never forwarded | only allowed at the end of a segment; anywhere else = stop the run |
| M | missing: ledger routing is null / `routing_error`, late split, sentinel row | drop the whole episode, counted by reason |
| X | invalid: duplicate id, out of range, all-zero row | stop the run |
| D_same | forwarded twice (recompute window), both at the same weight version | use (the first routing), write as R, count |
| D_cross | forwarded twice at different weight versions (or one side has two candidate versions) | server with session_exact_resume on: drop the whole episode and count (`routing_D_cross`); off: only count, write as R (`D_cross_unenforced_episodes`) |
| D_unknown | forwarded twice, one side has no recorded version | `--weight-version unversioned`: as D_same; in a versioned batch only allowed in episodes whose `weight_versions` is null (already dropped as `weight_version_unrecorded`), otherwise stop the run (only when the server has session_exact_resume on; off: as D_same) |

More than 0.5% of a round's episodes dropped for M: stop the run. More than `--max-d-cross-fraction` (default 2%; the orchestrator's
`prepare.max_d_cross_fraction`) dropped for D_cross: stop the run. A group with fewer than 2 left: drop the whole group. M / X take
precedence over D (that position cannot be used anyway).

**The D rules are enforced only for servers with exact resume on**: by each segment's ledger instance
(`server_args.session_exact_resume` in `server.json`; no key = old fork = off), and the segments of one episode must agree (otherwise
stop the run). Ledgers with it off work as before: every D is written as R, accepted, and only counted (the summary's
`quality.routing_D_cross` excludes them, so the orchestrator's batch-level threshold does not block them either), the same as prepare
before this rule existed: without exact resume, recompute windows are the norm (measurements in the next paragraph), and dropping them
would stop the current stream runs. Ledgers without `resume.pos` (old fork, flag off) use `cached_tokens` for reuse, which reflects
only the first admission (the part recomputed from the alignment point after a retract is invisible), so D is a **lower bound**.

**Forwarded twice** (`docs/plans/session_exact_resume.md`): each turn of a multi-turn session is a request = the previous turn's S +
tail, radix reattaches only part of the previous S (`usage.cached_tokens`, aligned to page 64 / mamba track 256), and
`[cached_tokens, prefix_len−1)` of the extension turn is prefilled again. Ledger routing is recorded from `prefix_len−1` on, so this
span keeps the routing of the **first** forward and R3 forces it, while the tokens after this turn were sampled with the KV and GDN
state of the second. At the same version the two differ only by serving's non-batch-invariance (as at any other position) → D_same is
used; across versions (an in_place sync landing between the two turns) the second was computed by another model → D_cross drops the
whole episode (dropping only the tokens after the window would also work; v1 drops the whole thing first). Version determination
(`arcforge.tito.ledger.segment_reforward`): recompute window = `[reuse, prefix_len−1)`, reuse = `resume.pos` (old fork / mode none:
`usage.cached_tokens`). Second = the prefill version of this append (`resume.prefill_version`; on old forks inferred from the labels:
every `weight_versions` span from gen position 0, empty spans included); first = the prefill version of the earlier append (position
in its tail or its prefix_len−1, same rule), or its version at decode: the step that takes gen token i as input samples token i+1, and
under overlap the label at the in_place sync step is one late, so {labels of tokens i and i+1} are taken, and it counts as same only
if both equal the second version. Forwards outside the ledger (aborted attempts, `parallel_sample_num > 1`) are invisible.
Measured (10-17 group_ready e2e tp1x2, stream + in_place sync, two replicas): recompute window per turn p50 125–130, p99 ~250, max
255; 17 / 328 segments had D_cross (1,680 positions), and the D_cross episode share of consumed batches 2 and 3 was 6.3% / 7.3%:
**without exact resume, the default 2% threshold would stop stream runs**, so the threshold is enforced only for servers with
session_exact_resume on (previous paragraph).

## 6. sglang fork (branched from fork main, all opt-in)

1. `update_weights_from_disk` gains the request field `keep_identity`: served_model_name / model_path unchanged, model_config
   untouched; on failure the server is marked broken (all requests aborted, generate refused until the next successful
   keep_identity), with no fake rollback; `/model_info` reports `weights_source`, `weight_version`, `poisoned` (bool),
   `weights_poisoned` (reason or null). `flush_cache=true` requires all ranks idle, otherwise it rejects verbatim (retryable);
   `flush_cache=false` does not require idle (the §8 in_place mid-flight sync); rejected under speculative decoding (exception from
   `exp/rl-dflash` on: DFlash, swapping only the target and leaving the drafter alone). Every update does a device sync before and
   after writing weights.
2. Ledger appends record `weight_versions` (format in `docs/plans/tito_session_ledger.md`).
3. routing sentinel: host buffer initialized to −1; a collected −1 or all-zero row → routing_error for that request (ledger routing
   null), without killing the server.
4. A failed flush returns `success=False` instead of asserting.

node1 deploys from a separate fork worktree (`$HOME/work/sglang-arc-2-rl`, branch `rl/v1`) + lock target `b200-qwen38fn-rl`, leaving the shared `$HOME/work/sglang-arc-2` untouched: `ARC_SGLANG_FORK=$HOME/work/sglang-arc-2-rl LOCK_TARGET=b200-qwen38fn-rl bash work/sglang_fork/serve_b200.sh`.
DFlash rollout uses `$HOME/work/sglang-arc-2-rl-dflash` (branch `exp/rl-dflash`) + lock target `b200-qwen38fn-rl-dflash`, via
`RL_SPEC=dflash` of `launch_rollout_server.sh`; prepare's `ARC_SGLANG_FORK` must point at the same tree (the media processor check
blocks otherwise).

## 7. CLI / API contracts (every agent writes to these; report any change)

**prepare_rl** (`work/qwen38fn_lora/prepare_rl.py`):

```
prepare_rl.py --ledger <instance dir> --manifest <tito_manifest.json> [--manifest ...] --out <new dir>
              --round N (--weight-version V[,V…]|unversioned | --current-version V --max-staleness S)
              --temperature 0.7 --tokenizer-dir <model dir>
              --minibatches K --world 4 --bin-tokens 120000 --seed S
              [--algo bpo] [--eps 0.1 --cap 3.0 --eps-low 0.2 --eps-high 0.28] | --algo mis [--mis-low 0.2 --mis-high 5.0]
              [--media-device cuda|cpu-exact|cpu-emulated [--allow-emulated-media]] [--workers 16]
              [--max-d-cross-fraction 0.02] [--defer-fraction-gates] [--error-json <new file>]
```
Success: exit 0, tito-rl-v1 written, and the last stdout line is a one-line JSON summary (episode / group / drop counts / token counts).
- join: classic = one sample run per manifest (s0…s7), group = task. group mode (streaming) = one group dir per manifest, item ids
  all `<tid>__s<k>` (same task, k = 0…n−1), group = `<dir name>:<tid>`: two groups of the same task in one batch are two groups. One
  kind per call.
- `--current-version V --max-staleness S` (streaming, mutually exclusive with `--weight-version`): allows [V−S, V]; episodes with
  older tokens are dropped as `stale`, newer than V raises; `rl.weight_version` = the versions actually present in the batch (one =
  string, several = list, as in §2).
Any fail-loud condition: nonzero exit. `--error-json`: a `ManifestError` that points to a manifest (kind `group_data`, exit 65) and
routing over `--max-gb` (kind `routing_over_max_gb`, exit 66) are written to that JSON (kind, message, manifests / groups or
routing_bytes / max_gb); everything else raises as before (the §10 rejection). `unversioned` is only for ledgers recorded before the
fork fix (e.g. the calibration round).
In arc2, the first message of each segment carries 5–13 images: media and M-RoPE need the deployed sglang processor, so use
`MEDIA_DEVICE=cuda|cpu-exact|cpu-emulated bash work/qwen38fn_lora/prepare_rl.sh <same arguments>` (sglang venv + fork; only text-only
batches can use venv-fn-train directly).
cuda = as deployed (one card); cpu-exact does not touch the GPU and reproduces the CUDA kernel's bicubic bit for bit (`cuda_bicubic_aa.py`), with frames identical to cuda, and is the B200 RL default (`configure_b200.py --prepare-media-device`; cuda on a trainer card hung the FSDP all-gather on 10-21); cpu-emulated raises on images scaled by bicubic (pixels may differ by ±1), and `--allow-emulated-media` is only for when that is known and accepted.
Extra rids are accepted only for "the last turn that hit the budget" (`arcforge.tito.join.rl_rid_join`).

**RLBatch reader** (`work/qwen38fn_lora/rl_batch.py`):

```python
b = RLBatch(path)          # checks format == "tito-rl-v1"; b.tokenizer_sha256, b.manifest, b.rl (manifest["rl"])
len(b); b.keys; b.lengths  # same semantics as TitoMMDataset
b.minibatches              # list of {"steps": steps}; steps[m][rank] = that rank's sample index list for this micro-step
b[i]                       # dict: key, input_ids(int64), loss_mask(bool), mrope_positions(int32,(3,T)), media(list),
                           #       routing(int16,(T,48,10)), routing_status(uint8,(T,)),
                           #       behavior_logprob(float32,(T,)), coef(float32,(T,)), weight_version(int32,(T,)),
                           #       plus the index.jsonl metadata (the batch-level version is in b.rl["weight_version"])
b.weight_versions          # allowed set list[str] (single element in v1)
b.window                   # streaming's (current_version, max_staleness); key absent or null = None (on open, checks all versions are inside the window)
b.pad                      # np.bool_: padding samples (copies to fill up the world, coef all 0, loss always 0); exclude from token counts and logp metrics
```

**trainer service**: `torchrun --nproc-per-node 4 work/qwen38fn_lora/train_rl_service.py --model <FP8 dir> --out <new dir>
--port P --lr 2e-5 [--init-lora <checkpoint> | --resume <checkpoint>]`, endpoints in §3.

## 8. v2: round-pipelined (the config value is still `"async"`; round gate, one step off-policy; not true async, which is §9 stream)

In synchronous rounds, rollout (GPU 0–3) and trainer (GPU 4–7) take turns idling. v2 overlaps them (config `"pipeline": "async"`,
default `"sync"` = v1):

```
rollout:  [ r0 @v0 ][ r1 @v0 ...↑v1 ...][ r2 @v1 ...↑v2 ...]
learner:            [prep r0 → train → publish v1 → sync][prep r1 → train → publish v2 → sync]
```

**Gate (the source of lag ≤ 1)**: round k's rollout waits until the server's weight_version ≥ k−1 (k=0 is v0). When the learner
trains round k the trainer is at v=k, and round k's tokens can only be v=k−1 (before the sync) or v=k (after). When training is slower
than rollout, rollout waits instead of drifting ever older.

**orchestrator**: the rollout loop and the learner loop run in parallel (two threads, each fail-loud; if either dies the other stops
and takes no further step).
- rollout round k: gate → 8 harness runs → record the ledger's `enqueued_events` at the end → hand to the learner.
- learner round k: wait for the ledger's `written_events` ≥ that value → audit → prepare (`--weight-version k-1,k`, `0` for k=0) →
  `/train` all minibatches (the trainer must be at v=k) → `/publish k+1` → sync:
  prepare runs at the same time as rollout: on the server's card (GPU 0, whose memory is static with a stable ~18 GB left), with
  `--workers 4` (measured 63 s, peak 3.6 GB, as fast as 16). Not on a trainer card: its allocator cache grows round by round
  (159→172 GB), and round 4's prepare OOMed there.
  - rollout running: `POST /pause_generation {"mode":"in_place"}` → `POST /update_weights_from_disk {model_path, weight_version:"k+1",
    flush_cache:false, keep_identity:true}` → `POST /continue_generation` **whatever the outcome**; only then judge: failed or poisoned
    stops the run.
  - server idle (rollout waiting at the gate): as in v1, `flush_cache:true`.
  - Both verify `/model_info` (version, name, poisoned, weights_source) and record the seconds from pause to continue.
- held-out: when due before round k's rollout, first wait for the server to reach v=k, then run (the same version as sync's round-k held-out, so rounds compare one to one; rollout pauses for the learner once every heldout_every rounds); the learner's sync waits while it runs, and the server version must be the same before and after held-out;
  the server is idle before held-out starts, so `POST /flush_cache` runs then (clearing old prefixes accumulated over many syncs without flush).
- Resume: only from a point where "every started round has fully finished and server version = trainer version" (the last row of
  rounds.jsonl); intermediate states are not guessed at, it raises for a human to handle.
- Each rounds.jsonl row also records: server version at rollout start and end, the second at which the sync happened, pause seconds,
  seconds rollout waited at the gate, seconds the learner waited for rollout.
- **tail overlap** (`"tail_overlap": {"remaining_episodes": N}`, absent = off): once round k has ≤ N episodes still running (done =
  the `results/<task>.json` the harness writes atomically as each task ends) and the gate is open, round k+1 starts; round k's tail
  finishes before going to the learner (groups must be complete). Reason: the toy's 512 episodes per round end together at ~420 s,
  often leaving one straggler running ~140 s longer (a tool waiting on a slow script) while the server is nearly idle. All invariants
  above hold (the flush snapshot is still taken when all of round k's harness runs have ended; audit / prepare only look at the threads
  of round k's manifests). At most two rounds at once (round k+2 needs v=k+1); by round k's sync it has ended, but "rollout is
  running" checks the slots of all rounds. No early start when round k+1 has a held-out. If either round fails, both rounds' harnesses
  are terminated. The row records `async.tail_overlap`.

**Ledger flush (fork)**: under async the ledger is always being written, so "queue empty + bytes unchanged" never comes.
`/session_ledger/status` gains `enqueued_events` and `written_events` (writer counters, monotonically increasing). The writer writes
media and routing before the event line, so the dependent data of every complete event line read is already on disk; the reader
(`arcforge.tito.ledger`) must handle a ledger "being written": a half line without a trailing newline at the end of the last events
file is ignored (allowed only at the end of the last file).

**prepare_rl**: `--weight-version` takes a comma-separated allowed set (e.g. `3,4`; a single value = v1 behavior). A token's version
comes from the append's `weight_versions` spans; not in the set = raise (the gate guarantees this cannot happen; if it does, the
pipeline has a bug); an append whose `weight_versions` is null = that episode is dropped and counted (`weight_version_unrecorded`),
stopping above 0.5% per round. `train/weight_version.i32` is written per token (−1 at non-generated positions);
`manifest.rl.weight_version` becomes a list (the sorted allowed set), plus the token count per version; each index.jsonl row records
its token count per version.

**trainer**: `/train` requires the batch's version set ⊆ {v−1, v} (v = the trainer's current version), and the report's |Δ|, masked
and ratio are also given split by "lag 0 / 1". A streaming batch (with both `rl.current_version` V and `rl.max_staleness` S) instead
requires V = v and every token's version ∈ [max(0, v−S), v], split by lag 0…S; only one of the keys, S not a non-negative integer,
V≠v, or anything outside the window raises.

**Already in the fork**: an update during an in_place pause does not take the writer lock (the `is_paused` branch of
`tokenizer_manager`); device sync before and after an update; under overlap, the step already issued before the pause samples one
extra old-weight token per request but labels it with the new version: its μ is correct (computed by the old weights), only the label
is one late.

v2's routing state S (recomputed with the new version after radix eviction, with the ledger keeping the old row) = §5's D_cross:
prepare judges it per position, drops the episode, and applies a threshold.
Acceptance: throughput (episodes / hour) higher than v1, held-out no worse than v1 at the same number of rounds, |logp − μ| not rising
with rounds.

## 9. v3: stream (true async: groups as the dispatch unit, no rounds; the agent-factory-3 design)

§8's async still has rounds: a round's 512 episodes start together and end together, the next round waits for the version gate,
held-out blocks rollout, and the server is often idle (half the syncs are idle syncs). v3 (config `"pipeline": "stream"` + a
`"stream"` section) follows agent-factory-3: generation waits for the learner only under explicit back-pressure.

```
dispatcher: [g0][g1]…[g63] launch the next as soon as any one ends (≤ max_groups_in_flight)
learner:    pool live episodes ≥ batch_live_episodes → batch b → audit → prepare → train → publish b+1 → in_place sync
```

- **group** = one task × `group_size` samples = one group-mode `run_arc2.py` (`--only <tid> --samples N --salt-prefix
  <run>/<session>/gNNNNNN`, dir `groups/gNNNNNN`). The harness config must have `cache_salt: "episode"`: every episode has its own
  prefix-cache salt, so syncs never flush; session = this start's time + random. Tasks follow the task list shuffled by (seed, epoch)
  continuously (gid → task is a pure function); the gid is written to `stream_state.json` before launch, an existing dir stops the run,
  and failures are not retried, so (run, gid) is never dispatched twice and the salt cannot hit KV from old weights.
- **dispatcher**: rc 0 → read `results/<tid>__s<k>.json` (k = 0..N−1, except games the manifest marks `status failed`: run_arc3_level
  still exits rc 0 when ≤ 25% of the games break, those games have no results file and no thread, the orchestrator records them in the
  row's `failed_episodes` and does not count them as episodes, and prepare drops each as harness_failed; none of the rest may be
  missing) → record a snapshot of the ledger's `enqueued_events` → one `groups.jsonl` row (append-only, complete only once written) →
  pool. rc≠0 or timeout (`group_timeout_seconds`: SIGTERM to it alone, SIGKILL to its process group after `terminate_grace_seconds`;
  bwrap / codex / gateway are all in the harness's session, so no child is left holding a slot; dispatch does not block meanwhile):
  quarantine that group (see below) and keep dispatching. Each group writes `groups/gNNNNNN.launch.json` at launch (pid, /proc
  starttime, launch version, backends, ledger roster), which the next orchestrator relies on for adoption (see resume). If the
  dispatcher itself fails (circuit breaker, reading results fails…), the learner dies, or on detach (SIGUSR1, or SIGTERM while the
  `<run>/DETACH` file exists; the DETACH file is deleted once used): in-flight groups **keep running** (`stream.keep_groups_on_failure`,
  default true); only cancellation (SIGTERM / SIGINT without a DETACH file) terminates the group processes it started (2026-10-24
  review: in-flight groups abandoned by one restart accounted for 42% of output tokens). **Before sending SIGTERM it writes
  `groups/gNNNNNN.terminated.json` (reason cancel / timeout / abandon)**: run_arc2 still writes results / manifest / EXIT.json rc 0
  after SIGTERM, and only this marker keeps a resume from treating it as a finished group. The unit of `launch_orchestrator.sh` uses
  `KillMode=process` (systemd does not kill groups; whether to kill is decided by the orchestrator per the rules above; a unit name
  with leftover groups stays loaded, and the launcher switches to `<unit>-N` automatically). When the orchestrator is SIGKILLed / OOMs,
  prepare / audit subprocesses become orphans and may keep writing packet dirs under batches/ (known; a resume moves the half batch
  aside). In the wind-down supply, in-flight groups count only if their age < 1.5 × this session's median completion seconds
  (stragglers do not count, otherwise the last batch cannot be filled).
- **Quarantine** (2026-10-21, `stream.quarantine`, required to run stream): run-level invariants still stop the run; a per-group anomaly
  quarantines that group, logs it loudly, and the run continues. Harness level (rc≠0, timeout): `groups/gNNNNNN`, `.out` and
  `.launch.json` move to `groups/quarantined/` (not in groups.jsonl, so a resume does not take it for a half-run group); prepare level
  (§10) and audit level (§10: if every blocking line can be attributed to a single manifest (rid join, a task with no segment, starts
  with a manifest path in the packet), bisect the packet and re-audit to find the offending group, quarantining it only if it fails on
  its own; if both halves pass at the bottom = a cross-group interaction, stop the run; ledger-level failures still stop the run):
  recorded as `abandoned` (reason quarantine). One `quarantine.jsonl` row per group (stage harness / prepare / audit, error signature =
  the skeleton with task ids / episodes / paths / numbers removed, original error, evidence path, rewards, `event` = the id of one
  quarantine occurrence), plus one `QUARANTINED` line in orchestrator.log. Circuit breaker: the `same_signature_max`-th **event** with
  the same signature within `same_signature_seconds` (naming several groups at once counts as one; the signature of an event's rows
  removes the words of every named group), or a quarantined share > `max_fraction` among the last `fraction_window_groups` finished
  groups (≥ `fraction_min_groups`): stop the run.
  Routing cleanup treats `groups/quarantined/` as moved-aside groups. wandb: `train/quarantined_groups` (cumulative),
  `train/quarantined_groups_batch`.
- **Dispatch and replicas** (2026-10-15): with `rollout.replica_urls`, each sample goes to the replica with the fewest running
  episodes at that moment (ties to the lower index; harness `--sample-backends`, the manifest's `servers` has one entry per server, and
  audit / prepare need each server's ledger). The old approach sent a whole group to one replica and counted load in groups, so slots
  freed by finished samples were invisible (4r4t g16 step 1: only 192/256 running on average). With `stream.rollout_slots` (x, a
  multiple of the replica count), the next group launches when running episodes (dispatched, results not yet written) + `group_size` ≤
  x, without waiting for a whole group to finish; `max_groups_in_flight` then only caps the number of harness processes. No replica
  exceeds x / replica count (the total before a launch is ≤ x − group_size, and each sample goes to the least-loaded one).
- **Start-up ramp** (2026-10-20, `stream.ramp_seconds`, may be absent = no ramp): after each dispatcher start (resume included), the
  cap on running episodes rises linearly from `group_size` to min(x, `max_groups_in_flight` × `group_size`), reaching it after this
  many seconds. v3 c64 filled all 256 slots at once at start-up, the episodes' contexts grew in lockstep, and KV peaked at 0.86–0.91;
  in a CPU simulation 1200 s brings the start-up peak down to around the steady-state p95, and longer brings it no lower.
- **back-pressure**: no launches while the pool's unconsumed **live** groups ≥ `max_pooled_groups`. Live only: if dead groups counted, a
  pool full of dead groups would never assemble a batch (deadlock). The config requires `max_pooled_groups × group_size ≥
  batch_live_episodes`.
- **Wind-down**: the live episodes the remaining batches need (each batch actually consumes ⌈batch_live_episodes / group_size⌉ live
  groups) = demand; the pool's live episodes + running groups × group_size × the recent live share (counting back to two batches' worth
  of live groups, or the whole run if there are not enough; no completed ones yet = running ones do not count) = supply. No launches
  while supply ≥ `end_margin` (≥ 1.0) × demand; recomputed every loop, launching again if running groups come back dead and fall short;
  once all running groups are back, supply is the pool, so the learner never starves. Each transition logs once and records
  `end_hold_seconds`. When the toy stream ended, the pool had 110 unconsumed groups left (8% of generated tokens).
- **Disk**: with less than `min_free_gb` free the dispatcher merely stops launching (running groups continue, `disk_wait_seconds` is
  recorded) and does not stop the run; on 10-08 a single stop threw away 63 running groups. Only below `stream.hard_min_free_gb` (≤
  min_free_gb) does the learner stop the run (prepare and publish write there).
- **learner**: once live episodes (episodes of groups whose rewards are not all equal; with the efficiency term on, groups that are all correct with output tokens not all equal also count) ≥ `batch_live_episodes`, take groups from the front in completion order until enough, with dead groups in between
  consumed too (their ledgers are still audited). Ledger flushed (written ≥ the largest snapshot of this batch) → audit → prepare (one
  `--manifest` per group, `--current-version V --max-staleness S`: accepts versions [V−S, V], older episodes dropped as `stale`; the
  group key is the group dir) → `/train` (trainer at V = the number of trained batches) → `/publish V+1` → **always** pause(in_place) →
  update(flush_cache:false, keep_identity) → continue whatever the outcome → verify `/model_info`.
- **held-out** (2026-10-25, `stream.heldout_every` E, default 0 = off; the old held-out keys of `tasks` still raise): a round opens
  when the server version reaches a multiple of E (v0 per `heldout_at_v0`, default true; requires `rollout_slots`): one group of
  `heldout_samples` samples per task in `tasks.heldout`, with the same harness argv / config as training (same budget), dir
  `heldout/vNNNNNN/<task>`, salt `<run>/<session>/hNNNNNN`. While a round still has undispatched groups, only those are dispatched (new
  training groups wait), under the same rollout_slots / ramp / disk rules and per-sample replica assignment; back-pressure, wind-down
  supply and max_groups_in_flight do not count it. For monitoring: an episode may span versions, and each group records its start and
  end server versions. `heldout_state.json` records rounds (launched is recorded before each group is dispatched: a restart neither
  re-dispatches nor misses one; if the version skipped a multiple (e.g. a restart at 13), the latest multiple 10 is opened right
  then, recording both nominal and actual versions); nominal version ≥ rounds does not open (the final weights are evaluated offline);
  when the learner is done, the open round is still dispatched to completion, while STOP dispatches nothing. Results go only to
  `heldout.jsonl`: one row per group (completed / failed, per-game score / stop / output tokens) and one summary row per round (pass
  rate = share with score > 0, share that hit the budget, output tokens, version span), not to groups.jsonl / pool / batch; a failed
  or timed-out group is only recorded as failed, with no quarantine and no stop. Adoption, cancel and detach work as for training
  groups (the adjacent `.launch.json` / `.terminated.json`).
  `eval_keep_every` E: merged weights whose version is a multiple of E are exempt from retention deletion and evaluated offline.
- **STOP**: no more launches; running groups finish; every batch that can still be assembled is trained; the remaining groups stay in
  groups.jsonl (unconsumed). Wind-down is the same once `rounds` batches have been trained.
- **Resume**: batches.jsonl has n rows; every `groups/gNNNNNN` is a completed group in groups.jsonl; unconsumed groups return to the
  pool (prepare drops too-old ones). `batches/` holds at most one batch not yet written, and it is b = n: if the trainer has taken a
  step from it (train metrics or publish present), it is **continued** (no flag): its groups must all be completed, unconsumed and not
  abandoned, trainer step = the previous row's `trainer_step_after` + the number of minibatches it records, and versions: not
  published = trainer n, server n; published but not synced = trainer n+1, server n; synced = both n+1; stages already done are
  skipped, and the row records `resumed`. With no such batch, trainer and server are both at n and step = the last row's.
  **Half-run group dirs** (2026-10-24, adoption): judged by their `.launch.json`: harness still alive (same pid + starttime, not a
  zombie) and ledger roster unchanged: **adopt** (the dispatcher manages it as its own, and its completion row gets `adopted: true`;
  no flag needed); a `.terminated.json` present (the previous orchestrator sent SIGTERM): if alive, wait for it to exit, then
  quarantine for timeout, move aside for cancel / abandon; already ended with `EXIT.json` rc 0 (the harness writes it on every exit
  path; missing = hard-killed = failure, not inferred from the manifest), complete results and an unchanged roster: write its
  groups.jsonl row once the server version has been reconciled; every other dead one moves to `groups/abandoned-*` automatically
  (certainly nobody is writing to it); alive but the roster changed: raise, and only `--abandon-incomplete-round` sends SIGTERM, waits
  for it to exit and moves it aside. Without `.launch.json` (started by an old orchestrator) the flag is still required. A half batch
  the trainer has not touched, or pool groups whose ledger instance has changed (the server restarted): raise; only
  `--abandon-incomplete-round` moves them aside / records them as `abandoned`. Died while /train was in flight, or anything that does
  not match: raise, for a human to handle.
- **batches.jsonl** rows: batch, version, groups (gid, task, live, start/end server versions, seconds waited in the pool), live / total
  episodes, prepare summary (staleness histogram, `stale` drop count), trainer metrics, seconds per stage (pool wait, ledger, audit,
  prepare, train, publish, sync, pause), groups running during the sync, dispatcher statistics since the previous row (launched /
  completed counts, time-averaged running groups / episodes, back-pressure, wind-down and disk-wait seconds).
- **trainer**: `/train` accepts versions [V−S, V] (the batch's `rl.current_version` / `rl.max_staleness`), not §8's {V−1, V}.
- **Routing cleanup** (`routing_gc.py`, when there is no NFS offload): when all of a shard's appends belong to groups consumed in batches.jsonl, abandoned in groups.jsonl or moved to `groups/abandoned-*`, or to held-out groups with a row in heldout.jsonl (identified by thread: manifest, run_arm manifest, proxy log), and the shard is covered by a later shard, it is deleted directly (games named by nothing are listed under `unnamed` in the report, and their shards are kept); prepare and the audit (with `-m arcforge.tito`) rebuild only the batch's own threads and never read them. Readers of the whole ledger (an audit without `--manifest-threads-only`, export) will afterwards report routing_file for these appends.

## 10. stream.group_ready (2026-10-17: train each group as soon as it is ready, so when the last group arrives only its accumulate + step remain)

§9's batch waits for the last group to enter the pool before preparing the whole batch and training: last group arrives → sync takes
about 1400 s, and 1–1.4 versions of the lag come from that wait. Config `"stream": {"group_ready": true}`; requires
`prepare.minibatches == 1` and cannot be combined with `prefetch_prepare`.

- **Taking groups**: the same set as §9 (taken from the front in completion order until there are enough live episodes); the only
  difference is when they are trained. A group is reserved for batch b as soon as it enters the pool, and while b trains, b+1 can
  already be reserved.
- **materializer** (one thread): when the trainer has no pending packet, pack all reserved, unprocessed groups into one packet
  `batches/bNNNNNN/packets/pNNN-<session>` → ledger flushed → audit → `prepare_rl --current-version b --max-staleness S
  --defer-fraction-gates` (one minibatch). The packet dir carries the session: the trainer raises on seeing the same (batch_dir,
  minibatch) a second time within one process.
- **trainer** (`train_rl_service.py`): `/begin {update_id}` → per packet `/accumulate {update_id, batch_dir, minibatch: 0}` (gradient
  multiplied by H_p: prompt = number of groups, sequence = number of episodes) → `/step {update_id, units: ΣH_p}` (gradient divided by
  units, then clip, AdamW) → `/publish b+1` → sync. `/abort {update_id}` clears gradients without stepping; `/health.update` is the
  open update_id. Wrong state (/train, /publish or /save while an update is open, mismatched id) always raises.
- **Thresholds**: prepare's share thresholds (routing M and unrecorded weight_version at 0.5% each; routing D_cross uses
  `prepare.max_d_cross_fraction`, default 2%, counting only episodes from servers with session_exact_resume on, §5) are judged on the
  sum over all packets before /step (the summary's `quality`); exceeding them → /abort, then stop the run.
- **Empty batch**: when no packet keeps any group, handled like EMPTY_ELIGIBLE; groups already reserved for b+1 go back to the front
  of the pool, and the batch is reassembled at the same version.
- **prepare rejections** (`prepare_rl --error-json`, `logs/prepare.error.json` + a dedicated exit code): `group_data` (the manifests
  a `ManifestError` points to) → those groups are quarantined (§9), and the packet's other groups are redone; the batch is not topped
  up (the sealed batch keeps the estimate from reservation time, and the missing groups are like episodes dropped by prepare; all
  quarantined = empty batch). `routing_over_max_gb` → the packet is split in half recursively until it fits, stopping the run only if a
  single group exceeds it. The rejected attempt is renamed `<packet dir>.rejected-<n>`. Other pipelines still stop.
- **Resume**: a batch that has already taken /step continues with publish / sync per §9; one that has not is moved aside like a
  prefetched batch (`prefetched-*`) and its groups return to the pool; if the trainer still has an update open, /abort first at
  start-up. Died while /step was in flight: raise, for a human to handle.
- **Records**: each batches.jsonl row also has `group_ready` (each packet's groups, units, seconds, reservation / sealing / last group
  completion times) and `seconds.last_group_to_sync`. train_log's /step rows have no minibatch, but have update_id, units,
  accumulate_seconds and packages.
- **Validation** (2026-10-17, node1 tp1x2 toy, 4 batches, orchestrator stopped after batch 1's accumulate and before sealing, then
  restarted): `probes/check_group_ready.py --resumed-batch 1` all passed; all 4 batches started accumulating before sealing,
  last_group_to_sync 51–76 s.
