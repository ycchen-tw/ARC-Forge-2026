"""RL trainer service for the Qwen3.8-Flash-Next LoRA (work/qwen38fn_rl/SPEC.md §1, §3, §7).

``--trainable full`` trains the same projections as FP32 full weights instead of a LoRA (dense_full_rl.py: FSDP-sharded
masters, the 144 block-FP8 ones through QAD's dense_full_qat, ``--freeze-first-layers``); /publish then writes the
trained weights in the same merged v2 layout (MergedExporter full mode) and /save a torch.distributed.checkpoint
(``checkpoint-<step>/dcp``, optionally a link into ``--checkpoint-store``) with a small trainer_state.pt.

``torchrun --nproc-per-node 4 train_rl_service.py --model <FP8 dir> --out <new dir> --port P --lr 2e-5
[--init-lora <checkpoint> | --resume <checkpoint> [--resume-existing-out]]`` (or train_rl_service.sh).
Explicit existing-output resume preserves original config/code and appends successful resume provenance.
Model, LoRA, FSDP, routing replay
(always ``force``) and merged export are train_tito_mm.load_trainer; the batch is a tito-rl-v1 directory read with
rl_batch.RLBatch.

Rank 0 serves stdlib HTTP on 127.0.0.1:P (bound at start, so a taken port fails before loading; the address goes to
``<out>/trainer_endpoint.json`` once the model is ready) and hands one command at a time to its main thread, which
broadcasts it to every rank over a gloo group (``broadcast_object_list``); all ranks execute it. Any exception is
logged (rank 0 also answers the pending request with 500 and the traceback) and kills the process; the orchestrator
sees the dead /health.

- ``GET /health`` -> ``{ok, step, weight_version, busy, update}`` (ok is false while loading; update = the open
  update_id or null)
- ``POST /train {batch_dir, minibatch}``: one optimizer step. Every micro-step of the minibatch runs one bin per
  rank: forward (rl_forward: the packed model and the fp32 log-probability of every supervised token under
  logits / T, T = the batch's sampling temperature) -> the batch's loss (``rl.algo``: BPO or MIS,
  rl_loss.algo_loss) outside the forward (the FSDP root casts float forward inputs to bf16, so behavior
  log-probabilities and coefficients never enter it) -> backward, accumulating
  into the LoRA gradients. Then sync_gradients (sum over ranks), clip 1.0, AdamW. No division by token counts:
  coef = A/(E*L) (sequence) or A/(H*T_g) (prompt) already normalizes. Returns the SPEC §1 report, per-rank
  seconds (micro_seconds; prep_seconds = host preparation before each micro-step) and peak memory, routing agreement.
  ``--host-prefetch`` (default on) reads micro-step k+1's rows and RL targets on a host thread during micro-step k.
- Group-ready updates split one optimizer step over several requests (the packages arrive as their groups finish):
  ``POST /begin {update_id}`` zeroes the gradients and opens the update; ``POST /accumulate {update_id, batch_dir,
  minibatch}`` runs one package (any tito-rl-v1 minibatch; its coef divides by its own H_p groups, or E_p episodes
  for sequence reduction) like /train but only adds H_p times its gradient; ``POST /step {update_id, units}`` needs
  units == the sum of H_p, divides the gradients by it (= the same rows as one minibatch), then sync, clip, AdamW
  and the /train report over all packages (``packages`` lists them; no batch_dir / minibatch / round);
  ``POST /abort {update_id}`` drops the gradients without a step. While an update is open /begin, /train,
  /publish and /save raise; packages must share loss reduction, temperature, loss algorithm and its parameters; a (batch_dir,
  minibatch) is trained at most once per process, aborted or not.
- ``POST /publish {weight_version}``: merged LoRA projections -> ``<out>/merged/v{weight_version:06d}/`` (DONE last;
  merged_export.py, whose manifest ``step`` field holds the weight version; the optimizer step is
  ``optimizer_step``) -> ``{path, manifest_sha256, seconds, ...}``. Versions must increase.
- ``POST /save`` -> ``<out>/checkpoint-<step>/`` (PEFT adapter + trainer_state.pt: LoRA, AdamW state, step,
  weight_version, args), written to ``.tmp`` and renamed. Its trainer_state.pt also seeds train_tito_mm --init-lora.
- ``POST /stop``.

``weight_version`` is the version of the trainer's last /publish: 0 at a fresh start, restored by --resume. /train
requires the batch's versions (``rl.weight_version``: one string in v1, a sorted list in the async v2 of SPEC §8) to be
within {v-1, v}; a batch with several versions must carry the per-token ``train/weight_version.i32``. A batch recorded
before the fork wrote versions says ``unversioned`` and is accepted with a log line. A streaming batch (SPEC §7:
``rl.current_version`` V and ``rl.max_staleness`` S, both set) instead needs V == v and every token version in
[max(0, V - S), V], per token. The report splits |delta|, masks and ratios by the tokens' lag (0 / 1, or 0..S) and
by their context position (``CONTEXT_BUCKETS``: the token's position in its segment).

``--dump-logp`` (debug): every rank writes the per-token logp, behavior log-probability and coefficient of each
micro-step to ``<out>/logp/step-<step>/`` (probes/rl_logp_parity.py compares them with --mode score).
``--dump-grads`` (debug): rank 0 writes the summed pre-clip gradient and the stepped LoRA of every optimizer step to
``<out>/grads/step-<step>.pt`` (probes/rl_accumulate_parity.py compares /train with /begin../step).
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MethodType

import fp8_dense_qat
import hang_watch
import numpy as np
import sparse_qsa
import torch
import torch.distributed as dist
import trainer_kernels
from rl_batch import staleness_window, window_low
from rl_loss import ALGOS, algo_loss, check_algo_params, selective_logprob, summarize_terms, support_logprob
from train_tito_mm import (
    add_lora_target_arguments,
    assert_step_kernels,
    build_batch,
    check_routing_spec,
    host_rss_gib,
    load_trainer,
    qsa_backend,
    resume_conflicts,
    supervised_hidden,
)

FORMAT = 'fn-rl-trainer-v1'
FORMAT_FULL = 'fn-rl-trainer-full-v1'   # --trainable full: trainer_state.pt without weights, the DCP in checkpoint-<step>/dcp
# Bookkeeping a resumed run may change; everything else shapes the update and must repeat exactly.
RESUME_MAY_DIFFER = frozenset({'out', 'resume', 'init_lora', 'port', 'audit_steps', 'offload_checkpoint_inputs',
                               'offload_mode', 'offload_keep_gib', 'host_prefetch', 'dump_logp', 'dump_grads', 'resume_existing_out',
                               'dense_qat_reference', 'hang_dump_seconds', 'hang_cuda_gdb', 'checkpoint_store',
                               'keep_checkpoints', 'resume_new_lr'})
# Settings added after runs had been saved: a checkpoint without the key was trained with this value.
ADDED_SETTINGS = {'resume_existing_out': False, 'ple_codec': 'fp8', 'dense_qat': 'none', 'fast_kernels': 'none',
                  'lora_shared_expert': False, 'adam_eps': 1e-8, 'int3_act': 'fp8', 'dense_qat_gemm': 'triton',
                  'trainable': 'lora', 'freeze_first_layers': 0, 'full_grad_kernel': 'quack',
                  'full_grad_precision': 'bf16'}
ADDED_SETTINGS.update(sparse_qsa.ADDED_SETTINGS)  # --qsa-kv-fp8, --qsa-indexer
QSA_KERNELS = ('sparse_flex_calls', 'sparse_block_calls', 'sparse_gather_calls', 'fa4_calls')
# Ledger log-probabilities may round a hair above 0 (same bound as rl_batch.py); anything larger is a broken record.
BEHAVIOR_LOGPROB_MAX = 1e-5
# Report buckets of a generated token's position in its segment (= the context it was sampled after): (label, low, high)
CONTEXT_BUCKETS = (('0-16k', 0, 16_384), ('16k-64k', 16_384, 65_536), ('64k-128k', 65_536, 131_072),
                   ('128k+', 131_072, None))


def rl_forward(self, input_ids=None, labels=None, position_ids=None, packed_meta=None, media=None,
               attention_mask=None, inputs_embeds=None, ple_input_ids=None, temperature=None, supervised=None,
               support=None, **kwargs):
    """train_tito_mm's packed model pass, then ``{'logp': (N,) fp32, 'entropy': (N,) fp32 detached, 'valid'}``:
    log_softmax(logits / T) of every supervised label in pack order and the full-vocabulary entropy of that softmax at
    the same positions (rl_loss.selective_logprob; the frozen lm_head gets no gradient). ``supervised``: the host-side
    count of supervised labels (supervised_hidden selects them without a host sync).

    ``support`` ((N, K) int64 on the device, -1 padded; a ``replay: mask`` batch, SPEC §1): ``logp`` is the
    log_softmax restricted to each label's sampling support (rl_loss.support_logprob, the gradient's path), and the
    full-vocabulary pass runs without a graph for the report only, both on fp32 GEMM outputs (two bf16 roundings of one
    logit differ by a bf16 step, up to 0.25 at |z| > 32, which put log P_pi(S) above 0): ``logp_full`` (its log-probability; log P_pi(S) =
    logp_full - logp), ``entropy`` (full vocabulary, as without replay) and ``entropy_support`` (of the restricted
    softmax)."""
    shifted, wanted, valid = supervised_hidden(self, input_ids, labels, position_ids, packed_meta, media,
                                               attention_mask, inputs_embeds, ple_input_ids, kwargs,
                                               supervised=supervised)
    if support is None:
        logp, entropy = selective_logprob(shifted, self.lm_head.weight, wanted, temperature, with_entropy=True)
        return {'logp': logp, 'entropy': entropy, 'valid': valid}
    with torch.no_grad():
        logp_full, entropy = selective_logprob(shifted.detach(), self.lm_head.weight, wanted, temperature,
                                               with_entropy=True, fp32_logits=True)
    logp, entropy_support = support_logprob(shifted, self.lm_head.weight, wanted, support, temperature)
    return {'logp': logp, 'entropy': entropy, 'valid': valid, 'logp_full': logp_full,
            'entropy_support': entropy_support}


def rl_support(rows):
    """A ``replay: mask`` bin's sampling supports in pack order (rl_forward's order): ``(support, behavior_full,
    lens)`` -- ``support`` (N, K) int64, each supervised token's kept ids padded with -1 (K = the bin's largest
    support), ``behavior_full`` (N,) the ledger's full-vocabulary logprob (report only), ``lens`` (N,) int32. RLBatch
    has checked every token is in its own support."""
    lens = np.concatenate([row['support_len'][row['loss_mask']] for row in rows]).astype(np.int64)
    ids = np.concatenate([row['support_ids'] for row in rows]).astype(np.int64)
    full = np.concatenate([row['behavior_logprob_full'][row['loss_mask']] for row in rows]).astype(np.float32)
    if len(ids) != int(lens.sum()) or not len(lens) or int(lens.min()) < 1:
        raise ValueError(f'{len(ids)} support ids for support sizes summing to {int(lens.sum())} '
                         f'(min {int(lens.min()) if len(lens) else None})')
    support = np.full((len(lens), int(lens.max())), -1, np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    col = np.arange(len(ids)) - np.repeat(starts, lens)
    support[np.repeat(np.arange(len(lens)), lens), col] = ids
    return support, full, lens.astype(np.int32)


def rl_targets(rows, num_experts, denominator, versions):
    """Check the RL fields of each row and return, in pack order (the order of rl_forward's logp), the behavior
    log-probabilities and coefficients at the supervised positions, whether each such token is real (False on pad
    rows: copies that fill a rank's bin with coef 0, left out of every metric), its lag behind the trainer's weights
    (0 or 1, or 0..S for a streaming batch; -1 = unversioned), and the supervised count per row.
    ``denominator(row)`` is a real row's coef denominator (RLBatch.coef_denominator: E * gen_tokens for a
    sequence-mean batch, H * T_group for a prompt-mean one): its coef must be advantage / denominator(row).
    ``versions`` = batch_versions(): {'trainer': int, 'batch': rl.weight_version, 'per_token': whether rows carry the
    int32 ``weight_version`` array (SPEC §8; -1 off the loss positions; without it the batch version is one scalar),
    'max_staleness': S of a streaming batch, else None}."""
    behavior, coef, real, lag, counts = [], [], [], [], []
    for row in rows:
        key, mask, n = row['key'], row['loss_mask'], len(row['input_ids'])
        b, c, status, routing = row['behavior_logprob'], row['coef'], row['routing_status'], row['routing']
        if b.dtype != np.float32 or c.dtype != np.float32 or b.shape != (n,) or c.shape != (n,):
            raise TypeError(f'{key}: behavior_logprob {b.dtype}{b.shape}, coef {c.dtype}{c.shape}, {n} tokens')
        if mask[0]:
            raise ValueError(f'{key}: the first token of a segment is never supervised (pack_segments)')
        if not mask.any():
            raise ValueError(f'{key}: no generated tokens')
        bm, cm = b[mask], c[mask]
        if not np.isfinite(bm).all() or (bm > BEHAVIOR_LOGPROB_MAX).any() or not np.isnan(b[~mask]).all():
            raise ValueError(f'{key}: behavior_logprob must be finite <= 0 on generated tokens and NaN elsewhere')
        if not np.isfinite(cm).all() or (c[~mask] != 0).any() or (cm != cm[0]).any():
            raise ValueError(f'{key}: coef must be one finite value on generated tokens and 0 elsewhere')
        if row['pad']:
            if cm[0] != 0:
                raise ValueError(f'{key}: a pad row must have coef 0')
        elif not np.isclose(cm[0], row['advantage'] / denominator(row), rtol=1e-6, atol=0):
            raise ValueError(f'{key}: coef {cm[0]} != advantage {row["advantage"]} / {denominator(row)}')
        if status.dtype != np.uint8 or status.shape != (n,) or not np.isin(status, (0, 1)).all():
            raise ValueError(f'{key}: routing_status must be uint8 R=0/N=1 per token')
        never = np.flatnonzero(status == 1)
        if len(never) and never[0] != n - len(never):
            raise ValueError(f'{key}: never-forwarded (N) positions only at the segment end, got {never[:4]}')
        if routing.dtype != np.int16 or routing.shape[0] != n:
            raise TypeError(f'{key}: routing {routing.dtype}{routing.shape} for {n} tokens')
        if not ((routing >= 0).all(axis=(1, 2)) == (status == 0)).all() \
                or not ((routing == -1).all(axis=(1, 2)) == (status == 1)).all():
            raise ValueError(f'{key}: routing rows disagree with routing_status (R = full row, N = all -1)')
        if int(routing.max()) >= num_experts:
            raise ValueError(f'{key}: expert id {int(routing.max())} >= {num_experts}')
        behavior.append(bm)
        coef.append(cm)
        real.append(np.full(len(bm), not row['pad']))
        lag.append(token_lag(row, mask, versions))
        counts.append(int(mask.sum()))
    return np.concatenate(behavior), np.concatenate(coef), np.concatenate(real), np.concatenate(lag), counts


def terms_report(algo, params, every):
    """SPEC §1 report of one optimizer step from its concatenated per-token ``terms`` (all ranks; ``lag`` and ``pos``
    besides the loss's): the whole step, then ``by_lag`` (``unversioned`` for lag -1) and ``by_position``
    (``CONTEXT_BUCKETS``, empty buckets left out)."""

    def summarize(sel):
        return summarize_terms(algo, params, {key: value[sel] for key, value in every.items()})

    by_lag = {'unversioned' if value < 0 else str(value): summarize(every['lag'] == value)
              for value in sorted(set(every['lag'].tolist()))}
    by_position = {}
    for label, low, high in CONTEXT_BUCKETS:
        sel = (every['pos'] >= low) if high is None else (every['pos'] >= low) & (every['pos'] < high)
        if sel.any():
            by_position[label] = summarize(sel)
    return {**summarize_terms(algo, params, every), 'by_lag': by_lag, 'by_position': by_position}


def token_lag(row, mask, versions):
    """Per generated token: trainer version minus the version that sampled it (0 or 1; a streaming batch 0..S, i.e. the
    version within [max(0, trainer - S), trainer]), -1 for an unversioned batch."""
    key, n, trainer, batch = row['key'], len(mask), versions['trainer'], versions['batch']
    if not versions['per_token']:
        if not isinstance(batch, str):
            raise TypeError(f'{key}: a multi-version batch needs the per-token weight_version array')
        return np.full(int(mask.sum()), -1 if batch == 'unversioned' else trainer - int(batch), dtype=np.int32)
    token = row['weight_version']
    if not isinstance(token, np.ndarray) or token.dtype != np.int32 or token.shape != (n,):
        raise TypeError(f'{key}: per-token weight_version must be int32 ({n},)')
    if (token[~mask] != -1).any():
        raise ValueError(f'{key}: per-token weight_version off the generated tokens must be -1')
    if batch == 'unversioned':
        if (token != -1).any():
            raise ValueError(f'{key}: an unversioned batch carries versions')
        return np.full(int(mask.sum()), -1, dtype=np.int32)
    lag = trainer - token[mask]
    staleness = versions['max_staleness']
    if staleness is None:
        if not np.isin(lag, (0, 1)).all():
            raise ValueError(f'{key}: token versions {sorted(set(token[mask].tolist()))} outside '
                             f'{{{trainer - 1}, {trainer}}} (trainer at {trainer})')
    elif not ((lag >= 0) & (token[mask] >= window_low((trainer, staleness)))).all():
        raise ValueError(f'{key}: token versions {sorted(set(token[mask].tolist()))} outside the staleness window '
                         f'[{window_low((trainer, staleness))}, {trainer}] (trainer at {trainer}, S={staleness})')
    return lag


def check_rl_batch(b, tokenizer_sha, replay, text_config, where):
    """Batch-level contract of a tito-rl-v1 directory against this trainer; returns its sampling temperature, loss
    algorithm and parameters (``rl.algo``, the values of ``rl_loss.ALGOS[algo]`` in order), round and weight_version,
    and ``sampling_replay`` (``rl.replay``: ``mask`` for top-k / top-p sampling, whose loss replays the supports;
    ``none`` otherwise, also for batches written before the key)."""
    if b.tokenizer_sha256 != tokenizer_sha:
        raise RuntimeError(f'{where}: batch tokenizer does not match the checkpoint')
    processor = b.manifest['processor']
    if processor is None:
        # prepare_rl records no processor for a batch without media (arc3 v8b b7: one group, 16 text-only episodes)
        if b.manifest['media']['unique'] != 0:
            raise RuntimeError(f'{where}: {b.manifest["media"]["unique"]} media but no media processor recorded')
    elif processor['device'] not in ('cuda', 'cpu-exact'):
        raise RuntimeError(f'{where}: media processor device {processor["device"]!r}, not as deployed '
                           '(cuda, or cpu-exact: its bit-exact CPU replay); the CPU emulation differs by +-1 u8')
    check_routing_spec(b.manifest['routing'], replay, text_config, where)
    rl = b.rl
    algo = rl['algo']
    if algo not in ALGOS:
        raise ValueError(f'{where}: algo {algo!r}, this trainer implements {sorted(ALGOS)}')
    temperature = float(rl['temperature'])
    if not temperature > 0:
        raise ValueError(f'{where}: temperature {rl["temperature"]!r}')
    params = tuple(float(rl[k]) for k in ALGOS[algo])
    check_algo_params(algo, params)
    version = rl['weight_version']  # v1: one string (or 'unversioned'); v2: sorted list of version strings (SPEC §8)
    if isinstance(version, str):
        listed = [version]
    elif isinstance(version, list) and len(version) >= 2 and 'unversioned' not in version:
        listed = [str(v) for v in version]
    else:
        raise ValueError(f'{where}: rl.weight_version {version!r}')
    per_token = (Path(b.data_dir) / 'train' / 'weight_version.i32').exists()
    if not per_token and not isinstance(version, str):
        raise ValueError(f'{where}: a multi-version batch needs train/weight_version.i32')
    window = staleness_window(rl, where)  # (V, S) of a streaming batch (SPEC §7), else None
    if window is not None:
        if not per_token:
            raise ValueError(f'{where}: a streaming batch (current_version / max_staleness) needs '
                             'train/weight_version.i32')
        low = window_low(window)
        if not all(v.isdigit() and low <= int(v) <= window[0] for v in listed):
            raise ValueError(f'{where}: rl.weight_version {version!r} outside the staleness window [{low}, {window[0]}]')
    sampling_replay = rl.get('replay', 'none')     # no key: written before prepare_rl had --top-p / --top-k
    truncated = rl.get('top_p', 1.0) < 1 or rl.get('top_k') is not None
    if sampling_replay not in ('none', 'mask') or truncated != (sampling_replay == 'mask'):
        raise ValueError(f'{where}: rl.replay {sampling_replay!r} with top_p {rl.get("top_p")!r} and top_k '
                         f'{rl.get("top_k")!r} (truncated sampling replays its supports: mask)')
    return {'temperature': temperature, 'algo': algo, 'params': params, 'round': rl['round'], 'weight_version': version,
            'versions': listed, 'per_token': per_token, 'window': window, 'sampling_replay': sampling_replay,
            'top_p': rl.get('top_p', 1.0), 'top_k': rl.get('top_k')}


def batch_versions(info, trainer, where):
    """check_rl_batch's ``info`` against the trainer at version ``trainer`` -> the ``versions`` argument of rl_targets.
    A streaming batch (``info['window']`` = (V, S)) must have been prepared for this trainer: V == trainer (its
    versions are within [max(0, V - S), V]: check_rl_batch, and per token token_lag). Any other batch keeps the v1/v2
    rule: unversioned, or every version in {trainer - 1, trainer}."""
    window = info['window']
    if window is not None:
        if window[0] != trainer:
            raise RuntimeError(f'{where}: prepared for current_version {window[0]} (max_staleness {window[1]}), the '
                               f'trainer is at {trainer}')
    elif info['versions'] != ['unversioned'] and not {int(v) for v in info['versions']} <= {trainer - 1, trainer}:
        raise RuntimeError(f'{where}: rolled out with weight_version {info["weight_version"]}, the trainer is '
                           f'at {trainer} (SPEC §8: lag <= 1)')
    return {'trainer': trainer, 'batch': info['weight_version'], 'per_token': info['per_token'],
            'max_staleness': None if window is None else window[1]}


class CommandServer:
    """Rank 0's HTTP front: GET /health from a status snapshot, POSTs handed one at a time to the main thread."""

    FIELDS = {'/train': {'batch_dir': str, 'minibatch': int}, '/begin': {'update_id': str},
              '/accumulate': {'update_id': str, 'batch_dir': str, 'minibatch': int},
              '/step': {'update_id': str, 'units': int}, '/abort': {'update_id': str},
              '/publish': {'weight_version': int}, '/save': {}, '/stop': {}}

    def __init__(self, port, status):
        self.status = {'ok': False, 'busy': True, **status}
        self.lock = threading.Lock()
        self.inbox = queue.Queue()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != '/health':
                    return self.answer(404, {'error': f'no GET {self.path}'})
                with server.lock:
                    snapshot = dict(server.status)
                self.answer(200, snapshot)

            def do_POST(self):
                fields = CommandServer.FIELDS.get(self.path)
                if fields is None:
                    return self.answer(404, {'error': f'no POST {self.path}'})
                size = self.headers['Content-Length']
                raw = self.rfile.read(int(size)) if size is not None else b''  # no header = no body (curl -X POST)
                try:
                    body = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError as error:
                    return self.answer(400, {'error': f'invalid JSON: {error}'})
                problem = CommandServer.invalid(body, fields)
                if problem:
                    return self.answer(400, {'error': problem})
                job, refusal = server.submit(self.path.removeprefix('/'), body)
                if job is None:
                    return self.answer(*refusal)
                job['done'].wait()
                try:
                    self.answer(job['code'], job['result'])
                finally:
                    job['sent'].set()

            def answer(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt, *args):
                print(f'[http] {self.address_string()} {fmt % args}', flush=True)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name='rl-http', daemon=True)
        self.thread.start()

    @staticmethod
    def invalid(body, fields):
        if not isinstance(body, dict) or set(body) != set(fields):
            return f'body must be an object with exactly {sorted(fields)}'
        for name, kind in fields.items():
            if not isinstance(body[name], kind) or isinstance(body[name], bool):
                return f'{name} must be {kind.__name__}'
        if any(name in body and body[name] < 0 for name in ('minibatch', 'weight_version', 'units')):
            return 'indices are non-negative'
        if 'update_id' in body and not body['update_id']:
            return 'update_id must be non-empty'
        return None

    def submit(self, op, body):
        """Queue a command for the main thread: (job, None), or (None, (code, error)) while loading or busy."""
        with self.lock:
            if not self.status['ok']:
                return None, (503, {'error': 'loading'})
            if self.status['busy']:
                return None, (409, {'error': 'busy'})
            self.status['busy'] = True
        job = {'op': op, 'body': body, 'done': threading.Event(), 'sent': threading.Event()}
        self.inbox.put(job)
        return job, None

    def ready(self, **status):
        with self.lock:
            self.status.update(status, ok=True, busy=False)

    def next_job(self):
        return self.inbox.get()

    def finish(self, job, code, result, **status):
        """Answer ``job`` and wait until the handler wrote it (bounded: a vanished client must not hang the loop)."""
        with self.lock:
            self.status.update(status, busy=False)
        job['code'], job['result'] = code, result
        job['done'].set()
        if not job['sent'].wait(timeout=60):
            print(f'[rank0] reply to {job["op"]} not written within 60 s', flush=True)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def warmup_lr(lr, warmup_steps, step):
    """The LR of optimizer step ``step`` (1-based): linear warmup ``lr·step/(warmup_steps+1)`` over the first
    ``warmup_steps`` steps (2 -> 1/3, 2/3, then 1), then ``lr``; 0 warmup steps = constant ``lr``."""
    if warmup_steps < 0 or step < 1:
        raise ValueError(f'warmup steps {warmup_steps} must be >= 0 and the optimizer step {step} >= 1')
    return lr * min(1.0, step / (warmup_steps + 1))


def write_json_atomic(path, payload):
    staging = path.with_name(path.name + '.tmp')
    staging.write_text(json.dumps(payload, indent=1) + '\n')
    staging.rename(path)


def load_resume_state(args):
    """CPU preflight before initializing CUDA; existing-output reuse must continue its exact latest state."""
    if args.resume_existing_out and (not args.resume or args.init_lora):
        raise ValueError('--resume-existing-out requires --resume and no --init-lora')
    if not args.resume:
        return None
    checkpoint = Path(args.resume).resolve()
    saved = torch.load(checkpoint / 'trainer_state.pt', map_location='cpu', weights_only=True)
    want = FORMAT_FULL if args.trainable == 'full' else FORMAT
    if saved['format'] != want:
        raise RuntimeError(f'{checkpoint}: format {saved["format"]!r}, this service (--trainable {args.trainable}) '
                           f'writes {want}')
    may_differ = (RESUME_MAY_DIFFER | {'lr'}) if args.resume_new_lr else RESUME_MAY_DIFFER   # a deliberate LR change
    conflicts = resume_conflicts(saved['args'], vars(args), may_differ=may_differ,
                                added=ADDED_SETTINGS)
    if conflicts:
        raise RuntimeError(f'resume would change training settings (saved, now): {conflicts}')
    if not args.resume_existing_out:
        return saved
    out = Path(args.out).resolve()
    if (checkpoint.parent != out or checkpoint.name != f'checkpoint-{saved["step"]}'
            or Path(saved['args']['out']).resolve() != out):
        raise RuntimeError('existing-output resume checkpoint must belong to the same OUT')
    original = json.loads((out / 'config.json').read_text())
    if Path(original['out']).resolve() != out or not (out / 'code').is_dir():
        raise RuntimeError('existing OUT original config/code provenance is missing or belongs to another run')
    conflicts = resume_conflicts(original, vars(args), may_differ=may_differ,
                                added=ADDED_SETTINGS)
    if conflicts:
        raise RuntimeError(f'resume would change original training settings (original, now): {conflicts}')
    raw = (out / 'train_log.jsonl').read_bytes()
    if not raw.endswith(b'\n'):
        raise RuntimeError('existing train_log has an incomplete final line')
    rows = [json.loads(line) for line in raw.splitlines()]
    steps = [row['step'] for row in rows]
    if not steps or any(type(step) is not int for step in steps) or any(a >= b for a, b in zip(steps, steps[1:])):
        raise RuntimeError('existing train_log optimizer steps must increase strictly')
    if steps[-1] != saved['step']:
        raise RuntimeError(f'checkpoint step {saved["step"]} != latest train_log step {steps[-1]}: refuse rollback')
    exports = list((out / 'merged').glob('v[0-9][0-9][0-9][0-9][0-9][0-9]'))
    if not exports:
        raise RuntimeError('existing OUT has no published export evidence')
    latest = max(exports, key=lambda path: int(path.name[1:]))
    from merged_export import FORMAT as MERGED_V1, FORMAT_V2 as MERGED_V2
    manifest = json.loads((latest / 'merged_manifest.json').read_text())
    if (not (latest / 'DONE').is_file() or manifest['format'] != (MERGED_V2 if args.dense_qat == 'fp8' else MERGED_V1)
            or int(latest.name[1:]) != saved['weight_version']
            or str(manifest['weight_version']) != str(saved['weight_version'])
            or manifest['optimizer_step'] != saved['step'] or Path(manifest['run']).resolve() != out
            or manifest['base_model'] != args.model):
        raise RuntimeError('checkpoint does not match the latest completed publication; refuse rollback')
    return saved


def append_resume_provenance(args, saved, reader_path):
    """Called by rank0 only after LoRA/Adam load and replicated-state verification succeeded."""
    out = Path(args.out)
    sources = sorted(set(Path(__file__).parent.glob('*.py')) | {Path(reader_path)})
    record = {'utc': time.strftime('%FT%TZ', time.gmtime()), 'checkpoint': str(Path(args.resume).resolve()),
              'step': saved['step'], 'weight_version': saved['weight_version'], 'args': vars(args),
              'original_config_sha256': hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(),
              'source_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    with (out / 'resume_history.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True, help='new directory')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--lr', type=float, default=2e-5, help='AdamW LR (after --lr-warmup-steps), betas (0.9, --adam-beta2), wd 0')
    parser.add_argument('--lr-warmup-steps', type=int, default=0,
                        help='linear LR warmup over the first N optimizer steps: step k runs at lr·k/(N+1) (warmup_lr). '
                        'The step count is the checkpoint\'s on --resume, so a resumed run keeps the schedule. Default 0: constant')
    parser.add_argument('--adam-beta2', type=float, default=0.98, help='AdamW beta2 (Tinker uses 0.95)')
    parser.add_argument('--adam-eps', type=float, default=1e-8,
                        help='AdamW eps (default torch 1e-8). The prompt-mean loss gives LoRA-A elements gradients near 1e-8, '
                             'where eps damps their step (0.58-0.74x, docs/research/qwen38fn_rl_implementation_audit_20261018.md '
                             'P4); Tinker uses 1e-12')
    parser.add_argument('--init-lora', help='start from the LoRA of a checkpoint-<step> directory (trainer_state.pt)')
    parser.add_argument('--resume', help='continue a checkpoint-<step> written by /save (LoRA, AdamW, step, version)')
    parser.add_argument('--resume-existing-out', action='store_true',
                        help='explicitly reuse the checkpoint original OUT; preserve config/code and append provenance')
    parser.add_argument('--resume-new-lr', action='store_true',
                        help='with --resume: continue at --lr although the checkpoint (and the original config) ran another '
                             'LR, a deliberate mid-run LR change (10-31 arc2-hq741b: 3e-6 -> 1e-5 from step 14). Adam '
                             'moments carry over, the warmup keeps the checkpoint step count, and resume_history.jsonl '
                             'records the new LR. Without it a changed LR is a resume conflict')
    parser.add_argument('--trainable', choices=['lora', 'full'], default='lora',
                        help='lora: a LoRA on the projections below; full: the projections themselves as FP32 full '
                             'weights (dense_full_rl.py; needs --base-storage int3 --dense-qat fp8; --lora-* unused)')
    parser.add_argument('--freeze-first-layers', type=int, default=0,
                        help='--trainable full: the projections of decoder layers 0..N-1 stay frozen; no gradient reaches '
                             'those layers, so they keep no checkpointed inputs and run no backward (2 = through the '
                             'PLE layer). Default 0')
    parser.add_argument('--full-grad-kernel', choices=['quack', 'fused', 'fp32'], default='quack',
                        help='--trainable full: weight-gradient kernel of the 144 block-FP8 projections (dense_full_qat '
                             'DENSE_GRAD; quack = QAD v2\'s default, fp32 = the chunked FP32 reference)')
    parser.add_argument('--full-grad-precision', choices=['bf16', 'bf16x2'], default='bf16',
                        help='--trainable full, fused / quack: operand precision of that kernel (QAD v2 default bf16)')
    parser.add_argument('--checkpoint-store',
                        help='--trainable full: /save writes the ~35 GB DCP to <store>/checkpoint-<step> and links it as '
                             '<out>/checkpoint-<step>/dcp (e.g. tmpfs when /data is short). Default: inside <out>')
    parser.add_argument('--keep-checkpoints', type=int, default=0,
                        help='--trainable full: after each /save keep only the newest N checkpoint-<step> (and their '
                             'store directories); 0 = keep all')
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    add_lora_target_arguments(parser)
    parser.add_argument('--seed', type=int, default=20261006, help='LoRA initialization')
    parser.add_argument('--bin-tokens', type=int, default=120_000, help='the largest bin (sizes the offload slots)')
    parser.add_argument('--base-storage', choices=['bf16', 'fp8', 'int3'], default='fp8',
                        help='int3 = gEAN320 INT3 package with BF16 dense weights (W3A8 experts, dX on dequantized BF16)')
    parser.add_argument('--offload-checkpoint-inputs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--offload-mode', choices=['sync', 'async'], default='async')
    parser.add_argument('--offload-keep-gib', type=float, default=0.0,
                        help='async: the newest checkpointed layer inputs stay on the GPU while they fit in this many GiB; only older ones go to host memory (checkpoint_offload.py keep_bytes)')
    parser.add_argument('--fp8-dx', choices=['fp8', 'bf16'], default='fp8')
    parser.add_argument('--int3-act', choices=['fp8', 'bf16'], default='fp8',
                        help='INT3 base: activation format of the routed experts\' forward (int3_moe.py). fp8 = W3A8, '
                        'per-row E4M3 like Humming W3A8; bf16 = W3A16, BF16 activations into bf16((code - 4) * scale) '
                        'like Humming A16 (the rollout\'s ARC_INT3_ACT=bf16). The backward is the same for both')
    parser.add_argument('--ple-media', choices=['deployed', 'semantic'], default='deployed')
    parser.add_argument('--ple-codec', choices=['fp8', 'int5'], default='fp8',
                        help='PLE table storage: fp8 = the checkpoint rows x weight_scale; int5 = re-quantized at load to the INT5 table '
                        'the deployed server builds with ARC_PLE_INT5=1 (lookups bit-identical to it); needs --base-storage fp8 or int3')
    parser.add_argument('--vit-interp', choices=['deployed', 'hf'], default='deployed')
    parser.add_argument('--qsa-backend', default='auto')
    parser.add_argument('--audit-steps', type=int, default=2, help='QSA coverage audit on the first N micro-steps')
    parser.add_argument('--host-prefetch', action=argparse.BooleanOptionalAction, default=True,
                        help="read micro-step k+1's rows and RL targets on a host thread while micro-step k runs")
    parser.add_argument('--dump-logp', action='store_true', help='debug: per-token logp of every micro-step')
    parser.add_argument('--dump-grads', action='store_true',
                        help='debug: rank 0 writes <out>/grads/step-<step>.pt per optimizer step: the summed gradient before '
                        'clipping, its norm and the LoRA before and after AdamW (flat fp32, parameter order)')
    parser.add_argument('--gpu-memory-fraction', type=float, default=None,
                        help='cap each rank\'s CUDA caching allocator at this fraction of the card '
                        '(torch.cuda.set_per_process_memory_fraction): the allocator frees its cached blocks before it '
                        'passes the cap, so what it holds but does not use is left to another process on the card '
                        '(v3 c64: peak allocated ~141 GiB, reserved ~175 GiB of 178 GiB). Default: no cap')
    parser.add_argument('--hang-dump-seconds', type=float, default=180.0,
                        help='a micro-step, optimizer step, publish or save running longer than this writes thread '
                        'stacks, wchan and allocator counters to <out>/hang/ once (hang_watch.py), before NCCL\'s 600 s '
                        'watchdog ends the process; 0 = off')
    parser.add_argument('--hang-cuda-gdb', action=argparse.BooleanOptionalAction, default=False,
                        help='also attach cuda-gdb for the resident kernels. Off: 10-21, 10-22 and 10-26 it printed '
                        'nothing in 240 s on every rank, the ranks it had attached to died of SIGTRAP after it was '
                        'killed, and its NvDebugAgent kept the trainer unit alive after torchrun exited')
    fp8_dense_qat.add_arguments(parser)
    trainer_kernels.add_arguments(parser)
    sparse_qsa.add_arguments(parser)
    args = parser.parse_args()
    fp8_dense_qat.check_arguments(parser, args)
    trainer_kernels.check_arguments(parser, args)
    sparse_qsa.apply_arguments(args)
    if not args.lr > 0 or not 0 < args.adam_beta2 < 1 or args.bin_tokens < 4096 or not 0 < args.port < 65536:
        parser.error('invalid LR, beta2, bin size or port')
    if not 0 < args.adam_eps < 1e-4:
        parser.error('adam eps must be in (0, 1e-4)')
    if args.lr_warmup_steps < 0:
        parser.error('--lr-warmup-steps must be >= 0')
    if args.gpu_memory_fraction is not None and not 0.5 <= args.gpu_memory_fraction <= 1.0:
        parser.error('--gpu-memory-fraction must be in [0.5, 1]')
    if args.init_lora and args.resume:
        parser.error('--init-lora starts a run; --resume continues one')
    if args.resume_new_lr and not args.resume:
        parser.error('--resume-new-lr needs --resume')
    if args.int3_act != 'fp8' and args.base_storage != 'int3':
        parser.error('--int3-act bf16 needs --base-storage int3')
    if args.trainable == 'full':
        if args.base_storage != 'int3' or args.dense_qat != 'fp8':
            parser.error('--trainable full needs --base-storage int3 --dense-qat fp8')
        if args.init_lora:
            parser.error('--trainable full has no --init-lora')
        if args.keep_checkpoints < 0 or args.freeze_first_layers < 0:
            parser.error('--keep-checkpoints / --freeze-first-layers must be >= 0')
    elif (args.freeze_first_layers or args.checkpoint_store or args.keep_checkpoints
          or (args.full_grad_kernel, args.full_grad_precision) != ('quack', 'bf16')):
        parser.error('--freeze-first-layers / --checkpoint-store / --keep-checkpoints / --full-grad-* need --trainable full')
    if not 0 <= args.hang_dump_seconds < 600:
        parser.error('--hang-dump-seconds must be in [0, 600): it has to fire before the NCCL watchdog')
    saved = load_resume_state(args)

    import dense_full_rl
    import rl_batch
    from checkpoint_offload import CheckpointInputOffload
    from distributed_utils import assert_replicated, sync_and_clip_gradients
    from merged_export import changed_elements, lora_pairs

    from arcforge.train.preflight import assert_flash_qla_stack, assert_gdn_backend, assert_liger_moe_patch

    started_unix = time.time()
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        cap = args.gpu_memory_fraction * torch.cuda.get_device_properties(device).total_memory / 2**30
        print(f'[rank{rank}] CUDA caching allocator capped at {args.gpu_memory_fraction} of the card = {cap:.1f} GiB',
              flush=True)
    hang_watch.allow_ptrace()  # cuda-gdb / py-spy may attach to a stuck rank (yama ptrace_scope 1)
    watch = hang_watch.HangWatch(rank, device, args.out, args.hang_dump_seconds,
                                 cuda_gdb=hang_watch.CUDA_GDB if args.hang_cuda_gdb else None)
    dist.init_process_group('nccl', device_id=device)
    world = dist.get_world_size()
    # 3 ranks since 10-15 (5 rollout + 3 trainer on one B200 box): FSDP2 pads the uneven dim-0 shards (320 experts ->
    # 107 each), the PLE/embedding row shards split total*rank//world. One rank (a one-device FSDP mesh, same DTensor
    # path) is for single-card probes (work/qwen38fn_rl/probes/topp_mask/).
    if world not in (1, 2, 3, 4):
        raise RuntimeError('FSDP over 1, 2, 3 or 4 ranks only')
    # Commands travel on gloo; ranks wait in its broadcast for as long as rollouts take.
    commands = dist.new_group(backend='gloo', timeout=timedelta(days=30))
    assert_flash_qla_stack()
    assert_gdn_backend('flash_qla')  # the fla Triton fallback is not run-to-run deterministic on B200
    assert_liger_moe_patch()
    out = Path(args.out)
    server = None
    if rank == 0:
        server = CommandServer(args.port, {
            'step': None, 'weight_version': None, 'update': None,
            'policy_identity': {
                'model_path': str(Path(args.model).resolve()),
                'init_lora': str(Path(args.init_lora).resolve()) if args.init_lora else None,
                'trainable': args.trainable, 'freeze_first_layers': args.freeze_first_layers,
                'resume': str(Path(args.resume).resolve()) if args.resume else None,
            },
        })
        if not args.resume_existing_out:
            out.mkdir(parents=True, exist_ok=False)
            (out / 'code').mkdir()
            for path in Path(__file__).parent.glob('*.py'):
                shutil.copy2(path, out / 'code' / path.name)
            shutil.copy2(rl_batch.__file__, out / 'code' / 'rl_batch.py')
            if args.trainable == 'full':
                import dense_full_rl
                for name in dense_full_rl.QAD_MODULES:
                    shutil.copy2(dense_full_rl.QAD_DIR / name, out / 'code' / f'qwen38fn_qad__{name}')
            (out / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
        print(f'[rank0] RLBatch reader: {rl_batch.__file__}', flush=True)
    dist.barrier()
    tokenizer_sha = hashlib.sha256((Path(args.model) / 'tokenizer.json').read_bytes()).hexdigest()
    trainer = load_trainer(args, rank, device, world, 'force', merged_export=True)
    model, base, parameters, names = trainer.model, trainer.base, trainer.parameters, trainer.names
    kernels, gdn, replay, exporter = trainer.kernels, trainer.gdn, trainer.replay, trainer.exporter
    text_config, vision_config, width, slots = trainer.text_config, trainer.vision_config, trainer.width, trainer.slots
    base.forward = MethodType(rl_forward, base)
    full = trainer.full   # None for the LoRA
    # full: the masters are FSDP DTensor shards (foreach AdamW, as QAD v2); the LoRA is replicated plain tensors
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, args.adam_beta2), eps=args.adam_eps,
                                  weight_decay=0, **({'foreach': True} if full else {'fused': True}))
    state = {'step': 0, 'weight_version': 0}
    if saved is not None:
        if full is not None:
            if saved['names'] != names:
                raise RuntimeError('saved full-weight parameter names differ from this model')
            dense_full_rl.restore(model, optimizer, names, Path(args.resume) / 'dcp')
            if args.resume_existing_out:   # continuing this run: its merged/v<weight_version> is the reference
                dense_full_rl.check_resumed_export(names, parameters, rank, exporter, out, saved['weight_version'],
                                                   args.checkpoint_store or out, commands)
        else:
            if set(saved['lora']) != set(names):
                raise RuntimeError('saved LoRA parameter names differ from this model')
            with torch.no_grad():
                for name, p in zip(names, parameters, strict=True):
                    p.copy_(saved['lora'][name].to(p.device))
            optimizer.load_state_dict(saved['optimizer'])
        state = {'step': saved['step'], 'weight_version': saved['weight_version']}
    if full is None:
        assert_replicated(parameters)
    if rank == 0 and args.resume_existing_out:
        append_resume_provenance(args, saved, rl_batch.__file__)
    del saved
    print(f'[rank{rank}] ready: {sum(p.numel() for p in parameters)} trainable, step {state["step"]}, '
          f'weight_version {state["weight_version"]}, {time.time() - started_unix:.0f} s', flush=True)

    # one host thread reads the next micro-step's rows while the GPU runs the current one
    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rl-rows') if args.host_prefetch else None
    batches = {}
    trained = set()
    runs = [0]
    update = {}  # the open /begin .. /step update (empty when none)

    def open_batch(path):
        if path not in batches:
            batches.clear()
            b = rl_batch.RLBatch(path)
            batches[path] = (b, check_rl_batch(b, tokenizer_sha, replay, text_config, path))
        return batches[path]

    def units_of(b, minibatch):
        """The count the minibatch's coef divides by: H groups (prompt reduction) or E episodes (sequence)."""
        mb = b.minibatches[minibatch]
        return len(mb['groups']) if b.loss_reduction == 'prompt' else mb['episodes']

    def run_minibatch(batch_dir, minibatch, scale):
        """Forward, the batch's loss and backward over every micro-step of one minibatch, adding ``scale`` times its
        gradient to the LoRA gradients (no sync, no step). Returns this rank's record and the batch info."""
        b, info = open_batch(batch_dir)
        temperature, algo, params = info['temperature'], info['algo'], info['params']
        if (batch_dir, minibatch) in trained:
            raise RuntimeError(f'{batch_dir} minibatch {minibatch} was already trained by this process')
        if info['versions'] == ['unversioned']:
            print(f'[rank{rank}] {batch_dir}: unversioned batch (recorded before the fork wrote versions)', flush=True)
        versions = batch_versions(info, state['weight_version'], batch_dir)
        if not 0 <= minibatch < len(b.minibatches):
            raise IndexError(f'{batch_dir}: minibatch {minibatch} of {len(b.minibatches)}')
        micro = b.minibatches[minibatch]['steps']
        if not micro or any(len(bins) != world or not all(bins) for bins in micro):
            raise ValueError(f'{batch_dir} minibatch {minibatch}: every micro-step needs a non-empty bin per rank')
        model.train(True)
        base.model.visual.eval()
        prior = dict(kernels)
        where = f'{Path(batch_dir).parent.name} mb{minibatch}'
        with watch.phase(f'barrier before {where}'):
            dist.barrier()
            torch.cuda.synchronize()
        started = time.perf_counter()
        terms, seconds, prep, routing = [], [], [], {'rows': 0, 'equal': 0, 'overlap': 0}
        tokens = loss = 0

        def host_rows(k):
            """Micro-step k's rows on this rank, their RL targets, the supervised tokens' positions in their segments
            (pack order, as the targets) and recorded routing (host only, so it can run on the prefetch thread while
            the GPU works on micro-step k - 1)."""
            rows = [b[i] for i in micro[k][rank]]
            targets = rl_targets(rows, text_config.num_experts, lambda row: b.coef_denominator(minibatch, row),
                                 versions)
            positions = np.concatenate([np.flatnonzero(row['loss_mask']) for row in rows]).astype(np.int32)
            if len(positions) != len(targets[0]):
                raise RuntimeError(f'{len(positions)} supervised positions for {len(targets[0])} RL targets')
            support = rl_support(rows) if info['sampling_replay'] == 'mask' else None
            if support is not None and len(support[0]) != len(targets[0]):
                raise RuntimeError(f'{len(support[0])} sampling supports for {len(targets[0])} RL targets')
            return rows, targets, positions, np.concatenate([row['routing'] for row in rows]), support

        pending = None if prefetch is None else prefetch.submit(host_rows, 0)
        for k in range(len(micro)):
            prep_started = time.perf_counter()
            if prefetch is None:
                rows, (behavior, coef, real, lag, counts), positions, recorded, support = host_rows(k)
            else:
                rows, (behavior, coef, real, lag, counts), positions, recorded, support = pending.result()
                if k + 1 < len(micro):
                    pending = prefetch.submit(host_rows, k + 1)
            audit = runs[0] < args.audit_steps
            runs[0] += 1
            batch, media = build_batch(rows, 0, device, vision_config, text_config.vocab_size,
                                       ple_media=args.ple_media, vit_interp=args.vit_interp, audit=audit)
            batch['packed_meta']['qsa_mask_backend'] = qsa_backend(args.qsa_backend, rows)
            length = batch['input_ids'].shape[1]
            tokens += length
            replay.begin(torch.from_numpy(recorded).to(device))
            offload = None
            if slots is not None:
                offload = CheckpointInputOffload(length, width, slots, asynchronous=args.offload_mode == 'async',
                                                 keep_bytes=int(args.offload_keep_gib * 2**30))
            gdn_before = gdn['calls']
            micro_started = time.perf_counter()
            prep.append(micro_started - prep_started)
            # No host sync between the forward and backward(): one there drained each rank's GPU queue before it
            # launched the logprob and the backward's first all-gather (10-21 v3.1 p001 profile: a 0.44 s host wait
            # and a 60 ms compute gap per micro step; the 0.17-0.2 s gap left at the first all-gather is the faster
            # ranks waiting for the slowest). The two checks that need the device run after the step's own
            # synchronize, before any optimizer step.
            with watch.phase(f'micro {k + 1}/{len(micro)} of {where} ({length} tokens)'):
                with torch.enable_grad(), (nullcontext() if offload is None else offload.context()):
                    forward = model(**batch, media=media, temperature=temperature, supervised=len(behavior),
                                    support=None if support is None else torch.from_numpy(support[0]).to(device))
                    logp = forward['logp']
                    if logp.shape != behavior.shape:
                        raise RuntimeError(f'{logp.shape[0]} supervised logp for {len(behavior)} generated tokens')
                    part, part_terms = algo_loss(algo, params, logp, torch.from_numpy(behavior).to(device),
                                                 torch.from_numpy(coef).to(device))
                    (part if scale == 1 else part * scale).backward()
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - micro_started)
            found = int(forward['valid'].sum())
            if found != len(behavior):
                raise RuntimeError(f'{found} supervised labels in the bin for {len(behavior)} generated tokens')
            if not torch.isfinite(part):
                raise RuntimeError('nonfinite loss')
            entropy = forward['entropy'].cpu().numpy()
            sampled = {}
            if support is not None:
                logp_full = forward['logp_full'].cpu().numpy()
                sampled = {'support_len': support[2][real], 'entropy_support': forward['entropy_support'].cpu().numpy()[real],
                           # log P_pi(S) and log P_mu(S): the full-vocabulary minus the support log-probabilities
                           'log_pi_kept': (logp_full - logp.detach().cpu().numpy())[real],
                           'log_mu_kept': (support[1] - behavior)[real],
                           'delta_full': (logp_full - support[1])[real]}
            del forward
            stats = replay.finish()
            routing = {key: routing[key] + int(stats[key].sum()) for key in routing}
            assert_step_kernels(batch, audit, gdn['calls'] - gdn_before, True, offload,
                                frozen=None if full is None else full.frozen)
            loss += float(part.detach()) * scale
            terms.append({**{key: value.cpu().numpy()[real] for key, value in part_terms.items() if key != 'weight'},
                          'entropy': entropy[real], 'nll': -logp.detach().cpu().numpy()[real],
                          'lag': lag[real], 'pos': positions[real], **sampled})
            if args.dump_logp:
                target = out / 'logp' / f'step-{state["step"] + 1:06d}'
                target.mkdir(parents=True, exist_ok=True)
                np.savez(target / f'micro-{k:03d}-rank{rank}.npz', keys=np.array([row['key'] for row in rows]),
                         pad=np.array([row['pad'] for row in rows]), counts=np.array(counts),
                         logp=logp.detach().cpu().numpy(), behavior=behavior, coef=coef, lag=lag)
        if not any(kernels[k] > prior[k] for k in QSA_KERNELS):
            raise RuntimeError('no QSA kernel executed')
        trained.add((batch_dir, minibatch))
        record = {'started': started, 'seconds': seconds, 'prep': prep, 'tokens': tokens, 'loss': loss,
                  'routing': routing, 'terms': {key: np.concatenate([t[key] for t in terms]) for key in terms[0]}}
        return record, info

    def optimizer_step(records, units, started, info, head):
        """Divide the accumulated gradients by ``units`` (1 for /train), sum them over ranks, clip, AdamW; on rank 0
        the SPEC §1 report over every record (one per minibatch), ``head`` first, appended to train_log.jsonl."""
        if full is not None:
            dense_full_rl.assert_gradients(names, parameters)
        if units != 1:
            for p in parameters:
                p.grad.div_(units)
        dumped = {}

        def observe(synced):
            dumped['grad'] = torch.cat([p.grad.flatten().float() for p in synced]).cpu()
            dumped['lora_before'] = torch.cat([p.detach().flatten().float() for p in synced]).cpu()

        def observe_full():
            """--dump-grads of the sharded masters: every rank joins the gathers, rank 0 keeps the result."""
            grads, weights = [], []
            for p in parameters:
                g, w = p.grad.full_tensor().float(), p.detach().full_tensor().float()
                if rank == 0:
                    grads.append(g.flatten().cpu())
                    weights.append(w.flatten().cpu())
                del g, w
            if rank == 0:
                dumped['grad'], dumped['lora_before'] = torch.cat(grads), torch.cat(weights)

        with watch.phase(f'optimizer step {state["step"] + 1}'):
            if full is None:
                norm = sync_and_clip_gradients(parameters, observe=observe if args.dump_grads and rank == 0 else None)
            else:   # FSDP already summed the gradients over ranks in each backward (divide factor 1)
                if args.dump_grads:
                    observe_full()
                norm = dense_full_rl.clip_gradients(names, parameters, 1.0)
            for group in optimizer.param_groups:   # every step: a resumed optimizer carries the LR it saved
                group['lr'] = warmup_lr(args.lr, args.lr_warmup_steps, state['step'] + 1)
            optimizer.step()
            if full is None:
                assert_replicated(parameters)
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        state['step'] += 1
        if args.dump_grads:
            after = (torch.cat([p.detach().flatten().float() for p in parameters]).cpu() if full is None else
                     torch.cat([w.float().flatten() for w in dense_full_rl.full_tensors(names, parameters, rank).values()])
                     if rank == 0 else dense_full_rl.full_tensors(names, parameters, rank))
        if dumped:
            target = out / 'grads'
            target.mkdir(exist_ok=True)
            # full: 'lora' / 'lora_before' hold the full weights (after: the BF16 cast, before: the FP32 masters)
            torch.save({'names': names, 'numel': [p.numel() for p in parameters], 'grad': dumped['grad'],
                        'grad_norm': norm, 'lora_before': dumped['lora_before'], 'lora': after},
                       target / f'step-{state["step"]:06d}.pt')
        local = {'rank': rank, 'seconds': sum(s for r in records for s in r['seconds']),
                 'micro_seconds': [s for r in records for s in r['seconds']],
                 'prep_seconds': [s for r in records for s in r['prep']], 'tokens': sum(r['tokens'] for r in records),
                 'loss_tokens': int(sum(len(r['terms']['delta']) for r in records)),
                 'loss': sum(r['loss'] for r in records) / units,
                 'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
                 'reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'host_rss_gib': host_rss_gib(),
                 'routing': {key: sum(r['routing'][key] for r in records) for key in records[0]['routing']},
                 'terms': {key: np.concatenate([r['terms'][key] for r in records]) for key in records[0]['terms']}}
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(local, gathered, dst=0, group=commands)
        if rank != 0:
            return None
        algo, params = info['algo'], info['params']
        every = {key: np.concatenate([r['terms'][key] for r in gathered]) for key in local['terms']}
        rows_compared = sum(r['routing']['rows'] for r in gathered)
        report = {'step': state['step'], **head, 'weight_version': state['weight_version'],
                  'temperature': info['temperature'], 'algo': algo, 'algo_params': dict(zip(ALGOS[algo], params)),
                  'lr': optimizer.param_groups[0]['lr'],
                  'loss': sum(r['loss'] for r in gathered), 'grad_norm': norm,
                  **terms_report(algo, params, every),
                  'sequence_tokens': sum(r['tokens'] for r in gathered),
                  'routing_equal_rate': sum(r['routing']['equal'] for r in gathered) / rows_compared,
                  'routing_mean_shared': sum(r['routing']['overlap'] for r in gathered) / rows_compared,
                  'seconds': wall,
                  'ranks': [{k: v for k, v in r.items() if k not in ('terms', 'routing')} for r in gathered]}
        with open(out / 'train_log.jsonl', 'a') as f:
            f.write(json.dumps(report) + '\n')
        return report

    def refuse_in_update(op):
        if update:
            raise RuntimeError(f'/{op} while update {update["id"]!r} is open (POST /step or /abort first)')

    def train(batch_dir, minibatch):
        refuse_in_update('train')
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        record, info = run_minibatch(batch_dir, minibatch, 1)
        return optimizer_step([record], 1, record['started'], info, {
            'batch_dir': batch_dir, 'minibatch': minibatch, 'round': info['round'],
            'batch_weight_version': info['weight_version'], 'micro_steps': len(record['seconds'])})

    def begin(update_id):
        refuse_in_update('begin')
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        update.update(id=update_id, records=[], packages=[], units=0, info=None, rule=None)
        return {'update_id': update_id, 'step': state['step'], 'weight_version': state['weight_version']}

    def open_update(update_id, op):
        if not update:
            raise RuntimeError(f'/{op} {update_id!r} without /begin')
        if update['id'] != update_id:
            raise RuntimeError(f'/{op} for update {update_id!r}, the open update is {update["id"]!r}')

    def accumulate(update_id, batch_dir, minibatch):
        """One package (a minibatch whose coef divides by its own H_p groups or E_p episodes) into the open update:
        its gradient enters times H_p, and /step divides the sum by H = sum of H_p, which gives the gradient of the
        same rows trained as one minibatch."""
        open_update(update_id, 'accumulate')
        b, info = open_batch(batch_dir)
        if not 0 <= minibatch < len(b.minibatches):
            raise IndexError(f'{batch_dir}: minibatch {minibatch} of {len(b.minibatches)}')
        rule = (b.loss_reduction, info['temperature'], info['algo'], info['params'], info['sampling_replay'],
                info['top_p'], info['top_k'])
        if update['rule'] is not None and rule != update['rule']:
            raise ValueError(f'{batch_dir}: loss reduction, temperature, loss algorithm and parameters, sampling replay {rule} differ '
                             f'from the update\'s {update["rule"]}')
        units = units_of(b, minibatch)
        if units <= 0:
            raise ValueError(f'{batch_dir} minibatch {minibatch}: {units} units')
        record, info = run_minibatch(batch_dir, minibatch, units)
        wall = time.perf_counter() - record['started']
        package = {'batch_dir': batch_dir, 'minibatch': minibatch, 'units': units, 'round': info['round'],
                   'batch_weight_version': info['weight_version'], 'micro_steps': len(record['seconds']),
                   'seconds': wall}
        update['records'].append(record)
        update['packages'].append(package)
        update['units'] += units
        update['rule'], update['info'] = rule, update['info'] or info
        local = {'rank': rank, 'seconds': sum(record['seconds']), 'prep_seconds': sum(record['prep']),
                 'tokens': record['tokens'],
                 'loss_tokens': len(record['terms']['delta']), 'peak_gib': torch.cuda.max_memory_allocated() / 2**30}
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(local, gathered, dst=0, group=commands)
        if rank != 0:
            return None
        return {'update_id': update_id, **package, 'loss_reduction': b.loss_reduction,
                'tokens': sum(r['tokens'] for r in gathered), 'loss_tokens': sum(r['loss_tokens'] for r in gathered),
                'accumulated_units': update['units'], 'packages': len(update['packages']), 'ranks': gathered}

    def step(update_id, units):
        open_update(update_id, 'step')
        if not update['records']:
            raise RuntimeError(f'/step {update_id!r} with no accumulated package (POST /abort instead)')
        if units != update['units']:
            raise ValueError(f'/step units {units} != {update["units"]}, the sum over the accumulated packages')
        started = time.perf_counter()
        report = optimizer_step(update['records'], units, started, update['info'], {
            'update_id': update_id, 'units': units, 'loss_reduction': update['rule'][0],
            'packages': update['packages'], 'micro_steps': sum(p['micro_steps'] for p in update['packages']),
            'accumulate_seconds': sum(p['seconds'] for p in update['packages'])})
        update.clear()
        return report

    def abort(update_id):
        open_update(update_id, 'abort')
        model.zero_grad(set_to_none=True)
        result = {'update_id': update_id, 'packages': len(update['packages']), 'units': update['units'],
                  'step': state['step']}
        update.clear()
        return result

    def publish(weight_version):
        refuse_in_update('publish')
        if weight_version <= state['weight_version']:
            raise ValueError(f'weight_version {weight_version} must exceed the current {state["weight_version"]}')
        result = None
        started = time.perf_counter()
        weights = None if full is None else dense_full_rl.full_tensors(names, parameters, rank)   # every rank gathers
        if rank == 0:
            target = out / 'merged' / f'v{weight_version:06d}'
            extra = {'run': str(out), 'optimizer_step': state['step']}
            if full is None:
                pairs = lora_pairs({n: p.detach().cpu() for n, p in zip(names, parameters, strict=True)})
                manifest = exporter.write(target, pairs, weight_version, extra)
            else:
                manifest = exporter.write_full(target, weights, weight_version,
                                               {**extra, 'frozen_layers': full.frozen['layers']})
            del weights
            result = {'path': str(target), 'weight_version': weight_version, 'step': state['step'],
                      'manifest_sha256': hashlib.sha256((target / 'merged_manifest.json').read_bytes()).hexdigest(),
                      'bytes': manifest['bytes'], 'seconds': time.perf_counter() - started,
                      'changed_bf16_elements': changed_elements(manifest)}  # stored elements (v2: e4m3 + scales too)
        state['weight_version'] = weight_version
        return result

    def save_full():
        """checkpoint-<step>/{trainer_state.pt, dcp} (dcp a link into --checkpoint-store when given), staged as .tmp and
        renamed; then --keep-checkpoints prunes the oldest (with their store directories)."""
        started = time.perf_counter()
        target = out / f'checkpoint-{state["step"]}'
        staging = target.with_name(target.name + '.tmp')
        store = None if args.checkpoint_store is None else Path(args.checkpoint_store) / f'checkpoint-{state["step"]}'
        if rank == 0:
            if target.exists() or staging.exists() or (store is not None and store.exists()):
                raise FileExistsError(f'{target}, its staging directory or its store {store} exists')
            staging.mkdir()
            if store is not None:
                store.mkdir(parents=True)
                (staging / 'dcp').symlink_to(store)
        dist.barrier()
        dense_full_rl.save(model, optimizer, names, staging / 'dcp')
        dist.barrier()
        result = None
        if rank == 0:
            torch.save({'format': FORMAT_FULL, 'step': state['step'], 'weight_version': state['weight_version'],
                        'names': names, 'args': vars(args)}, staging / 'trainer_state.pt')
            staging.rename(target)
            removed = []
            if args.keep_checkpoints:
                done = sorted((int(d.name.split('-')[1]), d) for d in out.glob('checkpoint-*')
                              if d.name.split('-')[1].isdigit() and (d / 'trainer_state.pt').exists())
                if done[-1][1] != target:
                    raise RuntimeError(f'newest checkpoint in {out} is {done[-1][1]}, not the one just saved')
                for _, old in done[:-args.keep_checkpoints]:
                    link = old / 'dcp'
                    if link.is_symlink():
                        shutil.rmtree(link.resolve())
                    shutil.rmtree(old)
                    removed.append(str(old))
            result = {'path': str(target), 'step': state['step'], 'weight_version': state['weight_version'],
                      'store': None if store is None else str(store), 'removed': removed,
                      'seconds': time.perf_counter() - started}
        with watch.phase(f'barrier after save of step {state["step"]}'):
            dist.barrier()
        return result

    def save():
        refuse_in_update('save')
        if full is not None:
            return save_full()
        result = None
        if rank == 0:
            started = time.perf_counter()
            target = out / f'checkpoint-{state["step"]}'
            staging = target.with_name(target.name + '.tmp')
            if target.exists() or staging.exists():
                raise FileExistsError(f'{target} or its staging directory exists')
            adapter = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
            model.save_pretrained(staging, state_dict=adapter, safe_serialization=True, save_embedding_layers=False)
            torch.save({'format': FORMAT, 'step': state['step'], 'weight_version': state['weight_version'],
                        'lora': {n: p.detach().cpu() for n, p in zip(names, parameters, strict=True)},
                        'optimizer': optimizer.state_dict(), 'args': vars(args)}, staging / 'trainer_state.pt')
            staging.rename(target)
            result = {'path': str(target), 'step': state['step'], 'weight_version': state['weight_version'],
                      'seconds': time.perf_counter() - started}
        with watch.phase(f'barrier after save of step {state["step"]}'):
            dist.barrier()
        return result

    operations = {'train': lambda body: train(body['batch_dir'], body['minibatch']),
                  'begin': lambda body: begin(body['update_id']),
                  'accumulate': lambda body: accumulate(body['update_id'], body['batch_dir'], body['minibatch']),
                  'step': lambda body: step(body['update_id'], body['units']),
                  'abort': lambda body: abort(body['update_id']),
                  'publish': lambda body: publish(body['weight_version']),
                  'save': lambda body: save(),
                  'stop': lambda body: {'stopping': True, 'step': state['step']}}
    if rank == 0:
        write_json_atomic(out / 'trainer_endpoint.json', {
            'url': f'http://127.0.0.1:{args.port}', 'host': '127.0.0.1', 'port': args.port, 'pid': os.getpid(),
            'out': str(out), 'started_unix': started_unix, 'ready_unix': time.time()})
        server.ready(step=state['step'], weight_version=state['weight_version'], update=None)
        print(f'[rank0] serving http://127.0.0.1:{args.port}', flush=True)
    while True:
        job = server.next_job() if rank == 0 else None
        command = [None if job is None else {'op': job['op'], 'body': job['body']}]
        dist.broadcast_object_list(command, src=0, group=commands)
        op, body = command[0]['op'], command[0]['body']
        started = time.perf_counter()
        try:
            result = operations[op](body)
        except BaseException:
            print(f'[rank{rank}] {op} {body} failed:\n{traceback.format_exc()}', flush=True)
            if rank == 0:
                server.finish(job, 500, {'error': traceback.format_exc()})
            raise
        if rank == 0:
            print(f'[rank0] {op} {body} done in {time.perf_counter() - started:.1f} s', flush=True)
            server.finish(job, 200, result, step=state['step'], weight_version=state['weight_version'],
                          update=update.get('id'))
        if op == 'stop':
            break
    if rank == 0:
        server.close()
    if slots is not None:
        slots.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
