"""Verify complete prompt coverage and export exact-token SpecForge JSONL."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from generate import EFFORTS, atomic_json, canonical, digest, exclusive, model_eos_ids, render


def assigned_efforts(uid, config):
    if config['all_efforts']:
        return EFFORTS
    return (EFFORTS[int(digest([config['seed'], uid, 'effort'])[:8], 16) % 3],)


def check_row(row, messages, config, tok, eos=None):
    uid, effort = row['uid'], row['effort']
    if digest(messages) != uid or effort not in assigned_efforts(uid, config):
        raise ValueError(f'prompt/effort mismatch: {uid}/{effort}')
    prefix = render(tok, messages, effort)
    n, m = row['n_prompt'], row['n_completion']
    ids, mask = row['input_ids'], row['loss_mask']
    if n != len(prefix) or ids[:n] != prefix:
        raise ValueError(f'prompt tokens differ: {uid}/{effort}')
    if m <= 0 or len(ids) != n + m or len(mask) != n + m:
        raise ValueError(f'bad token/mask lengths: {uid}/{effort}')
    if mask != [0] * n + [1] * m:
        raise ValueError(f'incorrect supervision: {uid}/{effort}')
    if len(ids) > config['max_length'] or m > config['max_new_tokens']:
        raise ValueError(f'budget exceeded: {uid}/{effort}')
    if any(type(t) is not int or not 0 <= t < len(tok) for t in ids):
        raise ValueError(f'invalid token IDs: {uid}/{effort}')
    meta = json.loads(row['meta_info'])
    if meta['prompt_tokens'] != n or meta['completion_tokens'] != m:
        raise ValueError(f'API token accounting mismatch: {uid}/{effort}')
    if meta['finish_reason']['type'] != row['finish_reason']:
        raise ValueError(f'finish reason mismatch: {uid}/{effort}')
    if row['finish_reason'] == 'stop':
        if ids[-1] not in ({tok.eos_token_id} if eos is None else eos):
            raise ValueError(f'completed row missing EOS: {uid}/{effort}')
        if meta['finish_reason'].get('matched') != ids[-1]:
            raise ValueError(f'non-token stop or invalid synthesized EOS: {uid}/{effort}')
    elif row['finish_reason'] != 'length':
        raise ValueError(f'unexpected finish reason: {uid}/{effort}')
    split = 'validation' if int(uid[:8], 16) % 100 == 0 else 'train'
    if row['split'] != split:
        raise ValueError(f'train/validation assignment mismatch: {uid}/{effort}')
    provenance = json.loads(row['provenance'])
    if not provenance or any(not p['uuid'] or not p['license'] for p in provenance):
        raise ValueError(f'missing source attribution: {uid}/{effort}')
    return {'id': f'{uid}:{effort}', 'input_ids': ids, 'loss_mask': mask,
            'metadata': {'uid': uid, 'effort': effort, 'n_prompt': n, 'n_completion': m,
                         'finish_reason': row['finish_reason'], 'sources': provenance}}


def finalize(args, *, supplements=(), export_name='specforge', require_all=False):
    out = args.out.resolve()
    # A live generator holds the same lock; never label a changing snapshot complete.
    with exclusive(out, 'generate'), exclusive(out, 'finalize'):
        config = json.loads((out / 'generation.json').read_text())
        prepared = json.loads((out / 'prepared.json').read_text())
        invocation = json.loads((out / 'invocation_complete.json').read_text())
        if invocation['limit']:
            raise ValueError('limited invocation is not complete collection')
        if config['source'] != prepared:
            raise ValueError('generation used a different prompt selection')
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        eos = model_eos_ids(args.model, tok)
        if digest(tok.chat_template) != config['template_sha256']:
            raise ValueError('teacher template changed')
        model = Path(args.model).resolve()
        if str(model) != config['model']:
            raise ValueError('teacher checkpoint path changed')
        for name, expected in config['tokenizer_files'].items():
            if hashlib.sha256((model / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'teacher asset changed: {name}')
        if not export_name or export_name in ('.', '..') or Path(export_name).name != export_name:
            raise ValueError('export_name must be one directory name inside the run')
        export = out / export_name
        export.mkdir(exist_ok=True)
        if (export / 'complete.json').exists():
            raise FileExistsError('already finalized; inspect specforge/complete.json')
        db = sqlite3.connect(f'file:{out / "prompts.sqlite"}?mode=ro', uri=True)
        expected = {(uid, e) for (uid,) in db.execute('SELECT uid FROM prompts')
                    for e in assigned_efforts(uid, config)}
        if db.execute('SELECT COUNT(*) FROM prompts').fetchone()[0] != prepared['prompts']:
            raise ValueError('source DB count changed')
        skipped = set()
        skip_file = out / 'skipped.jsonl'
        if skip_file.exists():
            for line in skip_file.read_text().splitlines():
                record = json.loads(line)
                key = record['uid'], record['effort']
                if key not in expected or key in skipped:
                    raise ValueError(f'invalid skipped key: {key}')
                messages = json.loads(db.execute('SELECT messages FROM prompts WHERE uid=?', (key[0],)).fetchone()[0])
                ids = render(tok, messages, key[1])
                if record['n_prompt'] != len(ids) or config['max_length'] - len(ids) >= 128:
                    raise ValueError(f'unjustified skipped prompt: {key}')
                skipped.add(key)
        runs = [(out, config, invocation)]
        same = ('model', 'tokenizer_files', 'template_sha256', 'temperature', 'top_p', 'top_k',
                'all_efforts', 'seed', 'max_new_tokens')
        for path in supplements:
            path = Path(path).resolve()
            if path == out or any(path == r[0] for r in runs):
                raise ValueError('duplicate supplement run')
            extra = json.loads((path / 'generation.json').read_text())
            finished = json.loads((path / 'invocation_complete.json').read_text())
            if finished['limit'] or any(extra[key] != config[key] for key in same):
                raise ValueError('supplement changed teacher/sampling or is not fully finished')
            if extra['max_length'] < config['max_length']:
                raise ValueError('backfill context must not be smaller')
            runs.append((path, extra, finished))
        seen, stats, run_reports = set(), Counter(), []
        paths = {s: export / f'{s}.jsonl.tmp' for s in ('train', 'validation')}
        streams = {s: p.open('wb') for s, p in paths.items()}
        hashes = {s: hashlib.sha256() for s in paths}
        try:
            for path, run_config, finished in runs:
                run_rows = 0
                for file in sorted((path / 'tokens').glob('part-*.parquet')):
                    if shutil.disk_usage(out).free < 5 * 1024**3:
                        raise RuntimeError('less than 5 GiB disk free; export stopped safely')
                    for row in pq.read_table(file).to_pylist():
                        key = row['uid'], row['effort']
                        if key not in expected or key in seen:
                            raise ValueError(f'unexpected/duplicate generated key: {key}')
                        if (path == out and key in skipped) or (path != out and key not in skipped):
                            raise ValueError(f'backfill must match the main skipped inventory: {key}')
                        messages = json.loads(db.execute('SELECT messages FROM prompts WHERE uid=?', (key[0],)).fetchone()[0])
                        record = check_row(row, messages, run_config, tok, eos)
                        record['metadata']['generation_config_sha256'] = digest(run_config)
                        seen.add(key)
                        run_rows += 1
                        split = row['split']
                        encoded = (canonical(record) + '\n').encode()
                        streams[split].write(encoded)
                        hashes[split].update(encoded)
                        stats['rows'] += 1
                        stats[split + '_rows'] += 1
                        stats['prompt_tokens'] += row['n_prompt']
                        stats['completion_tokens'] += row['n_completion']
                        stats['effort_' + row['effort']] += 1
                        stats['finish_' + row['finish_reason']] += 1
                    if stats['rows'] % 1024 == 0:
                        print(canonical(dict(stats)), flush=True)
                if finished['total_completed'] != run_rows:
                    raise ValueError('invocation completed count differs from verified rows')
                run_reports.append({'run': str(path), 'rows': run_rows, 'generation_sha256': digest(run_config)})
            missing = expected - seen - skipped
            if missing:
                raise ValueError(f'{len(missing)} requested prompt/effort pairs not collected')
            if invocation['total_skipped'] != len(skipped):
                raise ValueError('invocation skipped count differs from verified inventory')
            unresolved = skipped - seen
            if require_all and unresolved:
                raise ValueError(f'{len(unresolved)} selected prompt/effort pairs still need backfill')
        finally:
            for stream in streams.values():
                stream.close()
            db.close()
        outputs = {}
        for split, tmp in paths.items():
            dest = export / f'{split}.jsonl'
            os.replace(tmp, dest)
            outputs[split] = {'file': dest.name, 'bytes': dest.stat().st_size,
                              'sha256': hashes[split].hexdigest()}
        report = {'complete': True, 'expected': len(expected), 'verified_rows': len(seen),
                  'verified_skips': len(unresolved), 'all_prompts_generated': not unresolved,
                  'source_sha256': prepared['source_sha256'], 'runs': run_reports,
                  'generation_config_sha256': digest(config), 'outputs': outputs, **dict(stats)}
        atomic_json(export / 'complete.json', report)
        print(canonical(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--wait-supervisor', type=int)
    parser.add_argument('--supplement', type=Path, action='append', default=[])
    parser.add_argument('--export-name', default='specforge')
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    if args.wait_supervisor:
        from supervise import identity
        started = identity(args.wait_supervisor)
        if started is None:
            raise RuntimeError('requested supervisor is not live')
        print(f'waiting for live supervisor {args.wait_supervisor}', flush=True)
        while identity(args.wait_supervisor) == started:
            time.sleep(30)
        status = json.loads((args.out / 'supervisor.json').read_text())
        if status['status'] != 'complete':
            raise RuntimeError(f'generation did not complete: {status}')
    finalize(args, supplements=args.supplement, export_name=args.export_name, require_all=args.require_all)
