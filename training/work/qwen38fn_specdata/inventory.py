"""Inventory frozen Parquet runs without claiming full token/template validation."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from generate import atomic_json, exclusive


def inventory(runs, out):
    runs = [Path(p).resolve() for p in runs]
    if len(set(runs)) != len(runs) or out.exists():
        raise ValueError('unique runs and a new inventory output directory required')
    with ExitStack() as stack:
        for run in runs:
            for lock in ('generate', 'finalize'):
                stack.enter_context(exclusive(run, lock))
        seen, reports, checksums = set(), [], []
        for run in runs:
            config = json.loads((run/'generation.json').read_text())
            counts = Counter()
            files = sorted((run/'tokens').glob('part-*.parquet'))
            if not files:
                raise ValueError(f'no shards: {run}')
            with sqlite3.connect(f'file:{run / "prompts.sqlite"}?mode=ro', uri=True) as db:
                expected = {r[0] for r in db.execute('SELECT uid FROM prompts')}
            for file in files:
                columns = ['uid', 'effort', 'n_prompt', 'n_completion', 'finish_reason', 'split']
                for row in pq.read_table(file, columns=columns).to_pylist():
                    uid = row['uid']
                    if uid in seen or uid not in expected:
                        raise ValueError(f'duplicate or unknown prompt: {uid}')
                    if not 0 < row['n_completion'] <= config['max_new_tokens'] or row['n_prompt'] + row['n_completion'] > config['max_length']:
                        raise ValueError(f'budget metadata exceeded: {uid}')
                    seen.add(uid)
                    counts['rows'] += 1
                    counts['prompt_tokens'] += row['n_prompt']
                    counts['completion_tokens'] += row['n_completion']
                    for key in ('effort', 'finish_reason', 'split'):
                        counts[f'{key}_{row[key]}'] += 1
                h = hashlib.md5()
                with file.open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b''):
                        h.update(block)
                checksums.append(f'{h.hexdigest()}  {run.name}/tokens/{file.name}')
            reports.append({'run': run.name, 'max_new_tokens': config['max_new_tokens'],
                            'shards': len(files), **dict(counts)})
        report = {'validation': 'all-row metadata, cross-run UID uniqueness, Parquet MD5; not full token/template audit',
                  'rows': len(seen), 'runs': reports}
        out.mkdir(parents=True)
        atomic_json(out/'inventory.json', report)
        (out/'MANIFEST.md5').write_text('\n'.join(checksums)+'\n')
        print(json.dumps(report), flush=True)
        return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    inventory(a.runs, a.out)
