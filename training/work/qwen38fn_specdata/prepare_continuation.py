"""Freeze a stopped partial run and select only its uncollected prompts."""
import argparse
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from finalize import assigned_efforts
from generate import atomic_json, digest, exclusive


def prepare(parent, out, max_new_tokens):
    parent, out = parent.resolve(), out.resolve()
    if out.exists() or max_new_tokens <= 0:
        raise ValueError('new output directory and positive output budget required')
    with exclusive(parent, 'supervise'), exclusive(parent, 'generate'):
        config = json.loads((parent / 'generation.json').read_text())
        if config['all_efforts'] or (parent / 'invocation_complete.json').exists():
            raise ValueError('requires a partial single-effort run')
        source = sqlite3.connect(f'file:{parent / "prompts.sqlite"}?mode=ro', uri=True)
        try:
            expected = {uid for (uid,) in source.execute('SELECT uid FROM prompts')}
            done = set()
            for file in sorted((parent / 'tokens').glob('part-*.parquet')):
                for row in pq.read_table(file, columns=['uid', 'effort']).to_pylist():
                    uid = row['uid']
                    if uid not in expected or uid in done or (row['effort'],) != assigned_efforts(uid, config):
                        raise ValueError('invalid completed prompt inventory')
                    done.add(uid)
            remaining = expected - done
            if not remaining:
                raise ValueError('no remaining prompts')
            with exclusive(out, 'prepare'):
                db = sqlite3.connect(out / 'prompts.sqlite')
                with db:
                    db.executescript('''
                        CREATE TABLE prompts(uid TEXT PRIMARY KEY,messages TEXT NOT NULL);
                        CREATE TABLE sources(uid TEXT,file TEXT,row_index INT,turn INT,metadata TEXT,
                            PRIMARY KEY(file,row_index,turn));
                        CREATE INDEX sources_uid ON sources(uid);
                    ''')
                    for uid in sorted(remaining):
                        message = source.execute('SELECT messages FROM prompts WHERE uid=?', (uid,)).fetchone()[0]
                        provenance = source.execute('SELECT uid,file,row_index,turn,metadata FROM sources WHERE uid=?', (uid,)).fetchall()
                        if digest(json.loads(message)) != uid or not provenance:
                            raise ValueError('invalid prompt or missing provenance')
                        db.execute('INSERT INTO prompts VALUES (?,?)', (uid, message))
                        db.executemany('INSERT INTO sources VALUES (?,?,?,?,?)', provenance)
                source_turns = db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]
                db.close()
                lineage = {'parent': str(parent), 'parent_generation_sha256': digest(config),
                           'completed_parent_rows': len(done), 'remaining_prompts': len(remaining),
                           'remaining_uids_sha256': digest(sorted(remaining)),
                           'reason': 'user requested shorter outputs and prompt diversity',
                           'max_new_tokens': max_new_tokens}
                atomic_json(out / 'source.json', lineage)
                atomic_json(out / 'prepared.json', {'source_sha256': digest(lineage),
                            'prompts': len(remaining), 'source_turns': source_turns})
                atomic_json(parent / 'partial.json', {**lineage, 'continuation': str(out),
                            'status': 'superseded_partial', 'expected_prompts': len(expected)})
                print(json.dumps(lineage), flush=True)
                return lineage
        finally:
            source.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, required=True)
    a = p.parse_args()
    prepare(a.parent, a.out, a.max_new_tokens)
