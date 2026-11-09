#!/usr/bin/env python3
"""Replay, in isolation, the retained per-task independent-construction comparisons and semantic counterexamples; missing tests cannot count as PASS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / 'work/seed_bank'
sys.path.insert(0, str(ROOT / 'src'))
from arcforge import paths
SPECIAL = {'00576224','007bbfb7','00d62c1b'}


def source_hashes(directory):
    sources = [p for p in directory.rglob('*.py') if '__pycache__' not in p.parts]
    sources += [directory / 'seed.json']
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources) if p.is_file()}


def command(task_id):
    directory = BANK / 'seeds/training' / task_id
    scripts = sorted(p for p in directory.glob('*.py') if 'semantic' in p.name)
    if scripts:
        raw = paths.data() / 'ARC-AGI-2/data/training' / f'{task_id}.json'
        return [[sys.executable,str(script),str(raw)] for script in scripts]
    if task_id in SPECIAL:
        return [[sys.executable,str(BANK/'batches/20261003-training-lead-checks.py'),task_id]]
    if task_id == '98c475bf':
        return [[sys.executable,'-m','pytest',str(ROOT/'tests/test_training_seed_context.py'),'-q']]
    for runner in sorted((BANK/'batches').glob('20261003-training-astra*-checks.py')):
        assignment=runner.with_name(runner.name.replace('-checks.py','.json'))
        if assignment.is_file() and task_id in json.loads(assignment.read_text()).get('task_ids',[]):
            return [[sys.executable,str(runner),task_id]]
    raise ValueError('no retained semantic test source')


def run(task_id, timeout):
    row = {'task_id':task_id,'status':'FAIL','checks':[]}
    directory=BANK/'seeds/training'/task_id
    before=source_hashes(directory)
    replay_before=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    row['external_check_sha256']={}
    try:
        for args in command(task_id):
            for value in args[1:]:
                path=Path(value)
                if path.suffix=='.py' and path.is_file():
                    row['external_check_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            result = subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=timeout,
                                    env=dict(os.environ,PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
            row['checks'].append({'command':[str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x for x in args[1:]],
                                  'returncode':result.returncode,'stdout':result.stdout[-5000:],'stderr':result.stderr[-5000:]})
        row['status'] = 'PASS' if all(x['returncode']==0 for x in row['checks']) else 'FAIL'
    except (ValueError,subprocess.TimeoutExpired) as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    after=source_hashes(directory)
    row['source_sha256']=before
    row['replay_sha256']=replay_before
    raw=paths.data()/'ARC-AGI-2/data/training'/f'{task_id}.json'
    if raw.is_file():row['official_task_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
    if after!=before:
        row.update(status='FAIL',error='source changed during semantic replay')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=replay_before:
        row.update(status='FAIL',error='replay validator changed during execution')
    for relative, digest in row['external_check_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            row.update(status='FAIL',error='external semantic check changed during execution')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_ids',nargs='*')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--timeout',type=float,default=180)
    args=parser.parse_args()
    ids=args.task_ids or sorted(p.parent.name for p in (BANK/'seeds/training').glob('*/seed.json'))
    if not ids or len(set(ids))!=len(ids) or any(not re.fullmatch('[0-9a-f]{8}',x) for x in ids):
        parser.error('task IDs must be unique canonical training IDs')
    args.out.mkdir(parents=True,exist_ok=True)
    def one(task_id):
        row=run(task_id,args.timeout)
        (args.out/f'{task_id}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(task_id,row['status'],flush=True)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(one,ids))
    summary={'task_count':len(rows),'passed':sum(r['status']=='PASS' for r in rows),'tasks':rows,
             'semantic_completeness':'construction agreement and counterexamples, not text-only reconstruction'}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return 0 if summary['passed']==len(rows) else 1


if __name__=='__main__':
    sys.exit(main())
