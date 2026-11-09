#!/usr/bin/env python3
"""Replay a collection's saved examples in isolated package processes.

Only standard Python and NumPy are required. This is a finite output/format
check; it does not certify semantic originality, difficulty, or universal input
validity. No task files are modified.
"""
import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def check_package(directory, generated):
    import numpy as np
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(directory))
    transform=importlib.import_module('transform').transform
    generator=importlib.import_module('generate_input')
    check_input=generator.check_input
    task=json.loads((directory/'task.json').read_text())
    saved=0;seen=set()
    def legal(grid):
        return grid.ndim==2 and 1<=min(grid.shape)<=max(grid.shape)<=30 and np.issubdtype(grid.dtype,np.integer) and bool(np.all((grid>=0)&(grid<=9)))
    for split in ('train','test'):
        if not task.get(split):raise AssertionError(f'empty {split}')
        for index,pair in enumerate(task[split]):
            a=np.array(pair['input']);expected=np.array(pair['output']);before=a.copy()
            if not legal(a) or not legal(expected):raise AssertionError(f'illegal stored grid {split}/{index}')
            if not check_input(a):raise AssertionError(f'rejected stored input {split}/{index}')
            actual=np.asarray(transform(a));again=np.asarray(transform(a.tolist()))
            if not np.array_equal(a,before):raise AssertionError(f'input mutation {split}/{index}')
            if not legal(actual) or not np.array_equal(actual,expected) or not np.array_equal(again,expected):raise AssertionError(f'output mismatch {split}/{index}')
            digest=hashlib.sha256(str(a.shape).encode()+a.astype(np.int64).tobytes()).hexdigest()
            if digest in seen:raise AssertionError(f'duplicate saved input {split}/{index}')
            seen.add(digest);saved+=1
    rng=np.random.default_rng(202610401);replay=np.random.default_rng(202610401)
    for index in range(generated):
        a=np.asarray(generator.generate_input(rng));b=np.asarray(generator.generate_input(replay));before=a.copy()
        if not legal(a) or not np.array_equal(a,b) or not check_input(a):raise AssertionError(f'generator replay/domain failure {index}')
        result=np.asarray(transform(a));again=np.asarray(transform(a))
        if not legal(result) or not np.array_equal(a,before) or not np.array_equal(result,again):raise AssertionError(f'generated transform contract failure {index}')
    return {'package':directory.name,'status':'PASS','saved_pairs':saved,'generated_contract_cases':generated,'generated_seed':202610401 if generated else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--ids',nargs='*',help='Optional exact package IDs')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--generated',type=int,default=0,help='Additional replay/domain/format cases, without an independent expected-output claim')
    parser.add_argument('--output',type=Path,help='Optional JSON result file')
    parser.add_argument('--package',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.generated<0 or args.jobs<1:parser.error('nonnegative --generated and positive --jobs required')
    if args.package:
        try:result=check_package(args.package.resolve(),args.generated)
        except Exception as error:result={'package':args.package.name,'status':'FAIL','error':f'{type(error).__name__}: {error}'}
        print(json.dumps(result,ensure_ascii=False));return 0 if result['status']=='PASS' else 1
    directory=args.root.resolve()/'packages'
    if not directory.is_dir():parser.error(f'no packages directory under {args.root}')
    paths=sorted(p.parent for p in directory.glob('ind_*/task.json'))
    if args.ids:
        wanted=set(args.ids);paths=[p for p in paths if p.name in wanted]
        missing=wanted-{p.name for p in paths}
        if missing:parser.error('unknown package IDs: '+','.join(sorted(missing)))
    def run(path):
        environment=dict(os.environ);environment['PYTHONDONTWRITEBYTECODE']='1'
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--package',str(path),'--generated',str(args.generated)],cwd=path,capture_output=True,text=True,env=environment,timeout=120)
            return json.loads(result.stdout)
        except Exception as error:return {'package':path.name,'status':'FAIL','error':f'{type(error).__name__}: {error}'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs,16)) as pool:results=list(pool.map(run,paths))
    summary={'status':'PASS' if results and all(r['status']=='PASS' for r in results) else 'FAIL','packages':len(results),'saved_pairs':sum(r.get('saved_pairs',0) for r in results),'generated_contract_cases':sum(r.get('generated_contract_cases',0) for r in results),'scope':'Stored expected-output replay plus optional generator replay/domain/format checks; no semantic or difficulty certification','results':results}
    if args.output:args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False))
    for result in results:
        if result['status']!='PASS':print(json.dumps(result,ensure_ascii=False))
    return 0 if summary['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
