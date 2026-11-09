"""Audit saved tokens against source prompts and report corpus progress."""
import argparse
from collections import Counter
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from generate import atomic_json, model_eos_ids, render


def main(a):
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    eos=model_eos_ids(a.model,tok)
    files=sorted((a.out/'tokens').glob('part-*.parquet'))
    if not files:
        raise RuntimeError('no completed shards')
    counts=Counter()
    seen=set()
    for file in files:
        for row in pq.read_table(file,columns=['uid','effort','n_prompt','n_completion','finish_reason']).to_pylist():
            key=(row['uid'],row['effort'])
            if key in seen:
                raise ValueError(f'duplicate key: {key}')
            seen.add(key)
            counts['rows']+=1
            counts['prompt_tokens']+=row['n_prompt']
            counts['completion_tokens']+=row['n_completion']
            counts['effort_'+row['effort']]+=1
            counts['finish_'+row['finish_reason']]+=1
    db=sqlite3.connect(f'file:{a.out / "prompts.sqlite"}?mode=ro',uri=True)
    selected=sorted({files[round(i*(len(files)-1)/max(1,min(a.samples,len(files))-1))]
                     for i in range(min(a.samples,len(files)))})
    checked=0
    for file in selected:
        rows=pq.read_table(file).to_pylist()
        for row in rows[:max(1,a.samples//len(selected))]:
            message=db.execute('SELECT messages FROM prompts WHERE uid=?',(row['uid'],)).fetchone()
            assert message is not None
            ids=render(tok,json.loads(message[0]),row['effort'])
            n=row['n_prompt']
            assert n==len(ids) and row['input_ids'][:n]==ids
            assert len(row['input_ids'])==len(row['loss_mask'])==n+row['n_completion']
            assert row['loss_mask']==[0]*n+[1]*row['n_completion']
            assert row['n_completion']>0
            if row['finish_reason']=='stop':
                assert row['input_ids'][-1] in eos
                assert json.loads(row['meta_info'])['finish_reason']['matched']==row['input_ids'][-1]
            assert json.loads(row['provenance'])
            checked+=1
    db.close()
    report={**dict(counts),'prompt_token_checks':checked,'shards':len(files)}
    atomic_json(a.out/'audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--samples',type=int,default=64)
    main(p.parse_args())
