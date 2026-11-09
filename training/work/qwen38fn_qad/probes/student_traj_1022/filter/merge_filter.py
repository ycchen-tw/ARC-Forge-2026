"""Merge tito-mm datasets into one, dropping listed keys: python merge_filter.py OUT DROP_KEYS.json DATA [DATA...]
Rows keep their input order (inputs in argument order); media npz are hard links (same tmpfs)."""
import json, os, sys, time
from pathlib import Path
import numpy as np

out, drop, ins = Path(sys.argv[1]), set(json.load(open(sys.argv[2]))), [Path(p) for p in sys.argv[3:]]
out.mkdir(parents=False, exist_ok=False)
mans = [json.loads((d / 'manifest.json').read_text()) for d in ins]
import copy
def proc(m):  # the serving model's location differs between collections (same files: model_files_sha256 is compared)
    p = copy.deepcopy(m['processor']); p['server_args_mm'].pop('tokenizer_path'); return p
if any(proc(m) != proc(mans[0]) for m in mans):
    raise RuntimeError('inputs differ in processor (beyond server_args_mm.tokenizer_path)')
for k in ('format', 'tokenizer_sha256', 'vision', 'max_tokens', 'split', 'loss_mask', 'mrope', 'routing'):
    if any(m.get(k) != mans[0].get(k) for m in mans):
        raise RuntimeError(f'inputs differ in {k}')
man = dict(mans[0])
man.update(created_unix=time.time(), argv=sys.argv, ledger=sum((m['ledger'] for m in mans), []),
           manifests=sum((m['manifests'] for m in mans), []), merged_from=[str(d) for d in ins],
           dropped_keys=len(drop), processor_digests=[m['processor_digest'] for m in mans],
           processors=[m['processor'] for m in mans], parity_set=sum((m.get('parity_set', []) for m in mans), []))
seen, used = set(), set()
for split in ('train', 'validation'):
    (out / split).mkdir()
    fi, fl, fp = (open(out / split / n, 'wb') for n in ('input_ids.i32', 'loss_mask.u8', 'mrope.i32'))
    idx, off, n_drop = open(out / split / 'index.jsonl', 'w'), 0, 0
    st = {'samples': 0, 'tokens': 0, 'loss_tokens': 0, 'groups': set(), 'with_images': 0, 'with_videos': 0}
    for d in ins:
        ids = np.memmap(d / split / 'input_ids.i32', dtype='<i4', mode='r')
        lm = np.memmap(d / split / 'loss_mask.u8', dtype=np.uint8, mode='r')
        pos = np.memmap(d / split / 'mrope.i32', dtype='<i4', mode='r')
        for line in open(d / split / 'index.jsonl'):
            r = json.loads(line)
            if r['key'] in seen:
                raise RuntimeError(f"duplicate key {r['key']}")
            seen.add(r['key'])
            if r['key'] in drop:
                n_drop += 1
                continue
            o, n = r['offset'], r['length']
            fi.write(np.asarray(ids[o:o + n]).tobytes()); fl.write(np.asarray(lm[o:o + n]).tobytes())
            fp.write(np.asarray(pos[3 * o:3 * (o + n)]).tobytes())
            r['offset'] = off; off += n
            idx.write(json.dumps(r, ensure_ascii=False) + '\n')
            st['samples'] += 1; st['tokens'] += n; st['loss_tokens'] += r['n_loss']; st['groups'].add(r['group'])
            st['with_images'] += any(m['modality'] == 'image' for m in r['media'])
            st['with_videos'] += any(m['modality'] == 'video' for m in r['media'])
            for m in r['media']:
                if m['sha256'] not in used:
                    used.add(m['sha256'])
                    src = d / 'media' / m['sha256'][:2] / f"{m['sha256']}.npz"
                    dst = out / 'media' / m['sha256'][:2] / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.link(src, dst)
    for f in (fi, fl, fp, idx):
        f.close()
    st['groups'] = len(st['groups'])
    man['splits'][split] = st
    print(split, st, 'dropped', n_drop, flush=True)
missing = drop - seen
if missing:
    raise RuntimeError(f'{len(missing)} drop keys not in the inputs, e.g. {sorted(missing)[:3]}')
with open(out / 'media.jsonl', 'w') as f:
    for d in ins:
        for line in open(d / 'media.jsonl'):
            m = json.loads(line)
            if m['sha256'] in used:
                used.discard(m['sha256']); f.write(line)
if used:
    raise RuntimeError(f'{len(used)} media without a media.jsonl row')
(out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1))
