"""Per-segment stats for filtering button-mashing / looping trajectories (data1019 + data1022).
Out: segstats.jsonl (one row per train/validation segment)."""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool

S = f'/dev/shm/uid-{os.getuid()}/arcforge-runs/tito/20261022-int3-qad-collect'
N = 16
R = np.random.default_rng(0).integers(1, 2**63, size=N, dtype=np.uint64) | np.uint64(1)

def outcomes():
    mans = glob.glob(os.environ['ARC_NFS'] + '/runs/tito/20261019-int3-collect/arc2/*/tito_manifest.json') \
        + glob.glob(f'{S}/manifests1019/arc3/*/run/tito_manifest.json') \
        + glob.glob(f'{S}/manifests1022/arc2/*/tito_manifest.json') + glob.glob(f'{S}/manifests1022/arc3/*/run/tito_manifest.json')
    th = {}
    for m in mans:
        d = json.load(open(m))
        rnd = m.split('/manifests10')[-1].split('/')[2] if '/manifests10' in m else m.split('/')[-2]
        cards = {}
        for c in glob.glob(f'{S}/arc3cards/*/{rnd}/run/gateway/official_out/scorecard_official.json'):
            for e in json.load(open(c))['environments']:
                r = e['runs'][-1]
                cards[e['id'].split('-')[0]] = {k: r[k] for k in ('levels_completed', 'actions', 'resets', 'level_actions', 'state')}
        for it in d['items']:
            for t in it['threads']:
                th[t['thread']] = {'round': rnd, 'item': it['id'], 'outcome': it.get('outcome'), 'card': cards.get(it['id'])}
    return th

def seg(args):
    d, split, r = args
    ids = np.memmap(f'{d}/{split}/input_ids.i32', dtype=np.int32, mode='r')
    lm = np.memmap(f'{d}/{split}/loss_mask.u8', dtype=np.uint8, mode='r')
    a = np.asarray(ids[r['offset']:r['offset'] + r['length']]).astype(np.uint64)
    m = np.asarray(lm[r['offset']:r['offset'] + r['length']])
    e = np.flatnonzero(np.diff(np.r_[0, m, 0].astype(np.int8)))
    spans = list(zip(e[::2], e[1::2]))
    seen, dup, dup_tok, loop_tok, maxrep, lens = set(), 0, 0, 0, 0.0, []
    for s, t in spans:
        x = a[s:t]; lens.append(int(t - s))
        h = hash(x.tobytes())
        if h in seen:
            dup += 1; dup_tok += int(t - s)
        seen.add(h)
        if t - s >= 512:
            g = np.zeros(len(x) - N + 1, dtype=np.uint64)
            for j in range(N):
                g += x[j:len(x) - N + 1 + j] * R[j]
            rep = 1 - len(np.unique(g)) / len(g)
            maxrep = max(maxrep, rep)
            if rep > 0.5:
                loop_tok += int(t - s)
    return {'data': os.path.basename(d), 'split': split, 'key': r['key'], 'harness': r['harness'], 'task': r['task'],
            'thread': r['thread'], 'length': r['length'], 'n_loss': r['n_loss'], 'n_spans': len(spans),
            'dup_spans': dup, 'dup_tok': dup_tok, 'loop_tok': loop_tok, 'max_rep16': round(maxrep, 4),
            'max_span': max(lens, default=0), 'close_reason': r['close_reason']}

if __name__ == '__main__':
    th = outcomes()
    jobs = []
    for d in (f'{S}/data1019', f'{S}/data1022'):
        for split in ('train', 'validation'):
            jobs += [(d, split, json.loads(l)) for l in open(f'{d}/{split}/index.jsonl')]
    with Pool(32) as p, open(sys.argv[1], 'w') as f:
        for row in p.imap(seg, jobs, chunksize=16):
            row.update(th.get(row['thread']) or {})
            f.write(json.dumps(row) + '\n')
