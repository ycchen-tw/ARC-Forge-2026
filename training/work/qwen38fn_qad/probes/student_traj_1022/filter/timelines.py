"""arc3 per-game action timelines (timestamp, levels_completed) from each round archive's recordings -> timelines.jsonl;
ledger segment open/close times -> segtimes.json. One archive at a time through a temp dir in shm."""
import glob, json, os, re, shutil, subprocess, sys
from datetime import datetime

S = f'/dev/shm/uid-{os.getuid()}/arcforge-runs/tito/20261022-int3-qad-collect'
out = sys.argv[1]
tars = sorted(glob.glob(os.environ['ARC_NFS'] + '/runs/tito/20261019-int3-collect/arc3/*.tar.zst')) + \
    sorted(glob.glob(os.environ['ARC_NFS'] + '/runs/tito/20261022-int3-qad-collect/arc3/*.tar.zst'))
tmp = f'{S}/tl-tmp'
ts_re = re.compile(r'"timestamp": "([^"]+)"')
lv_re = re.compile(r'"levels_completed": (\d+)')
with open(f'{out}/timelines.jsonl', 'w') as f:
    for t in tars:
        shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        subprocess.run(['tar', '-I', 'zstd', '-xf', t, '-C', tmp, '--wildcards', '*/recordings/*.jsonl'], check=True)
        rnd = os.path.basename(t)[:-len('.tar.zst')]
        for rec in glob.glob(f'{tmp}/*/run/recordings/*/*.jsonl'):
            ts, lv = [], []
            for line in open(rec):
                head = line[:300]
                m = ts_re.search(head); l = lv_re.search(line[:400])
                ts.append(datetime.fromisoformat(m.group(1)).timestamp()); lv.append(int(l.group(1)) if l else -1)
            f.write(json.dumps({'round': rnd, 'game': os.path.basename(rec).split('-')[0], 'file': os.path.basename(rec),
                                'ts': ts, 'lv': lv}) + '\n')
        print(rnd, flush=True)
shutil.rmtree(tmp, ignore_errors=True)
seg = {}
for ev in glob.glob(f'{S}/ledger1019/*/*/events.*.jsonl') + glob.glob(f'{S}/ledger/*/*/events.*.jsonl'):
    inst = ev.split('/')[-2]
    for line in open(ev):
        if line.startswith('{"ev":"seg_open"') or line.startswith('{"ev":"seg_close"'):
            d = json.loads(line)
            seg.setdefault(f"{inst}:{d['seg']}", {})['open' if d['ev'] == 'seg_open' else 'close'] = d['t']
json.dump(seg, open(f'{out}/segtimes.json', 'w'))
print('segs', len(seg))
