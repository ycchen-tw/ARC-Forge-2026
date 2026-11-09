"""Close time of segments that never got seg_close (rounds stopped mid-game): their last append's t1 -> segtimes.json"""
import glob, json, os, re
S = f'/dev/shm/uid-{os.getuid()}/arcforge-runs/tito/20261022-int3-qad-collect'
seg = json.load(open('segtimes.json'))
want = {k for k, v in seg.items() if 'close' not in v}
pat = re.compile(r'"seg":(\d+),.*?"t1":([0-9.]+)')
n = 0
for ev in glob.glob(f'{S}/ledger1019/*/*/events.*.jsonl') + glob.glob(f'{S}/ledger/*/*/events.*.jsonl'):
    inst = ev.split('/')[-2]
    for line in open(ev):
        if line.startswith('{"ev":"append"'):
            m = pat.search(line[:600])
            k = f'{inst}:{m.group(1)}'
            if k in want:
                seg[k]['close'] = max(seg[k].get('close', 0), float(m.group(2))); seg[k]['close_from'] = 'last_append'; n += 1
json.dump(seg, open('segtimes.json', 'w'))
print('unclosed', len(want), 'appends', n, 'still open', sum('close' not in seg[k] for k in want))
