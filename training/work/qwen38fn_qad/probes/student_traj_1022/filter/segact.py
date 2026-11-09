"""Join arc3 segments with action timelines: actions taken and levels gained while each segment was open -> segact.jsonl"""
import bisect, collections, json
rows = [json.loads(l) for l in open('segstats.jsonl')]
seg = json.load(open('segtimes.json'))
tl = collections.defaultdict(list)
for l in open('timelines.jsonl'):
    d = json.loads(l)
    tl[(d['round'], d['game'])] += list(zip(d['ts'], d['lv']))
for k in tl:
    tl[k].sort()
miss = collections.Counter()
with open('segact.jsonl', 'w') as f:
    for r in rows:
        if r['harness'] != 'arc3':
            continue
        t = seg.get(r['key']); g = tl.get((r.get('round'), r['task']))
        if not t or 'close' not in t or not g:
            miss['no-time' if not t or 'close' not in t else 'no-timeline'] += 1; continue
        ts = [x[0] for x in g]
        a, b = bisect.bisect_left(ts, t['open']), bisect.bisect_right(ts, t['close'])
        lv0 = max([x[1] for x in g[:a]] or [0]); lv1 = max([x[1] for x in g[:b]] or [0])
        f.write(json.dumps({'key': r['key'], 'round': r['round'], 'game': r['task'], 'n_loss': r['n_loss'],
                            'actions': b - a, 'lv0': lv0, 'gained': lv1 - lv0, 'minutes': (t['close'] - t['open']) / 60}) + '\n')
print(miss)
