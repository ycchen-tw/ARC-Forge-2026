"""Build the DFlash data plan: which source items become capture jobs, how often, in which split, in what order.

    python make_data_plan.py --out <new dir> --target-profile <profile.json> --seed 0 \
        --node1 <ledger instance dir> ... --agentic <parquet dir> --naive <parquet dir> --nemotron <tokens dir> ... \
        --fraction node1=1 agentic=0.5 naive=1 nemotron=0.25 --passes node1=2 agentic=1 naive=1 nemotron=1 \
        [--context-length 220000] [--heldout-tokens 3000000] [--arc3-heldout-games 2] [--heldout-parents 8] \
        [--image-detail high]

CPU only (anchors come from the masks). Writes ``plan.jsonl`` (one job a line, in the order producers capture them),
``plan_meta.json`` (settings, sources, the held-out choice, plan.jsonl's sha256) and ``summary.json`` (also printed).
data_plan.py documents the masks, groups and splits.

- One source item (a ledger segment or a parquet row) is one job. Items that cannot be one are skipped and counted:
  longer than ``--context-length`` - 1, a media pad in a text row, fewer than 7 supervised tokens, no eligible anchor,
  media recorded by reference, a group held out in another corpus, or the unsampled rest of a held-out NVARC family.
- ``--codex1015`` instances (the 10-14..16 codex collection) become corpora arc2 and arc3 by lane; their held-out is
  ``--codex-heldout`` per bucket (arc2, and arc3's subsets): that many tasks / games, ``--codex-heldout-groups`` first,
  each a fixed sample of its segments (choose_codex_heldout).
- ``--int3c1019`` (the 10-19 INT3 collection) and ``--bench`` (the 10-06 / 10-08 FP8 bench rounds) instances join
  arc2 and arc3 the same way, train only, their source in the subset name (data_plan.classify_int3c1019 /
  classify_bench).
- ``--heldout-from <plan dir>``: that plan's held-out jobs, exactly (same key, same ids_sha256), are this plan's
  held-out, and nothing else is: no other held-out choice, no group excluded from training (a drafter's held-out only
  has to keep the scored segments out of training; the plans stay comparable on the same features).
- Held-out: about ``--heldout-tokens`` per corpus, groups in a fixed hash order (independent of --seed). node1 holds
  out whole ARC-AGI-2 training1000 tasks without an NVARC family, in that order to the size, plus
  ``--arc3-heldout-games`` whole ARC-AGI-3 games; agentic and naive hold out the same first ``--heldout-parents``
  parents both have, each corpus a fixed sample of each family worth size / parents tokens (choose_families); nemotron
  takes the smallest data.split_for_key permyriad that reaches the size. Held-out jobs are pass 1 and first in the
  plan.
- Train: each corpus keeps an item with probability ``--fraction`` (per item, by hash of seed and key) and repeats it
  ``--passes`` times. The order mixes corpora: pass p of a corpus with P passes sits at (p - 1 + u) / P of the plan
  (u per job from the seed), so every stretch of the plan carries each corpus in its overall proportion and two passes
  of one item are never close.
- Deterministic: same sources, arguments and seed, same plan.jsonl bytes.
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import capture_tito as ct
import capture_tito_runtime as rt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from data import split_for_key
from data_plan import (
    ANCHOR_RULE,
    BLOCK_SIZE,
    CORPORA,
    HELDOUT_SALT,
    MIN_SUPERVISED,
    PARQUET_COLUMNS,
    SKIP_REASONS,
    ContractError,
    Markers,
    anchor_count,
    CODEX_ARC3_SUBSETS,
    node1_group,
    classify_bench,
    classify_codex1015,
    classify_int3c1019,
    classify_threads,
    classify_workspace,
    codex1015_group,
    ids_sha256,
    kept_rounds,
    parquet_group,
    parquet_key,
    parquet_subset,
    row_ids_mask,
    text_skip,
    tito_mask,
    train_only_group,
    unit_hash,
    workspace,
)

from arcforge.tito.export import verify_media_blobs
from arcforge.tito.ledger import Vocab, load_instance, sha256_file

_MARKERS = None


def _init(model_dir):
    global _MARKERS
    _MARKERS = Markers(Vocab(model_dir))


def _columns(table):
    """name -> python list, or (offsets, values) for a list column (decoding millions of ids to python is too slow)."""
    out = {}
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        if col.null_count:
            raise ContractError(f"{name}: {col.null_count} nulls")
        if pa.types.is_list(col.type):
            if col.offset != 0:
                raise ValueError(f"{name}: a sliced list column")
            out[name] = (col.offsets.to_numpy(), col.values.to_numpy())
        else:
            out[name] = col.to_pylist()
    return out


def _row(cols, i):
    return {k: (v[1][v[0][i]:v[0][i + 1]] if isinstance(v, tuple) else v[i]) for k, v in cols.items()}


def scan_parquet(task):
    """Items of one parquet file. An agentic file holds whole tasks (render_segments.py writes a task's rows together),
    so every compacted segment's previous segment is in the same file."""
    corpus, path, file_index, batch, context_length = task
    table = pq.read_table(path, columns=PARQUET_COLUMNS[corpus])
    cols = _columns(table)
    items, digests = [], {}
    for i in range(table.num_rows):
        row = _row(cols, i)
        key = parquet_key(corpus, row, batch)
        where = f"{path}:{i} ({key})"
        ids, mask, spans, cut = row_ids_mask(corpus, row, _MARKERS, where)
        group, bucket = parquet_group(corpus, row)
        item = {"corpus": corpus, "key": key, "group": group, "bucket": bucket,
                "subset": parquet_subset(corpus, row, batch), "ref": {"file": file_index, "row": i},
                "num_tokens": len(ids), "ids_sha256": ids_sha256(ids), "mask": mask, "ids": ids,
                "evidence": {"assistant_turns": len(spans) if corpus != "nemotron" else 1, "cut_turns": cut}}
        if corpus == "agentic":
            episode = (row["uid"], row["attempt"], row["run"], row["episode"], row["lineage"])
            k = row["segment_idx"]
            if (episode, k) in digests:
                raise ContractError(f"{where}: segment {k} of {episode} twice")
            digests[(episode, k)] = [hashlib.blake2b(ids[s:e].tobytes(), digest_size=16).digest() for s, e in spans]
            item["spans"] = spans
            item["episode"] = (episode, k, row["n_segments"])
        items.append(item)
    if corpus == "agentic":
        segments = defaultdict(set)
        for episode, k in digests:
            segments[episode].add(k)
        for item in items:
            episode, k, n = item.pop("episode")
            if segments[episode] != set(range(n)):
                raise ContractError(f"{path}: episode {episode} has segments {sorted(segments[episode])} of {n}")
            kept = kept_rounds(digests[(episode, k - 1)], digests[(episode, k)]) if k else 0
            spans = item.pop("spans")
            item["ref"]["kept_rounds"] = kept
            item["kept_tokens"] = sum(e - s for s, e in spans[:kept])
            item["mask"] = np.zeros(item["num_tokens"], bool)
            for s, e in spans[kept:]:
                item["mask"][s:e] = True
    for item in items:
        mask = item.pop("mask")
        item["skip"] = text_skip(item.pop("ids"), mask, _MARKERS, context_length)
        item["supervised"] = int(mask.sum())
        item["eligible_anchors"] = anchor_count(mask)
    return items


LEDGER_KINDS = {"node1": classify_workspace, "codex1015": classify_codex1015, "int3c1019": classify_int3c1019,
                "bench": classify_bench}
LANE_KINDS = ("codex1015", "int3c1019", "bench")  # their items become corpora arc2 / arc3 by lane


def scan_instance(task):
    """Items of one ledger instance, with each segment's workspace (classified later, per thread). A codex1015 item's
    corpus and key are provisional ("codex1015") until its thread's lane is known (main)."""
    path, model_dir, context_length, kind = task
    classify = LEDGER_KINDS[kind]
    vocab = Vocab(model_dir)
    inst = load_instance(path, vocab)
    bad = verify_media_blobs(inst)
    if bad:
        raise RuntimeError(f"{path}: {len(bad)} media blobs do not match their sha256, e.g. {bad[0]}")
    items = []
    for seg in sorted(inst.segments.values(), key=lambda s: s.seg):
        mask, substituted = tito_mask(seg, f"{inst.name}:{seg.seg}")
        if not np.array_equal(mask, ct.loss_mask(seg)):
            raise RuntimeError("tito_mask and capture_tito.loss_mask disagree")
        skip = ct.skip_reason(seg, context_length)  # capture_tito's own rule, so capture_plan can assert it again
        if skip == "few_gen":
            skip = "few_supervised"
        if skip is None and anchor_count(mask) == 0:
            skip = "no_anchors"
        found = workspace(seg, vocab)
        items.append({"corpus": kind, "key": f"{kind}:{inst.name}:{seg.seg}", "ref": {"instance": inst.name,
                      "seg": seg.seg}, "num_tokens": len(seg), "ids_sha256": ids_sha256(seg.ids),
                      "supervised": int(mask.sum()), "eligible_anchors": anchor_count(mask), "skip": skip,
                      "thread": seg.thread if seg.sid is not None else None,
                      "where": None if found is None else classify(found),
                      "n_images": sum(m.modality == "image" for m in seg.media),
                      "n_videos": sum(m.modality == "video" for m in seg.media),
                      "evidence": {"gen_spans": len(seg.gen_spans), "stop_substituted": substituted,
                                   "rewound_gen_tokens": sum(s.dropped for s in seg.gen_spans)}})
    return items


def pairs(values, name, kind):
    out = {}
    for value in values:
        corpus, _, number = value.partition("=")
        if corpus not in CORPORA or corpus in out or not number:
            raise ValueError(f"--{name} {value}: want <corpus>=<{kind.__name__}>, each corpus once")
        out[corpus] = kind(number)
    return out


FAMILY_BUCKETS = (("agentic", "nvarc"), ("naive", "nvarc"))  # NVARC families: one parent set for both corpora


def heldout_order(groups):
    return sorted(groups, key=lambda g: (unit_hash(HELDOUT_SALT, g), g))


def choose_families(members, target_tokens, parents):
    """agentic and naive: the first ``parents`` parents (hash order) that both corpora have, the same in both, and per
    corpus a fixed sample of each family -- its items in hash order until target / parents tokens. A whole family is
    0.05-12M tokens (agentic median 2.7M, naive 1.7M) against a target of a few M, so holding out whole families
    either takes one or two parents or only the smallest ones; sampling inside the families keeps several parents,
    each with an equal share of the target. The unsampled rest of a held-out family is not trained either.
    Returns {bucket: (groups, item keys, report)}."""
    common = set.intersection(*(set(m) for m in members.values()))
    order = heldout_order(common)
    if len(order) < parents:
        raise ValueError(f"{len(order)} parents in every one of {sorted(members)}, {parents} to hold out")
    pick = order[:parents]
    share = target_tokens / parents
    out = {}
    for bucket, families in members.items():
        keys, per_parent = set(), []
        for g in pick:
            got = 0
            for it in sorted(families[g], key=lambda it: (unit_hash(HELDOUT_SALT, it["key"]), it["key"])):
                if got >= share:
                    break
                keys.add(it["key"])
                got += it["num_tokens"]
            per_parent.append(got)
        out[bucket] = (set(pick), keys, {
            "tokens_target": target_tokens, "parents": parents, "per_parent_target": share,
            "joint_with": [f"{c}/{b}" for c, b in sorted(members) if (c, b) != bucket], "items": len(keys),
            "per_parent_tokens": per_parent,
            "family_tokens": sum(it["num_tokens"] for g in pick for it in families[g])})
    return out


CODEX_CORPORA = ("arc2", "arc3")


def codex_rules(values):
    """{(corpus, bucket): (groups, items per group)} from <corpus>/<bucket>=<groups>x<items>, e.g. arc3/off25=2x6."""
    out = {}
    for value in values:
        bucket, _, rule = value.partition("=")
        corpus, _, sub = bucket.partition("/")
        groups, _, per = rule.partition("x")
        ok = corpus in CODEX_CORPORA and (sub == "arc2" if corpus == "arc2" else sub in CODEX_ARC3_SUBSETS)
        if not ok or (corpus, sub) in out or not groups.isdigit() or not per.isdigit() or not int(groups) or not int(per):
            raise ValueError(f"--codex-heldout {value}: want arc2/arc2=<groups>x<items> or arc3/<subset>=..., "
                             f"subsets {CODEX_ARC3_SUBSETS}, each bucket once, both numbers >= 1")
        out[(corpus, sub)] = (int(groups), int(per))
    return out


def choose_codex_heldout(members, rules, forced):
    """arc2 / arc3: per bucket, ``groups`` groups -- the forced ones first, then hash order -- and of each a fixed
    sample of ``items`` segments in hash order; a held-out group's other segments are neither held out nor trained.
    A codex segment is up to ~0.2M tokens of 20 KiB GR each, so the held-out is counted in segments, not tokens.
    Returns {bucket: (groups, item keys, report)}."""
    if set(members) != set(rules):
        raise ValueError(f"--codex-heldout names {sorted(rules)}, the plan's codex buckets are {sorted(members)}")
    homes = {g: b for b, groups in members.items() for g in groups}
    for g in forced:
        if g not in homes:
            raise ValueError(f"--codex-heldout-groups {g}: no codex segment can be a job in that group")
    out = {}
    for bucket, groups in sorted(members.items()):
        n, per = rules[bucket]
        mine = sorted(g for g in forced if homes[g] == bucket)
        order = mine + [g for g in heldout_order(groups) if g not in mine]
        if len(mine) > n or len(order) < n:
            raise ValueError(f"{'/'.join(bucket)}: {len(order)} groups ({len(mine)} forced), {n} to hold out")
        pick = order[:n]
        keys = set()
        for g in pick:
            ranked = sorted(groups[g], key=lambda it: (unit_hash(HELDOUT_SALT, it["key"]), it["key"]))
            keys.update(it["key"] for it in ranked[:per])
        out[bucket] = (set(pick), keys, {
            "groups_rule": n, "items_per_group": per, "forced": mine, "items": len(keys),
            "tokens": sum(it["num_tokens"] for g in pick for it in groups[g] if it["key"] in keys),
            "group_tokens": sum(it["num_tokens"] for g in pick for it in groups[g])})
    return out


def pinned_heldout(items, plan_dir):
    """--heldout-from: (the held-out item keys of the plan in plan_dir, {"pinned": what was pinned}). Every one must be
    an item here with the same ids_sha256 that can be a job (not skipped); raises otherwise -- a held-out that changed
    is not the one the earlier drafters were scored on."""
    meta = json.loads((plan_dir / "plan_meta.json").read_text())
    data = (plan_dir / "plan.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["plan_sha256"]:
        raise ContractError(f"{plan_dir}/plan.jsonl does not match its plan_meta.json sha256")
    want = {}
    for line in data.decode().splitlines():
        job = json.loads(line)
        if job["split"] == "heldout":
            if job["key"] in want:
                raise ContractError(f"{plan_dir}: held-out key {job['key']} twice")
            want[job["key"]] = job["ids_sha256"]
    if not want:
        raise ContractError(f"{plan_dir} holds no held-out job")
    by_key = {it["key"]: it for it in items}
    missing = sorted(k for k in want if k not in by_key)
    changed = sorted(k for k in want if k in by_key and by_key[k]["ids_sha256"] != want[k])
    skipped = sorted(k for k in want if k in by_key and by_key[k]["skip"] is not None)
    if missing or changed or skipped:
        raise ContractError(f"--heldout-from {plan_dir}: {len(missing)} held-out keys missing (e.g. {missing[:2]}), "
                            f"{len(changed)} with other ids (e.g. {changed[:2]}), {len(skipped)} skipped "
                            f"(e.g. {skipped[:2]})")
    tokens = sum(by_key[k]["num_tokens"] for k in want)
    return set(want), {"pinned": {"plan": str(plan_dir.resolve()), "plan_sha256": meta["plan_sha256"],
                                  "items": len(want), "tokens": tokens}}


def choose_heldout(items, target_tokens, arc3_games, parents, codex=None, codex_forced=()):
    """({(corpus, bucket): set of held-out groups}, {(corpus, bucket): held-out item keys, for a bucket that holds out
    a sample of its groups' items}, what was chosen), over the items that can be jobs. ``codex``: the arc2 / arc3
    buckets' rules (codex_rules), required when the plan has codex1015 items."""
    tokens = defaultdict(lambda: defaultdict(int))
    members = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it["skip"] is None and it["bucket"] is not None:
            tokens[(it["corpus"], it["bucket"])][it["group"]] += it["num_tokens"]
            members[(it["corpus"], it["bucket"])][it["group"]].append(it)
    # every group with an item outside node1's training1000 pool (skipped or not): an ARC-AGI-2 task with an NVARC
    # family, which node1 does not hold out -- its family would be excluded from training everywhere else
    family = {it["group"] for it in items if (it["corpus"], it["subset"]) != ("node1", "training1000")}
    chosen, sample, report = {}, {}, {}
    joint = {b: members[b] for b in FAMILY_BUCKETS if b in tokens}
    if joint:
        for bucket, (pick, keys, rule) in choose_families(joint, target_tokens, parents).items():
            chosen[bucket], sample[bucket] = pick, keys
            report["/".join(bucket)] = {**rule, "groups": len(pick), "eligible_groups": len(tokens[bucket]),
                                        "tokens": sum(rule["per_parent_tokens"]), "group_list": sorted(pick)}
    coded = {b: members[b] for b in tokens if b[0] in CODEX_CORPORA}
    if coded or codex:
        if codex is None:
            raise ValueError("codex1015 items need --codex-heldout")
        for bucket, (pick, keys, rule) in choose_codex_heldout(coded, codex, codex_forced).items():
            chosen[bucket], sample[bucket] = pick, keys
            report["/".join(bucket)] = {**rule, "groups": len(pick), "eligible_groups": len(tokens[bucket]),
                                        "group_list": sorted(pick)}
    for (corpus, bucket), groups in sorted(tokens.items()):
        if (corpus, bucket) in joint or corpus in CODEX_CORPORA:
            continue
        if (corpus, bucket) == ("node1", "arc3"):
            order = sorted(groups, key=lambda g: (unit_hash(HELDOUT_SALT, g), g))
            if len(order) < arc3_games:
                raise ValueError(f"{len(order)} ARC-AGI-3 games, {arc3_games} to hold out")
            pick = order[:arc3_games]
            rule = {"games": arc3_games}
        elif corpus == "nemotron":
            # data.split_for_key: heldout iff sha256("mtp3-v1:" + uid)[:16] as an int, mod 10000, < permyriad
            value = {g: int(hashlib.sha256(("mtp3-v1:" + g.split(":", 1)[1]).encode()).hexdigest()[:16], 16) % 10000
                     for g in groups}
            total, permyriad = 0, None
            for g in sorted(groups, key=lambda g: (value[g], g)):
                total += groups[g]
                if total >= target_tokens:
                    permyriad = value[g] + 1
                    break
            if permyriad is None or permyriad >= 10000:
                raise ValueError(f"nemotron has {total} tokens, under the held-out target")
            pick = [g for g in groups if split_for_key(g.split(":", 1)[1], permyriad) == "heldout"]
            if sorted(pick) != sorted(g for g in groups if value[g] < permyriad):
                raise RuntimeError("data.split_for_key disagrees with its reimplementation here")
            rule = {"split_for_key_permyriad": permyriad}
        elif (corpus, bucket) == ("node1", "arc2"):
            candidates = [g for g in heldout_order(groups) if g not in family]
            total, pick = 0, []
            for g in candidates:
                if total >= target_tokens:
                    break
                pick.append(g)
                total += groups[g]
            if total < target_tokens:
                raise ValueError(f"node1's {len(candidates)} ARC-AGI-2 tasks without an NVARC family have {total} "
                                 f"tokens, under the held-out target")
            rule = {"tokens_target": target_tokens, "last_score": unit_hash(HELDOUT_SALT, pick[-1]),
                    "without_family": len(candidates)}
        else:
            raise ValueError(f"no held-out rule for {corpus}/{bucket}")
        chosen[(corpus, bucket)] = set(pick)
        report[f"{corpus}/{bucket}"] = {**rule, "groups": len(pick), "eligible_groups": len(groups),
                                        "tokens": sum(groups[g] for g in pick), "group_list": sorted(pick)}
    return chosen, sample, report


def add(table, *where, job=None, item=None):
    row = table
    for k in where:
        row = row.setdefault(k, {})
    for k in ("jobs", "items", "tokens", "supervised", "eligible_anchors"):
        row.setdefault(k, 0)
    if job is not None:
        row["jobs"] += 1
        for k in ("tokens", "supervised", "eligible_anchors"):
            row[k] += job["num_tokens" if k == "tokens" else k]
    if item is not None:
        row["items"] += 1


def git_head():
    here = Path(__file__).resolve().parent
    head = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"], check=True,
                           capture_output=True, text=True)
    return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, type=Path, help="a new directory")
    p.add_argument("--target-profile", required=True, type=Path, help="the capture servers' profile (server.json)")
    p.add_argument("--node1", nargs="*", default=[], type=Path, help="ledger instance directories")
    p.add_argument("--codex1015", nargs="*", default=[], type=Path,
                   help="ledger instance directories of the 10-14..16 codex collection: corpora arc2 and arc3")
    p.add_argument("--codex-heldout", nargs="*", default=[],
                   help="with --codex1015, every bucket once: arc2/arc2=<groups>x<items> arc3/<subset>=<groups>x<items>")
    p.add_argument("--codex-heldout-groups", nargs="*", default=[],
                   help="groups the codex held-out takes first (e.g. arc3:tn36), counted in their bucket's groups")
    p.add_argument("--int3c1019", nargs="*", default=[], type=Path,
                   help="ledger instance directories of the 10-19 INT3 codex collection: arc2 and arc3, train only")
    p.add_argument("--bench", nargs="*", default=[], type=Path,
                   help="ledger instance directories of the 10-06 / 10-08 FP8 bench rounds: arc2 and arc3, train only")
    p.add_argument("--heldout-from", type=Path,
                   help="a plan directory: its held-out jobs (key and ids_sha256) are this plan's whole held-out")
    p.add_argument("--agentic", nargs="*", default=[], type=Path, help="agentic_distill parquet directories")
    p.add_argument("--naive", nargs="*", default=[], type=Path, help="qwen38fn_distill parquet directories")
    p.add_argument("--nemotron", nargs="*", default=[], type=Path, help="qwen38fn_specdata <batch>/tokens directories")
    p.add_argument("--fraction", nargs="+", required=True, help="<corpus>=<fraction in (0, 1]> for every corpus given")
    p.add_argument("--passes", nargs="+", required=True, help="<corpus>=<passes >= 1> for every corpus given")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--context-length", type=int, default=220000, help="the capture servers' context length")
    p.add_argument("--heldout-tokens", type=float, default=3e6, help="held-out size per corpus")
    p.add_argument("--arc3-heldout-games", type=int, default=2)
    p.add_argument("--heldout-parents", type=int, default=8, help="NVARC parent tasks agentic and naive hold out")
    p.add_argument("--image-detail", default="high", choices=("auto", "low", "high"),
                   help="the detail every ledger image was sent with (codex: high)")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    given = {c: getattr(args, c) for c in CORPORA if c not in CODEX_CORPORA and getattr(args, c)}
    lane_dirs = args.codex1015 + args.int3c1019 + args.bench
    if lane_dirs:
        given.update({c: lane_dirs for c in CODEX_CORPORA})
    if args.heldout_from is not None:
        if args.codex_heldout or args.codex_heldout_groups:
            raise ValueError("--heldout-from is the whole held-out: no --codex-heldout / --codex-heldout-groups")
        codex = None
    else:
        codex = codex_rules(args.codex_heldout) if args.codex1015 else None
        if not args.codex1015 and (args.codex_heldout or args.codex_heldout_groups):
            raise ValueError("--codex-heldout / --codex-heldout-groups qualify --codex1015")
    fraction = pairs(args.fraction, "fraction", float)
    passes = pairs(args.passes, "passes", int)
    if set(fraction) != set(given) or set(passes) != set(given):
        raise ValueError(f"--fraction {sorted(fraction)} and --passes {sorted(passes)} must name exactly the corpora "
                         f"given: {sorted(given)}")
    for c in given:
        if not 0 < fraction[c] <= 1 or passes[c] < 1:
            raise ValueError(f"{c}: fraction {fraction[c]} must be in (0, 1], passes {passes[c]} >= 1")
    if args.heldout_parents < 1:
        raise ValueError("--heldout-parents must be >= 1")
    args.out.mkdir(parents=True, exist_ok=False)

    profile = json.loads(args.target_profile.read_text())
    model_dir = Path(profile["server_args"]["model_path"])
    rt.check_model_files(profile, model_dir)
    vocab = Vocab(model_dir)

    sources, tasks = [], []
    for corpus in ("agentic", "naive", "nemotron"):
        for directory in given.get(corpus, []):
            files = sorted(directory.glob("part-*.parquet"))
            if not files:
                raise ValueError(f"no part-*.parquet in {directory}")
            batch = directory.parent.name if corpus == "nemotron" else None
            for path in files:
                meta = pq.ParquetFile(path).metadata
                sources.append({"corpus": corpus, "path": str(path.resolve()), "rows": meta.num_rows,
                                "bytes": path.stat().st_size, "batch": batch})
                tasks.append((corpus, str(path), len(sources) - 1, batch, args.context_length))
    instances = []
    for kind, directories in (("node1", args.node1), ("codex1015", args.codex1015), ("int3c1019", args.int3c1019),
                              ("bench", args.bench)):
        for directory in directories:
            recorded = json.loads((directory / "server.json").read_text())
            rt.check_target(recorded, profile, model_dir)  # the profile reads this ledger's segments as recorded
            instances.append({"name": recorded["instance"], "path": str(directory.resolve()), "kind": kind,
                              "server_json_sha256": sha256_file(directory / "server.json")})
    if len({i["name"] for i in instances}) != len(instances):
        raise ContractError("two ledger directories are one instance")

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_init, initargs=(str(model_dir),)) as pool:
        ledgers = pool.map_async(scan_instance, [(i["path"], str(model_dir), args.context_length, i["kind"])
                                                 for i in instances], chunksize=1)
        items = [it for part in pool.map(scan_parquet, tasks, chunksize=4) for it in part]
        tito = [it for part in ledgers.get() for it in part]
    where = classify_threads([(it["key"], it["thread"], it["where"]) for it in tito])
    dropped = {}
    kept = []
    for it in tito:
        it["where"] = where[it["key"]]
        if it["corpus"] in LANE_KINDS:
            kind, pool = it["corpus"], it["where"]["pool"]
            if pool in ("probe", "test"):  # outside any workspace (no lane, so no corpus), or a harness test
                reason = it["skip"] or ("no_workspace" if pool == "probe" else "harness_test")
                d = dropped.setdefault(kind, {}).setdefault(reason, {"segments": 0, "tokens": 0, "supervised": 0})
                d["segments"] += 1
                d["tokens"] += it["num_tokens"]
                d["supervised"] += it["supervised"]
                continue
            it["corpus"] = it["where"]["lane"]
            it["key"] = f"{it['corpus']}:{it['ref']['instance']}:{it['ref']['seg']}"
            it["group"], it["bucket"] = (codex1015_group if kind == "codex1015" else train_only_group)(it["where"])
        else:
            it["group"], it["bucket"] = node1_group(it["where"])
        it["subset"] = it["where"]["pool"]
        kept.append(it)
    items = kept + items
    keys = [it["key"] for it in items]
    if len(set(keys)) != len(keys):
        raise ContractError("two source items share a key")
    public_eval = {it["group"] for it in items if it["corpus"] == "agentic" and it["subset"] == "nvarc_full"}
    both = public_eval & {it["group"] for it in items if it["bucket"] is not None}
    if both:
        raise ContractError(f"{len(both)} ARC tasks are both an nvarc_full (public-eval) parent and a held-out pool's "
                            f"parent, e.g. {sorted(both)[:3]}")

    if args.heldout_from is not None:
        pinned, heldout_report = pinned_heldout(items, args.heldout_from)
        chosen = {}
    else:
        pinned = None
        chosen, sample, heldout_report = choose_heldout(items, args.heldout_tokens, args.arc3_heldout_games,
                                                        args.heldout_parents, codex, args.codex_heldout_groups)
    everywhere = set().union(*chosen.values())
    jobs = []
    for it in items:
        if it["skip"] is not None:
            continue
        bucket = (it["corpus"], it["bucket"])
        if pinned is not None:  # --heldout-from: the pinned segments are held out, every other one trains
            held = it["key"] in pinned
        elif it["group"] in chosen.get(bucket, ()):
            if bucket in sample and it["key"] not in sample[bucket]:
                it["skip"] = "heldout_unsampled"
                continue
            held = True
        elif it["group"] in everywhere:
            it["skip"] = "heldout_elsewhere"
            continue
        else:
            held = False
        if held:
            it["split"] = "heldout"
            jobs.append((unit_hash(args.seed, "order", it["key"], 1), it, 1))
        else:
            it["split"] = "train"
            if unit_hash(args.seed, "keep", it["key"]) < fraction[it["corpus"]]:
                n = passes[it["corpus"]]
                jobs += [((k - 1 + unit_hash(args.seed, "order", it["key"], k)) / n, it, k) for k in range(1, n + 1)]
    jobs.sort(key=lambda j: (j[1]["split"] != "heldout", j[0], j[1]["key"], j[2]))

    lines, summary = [], {"by_split": {}, "by_subset": {}, "sources": {}, "skipped": {}, "skipped_tokens": {}}
    for index, (_, it, k) in enumerate(jobs):
        job = {"index": index, "key": it["key"], "pass": k, "split": it["split"], "corpus": it["corpus"],
               "subset": it["subset"], "group": it["group"], "ref": it["ref"], "num_tokens": it["num_tokens"],
               "supervised": it["supervised"], "eligible_anchors": it["eligible_anchors"],
               "ids_sha256": it["ids_sha256"]}
        lines.append(json.dumps(job, separators=(",", ":")))
        add(summary["by_split"], it["corpus"], it["split"], job=job, item=it if k == 1 else None)
        add(summary["by_subset"], it["corpus"], it["subset"], it["split"], job=job, item=it if k == 1 else None)
    for it in items:
        src = summary["sources"].setdefault(it["corpus"], {"items": 0, "tokens": 0, "supervised": 0,
                                                            "eligible_anchors": 0, "kept_round_tokens": 0})
        src["items"] += 1
        src["tokens"] += it["num_tokens"]
        src["supervised"] += it["supervised"]
        src["eligible_anchors"] += it["eligible_anchors"]
        src["kept_round_tokens"] += it.get("kept_tokens", 0)
        if it["corpus"] == "agentic":
            src["kept_rounds"] = src.get("kept_rounds", 0) + it["ref"]["kept_rounds"]
        for k, v in it["evidence"].items():
            src[k] = src.get(k, 0) + v
        if it["skip"] is not None:
            skipped = summary["skipped"].setdefault(it["corpus"], {})
            skipped[it["skip"]] = skipped.get(it["skip"], 0) + 1
            skipped_tokens = summary["skipped_tokens"].setdefault(it["corpus"], {})
            skipped_tokens[it["skip"]] = skipped_tokens.get(it["skip"], 0) + it["num_tokens"]
    groups = defaultdict(set)
    for it in items:
        groups[f"{it['corpus']}/{it['subset']}"].add(it["group"])
    summary["groups"] = {k: len(v) for k, v in sorted(groups.items())}
    summary["heldout"] = {k: {kk: vv for kk, vv in v.items() if kk != "group_list"} for k, v in heldout_report.items()}
    if lane_dirs:
        summary["dropped"] = dropped
    if any(r not in SKIP_REASONS for c in summary["skipped"].values() for r in c):
        raise RuntimeError(f"unlisted skip reason in {summary['skipped']}")

    data = ("\n".join(lines) + "\n").encode() if lines else b""
    (args.out / "plan.jsonl").write_bytes(data)
    meta = {"version": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "argv": sys.argv, "code": git_head(), "seed": args.seed, "context_length": args.context_length,
            "block_size": BLOCK_SIZE, "anchor_rule": ANCHOR_RULE, "min_supervised": MIN_SUPERVISED,
            "heldout_tokens": args.heldout_tokens,
            "heldout_salt": HELDOUT_SALT, "arc3_heldout_games": args.arc3_heldout_games,
            "heldout_parents": args.heldout_parents, "image_detail": args.image_detail,
            "codex_heldout": None if codex is None else {"/".join(b): {"groups": n, "items_per_group": k}
                                                         for b, (n, k) in sorted(codex.items())},
            "codex_heldout_groups": sorted(args.codex_heldout_groups),
            "heldout_from": None if args.heldout_from is None else heldout_report["pinned"],
            "known_approximations": (["codex1015: think-budget-forced tokens stay supervised (not sampled; 436 of "
                                      "185M arc3 tokens where the ledger kept flags)"] if args.codex1015 else [])
            + (["int3c1019: think-budget-forced tokens stay supervised (arc3 ran ARC_THINK_BUDGET=24576)"]
               if args.int3c1019 else []),
            "corpora": {c: {"fraction": fraction[c], "passes": passes[c], "sources": [str(d.resolve()) for d in given[c]]}
                        for c in given},
            "target_profile": {"path": str(args.target_profile.resolve()), "instance": profile["instance"],
                               "sha256": hashlib.sha256(args.target_profile.read_bytes()).hexdigest(),
                               "served_model_name": rt.profile_served_name(args.target_profile),
                               "model_path": str(model_dir)},
            "tokenizer": {"path": str(vocab.path), "sha256": vocab.sha256},
            "tito_instances": instances, "sources": sources, "heldout": heldout_report,
            "plan_sha256": hashlib.sha256(data).hexdigest(), "jobs": len(lines)}
    (args.out / "plan_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_summary(summary)


def print_summary(summary):
    print(f"{'corpus':9} {'split':8} {'items':>8} {'jobs':>8} {'tokens':>14} {'supervised':>14} {'anchors':>14}")
    for corpus, splits in summary["by_split"].items():
        for split, r in sorted(splits.items()):
            print(f"{corpus:9} {split:8} {r['items']:8d} {r['jobs']:8d} {r['tokens']:14,d} {r['supervised']:14,d} "
                  f"{r['eligible_anchors']:14,d}")
    for corpus, src in summary["sources"].items():
        print(f"source {corpus}: {src} skipped {summary['skipped'].get(corpus, {})} "
              f"(tokens {summary['skipped_tokens'].get(corpus, {})})")
    for bucket, r in summary["heldout"].items():
        print(f"held-out {bucket}: {r}")


if __name__ == "__main__":
    main()
