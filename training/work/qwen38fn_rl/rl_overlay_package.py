"""An RL LoRA-merged version (trainer/merged/vNNNNNN, fn-lora-merged-v2) + the deploy package it was trained on -> a Kaggle
model overlay: only the package shards that hold a merged tensor, rewritten, the package's deploy_manifest.json with
those shards' hashes and an ``rl_merged`` block, and overlay.json (what the notebook checks).

The notebook (02_stage find_model, config ``model_overlay``) mounts the unchanged package beside the overlay and builds
the servable directory out of symlinks, so Kaggle stores ~20 GB instead of another 89 GB package. Nothing is added,
renamed or re-quantized: every merged tensor must exist in the package under the same name, dtype and shape (the v2
export already holds the deployed block-FP8 bytes and their scales, the rollout servers load exactly these). Checked:
the package's deploy_manifest.json against --source-manifest-sha256, the source shards against that manifest, the
merged bytes against merged_manifest.json, and every rewritten shard read back (merged tensors against the merged hash,
the others against the source). No config.json is written, so the overlay is never mistaken for a checkpoint.
Writes <out>.partial, renames to <out>, then DONE.

    python rl_overlay_package.py --source <package> --source-manifest-sha256 <sha> --merged <trainer/merged/v000030> \
        --out <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MERGED_FORMAT = 'fn-lora-merged-v2'
OVERLAY_SCHEMA = 'arcforge.kaggle.model_overlay.v1'


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--source-manifest-sha256', required=True)
    ap.add_argument('--merged', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--step-is-optimizer-step', action='store_true',
                    help='the merged manifest has no optimizer_step and its step is one (train_tito_mm exports)')
    a = ap.parse_args()
    src, merged, out = a.source, a.merged, a.out
    partial = out.with_name(out.name + '.partial')
    if out.exists() or partial.exists():
        raise FileExistsError(out)
    if not (merged / 'DONE').exists():
        raise RuntimeError(f'{merged}: no DONE marker')
    got = file_sha(src / 'deploy_manifest.json')
    if got != a.source_manifest_sha256:
        raise RuntimeError(f'{src}/deploy_manifest.json is {got}, not {a.source_manifest_sha256}')
    deploy = json.loads((src / 'deploy_manifest.json').read_text())
    mman_path = merged / 'merged_manifest.json'
    mman = json.loads(mman_path.read_text())
    if mman['format'] != MERGED_FORMAT:
        raise RuntimeError(f"{merged}: format {mman['format']}, not {MERGED_FORMAT}")
    files = sorted(merged.glob('*.safetensors'))
    if [f.name for f in files] != ['model.safetensors']:
        raise RuntimeError(f'{merged}: expected one model.safetensors, found {[f.name for f in files]}')
    new = {}
    with safe_open(str(files[0]), 'pt') as h:
        for k in h.keys():
            new[k] = h.get_tensor(k)
    if set(new) != set(mman['keys']):
        raise RuntimeError('merged file and merged_manifest.json disagree on the keys')
    for k, t in new.items():
        if tensor_sha(t) != mman['keys'][k]['merged_sha256']:
            raise RuntimeError(f'{k}: merged tensor does not match merged_manifest.json')
    index = json.loads((src / 'model.safetensors.index.json').read_text())['weight_map']
    missing = sorted(k for k in new if k not in index)
    if missing:
        raise KeyError(f'{len(missing)} merged tensors are not in the package (e.g. {missing[:3]})')
    by_shard = {}
    for k in new:
        by_shard.setdefault(index[k], []).append(k)
    for shard in sorted(by_shard):
        want = deploy['files'][shard]
        if (src / shard).stat().st_size != want['bytes'] or file_sha(src / shard) != want['sha256']:
            raise RuntimeError(f'{src}/{shard} does not match its deploy_manifest.json entry')
    partial.mkdir(parents=True)
    report = {'replaced': 0, 'changed_bytes': 0, 'rewritten_shards': sorted(by_shard)}
    for shard in sorted(by_shard):
        tensors = {}
        with safe_open(str(src / shard), 'pt') as h:
            meta = h.metadata()
            for k in h.keys():
                t = h.get_tensor(k)
                if k in new:
                    if new[k].dtype != t.dtype or new[k].shape != t.shape:
                        raise TypeError(f'{k}: merged {new[k].dtype} {tuple(new[k].shape)}, package {t.dtype} '
                                        f'{tuple(t.shape)}')
                    report['changed_bytes'] += int((new[k].view(torch.uint8) != t.view(torch.uint8)).sum())
                    t = new[k]
                    report['replaced'] += 1
                tensors[k] = t
        save_file(tensors, str(partial / shard), metadata=meta)
    if report['replaced'] != len(new):
        raise RuntimeError(f"replaced {report['replaced']} of {len(new)} merged tensors")
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(partial / shard), 'pt') as h, safe_open(str(src / shard), 'pt') as s:
            if set(h.keys()) != set(s.keys()):
                raise RuntimeError(f'{shard}: tensor set changed')
            for k in h.keys():
                want = mman['keys'][k]['merged_sha256'] if k in new else tensor_sha(s.get_tensor(k))
                if tensor_sha(h.get_tensor(k)) != want:
                    raise RuntimeError(f'{shard}/{k}: read-back mismatch')
    overlay_files = {}
    for shard in sorted(by_shard):
        entry = {'sha256': file_sha(partial / shard), 'bytes': (partial / shard).stat().st_size}
        deploy['files'][shard] = entry
        overlay_files[shard] = entry
    if 'optimizer_step' not in mman:   # train_tito_mm exports before 10-30: one optimizer step per step
        if not a.step_is_optimizer_step:
            raise KeyError(f'{merged}: merged manifest has no optimizer_step (train_tito_mm export? pass '
                           '--step-is-optimizer-step)')
        mman = {**mman, 'optimizer_step': mman['step']}
    meta = {k: mman[k] for k in ('weight_version', 'step', 'optimizer_step', 'run', 'base_model', 'lora_rank',
                                 'lora_alpha', 'dense_qat')}
    meta['merged_dir'] = str(merged)
    meta['merged_manifest_sha256'] = file_sha(mman_path)
    deploy['rl_merged'] = {**meta, 'source_package': str(src), 'source_manifest_sha256': a.source_manifest_sha256,
                           'rewritten_shards': report['rewritten_shards']}
    (partial / 'deploy_manifest.json').write_text(json.dumps(deploy, indent=2) + '\n')
    overlay = {'schema': OVERLAY_SCHEMA, 'base_manifest_sha256': a.source_manifest_sha256,
               'deploy_manifest_sha256': file_sha(partial / 'deploy_manifest.json'),
               'files': overlay_files, 'rl_merged': meta, 'report': report}
    (partial / 'overlay.json').write_text(json.dumps(overlay, indent=1) + '\n')
    partial.rename(out)
    (out / 'DONE').write_text('ok\n')
    print(json.dumps({'out': str(out), 'deploy_manifest_sha256': overlay['deploy_manifest_sha256'], **report}))


if __name__ == '__main__':
    main()
