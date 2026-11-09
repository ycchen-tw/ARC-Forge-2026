"""An RL overlay (rl_overlay_package.py) + the package it was made against -> a Kaggle weight patch: only the tensors
whose bytes the overlay's rewritten shards changed, one patch-<shard> safetensors per shard, beside patch.json and the
overlay's deploy_manifest.json. Kaggle then stores ~3.4 GB (RL v6b v10: 516 tensors) instead of ~21 GB of whole shards.

The notebook (02_stage find_model, config ``model_patch``) serves the unchanged mounted package and points the int3
deploy runtime at the patch (ARC_INT3_WEIGHT_PATCH, int3_deploy_runtime.WeightPatch), which hands the loader the
patch's tensor wherever the package's would go -- nothing is written on the Kaggle machine.

Checked: the overlay's deploy_manifest.json against overlay.json; the base manifest the overlay names, rebuilt from the
overlay's manifest with the package's own entries for the rewritten shards (so the package is that base, byte for byte
in those shards); each rewritten shard's header identical in package and overlay; package and overlay shard sha256
against the base and overlay manifests; and, from the written patch files, base + patch == overlay shard sha256.
Writes <out>.partial, renames to <out>, then DONE.

    python rl_weight_patch.py --base <package> --overlay <overlay> --out <dir>
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

PATCH_SCHEMA = 'arcforge.kaggle.weight_patch.v1'
OVERLAY_SCHEMA = 'arcforge.kaggle.model_overlay.v1'
#: bytes per element, for laying the patch file out with every tensor aligned to its own element size
ELEMENT = {'F64': 8, 'I64': 8, 'F32': 4, 'I32': 4, 'BF16': 2, 'F16': 2, 'I16': 2, 'F8_E4M3': 1, 'F8_E5M2': 1,
           'U8': 1, 'I8': 1, 'BOOL': 1}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def raw_header(path):
    """(the 8 + n header bytes, the parsed header) of a safetensors file."""
    with open(path, 'rb') as f:
        head = f.read(8)
        n = struct.unpack('<Q', head)[0]
        if n > 16 << 20:
            raise ValueError(f'{path}: oversized safetensors header')
        body = f.read(n)
    return head + body, json.loads(body)


def tensors_in_order(header):
    """(name, entry) by data offset, and a check that they tile the payload without holes."""
    rows = sorted(((k, v) for k, v in header.items() if k != '__metadata__'), key=lambda kv: kv[1]['data_offsets'])
    at = 0
    for name, entry in rows:
        start, end = entry['data_offsets']
        if start != at:
            raise ValueError(f'{name}: payload hole or overlap at {at}')
        at = end
    return rows, at


def base_manifest_of(overlay_manifest, package_files, rewritten):
    """The manifest the overlay was made from: rl_overlay_package.py copies it, replaces the rewritten shards' entries
    and adds ``rl_merged``; undoing exactly that must give back the bytes overlay.json's base_manifest_sha256 pins."""
    base = json.loads(json.dumps(overlay_manifest))
    base.pop('rl_merged')
    for shard in rewritten:
        base['files'][shard] = package_files[shard]
    return (json.dumps(base, indent=2) + '\n').encode()


def write_patch(path, tensors, shard):
    """Raw safetensors with the given (name, dtype, shape, bytes): largest elements first, so every tensor starts on a
    multiple of its own element size; the header padded with spaces to 8 bytes, as safetensors writes it."""
    rows = sorted(tensors, key=lambda t: (-ELEMENT[t[1]], t[0]))
    header, at = {'__metadata__': {'format': 'pt', 'patch_of': shard}}, 0
    for name, dtype, shape, data in rows:
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [at, at + len(data)]}
        at += len(data)
    body = json.dumps(header, separators=(',', ':')).encode()
    body += b' ' * (-len(body) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(body)) + body)
        for row in rows:
            f.write(row[3])


def tensor_sha(t):
    return hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, type=Path, help='the package the overlay was made against')
    ap.add_argument('--overlay', required=True, type=Path, help='rl_overlay_package.py output')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    base, ov, out = a.base, a.overlay, a.out
    partial = out.with_name(out.name + '.partial')
    if out.exists() or partial.exists():
        raise FileExistsError(out)
    if not (ov / 'DONE').exists():
        raise RuntimeError(f'{ov}: no DONE marker')
    overlay = json.loads((ov / 'overlay.json').read_text())
    if overlay['schema'] != OVERLAY_SCHEMA:
        raise RuntimeError(f"{ov}: schema {overlay['schema']}")
    if file_sha(ov / 'deploy_manifest.json') != overlay['deploy_manifest_sha256']:
        raise RuntimeError(f'{ov}/deploy_manifest.json is not the one overlay.json names')
    target = json.loads((ov / 'deploy_manifest.json').read_text())
    package_files = json.loads((base / 'deploy_manifest.json').read_text())['files']
    rewritten = sorted(overlay['files'])
    rebuilt = hashlib.sha256(base_manifest_of(target, package_files, rewritten)).hexdigest()
    if rebuilt != overlay['base_manifest_sha256']:
        raise RuntimeError(f"the overlay's manifest with {base}'s entries for {rewritten} is {rebuilt}, "
                           f"overlay.json was made against {overlay['base_manifest_sha256']}")
    partial.mkdir(parents=True)
    tensors, files = {}, {}
    for shard in rewritten:
        head, header = raw_header(base / shard)
        if raw_header(ov / shard)[0] != head:
            raise RuntimeError(f'{shard}: package and overlay headers differ; a patch replaces bytes, not layout')
        rows, payload = tensors_in_order(header)
        h_base, h_ov, changed = hashlib.sha256(head), hashlib.sha256(head), []
        with open(base / shard, 'rb') as fb, open(ov / shard, 'rb') as fo:
            fb.seek(len(head))
            fo.seek(len(head))
            for name, entry in rows:
                n = entry['data_offsets'][1] - entry['data_offsets'][0]
                old, new = fb.read(n), fo.read(n)
                if len(old) != n or len(new) != n:
                    raise EOFError(f'{shard}:{name}')
                h_base.update(old)
                h_ov.update(new)
                if old != new:
                    changed.append((name, entry['dtype'], entry['shape'], new))
            if fb.read(1) or fo.read(1):
                raise RuntimeError(f'{shard}: bytes past the last tensor')
        if h_base.hexdigest() != package_files[shard]['sha256']:
            raise RuntimeError(f'{base}/{shard} does not match its deploy_manifest.json entry')
        if h_ov.hexdigest() != overlay['files'][shard]['sha256']:
            raise RuntimeError(f'{ov}/{shard} does not match overlay.json')
        if not changed:
            raise RuntimeError(f'{shard}: rewritten but no tensor changed')
        name = 'patch-' + shard
        write_patch(partial / name, changed, shard)
        for t in changed:
            tensors[t[0]] = {'file': name, 'shard': shard, 'dtype': t[1], 'shape': t[2],
                             'sha256': hashlib.sha256(t[3]).hexdigest()}
        files[name] = {'sha256': file_sha(partial / name), 'bytes': (partial / name).stat().st_size,
                       'shard': shard, 'tensors': len(changed)}
        print(f'{shard}: {len(changed)} tensors, {sum(len(t[3]) for t in changed) / 1e9:.3f} GB', flush=True)
        del changed
    # Read back through safetensors itself (what the runtime opens), then rebuild every shard from package + patch.
    for name, entry in files.items():
        with safe_open(str(partial / name), 'pt') as h:
            keys = set(h.keys())
            if keys != {k for k, t in tensors.items() if t['file'] == name}:
                raise RuntimeError(f'{name}: tensor set read back differs')
            for k in keys:
                if tensor_sha(h.get_tensor(k)) != tensors[k]['sha256']:
                    raise RuntimeError(f'{name}/{k}: read-back mismatch')
        head, header = raw_header(base / entry['shard'])
        p_head, p_header = raw_header(partial / name)
        h = hashlib.sha256(head)
        with open(base / entry['shard'], 'rb') as fb, open(partial / name, 'rb') as fp:
            fb.seek(len(head))
            for k, t in tensors_in_order(header)[0]:
                n = t['data_offsets'][1] - t['data_offsets'][0]
                data = fb.read(n)
                if k in p_header:
                    fp.seek(len(p_head) + p_header[k]['data_offsets'][0])
                    data = fp.read(n)
                h.update(data)
        if h.hexdigest() != overlay['files'][entry['shard']]['sha256']:
            raise RuntimeError(f"package {entry['shard']} + {name} is not the overlay's shard")
    (partial / 'deploy_manifest.json').write_bytes((ov / 'deploy_manifest.json').read_bytes())
    patch = {'schema': PATCH_SCHEMA, 'base_manifest_sha256': overlay['base_manifest_sha256'],
             'deploy_manifest_sha256': overlay['deploy_manifest_sha256'], 'overlay_files': overlay['files'],
             'rl_merged': overlay['rl_merged'], 'files': files, 'tensors': tensors,
             'report': {'tensors': len(tensors), 'bytes': sum(f['bytes'] for f in files.values())}}
    (partial / 'patch.json').write_text(json.dumps(patch, indent=1) + '\n')
    partial.rename(out)
    (out / 'DONE').write_text('ok\n')
    print(json.dumps({'out': str(out), 'patch_sha256': file_sha(out / 'patch.json'), **patch['report']}))


if __name__ == '__main__':
    main()
