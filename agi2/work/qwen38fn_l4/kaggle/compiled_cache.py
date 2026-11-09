"""CPU-only restoration of an identity-pinned, private SM89 compiled-kernel cache.

The exporter manifest must include the actual resolved launch fields; an old
export lacking them is rejected. Nothing restores weights, keys, or HF caches.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

CACHE_LEAVES = {
    'SGLANG_JIT_CACHE_DIR': 'jit', 'TRITON_CACHE_DIR': 'triton',
    'CUDA_CACHE_PATH': 'nv', 'FLASHINFER_WORKSPACE_BASE': 'flashinfer',
}
IDENTITY_FIELDS = ('source', 'runtime_archive', 'versions', 'gpu', 'nvcc',
                   'launch_profile', 'runtime_path', 'cache_paths')
# The serving shape every L4 cache must come from. Capacity fields (running, logical tokens, hot blocks, graph batch
# sizes, speculative algorithm) are not fixed here any more (v16): they are matched exactly against the run that
# restores, through the whole launch_profile equality in _identity.
REQUIRED_PROFILE = {
    'tp_size': 4, 'ep_size': 4, 'dp_size': 2, 'enable_dp_attention': True,
    'page_size': 64, 'index_dtype': 'fp8', 'kv_dtype': 'nvfp4', 'ssm_dtype': 'float16', 'graph': True,
}
FORBIDDEN_NAMES = {'.env', '.ssh', '.aws', '.kaggle', 'kaggle.json', 'runpod_key.txt',
                   'id_rsa', 'id_ed25519', 'credentials', 'credentials.json',
                   'secrets.json', 'api_key', 'key.txt', 'private'}
SHA_RE = re.compile(r'[0-9a-f]{64}')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise ValueError('cache member path must be a string')
    path = PurePosixPath(name)
    if not name or path == PurePosixPath('.') or path.is_absolute() or '..' in path.parts or '\\' in name or '\x00' in name:
        raise ValueError(f'unsafe cache member path: {name!r}')
    if any(part.casefold() in FORBIDDEN_NAMES for part in path.parts):
        raise ValueError('credential file is forbidden in a compiled cache')
    lower = path.name.casefold()
    if path.suffix.casefold() in ('.safetensors', '.pt', '.pth', '.ckpt', '.onnx', '.h5') or (
            lower.endswith('.bin') and lower.startswith(('pytorch_model', 'model-', 'weights', 'adapter_model'))):
        raise ValueError('model weights are forbidden in a compiled cache')
    return path


def _sha(value: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ValueError('cache SHA256 must be lowercase hex')
    return value


def _absolute(name: str) -> Path:
    path = Path(name)
    if not path.is_absolute() or '..' in path.parts or '\x00' in str(path) or '\\' in str(path):
        raise ValueError('cache runtime/root path must be a safe absolute path')
    return path


def _identity(manifest: dict, expected: dict) -> dict:
    for name in IDENTITY_FIELDS:
        if name not in manifest or name not in expected:
            raise ValueError(f'cache identity missing {name}')
        if manifest[name] != expected[name]:
            raise ValueError(f'cache identity mismatch: {name}')
    for key in ('digest', 'manifest_sha256'):
        _sha(manifest['source'][key])
    if not isinstance(manifest['source']['files'], int) or manifest['source']['files'] <= 0:
        raise ValueError('cache source identity has no whole-package file count')
    _sha(manifest['runtime_archive']['sha256'])
    _sha(manifest['runtime_archive']['provenance_sha256'])
    versions = {name.casefold(): value for name, value in manifest['versions'].items()}
    if not {'torch', 'triton', 'flashinfer-python'}.issubset(versions) or not ({'sgl-kernel', 'sglang-kernel'} & versions.keys()):
        raise ValueError('cache identity lacks runtime package versions')
    gpu = manifest['gpu']
    if gpu['capabilities'] != [[8, 9]] * 4 or len(gpu['gpus']) != 4 or any('L4' not in name for name in gpu['gpus']):
        raise ValueError('compiled cache requires actual four-L4 SM89 evidence')
    if gpu['python'].split()[0] != '3.12.13' or not manifest['nvcc'].strip():
        raise ValueError('cache identity lacks the exact Python/nvcc evidence')
    profile = manifest['launch_profile']
    for name, value in REQUIRED_PROFILE.items():
        if name not in profile or profile[name] != value:
            raise ValueError(f'cache launch identity missing/mismatched actual {name}')
    return {name: manifest[name] for name in IDENTITY_FIELDS}


def _roots(manifest: dict, allowed: dict[str, Path]) -> dict[str, Path]:
    if set(allowed) != set(CACHE_LEAVES) or set(manifest['cache_paths']) != set(CACHE_LEAVES):
        raise ValueError('restore requires exactly the four owned compiled-cache roots')
    runtime = _absolute(manifest['runtime_path'])
    roots = {}
    for name, leaf in CACHE_LEAVES.items():
        root = _absolute(str(allowed[name]))
        recorded = _absolute(manifest['cache_paths'][name])
        if root != recorded or root != runtime / 'cache-sm89-hisparse' / leaf:
            raise ValueError(f'compiled cache path mismatch: {name}')
        for path in (root, *root.parents):
            if path.is_symlink():
                raise ValueError('compiled cache destination traverses a symlink')
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError('compiled cache restoration requires fresh/empty owned roots')
        roots[name] = root
    return roots


def _files(manifest: dict) -> dict[str, dict]:
    rows = {}
    for record in manifest['files']:
        if record['cache'] not in CACHE_LEAVES:
            raise ValueError('manifest names a cache outside the four owned roots')
        relative = _relative(record['file'])
        name = record['cache'] + '/' + relative.as_posix()
        size = record['bytes']
        if name in rows or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('duplicate cache file or invalid byte count')
        _sha(record['sha256'])
        rows[name] = record
    return rows


def _stage(archive: Path, directory: Path, files: dict[str, dict]) -> None:
    with tarfile.open(archive, 'r:gz') as stream:
        members = {}
        directories = {parent.as_posix() for name in files for parent in PurePosixPath(name).parents
                       if parent != PurePosixPath('.')}
        for member in stream.getmembers():
            name = _relative(member.name).as_posix()
            if name in members or member.mode & 0o6000:
                raise ValueError('duplicate cache member or privileged file mode')
            if member.isdir():
                if name not in directories and name not in CACHE_LEAVES:
                    raise ValueError('unlisted cache directory')
            elif member.isfile() or member.islnk():
                if name not in files or (member.isfile() and member.size != files[name]['bytes']):
                    raise ValueError('archive file roster/size differs from cache manifest')
            else:
                raise ValueError('cache symlinks and special files are forbidden')
            members[name] = member
        if {name for name, member in members.items() if not member.isdir()} != set(files):
            raise ValueError('archive misses manifest cache files')
        for name, member in members.items():
            for parent in PurePosixPath(name).parents:
                if parent.as_posix() in members and not members[parent.as_posix()].isdir():
                    raise ValueError('cache member traverses a non-directory')
            if member.islnk():
                target = _relative(member.linkname).as_posix()
                if target not in members or not members[target].isfile():
                    raise ValueError('cache hardlink must target a regular manifest member')
        for name, member in members.items():
            if not member.isfile():
                continue
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 8 << 20)
            target.chmod(member.mode & 0o777)
        for name, member in members.items():
            if member.islnk():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(directory / _relative(member.linkname), target)
        for name, record in files.items():
            target = directory / name
            if target.stat().st_size != record['bytes'] or sha256_file(target) != record['sha256']:
                raise ValueError(f'compiled cache file SHA256/size mismatch: {name}')


def restore(cache_asset_path: Path, expected_identity: dict, allowed_cache_roots: dict[str, Path]) -> dict:
    """Validate the whole asset before copying into four fresh/empty approved cache roots."""
    asset = Path(cache_asset_path)
    manifest_path = asset / 'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('cache asset requires a regular manifest.json')
    manifest = json.loads(manifest_path.read_text())
    identity = _identity(manifest, expected_identity)
    roots = _roots(manifest, allowed_cache_roots)
    files = _files(manifest)
    name = _relative(manifest['archive'])
    if len(name.parts) != 1:
        raise ValueError('compiled cache archive must be directly under its asset root')
    archive = asset / name
    if archive.is_symlink() or not archive.is_file() or sha256_file(archive) != _sha(manifest['archive_sha256']):
        raise ValueError('compiled cache archive SHA256 mismatch')
    for field in ('archive_sha256', 'files'):
        if field in expected_identity and manifest[field] != expected_identity[field]:
            raise ValueError(f'compiled cache expected asset mismatch: {field}')
    published = []
    with tempfile.TemporaryDirectory(prefix='arcforge-compiled-cache-') as temporary:
        stage = Path(temporary)
        _stage(archive, stage, files)
        try:
            for name, root in roots.items():
                # All input hashes passed; each destination was proved fresh/empty.
                root.mkdir(parents=True, exist_ok=True)
                root.chmod(0o700)
                published.append(root)
                source = stage / name
                if source.is_dir():
                    shutil.copytree(source, root, dirs_exist_ok=True)
        except BaseException:
            # Only directories this call populated are touched; no foreign roots merge.
            for root in published:
                shutil.rmtree(root)
            raise
    return {'status': 'restored', 'manifest_sha256': sha256_file(manifest_path),
            'archive_sha256': manifest['archive_sha256'], 'identity': identity,
            'cache_paths': {name: str(root) for name, root in roots.items()},
            'files_restored': len(files), 'bytes_restored': sum(row['bytes'] for row in files.values()),
            'cuda_graphs_recapture_required': True, 'speedup_measured': False}
