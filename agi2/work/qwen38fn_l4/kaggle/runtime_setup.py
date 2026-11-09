"""Install the frozen h28 Kaggle assets offline; no keys, network, pip, or GPU work."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

INPUT_ROOT = Path('/kaggle/input')
SOURCE_DIGEST = 'b9cbd753acb18f0a40802b198f7d64fd92a48d5bc2117c2918bcff2b9b8ebb22'
SOURCE_FILES = 4901
SOURCE_MANIFEST_SHA = 'a80d8d481679e25c17c4f9c3089c9d77812ef5a676eaab371541df2187de7f84'
FAST_SOURCE_DIGEST = 'df5847fd1fb73f62c09e5432c7a263d85ea97832594a0836880dd3f02e252b28'
FAST_SOURCE_MANIFEST_SHA = '3e71105a9ebcb21dee2c030c65b5b24a94c8aeca825d2b381bc0290c8251c710'
RUNTIME_FILE = 'sglang-0520-94602c9-sm120-cp312.tar.zst'
RUNTIME_SHA = 'cbf427cb3677313cf060ec4e4c4e53d2cf7a9dbf06ac884ce31e93447345c1b3'
RUNTIME_BYTES = 3496285060
PROVENANCE_SHA = '812fd475d9056e54c107272d93960c8aa2770c8e4945b476846811e30b908680'
CREDENTIAL_NAMES = {'.env', '.ssh', '.aws', '.kaggle', 'kaggle.json', 'runpod_key.txt',
                    'id_rsa', 'id_ed25519', 'credentials.json', 'secrets.json'}
SHA_RE = re.compile(r'[0-9a-f]{64}')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path == PurePosixPath('.') or path.is_absolute() or '..' in path.parts or '\\' in name or '\x00' in name:
        raise ValueError(f'unsafe asset path: {name!r}')
    if any(part.lower() in CREDENTIAL_NAMES for part in path.parts):
        raise ValueError(f'credential file is forbidden in offline assets: {name!r}')
    return path


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _mounted(candidates: list[Path], label: str) -> Path:
    found = set()
    for path in candidates:
        if path.is_dir():
            resolved = path.resolve()
            if not resolved.is_relative_to(INPUT_ROOT.resolve()):
                raise ValueError(f'{label} mount escapes Kaggle input')
            found.add(resolved)
    if len(found) != 1:
        raise RuntimeError(f'{label}: expected one mounted asset, found {len(found)}')
    return found.pop()


def dataset_mount(reference: str) -> Path:
    parts = _relative(reference).parts
    if len(parts) != 2:
        raise ValueError('dataset reference must be owner/slug')
    return _mounted([INPUT_ROOT / 'datasets' / parts[0] / parts[1], INPUT_ROOT / parts[1]], reference)


def _model_reference(reference: str) -> tuple[str, ...]:
    parts = _relative(reference).parts
    if len(parts) != 5 or parts[2].casefold() != 'transformers' or not parts[-1].isdigit():
        raise ValueError('model reference must be owner/slug/transformers/variation/version')
    return (*parts[:2], 'transformers', *parts[3:])


def model_mount(reference: str) -> Path:
    parts = _model_reference(reference)
    return _mounted([INPUT_ROOT / 'models' / Path(*parts), INPUT_ROOT / parts[1] / Path(*parts[2:])], reference)


def asset_file(root: Path, name: str) -> Path:
    relative = _relative(name)
    # A dataset can retain a packager directory; never choose between duplicate basenames.
    matches = list(root.rglob(relative.name)) if len(relative.parts) == 1 else [root / relative]
    found = []
    for path in matches:
        if path.is_file():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'asset file escapes its mounted dataset: {name}')
            found.append(path)
    if len(found) != 1:
        raise RuntimeError(f'{name}: expected one mounted file, found {len(found)}')
    return found[0]


def verify_file(path: Path, expected: str, size: int | None = None) -> dict:
    if not isinstance(expected, str) or SHA_RE.fullmatch(expected) is None:
        raise ValueError('asset SHA256 must be lowercase hex')
    if not path.is_file() or (size is not None and (not isinstance(size, int) or size <= 0 or path.stat().st_size != size)):
        raise RuntimeError(f'asset missing or byte count mismatch: {path.name}')
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f'asset SHA256 mismatch: {path.name}')
    return {'file': path.name, 'sha256': actual, 'bytes': path.stat().st_size}


KAGGLE_NOEXTRACT_SUFFIX = '.noextract'


@contextlib.contextmanager
def _open_tar(path: Path, zstd: Path | None):
    if zstd is None:
        with tarfile.open(path, mode='r:*') as archive:
            yield archive
        return
    environment = dict(os.environ, LD_LIBRARY_PATH=str(zstd.parent) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    process = subprocess.Popen([str(zstd), '-d', '-c', '--', str(path)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            yield archive
        process.stdout.close()
        _, error = process.communicate(timeout=60)
        if process.returncode:
            raise RuntimeError(f'bundled zstd failed ({process.returncode}): {error.decode(errors="replace")[-500:]}')
    except BaseException:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=10)
        raise


def _members(archive: tarfile.TarFile, prefix: str, *, audited_runtime: bool = False) -> dict[str, tarfile.TarInfo]:
    expected = _relative(prefix).as_posix()
    rows = {}
    for member in archive:
        if member.isdir() and PurePosixPath(member.name) == PurePosixPath('.'):
            continue
        name = _relative(member.name).as_posix()
        if name != expected and not name.startswith(expected + '/'):
            raise ValueError(f'archive has unexpected root: {name}')
        if name in rows:
            raise ValueError(f'duplicate archive member: {name}')
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()) or member.mode & 0o6000:
            raise ValueError(f'archive special file or privileged mode: {name}')
        if member.issym() or member.islnk():
            link = member.linkname
            if not link or '\\' in link or '\x00' in link:
                raise ValueError(f'unsafe archive link: {name}')
            if PurePosixPath(link).is_absolute():
                if not audited_runtime or not member.issym() or '..' in PurePosixPath(link).parts:
                    raise ValueError(f'unsafe archive link: {name}')
            else:
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link) if member.issym() else link)
                if target != expected and not target.startswith(expected + '/'):
                    raise ValueError(f'archive link escapes extraction root: {name}')
                _relative(target)
        rows[name] = member
    if not rows:
        raise ValueError('empty archive')
    for name, member in rows.items():
        for parent in PurePosixPath(name).parents:
            ancestor = rows.get(parent.as_posix())
            if ancestor is not None and not ancestor.isdir():
                raise ValueError(f'archive child traverses non-directory: {name}')
        if member.islnk():
            target = rows.get(posixpath.normpath(member.linkname))
            if target is None or not target.isfile():
                raise ValueError(f'archive hardlink must target a regular member: {name}')
    return rows


def _runtime_symlinks(rows: dict[str, tarfile.TarInfo]) -> dict[str, dict]:
    changes = {}
    roots = [('site-packages/', 'package/lib/python3.12/site-packages/'),
             ('cuda-13.1/', 'package/cuda-13.1/')]
    for name, member in rows.items():
        if not member.issym() or not PurePosixPath(member.linkname).is_absolute():
            continue
        candidates = set()
        for marker, destination in roots:
            marker = '/' + marker
            if member.linkname.count(marker) == 1:
                target = destination + member.linkname.split(marker, 1)[1]
                _relative(target)
                if target in rows and rows[target].isfile():
                    candidates.add(target)
        if len(candidates) > 1:
            raise ValueError(f'ambiguous audited runtime symlink: {name}')
        target = candidates.pop() if candidates else None
        changes[name] = {'name': name, 'original_target': member.linkname,
                         'action': 'rewritten' if target else 'omitted',
                         'relocated_target': posixpath.relpath(target, posixpath.dirname(name)) if target else None,
                         'reason': 'target is a regular archive member' if target else 'absolute target is not carried as a regular archive member'}
    return changes


def extract_archive(path: Path, destination: Path, *, prefix: str, zstd: Path | None = None,
                    audited_runtime: bool = False) -> dict:
    """Validate every member before writing; then stream regular files without tar path following."""
    if destination.exists() and (destination.is_symlink() or any(destination.iterdir())):
        raise RuntimeError('archive destination must be a fresh directory')
    if audited_runtime:
        if prefix != 'package':
            raise ValueError('runtime relocation requires the package root')
        verify_file(path, RUNTIME_SHA, RUNTIME_BYTES)
    with _open_tar(path, zstd) as archive:
        expected = _members(archive, prefix, audited_runtime=audited_runtime)
    remapped = _runtime_symlinks(expected) if audited_runtime else {}
    destination.mkdir(parents=True, exist_ok=True)
    links = []
    with _open_tar(path, zstd) as archive:
        observed = set()
        for member in archive:
            if member.isdir() and PurePosixPath(member.name) == PurePosixPath('.'):
                continue
            name = _relative(member.name).as_posix()
            original = expected[name]
            if (member.type, member.linkname, member.size, member.mode) != (original.type, original.linkname, original.size, original.mode):
                raise RuntimeError('archive changed between member validation and extraction')
            observed.add(name)
            output = destination / name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, output.open('xb') as target:
                    shutil.copyfileobj(source, target, 8 << 20)
                output.chmod(member.mode & 0o777)
            else:
                links.append((output, member))
        if observed != set(expected):
            raise RuntimeError('archive member set changed during extraction')
    for output, member in links:
        output.parent.mkdir(parents=True, exist_ok=True)
        if member.issym():
            change = remapped.get(output.relative_to(destination).as_posix())
            if change is not None and change['action'] == 'omitted':
                continue
            output.symlink_to(change['relocated_target'] if change else member.linkname)
        else:
            os.link(destination / posixpath.normpath(member.linkname), output)
    result = {'members': len(expected), 'regular_bytes': sum(m.size for m in expected.values() if m.isfile())}
    if audited_runtime:
        result['absolute_symlinks'] = list(remapped.values())
        result['omitted_symlinks'] = sum(row['action'] == 'omitted' for row in remapped.values())
    return result


def _zstd_tool(root: Path, directory: Path, manifest: dict) -> tuple[Path, list[dict]]:
    binary = asset_file(root, 'zstd')
    specification = manifest.get('zstd_binary')
    if specification is not None:
        if specification['file'] != 'zstd':
            raise ValueError('bundled decompressor must be named zstd')
        verify_file(binary, specification['sha256'], specification['bytes'])
    with binary.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            raise RuntimeError('bundled zstd must be an ELF executable')
    directory.mkdir()
    sources = [binary, *sorted(binary.parent.glob('lib*.so*'))]
    rows = []
    names = set()
    for source in sources:
        _relative(source.name)
        if source.name in names or not source.is_file() or not source.resolve().is_relative_to(root.resolve()):
            raise ValueError('invalid bundled zstd library')
        names.add(source.name)
        target = directory / source.name
        shutil.copyfile(source, target)
        target.chmod(0o755 if source.name == 'zstd' else 0o644)
        rows.append({'file': source.name, 'sha256': sha256_file(target), 'bytes': target.stat().st_size})
    return directory / 'zstd', rows


def _runtime_module(payload_repo: Path):
    source = payload_repo / 'work/qwen38fn_l4/probes/runpod_runtime.py'
    if not source.is_file():
        raise RuntimeError('payload lacks frozen runpod_runtime.py')
    sys.path.insert(0, str(payload_repo / 'src'))
    specification = importlib.util.spec_from_file_location('_kaggle_h28_runpod_runtime', source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    expected = payload_repo / 'src/arcforge/sglang_tree.py'
    if sha256_file(Path(module.sglang_tree.__file__)) != sha256_file(expected):
        raise RuntimeError('ambient arcforge.sglang_tree differs from the frozen payload')
    return module


def _python_version() -> tuple[int, int, int]:
    return tuple(sys.version_info[:3])


def _source_manifest(source: Path, manifest: Path) -> None:
    names = []
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (\S+)', line)
        if match is None:
            raise ValueError('invalid source manifest row')
        digest, name = match.groups()
        names.append(_relative(name).as_posix())
        verify_file(source / 'sglang' / name, digest)
    if not names or names != sorted(set(names)) or 'srt/mem_cache/qsa_hisparse_distributed.py' not in names:
        raise ValueError('source manifest lacks sorted unique h25 distributed source')


CANDIDATE_ID = "r7-r4"
CANDIDATE_TREE = "4eeea3c1baebf8074b6822d3a42307a36938d1eea33a80ac06c449e3a7e8fb80"
CANDIDATE_SOURCE_SHA = "9a4fd68562d4c2915447eed5712bb86efac3d88a1884b1a114fab140a131204d"
CANDIDATES = {
    CANDIDATE_ID: {"tree": CANDIDATE_TREE, "source_sha": CANDIDATE_SOURCE_SHA},
    "r8-r2": {"tree": "cd5709493e4e450e1ffd277417f480bddc267d19389dba86c30c481a387f1e26",
              "source_sha": "e8c32d8f3a138bf5f0fae8560fd98da743443e17a51b3d3db96cb6b6b1b8d1f2",
              "archive_sha": "2dd580e94efac3c3fa7af61bd6b2e1b723c43fb94d446ed1cf092741b5f44761"},
    "r9": {"tree": "65be14ac07f720d02ead2ca7a29a49ef498d70cd66b8e4486f4c604f2fb57232",
           "source_sha": "34d43120c82ea3e443c900f76e0a23d0c22f68688a80139a14b76c99fcb69850",
           "archive_sha": "936e08e45f1728a31af403305d110c0fc43c896ca5529f7c98b481c0b66f7e3c"},
    "r9-r2": {"tree": "1742410a5edfd8c4875b877001e81efa93b6b0d33124b42e956297d30403e175",
              "source_sha": "0e87cb851e4991188f2b461d45d461909b4d2272093252f2d02f77ed7265de7a",
              "archive_sha": "7341f2532274c7a237906bf4d4d6c56eda727337c49aaab8b16f50689cef8245"},
    # v16: sglang-arc-2 arc/main dfd2fd8f (r9-r2 + l01 + h40-h43 + the shared DFlash worker), staged by
    # stage_fork_candidate.py; deref digest = deploy/sglang.lock.json arc-main digest_deref.
    "arc-main": {"tree": "ccbd293cbc715080ab1f543181915a0456b8ac3a4f4d7266fc1f18f337212706",
                 "source_sha": "b67d4dfb8a100887ff72a928d6d54d92fb98ad85e23060f606bed7b0856273c3",
                 "archive_sha": "ad2766e071d7ee44a502e74149ffc848e5579fe698d49c74f1febb77f6e0d4a9",
                 "files": 4915, "fork_commit": "dfd2fd8fb687ff1b0344c9ae311f7b1f88f5114e"},
    # v16 dense DFlash: sglang-arc-2 exp/dense-dflash 86932128 = arc/main dfd2fd8f + the dense-routed DFlash drafter
    # (ffn_type dense_shared, block-FP8 linears, L4 attention-TP pair); deref digest = lock g4-dense-dflash.
    "arc-dense": {"tree": "b6c0eb6df98ba34013181aa87155f7f4572167626ae4a06949d43754f6cf1187",
                  "source_sha": "78cc675e146ebf0b28e9a0ab5d838cfcab77de3c3f11cd2c9c5fe321087b38be",
                  "archive_sha": "cc65652921933fe70f956b6451447c6e8078cdff02778734a59fb210798329dd",
                  "files": 4916, "fork_commit": "86932128dd9be22504372a1c7a8f3eb064829a16"},
    # arc-dense + ARC_DFLASH_COMPILE=det: exp/dense-dflash-det 8aea7bd0 (fork 135068d5a cherry-picked onto 86932128);
    # deref digest = lock l4-dense-dflash-det.
    "arc-dense-det": {"tree": "5386de511da1664b404de2aae6cc0e627ff6b5f79dbe0137f5f54e614e31dcab",
                      "source_sha": "b51904bfc10af2c3689ae29774ad527f32b8daa99052aac310db142a514c97ae",
                      "archive_sha": "707f66b127201e46d89e7c121a4218d71584a9c767022dbe8003c05cf39a85ed",
                      "files": 4916, "fork_commit": "8aea7bd0b6ee9c49f85bdceb68b62dda7a9198c9"},
    # arc-dense-det + DFlash draft vocabulary (ARC_DFLASH_DRAFT_VOCAB, SubsetPairHead on the L4 pairs):
    # exp/l4-dense-det-vocab 03024560; deref digest = lock l4-dense-det-vocab.
    "arc-dense-vocab": {"tree": "8ab40eab7dc6546f041b71e2636db92d9dd6406cc962b8fb104ac0ab017ecca5",
                        "source_sha": "41a0c274a57c2e64dd0cf80ca736010fa42664e1f9a064a85f3494b1eb1fd461",
                        "archive_sha": "eef0edbc2e4667c338cb1c9c785235f135fc8408420d818195782625ef2adb2f",
                        "files": 4916, "fork_commit": "0302456069cffc1b296da4d155b6e3476e347825"},
    # arc-dense-vocab + pure TP4 attention under ARC_HISPARSE_TP4=1 (l4opt topology experiment): exp/l4-tp4-attn
    # 63bc045b; deref digest = lock l4-tp4-attn.
    "arc-dense-vocab-tp4": {"tree": "b98605fd0bec8da7290154b48ce94b96ce73f9d441d9ec3a283479a07b871d94",
                            "source_sha": "773bb6f4b6670f1963327f883d9242c255cf59c2345c7f5e6ba23010316af979",
                            "archive_sha": "04b10c448194b6868220ac0d26290171d2da6744aac6b70e7c58b2cefafcded1",
                            "files": 4916, "fork_commit": "63bc045b14b2e93c66ab9e045c87bff2607135e0"},
    # arc-dense-vocab-tp4 + exact vocab-sharded verify sampling under ARC_SHARDED_VERIFY=1 (l4opt) + h30 cache advice over
    # the identity's shard roster (gEAN 104 shards) + per-GPU DFlash memory profile dirs + the unpadded PLE extend / early
    # PLE frees + opt-in ARC_PLE_NVFP4 / ARC_PLE_INT5 host PLE tables (main 16ec1914 / ee585f82 / a6466c50 / 28530bd7 /
    # 9b24ae2b) + G4's scheduler / extend sync removals and ViT batches (main 3d17a8d4 / 62a873f2 / 5cafe074 / 0d808d61 /
    # d2e4567b / 338d5a64; opt-in ARC_GDN_HOST_CHUNK_INDICES / ARC_MM_NO_SYNC) + DFlash prefill ids and opt-in exact
    # prefill / session park / async refresh (main 8a9ae114 / 56d78f62 / 06fd8d9c) + int64 index_fill_ in the HiSparse
    # pool (main 391f2df3; without it 3d17a8d4 raised at the first finished request) + B200 decode opt-ins
    # ARC_DFLASH_RING_TILE / RING_OUT_T / BATCHED_COMMIT / FUSED_PROPOSE / FAST_TOPK and ARC_HC_GATE_PRECOMPUTE
    # (exp/b200-perf-extend 4e651d87 / e71c3fdc / e1c2167a / e6271148 / cb97a914) + opt-in ARC_QSA_SM89_DECODE /
    # ARC_QSA_SM89_STAGES (bf16 E2M1 dequant in the HiSparse sm_89 direct decode, same bits) + the PLE short-conv
    # verify step windows written in one launch (same bytes) + SubsetPairHead rows padded to a multiple of 64 + h24
    # packed GDN state on TP > 1 (off unless ARC_GDN_STATE_FMT) + the target lm_head restricted to a fixed output
    # vocabulary (off unless ARC_TARGET_VOCAB) + drafter graphs in the target pool on an attention-TP group (opt-in) +
    # main's multimodal host-RAM fixes (opt-in ARC_MM_RELEASE_AFTER_PREFILL / ARC_MM_LAZY_MEDIA, the retract media
    # refill, video id splice, --no-mm-literal-image-sentinel) + the drafter's per-card memory (attention slices cloned,
    # selector codebooks pruned under ARC_TARGET_VOCAB, opt-in ARC_DFLASH_W4A8 on Humming's batch-invariant tuning for the
    # linears replicated across the attention-TP group) + opt-in ARC_HEAPDIAG + opt-in ARC_HISPARSE_MISS_STATS (swap-in
    # counters, 10-13 review E5): exp/l4-shv-review c8c83127 = exp/l4-sharded-verify 75a590f9 + the counters (l4opt20 ran
    # 53192d1a, tree f74ed58c;
    # l4opt21 / 22 lacked the index fix; l4opt23-25 ran efbcbcc4, tree b195e71b; l4opt26 18882064, tree 956c4cd8;
    # l4opt27 21bd6674, tree a9d50e18; l4opt28 272bad3b, tree 53cf4066; l4opt29 5ec4e14a, tree b5e7d7fe; l4opt30 / 31
    # 2f31c0cb, tree 65368574; l4opt32 d94aef29, tree f389150a: the pruned head failed on a 0-row IDLE batch; l4opt33
    # 53453a5a, tree d8676a30; l4opt34 b0986281, tree b5abb286: W4A8 on Humming's stream-K diverged the pair; l4opt35
    # bba827f1, tree 1d2c7087: every W4A8 linear batch-invariant; l4opt36 3781e841, tree 913ce47f; l4opt37 3b48fa36, tree
    # 05846869: the heap_diag deep dump died on an object whose __sizeof__ is not an int; l4opt38 dfa092d0, tree fc446599:
    # it died on a FakeTensor's data pointer; l4opt39 / 40 ran 75a590f9, tree 44232dca; l4opt41 c8c83127, tree
    # cd2597c4, lock l4-shv-review: codex2b / codex2wr 57150367 -- the builder no longer accepts it).
    # Now (l4opt43, 10-14): fork all-merge 8d4dea09 = the python tree of fork main ed837597 (all-merge + a scripts/
    # test fix), deref digest = lock main digest_deref; stage_fork_candidate.py --lock-target main reproduces this
    # archive_sha and PACKAGE_SHA256.json. First run as the 10-13 Kaggle Save & Run allmerge-codex4-c96 v1 on an
    # unpushed pin commit (160f6e4e); this entry is that commit's. Flags unset it serves the arc-dense-vocab-tp4
    # paths (the PLE extend is bitwise).
    # Now (l4opt46, 10-16): fork exp/l4-qsa-index-shard 5c196331 = all-merge 8d4dea09 + opt-in ARC_QSA_INDEX_SHARD
    # (QSA index split over the attention-TP pair by layer), lock l4-qsa-index-shard; unset, the 8d4dea09 paths
    # (indexer outputs and index state bitwise, docs/research/qsa_index_layer_shard_l4_20261016.md). l4opt43-45 ran
    # 8d4dea09, tree 07b13538 -- the builder no longer accepts them.
    # Now (l4opt46pb, 10-20): fork exp/planb-serve-l4 c8500d9fe = exp/kverify-l4 85170572 (5c196331 + the QSA verify
    # indexer's padded grouped windows and grouped GEMM on the real index heads, opt-out ARC_QSA_GROUPED_VERIFY_PAD /
    # ARC_QSA_GROUPED_UNPAD_HEADS; payload l4opt46kv, tree 7d562b29, 10-19) + the plain / injected-context DFlash
    # drafter's serving (config residual plain, context inject, ffn_type dense; ARC_DFLASH_W4A8 on its linears), lock
    # l4-planb-serve. The dense_shared drafter loads as on 85170572. l4opt46 ran 5c196331, tree 4b6cb81b -- the builder
    # no longer accepts it.
    # Now (l4opt46pbo, 10-20): fork 60997af63a (exp/planb-opt-l4) = exp/planb-serve-l4 aed8f6fe4 (c8500d9fe + a test
    # script) + exp/planb-opt's plan B serving opt-ins, all default off (ARC_DFLASH_MERGE, ARC_DFLASH_INJECT_KV,
    # ARC_DFLASH_FUSE_ONCE, ARC_DFLASH_KV_DTYPE, ARC_DFLASH_RING_FP8_DOT, ARC_DFLASH_FP8_OP), lock l4-planb-opt.
    # l4opt46pb ran c8500d9fe, tree a8fc06fc -- the builder no longer accepts it.
    # Now (l4opt47am, 10-20): fork all-merge-1020 a5c67c41cd (fork main a6c7fddbb + qshard + plan B on B200 / G4 / L4),
    # lock l4-all-merge-1020. l4opt46pbo ran 60997af63a, tree 6ce650f1 -- the builder no longer accepts it.
    "arc-dense-vocab-tp4-shv": {"tree": "8aa4af5f6a3e55ba21c7836f6b29ea0a90b933b2498b73327d43e9fcc133e281",
                                "source_sha": "4cd0ebe2b31c0b01cddee96e38fe16448b986ed027bc118d3e265907f61b0c02",
                                "archive_sha": "fcae3d799a98aa1ab79464f5ff6104fad51f124c36a44a35332047487128acd6",
                                "files": 4943, "fork_commit": "a5c67c41cd22933a153ff82d8b6c5a6ca6f458fb"},
}


def candidate_files(candidate_id: str) -> int:
    """Whole-package file count of a registered candidate (patch-built trees keep the 4901-file h28 roster)."""
    return CANDIDATES[candidate_id].get("files", SOURCE_FILES)


def candidate_identity(manifest: dict) -> dict | None:
    """Explicit registered source pin; never infer candidate mode from a digest."""
    if 'candidate' not in manifest:
        return None
    candidate, source = manifest['candidate'], manifest['source']
    pin = CANDIDATES.get(candidate['id'])
    if pin is None or (candidate['expect_tree'], source['digest'], source['files'],
            source['manifest_sha256']) != (pin['tree'], pin['tree'], pin.get('files', SOURCE_FILES), pin['source_sha']):
        raise ValueError('candidate must explicitly pin registered tree and SOURCE SHA')
    if 'archive_sha' in pin and source['sha256'] != pin['archive_sha']:
        raise ValueError('candidate source archive SHA differs from explicit registry')
    for key in ('record', 'package_manifest', 'cpu_validation'):
        entry = candidate[key]
        _relative(entry['file'])
        if not isinstance(entry['sha256'], str) or SHA_RE.fullmatch(entry['sha256']) is None:
            raise ValueError('invalid candidate metadata SHA')
    if source['archive'] != 'sglang-candidate-' + pin['tree'][:12] + '.tar.gz':
        raise ValueError('candidate source archive must pin its tree')
    return candidate


def install(manifest: dict, *, work_dir: Path, output_dir: Path) -> dict:
    """Return exactly python/env/runtime/model/payload_repo/source_manifest after CPU identity gates."""
    output_dir = Path(output_dir)
    work_dir = Path(work_dir)
    if output_dir.is_symlink() or output_dir.resolve().is_relative_to(INPUT_ROOT.resolve()):
        raise ValueError('evidence directory must be writable and outside mounted input')
    if work_dir.resolve() == output_dir.resolve() or output_dir.resolve().is_relative_to(work_dir.resolve()) or work_dir.resolve().is_relative_to(INPUT_ROOT.resolve()):
        raise ValueError('work directory must be writable and separate from mounted input and evidence')
    output_dir.mkdir(parents=True, exist_ok=True)
    if work_dir.exists() and (work_dir.is_symlink() or any(work_dir.iterdir())):
        raise RuntimeError('offline install requires a fresh work directory; preserve existing evidence')
    if _python_version() != (3, 12, 13):
        raise RuntimeError('Kaggle image must supply Python 3.12.13; this installer does not download Python')
    source_spec = manifest['source']
    runtime_spec = manifest['runtime_archive']
    allowed_sources = {(SOURCE_DIGEST, SOURCE_FILES, SOURCE_MANIFEST_SHA),
                       (FAST_SOURCE_DIGEST, SOURCE_FILES, FAST_SOURCE_MANIFEST_SHA)}
    candidate = candidate_identity(manifest)
    if candidate is None and (source_spec['digest'], source_spec['files'], source_spec['manifest_sha256']) not in allowed_sources:
        raise ValueError('source must be frozen h28 or its startup-only h30 candidate; h29 is not validated')
    expected_digest = source_spec['digest']
    expected_manifest_sha = source_spec['manifest_sha256']
    expected_files = SOURCE_FILES if candidate is None else candidate_files(candidate['id'])
    if (runtime_spec['file'], runtime_spec['sha256'], runtime_spec['bytes'], runtime_spec['provenance_sha256']) != (RUNTIME_FILE, RUNTIME_SHA, RUNTIME_BYTES, PROVENANCE_SHA):
        raise ValueError('runtime must be the audited 0520 archive')
    record = {'status': 'installing', 'offline': True, 'python': sys.executable,
              'python_version': '3.12.13', 'source_digest': expected_digest, 'source_files': expected_files,
              'assets': {}, 'extraction': {}, 'h29_included': candidate is not None}
    _write_json(output_dir / 'install_receipt.json', record)
    try:
        mounts = {key: dataset_mount(manifest[key]) for key in ('source_dataset', 'runtime_dataset', 'zstd_dataset')}
        model_source = model_mount(manifest['model_source'])
        setup_spec = manifest['runtime_setup']
        setup = asset_file(mounts['source_dataset'], setup_spec['file'])
        record['assets']['runtime_setup'] = verify_file(setup, setup_spec['sha256'])
        if sha256_file(Path(__file__)) != setup_spec['sha256']:
            raise RuntimeError('executed setup differs from the manifest-pinned installer')
        payload_spec = manifest['payload']
        payload_archive = asset_file(mounts['source_dataset'], payload_spec['archive'])
        # Kaggle auto-extracts uploaded .tar.gz (the file disappears, its members appear instead), so the
        # candidate archive is uploaded under a suffixed name; bytes and SHA are the pinned archive's.
        source_file = source_spec['archive']
        if candidate is not None:
            if source_spec['dataset_file'] != source_spec['archive'] + KAGGLE_NOEXTRACT_SUFFIX:
                raise ValueError('candidate source dataset file must be the pinned archive name plus the no-extract suffix')
            source_file = source_spec['dataset_file']
        source_archive = asset_file(mounts['source_dataset'], source_file)
        source_manifest = asset_file(mounts['source_dataset'], source_spec['manifest'])
        runtime_archive = asset_file(mounts['runtime_dataset'], runtime_spec['file'])
        record['assets']['payload'] = verify_file(payload_archive, payload_spec['sha256'], payload_spec['bytes'])
        record['assets']['source'] = verify_file(source_archive, source_spec['sha256'], source_spec['bytes'])
        record['assets']['source_manifest'] = verify_file(source_manifest, expected_manifest_sha)
        record['assets']['runtime'] = verify_file(runtime_archive, RUNTIME_SHA, RUNTIME_BYTES)
        work_dir.mkdir(parents=True, exist_ok=True)
        zstd, record['assets']['zstd'] = _zstd_tool(mounts['zstd_dataset'], work_dir / 'bin', manifest)
        record['extraction']['payload'] = extract_archive(payload_archive, work_dir / 'payload', prefix='repo', zstd=zstd)
        payload_repo = work_dir / 'payload/repo'
        launcher = payload_repo / _relative(payload_spec['launcher'])
        record['assets']['launcher'] = verify_file(launcher, payload_spec['launcher_sha256'])
        if candidate is not None:
            for name, digest in payload_spec['files_sha256'].items():
                verify_file(payload_repo / _relative(name), digest)
        runtime_module = _runtime_module(payload_repo)
        runtime = work_dir / 'runtime'
        record['extraction']['runtime'] = extract_archive(runtime_archive, runtime, prefix='package', zstd=zstd, audited_runtime=True)
        provenance = runtime / 'package/runtime-provenance.json'
        record['assets']['provenance'] = verify_file(provenance, PROVENANCE_SHA)
        prefix = 'sglang' if candidate is not None else 'python/sglang'
        destination = runtime / 'hisparse/python' if candidate is not None else runtime / 'hisparse'
        record['extraction']['source'] = extract_archive(source_archive, destination, prefix=prefix, zstd=zstd)
        installed_manifest = runtime / 'SOURCE_SHA256'
        shutil.copyfile(source_manifest, installed_manifest)
        source = runtime / 'hisparse/python'
        identity = runtime_module.source_identity(source)
        if (identity['digest'], identity['files']) != (expected_digest, expected_files):
            raise RuntimeError('installed source differs from frozen whole-package h28 identity')
        _source_manifest(source, installed_manifest)
        if candidate is not None:
            metadata = {}
            for key in ('record', 'package_manifest', 'cpu_validation'):
                entry = candidate[key]
                mounted = asset_file(mounts['source_dataset'], entry['file'])
                record['assets'][key] = verify_file(mounted, entry['sha256'])
                metadata[key] = json.loads(mounted.read_text())
                shutil.copyfile(mounted, runtime / Path(entry['file']).name)
            expected = metadata['package_manifest']
            if runtime_module.sglang_tree.package_manifest(source / 'sglang') != expected:
                raise RuntimeError('candidate full package manifest mismatch')
            row = metadata['record']
            if (row['tree_sha256'], row['files'], row['source_manifest_sha256'], row['source_archive_sha256']) != (
                    expected_digest, expected_files, expected_manifest_sha, source_spec['sha256']):
                raise RuntimeError('candidate composition receipt mismatch')
            cpu = metadata['cpu_validation']
            if cpu['tree_sha256'] != expected_digest or cpu['files'] != expected_files or cpu['roundtrip_verified'] is not True:
                raise RuntimeError('candidate lacks actual CPU archive validation')
        runtime_record = {'kind': 'kaggle-offline-frozen-h28', 'arch': '8.9',
                          'package': str((runtime / 'package').resolve()), 'archive_sha256': RUNTIME_SHA,
                          'provenance_sha256': PROVENANCE_SHA, 'arms': {'hisparse': identity},
                          'source_manifest_sha256': expected_manifest_sha, 'distributed_gpu_validation': 'Runpod h28 passed; Kaggle and startup-only h30 pending'}
        if candidate is not None:
            runtime_record.update(kind='kaggle-offline-candidate', candidate=candidate,
                                  distributed_gpu_validation='pending on this Kaggle host')
        _write_json(runtime / 'RUNTIME.json', runtime_record)
        runtime_module.check_runtime(runtime)
        model = work_dir / 'model'
        model.mkdir()
        target_manifest = manifest['model_identity']
        if not target_manifest['target_sizes'] or not target_manifest['target_sha256'] or not set(target_manifest['target_sha256']).issubset(target_manifest['target_sizes']):
            raise ValueError('model metadata hashes must identify sized model files')
        for name, expected_hash in target_manifest['target_sha256'].items():
            _relative(name)
            if not isinstance(expected_hash, str) or SHA_RE.fullmatch(expected_hash) is None:
                raise ValueError('model metadata SHA256 must be lowercase hex')
        for name, size in target_manifest['target_sizes'].items():
            if not isinstance(size, int) or size <= 0:
                raise ValueError('model file sizes must be positive integers')
            relative = _relative(name)
            mounted_file = model_source / relative
            if not mounted_file.is_file() or mounted_file.stat().st_size != size or not mounted_file.resolve().is_relative_to(model_source.resolve()):
                raise RuntimeError(f'mounted model file missing, escaped, or size mismatch: {name}')
            target = model / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(mounted_file.resolve())
        verified = runtime_module.verify_model(model, {'MANIFEST': target_manifest,
                                                       'IDENTITY': '/'.join(_model_reference(manifest['model_source']))})
        if _model_reference(verified['identity']) != _model_reference(manifest['model_source']):
            raise RuntimeError('model asset reference differs from the verified checkpoint identity')
        _write_json(model / 'verified_identity.json', verified)
        _write_json(output_dir / 'verified_model_identity.json', verified)
        python, env, _ = runtime_module.runtime_env(runtime, 'hisparse', sys.executable)
        env['PYTHONPATH'] = str(payload_repo / 'src') + ':' + env['PYTHONPATH']
        env['RUNPOD_PYTHON'] = python
        env['PYTHONNOUSERSITE'] = '1'
        env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
        env['PIP_NO_INDEX'] = '1'
        # The runner needs these four paths; all model/server output remains owned by this notebook.
        env.update(ARC_ROOT=str(payload_repo), ARC_DATA=str(work_dir / 'data'), ARC_MODELS=str(work_dir),
                   ARC_RUNS=str(output_dir), ARC_RAM_RUNS=str(work_dir / 'runs'))
        for directory in ('data', 'runs'):
            (work_dir / directory).mkdir()
        record.update(status='installed', runtime=runtime_record, model=verified, payload_repo=str(payload_repo),
                      source_manifest_sha256=sha256_file(installed_manifest),
                      environment={key: env[key] for key in ('PYTHONPATH', 'CUDA_HOME', 'HF_HUB_OFFLINE',
                                                           'TRANSFORMERS_OFFLINE', 'PYTHONNOUSERSITE', 'PIP_NO_INDEX')})
        _write_json(output_dir / 'install_receipt.json', record)
        _write_json(output_dir / 'asset_manifest_used.json', manifest)
        return {'python': python, 'env': env, 'runtime': runtime, 'model': model,
                'payload_repo': payload_repo, 'source_manifest': installed_manifest}
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        _write_json(output_dir / 'install_receipt.json', record)
        raise
