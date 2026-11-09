#!/usr/bin/env python3
"""Local → NFS file mover. O_DIRECT throughout: no page cache. Written on 2026-10-06 while moving the TITO collection (270 GB) on node1.

Why: the two NFS hangs on 10-05 and 10-06 were both dirty pages left by buffered writes being resent forever in the
kernel; the writing process was stuck waiting on writeback (uninterruptible) and could not be killed. O_DIRECT writes
are issued in the process's own syscall and wait in a killable wait, leaving no dirty pages; read-back verification
also uses O_DIRECT, so it reads from the server, not the local cache.

Runs in a separate systemd user unit (`systemd-run --user --unit=<name> --collect ...`), so even if it hangs it does not
drag down the session's cgroup; `systemctl --user stop <name>` stops it. One file at a time, a single process (flock). Per file:
local read + sha256 → O_DIRECT write <dst>.partial → O_DIRECT read back and sha256 → only on match utime + rename
→ with --delete remove the local copy → append one line to <src-root>/ARCHIVED_TO_NFS.txt.
Before and after each file, compares READ/WRITE/COMMIT of the destination NFS mount in /proc/self/mountstats: any
increase in errors, timeouts, or retransmissions stops it. Watchdog: no progress for --stall seconds during NFS I/O → SIGKILL itself.
log / lock / pid go in <run-dir>/lanes/, packing scratch in <run-dir>/.offload_tmp/ (both local).

nfs_move.py --run-dir R probe       --dst-dir D [--mib N]
nfs_move.py --run-dir R verify      --local L --nfs N [--delete --note-root R]
nfs_move.py --run-dir R check-jsonl --nfs N
nfs_move.py --run-dir R check-zst   --nfs N [--expect-files K]
nfs_move.py --run-dir R files       --src-root S --dst-root D [--delete] REL...
nfs_move.py --run-dir R dirs        --src-root S --dst-root D [--delete] REL...   (pack .tar.zst, verify with tar -d, then move)
nfs_move.py --run-dir R ledger-live --instance I --dst-root D [--loop SEC]
A running ledger instance: routing.<n>.bin / topk.<n>.bin already rolled and closed (n smaller than the current largest n
of the same kind, which the writer never touches again) are moved to D/<instance>/,
and after verification replaced locally in place by a symlink to NFS (readers using stat()/open() follow it). Before each
file it checks whether anyone else on this machine wrote to the same NFS mount in the last 5 seconds, and if so waits
for the next round, keeping only one write stream at a time.

Measured on node1 2026-10-06 (single stream, 16 MiB chunks): write 174 MB/s, O_DIRECT read-back ~1.7 GB/s (the READ count
in mountstats rises accordingly, so it really reads from the server); 66 files of 4.29 GB + 75 directories, errors / timeouts / retransmissions all 0 throughout.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import mmap
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

OPS = ("READ", "WRITE", "COMMIT")
KEY_FILES = ("tito_manifest.json", "tito_audit.json", "tito_audit.txt", "audit_class.txt", "ledger_status.json",
             "summary.json", "submission.json", "STATE.json", "ARM_STATE.json", "FINAL.json", "FINAL.txt",
             "scorecard_official.json")
MOUNT_RE = re.compile(r"^(\S+) mounted on (\S+) with fstype (\S+)")


def mountstats_blocks() -> list[tuple[str, str, str]]:
    """(mount point, fstype, block text), one entry per mount."""
    out = []
    for b in ("\n" + Path("/proc/self/mountstats").read_text()).split("\ndevice ")[1:]:
        m = MOUNT_RE.match(b)
        if not m:
            raise RuntimeError(f"unexpected mountstats header: {b.splitlines()[0]!r}")
        out.append((m.group(2), m.group(3), b))
    return out


def nfs_mount_of(path: str) -> str:
    """The NFS mount point containing path (longest prefix). Raises if not on NFS."""
    p = os.path.abspath(path)
    hits = [mnt for mnt, fstype, _ in mountstats_blocks()
            if fstype.startswith("nfs") and (p == mnt or p.startswith(mnt.rstrip("/") + "/"))]
    if not hits:
        raise RuntimeError(f"{path} is not on an NFS mount")
    return max(hits, key=len)


class Ctx:
    def __init__(self, run_dir: Path, mount: str, chunk_mib: int, max_mbps: float, stall: float):
        self.run_dir = run_dir
        self.mount = mount
        self.tmp = run_dir / ".offload_tmp"
        self.chunk = chunk_mib << 20
        self.max_mbps = max_mbps
        self.stall = stall
        self.last = time.monotonic()
        self.armed = False
        self.log_f = open(run_dir / "lanes" / "nfs_move.log", "a")
        threading.Thread(target=self._watch, daemon=True).start()

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{os.getpid()}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def tick(self) -> None:
        self.last = time.monotonic()

    def arm(self, on: bool) -> None:
        self.tick()
        self.armed = on

    def _watch(self) -> None:
        while True:
            time.sleep(5)
            if self.armed and time.monotonic() - self.last > self.stall:
                self.log(f"STALL: {self.stall:.0f}s without NFS progress -> SIGKILL self")
                os.kill(os.getpid(), signal.SIGKILL)

    def throttle(self, done: int, t0: float) -> None:
        if self.max_mbps > 0:
            ahead = done / (self.max_mbps * 1e6) - (time.monotonic() - t0)
            if ahead > 0:
                time.sleep(ahead)
        self.tick()

    def assert_nfs(self, path: Path) -> None:
        p = os.path.abspath(path)
        if not p.startswith(self.mount.rstrip("/") + "/"):
            raise RuntimeError(f"{path} is not under NFS mount {self.mount}")

    def counters(self) -> dict:
        blocks = [b for mnt, _, b in mountstats_blocks() if mnt == self.mount]
        if len(blocks) != 1:
            raise RuntimeError(f"expected exactly one mountstats block for {self.mount}, got {len(blocks)}")
        return parse_ops(blocks[0])


def parse_ops(block: str) -> dict:
    out = {}
    for line in block.splitlines():
        s = line.strip()
        for op in OPS:
            if s.startswith(op + ":"):
                v = [int(x) for x in s.split()[1:]]
                if len(v) != 9:
                    raise RuntimeError(f"unexpected mountstats per-op line: {line!r}")
                out[op] = {"ops": v[0], "retrans": v[1] - v[0], "timeouts": v[2], "errors": v[8]}
    if set(out) != set(OPS):
        raise RuntimeError(f"mountstats missing ops: {set(OPS) - set(out)}")
    return out


def assert_nfs_clean(before: dict, after: dict, what: str) -> None:
    bad = {f"{op}.{k}": after[op][k] - before[op][k] for op in OPS for k in ("retrans", "timeouts", "errors")
           if after[op][k] != before[op][k]}
    if bad:
        raise RuntimeError(f"NFS counters moved during {what}: {bad} -> stop, nothing more is touched")


def sha_local(ctx: Ctx, path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def sha_direct(ctx: Ctx, path: Path, size: int) -> str:
    ctx.assert_nfs(path)
    buf = mmap.mmap(-1, ctx.chunk)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    ctx.arm(True)
    t0 = time.monotonic()
    off = 0
    try:
        while True:
            n = os.preadv(fd, [buf], off)
            if n == 0:
                break
            with memoryview(buf)[:n] as mv:
                h.update(mv)
            off += n
            ctx.tick()
    finally:
        ctx.arm(False)
        os.close(fd)
        buf.close()
    if off != size:
        raise RuntimeError(f"{path}: read back {off} bytes, expected {size}")
    dt = time.monotonic() - t0
    ctx.log(f"  read-back {path.name} {size / 1e9:.2f} GB in {dt:.0f}s ({size / 1e6 / max(dt, 1e-3):.0f} MB/s, O_DIRECT)")
    return h.hexdigest()


def copy_direct(ctx: Ctx, src: Path, dst: Path) -> str:
    """src (local) → dst (NFS). Returns sha256. Raises if dst already exists."""
    ctx.assert_nfs(dst)
    if dst.exists():
        raise RuntimeError(f"{dst} already exists on NFS")
    partial = dst.with_name(dst.name + ".partial")
    dst.parent.mkdir(parents=True, exist_ok=True)
    st = src.stat()
    size = st.st_size
    buf = mmap.mmap(-1, ctx.chunk)
    h = hashlib.sha256()
    fs = os.open(src, os.O_RDONLY)
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    ctx.arm(True)
    t0 = time.monotonic()
    off = 0
    next_report = 2 << 30
    try:
        while off < size:
            n = os.preadv(fs, [buf], off)
            if n <= 0:
                raise RuntimeError(f"{src}: short local read at {off}/{size}")
            with memoryview(buf)[:n] as mv:
                h.update(mv)
                w = os.pwritev(fd, [mv], off)
            if w != n:
                raise RuntimeError(f"{partial}: short write at {off}: {w}/{n}")
            off += n
            ctx.throttle(off, t0)
            if off >= next_report:
                dt = time.monotonic() - t0
                ctx.log(f"  {src.name} {off / 1e9:.1f}/{size / 1e9:.1f} GB ({off / 1e6 / dt:.0f} MB/s)")
                next_report += 2 << 30
        if os.fstat(fd).st_size != size:
            raise RuntimeError(f"{partial}: size {os.fstat(fd).st_size} != {size}")
    finally:
        ctx.arm(False)
        os.close(fd)
        os.close(fs)
        buf.close()
    dt = time.monotonic() - t0
    ctx.log(f"  wrote {src.name} {size / 1e9:.2f} GB in {dt:.0f}s ({size / 1e6 / max(dt, 1e-3):.0f} MB/s, O_DIRECT)")
    want = h.hexdigest()
    got = sha_direct(ctx, partial, size)
    if got != want:
        raise RuntimeError(f"sha256 mismatch: {src} {want} != {partial} {got} (partial left for inspection)")
    os.utime(partial, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.rename(partial, dst)
    return want


def note(root: Path, what: str, where: Path, sha: str) -> None:
    with open(root / "ARCHIVED_TO_NFS.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{what}\t{where}\tsha256={sha}\n")


def preflight(ctx: Ctx) -> dict:
    a = ctx.counters()
    time.sleep(5)
    b = ctx.counters()
    assert_nfs_clean(a, b, "5 s idle preflight")
    return b


def cmd_probe(ctx: Ctx, args) -> None:
    d = Path(args.dst_dir)
    ctx.assert_nfs(d)
    size = (args.mib << 20) + 12345  # deliberately unaligned tail, to confirm this mount accepts O_DIRECT
    src = ctx.tmp / "probe.bin"
    src.parent.mkdir(exist_ok=True)
    with open(src, "wb") as f:
        left = size
        while left:
            k = min(left, 64 << 20)
            f.write(os.urandom(k))
            left -= k
    dst = d / f"probe-{os.getpid()}.bin"
    before = preflight(ctx)
    ctx.log(f"probe {size} bytes chunk {ctx.chunk >> 20} MiB -> {dst}")
    copy_direct(ctx, src, dst)
    assert_nfs_clean(before, ctx.counters(), "probe")
    dst.unlink()
    src.unlink()
    ctx.log("probe ok, test files removed")


def cmd_verify(ctx: Ctx, args) -> None:
    local, nfs = Path(args.local), Path(args.nfs)
    size = local.stat().st_size
    if nfs.stat().st_size != size:
        raise RuntimeError(f"size differs: {local} {size} vs {nfs} {nfs.stat().st_size}")
    before = preflight(ctx)
    want = sha_local(ctx, local)
    got = sha_direct(ctx, nfs, size)
    assert_nfs_clean(before, ctx.counters(), f"verify {nfs.name}")
    if got != want:
        raise RuntimeError(f"sha256 mismatch: {local} {want} != {nfs} {got}")
    ctx.log(f"verify OK {nfs} sha256={got}")
    if args.delete:
        local.unlink()
        note(Path(args.note_root), str(local.relative_to(args.note_root)), nfs, got)
        ctx.log(f"  local removed {local}")


def cmd_check_jsonl(ctx: Ctx, args) -> None:
    path = Path(args.nfs)
    ctx.assert_nfs(path)
    size = path.stat().st_size
    before = preflight(ctx)
    buf = mmap.mmap(-1, ctx.chunk)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    ctx.arm(True)
    off, lines, rest = 0, 0, b""
    try:
        while True:
            n = os.preadv(fd, [buf], off)
            if n == 0:
                break
            data = rest + buf[:n]
            h.update(buf[:n])
            parts = data.split(b"\n")
            rest = parts.pop()
            for p in parts:
                json.loads(p)
                lines += 1
            off += n
            ctx.tick()
    finally:
        ctx.arm(False)
        os.close(fd)
        buf.close()
    assert_nfs_clean(before, ctx.counters(), f"check-jsonl {path.name}")
    if off != size:
        raise RuntimeError(f"{path}: read {off} != size {size}")
    if rest:
        raise RuntimeError(f"{path}: does not end with newline ({len(rest)} trailing bytes)")
    ctx.log(f"check-jsonl OK {path} {lines} lines, all parse; sha256={h.hexdigest()}")


def cmd_check_zst(ctx: Ctx, args) -> None:
    path = Path(args.nfs)
    ctx.assert_nfs(path)
    size = path.stat().st_size
    before = preflight(ctx)
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", "zstd -dc | tar -tf - | grep -vc '/$'"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    buf = mmap.mmap(-1, ctx.chunk)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    ctx.arm(True)
    off = 0
    try:
        while True:
            n = os.preadv(fd, [buf], off)
            if n == 0:
                break
            h.update(buf[:n])
            p.stdin.write(buf[:n])
            off += n
            ctx.tick()
    finally:
        ctx.arm(False)
        os.close(fd)
        buf.close()
    p.stdin.close()
    out = p.stdout.read().decode().strip()
    if p.wait() != 0:
        raise RuntimeError(f"{path}: zstd/tar check failed rc={p.returncode}")
    assert_nfs_clean(before, ctx.counters(), f"check-zst {path.name}")
    if off != size:
        raise RuntimeError(f"{path}: read {off} != size {size}")
    if args.expect_files is not None and int(out) != args.expect_files:
        raise RuntimeError(f"{path}: {out} non-dir entries, expected {args.expect_files}")
    ctx.log(f"check-zst OK {path} {out} non-dir entries, zstd checksum ok; sha256={h.hexdigest()}")


def cmd_files(ctx: Ctx, args) -> None:
    sr, dr = Path(args.src_root), Path(args.dst_root)
    ctx.assert_nfs(dr)
    before = preflight(ctx)
    for i, rel in enumerate(args.rel, 1):
        src, dst = sr / rel, dr / rel
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f"{src} is not a regular local file")
        ctx.log(f"[{i}/{len(args.rel)}] file {rel} {src.stat().st_size / 1e9:.2f} GB")
        sha = copy_direct(ctx, src, dst)
        after = ctx.counters()
        assert_nfs_clean(before, after, f"file {rel}")
        before = after
        if args.delete:
            src.unlink()
            note(sr, rel, dst, sha)
            ctx.log(f"  OK sha256={sha}; local removed")
        else:
            ctx.log(f"  OK sha256={sha}; local kept")
        time.sleep(args.pause)
    ctx.log(f"files done: {len(args.rel)}")


def cmd_dirs(ctx: Ctx, args) -> None:
    sr, dr = Path(args.src_root), Path(args.dst_root)
    ctx.assert_nfs(dr)
    tmpd = ctx.tmp
    tmpd.mkdir(exist_ok=True)
    before = preflight(ctx)
    for i, rel in enumerate(args.rel, 1):
        d = sr / rel
        if not d.is_dir() or d.is_symlink():
            raise RuntimeError(f"{d} is not a local directory")
        if not d.resolve().is_relative_to(sr.resolve()) or d.resolve() == sr.resolve():
            raise RuntimeError(f"{d} is not strictly inside {sr}")
        arch = tmpd / (rel.replace("/", "__") + ".tar.zst")
        if arch.exists():
            arch.unlink()
        n_files = sum(1 for p in d.rglob("*") if p.is_symlink() or p.is_file())
        t0 = time.monotonic()
        r = subprocess.run(["bash", "-o", "pipefail", "-c",
                            f"tar -C '{d.parent}' -cf - '{d.name}' | zstd -q -3 -T8 -o '{arch}'"],
                           capture_output=True, text=True)
        noise = [l for l in r.stderr.splitlines() if l.strip() and not l.endswith(": socket ignored")]
        if r.returncode != 0 or noise:
            raise RuntimeError(f"tar {rel} rc={r.returncode}: {r.stderr[-2000:]}")
        listed = int(subprocess.run(["bash", "-o", "pipefail", "-c", f"zstd -dc '{arch}' | tar -tf - | grep -vc '/$'"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        if listed != n_files:
            raise RuntimeError(f"{arch}: tar lists {listed} non-dir entries, directory has {n_files}")
        r = subprocess.run(["bash", "-o", "pipefail", "-c", f"zstd -dc '{arch}' | tar -d -C '{d.parent}' -f -"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"tar -d {rel} differs: {r.stdout[-2000:]} {r.stderr[-2000:]}")
        raw = sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and not p.is_symlink())
        ctx.log(f"[{i}/{len(args.rel)}] dir {rel}: {n_files} entries {raw / 1e9:.2f} GB -> "
                f"{arch.stat().st_size / 1e9:.2f} GB tar.zst, tar -d clean ({time.monotonic() - t0:.0f}s)")
        dst = dr / (rel + ".tar.zst")
        sha = copy_direct(ctx, arch, dst)
        keys = [k for k in KEY_FILES if (d / k).is_file() and not (d / k).is_symlink()]
        for k in keys:
            copy_direct(ctx, d / k, dr / rel / k)
        after = ctx.counters()
        assert_nfs_clean(before, after, f"dir {rel}")
        before = after
        arch.unlink()
        if args.delete:
            # The harness leaves read-only things in the workspace (refiner candidate dirs copied by seq --reuse are r-x): unlink needs a writable directory
            for sub in [d, *(p for p in d.rglob("*") if p.is_dir() and not p.is_symlink())]:
                sub.chmod(sub.stat().st_mode | 0o700)
            shutil.rmtree(d)
            note(sr, rel + "/", dst, sha)
            ctx.log(f"  OK sha256={sha}; {len(keys)} key files alongside; local dir removed")
        else:
            ctx.log(f"  OK sha256={sha}; {len(keys)} key files alongside; local dir kept")
        time.sleep(args.pause)
    ctx.log(f"dirs done: {len(args.rel)}")


def other_writer_active(ctx: Ctx) -> bool:
    a = ctx.counters()["WRITE"]["ops"]
    time.sleep(5)
    return ctx.counters()["WRITE"]["ops"] != a


def cmd_ledger_live(ctx: Ctx, args) -> None:
    inst, dr = Path(args.instance), Path(args.dst_root)
    ctx.assert_nfs(dr)
    if not (inst / "server.json").is_file():
        raise RuntimeError(f"{inst} is not a ledger instance (no server.json)")
    dst_dir = dr / inst.name
    before = preflight(ctx)
    while True:
        todo = []
        for kind in ("routing", "topk"):
            rolling = {}
            for f in inst.iterdir():
                m = re.fullmatch(kind + r"\.(\d+)\.bin", f.name)
                if m:
                    rolling[int(m.group(1))] = f
            top = max(rolling) if rolling else -1
            closed = [rolling[n] for n in sorted(rolling) if n < top and not rolling[n].is_symlink()]
            if args.release and kind == "routing":       # only shards no trainable episode can still read
                allowed = set(json.loads(Path(args.release).read_text())["instances"].get(inst.name, []))
                closed = [f for f in closed if int(f.name.split(".")[1]) in allowed]
            todo += [(f, top) for f in closed]
        for f, top in todo:
            if (ctx.run_dir / "OFFLOAD_STOP").exists():
                ctx.log("OFFLOAD_STOP present -> exit")
                return
            if other_writer_active(ctx):
                ctx.log(f"another writer is active on {ctx.mount} -> retry {f.name} next pass")
                break
            ctx.log(f"live {inst.name}/{f.name} {f.stat().st_size / 1e9:.2f} GB (top={top})")
            dst = dst_dir / f.name
            sha = copy_direct(ctx, f, dst)
            after = ctx.counters()
            assert_nfs_clean(before, after, f"live {f.name}")
            before = after
            tmp = f.with_name(f.name + ".nfslink")
            tmp.symlink_to(dst)
            os.replace(tmp, f)  # atomically swap in the symlink; the writer stopped touching this file long ago
            note(ctx.run_dir, f"{inst.name}/{f.name}", dst, sha)
            ctx.log(f"  OK sha256={sha}; local replaced by symlink")
            time.sleep(args.pause)
        if not args.loop:
            return
        time.sleep(args.loop)
        before = ctx.counters()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="local directory: lanes/ holds log / lock / pid, .offload_tmp/ holds packing scratch")
    ap.add_argument("--chunk-mib", type=int, default=16)
    ap.add_argument("--max-mbps", type=float, default=0, help="0 = no cap")
    ap.add_argument("--stall", type=float, default=300)
    ap.add_argument("--pause", type=float, default=5)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("--dst-dir", required=True); p.add_argument("--mib", type=int, default=1024)
    p = sub.add_parser("verify"); p.add_argument("--local", required=True); p.add_argument("--nfs", required=True)
    p.add_argument("--delete", action="store_true"); p.add_argument("--note-root")
    p = sub.add_parser("check-jsonl"); p.add_argument("--nfs", required=True)
    p = sub.add_parser("check-zst"); p.add_argument("--nfs", required=True); p.add_argument("--expect-files", type=int)
    p = sub.add_parser("ledger-live"); p.add_argument("--instance", required=True); p.add_argument("--dst-root", required=True)
    p.add_argument("--release", help="routing_release.py output; move only the shards it lists for this instance")
    p.add_argument("--loop", type=float, default=0, help="rescan every this many seconds; 0 = scan once and exit")
    for name in ("files", "dirs"):
        p = sub.add_parser(name)
        p.add_argument("--src-root", required=True); p.add_argument("--dst-root", required=True)
        p.add_argument("--delete", action="store_true"); p.add_argument("rel", nargs="+")
    args = ap.parse_args()
    if args.cmd == "verify" and args.delete and not args.note_root:
        ap.error("verify --delete needs --note-root")
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "lanes").is_dir():
        raise RuntimeError(f"{run_dir}/lanes does not exist")
    nfs_side = {"probe": "dst_dir", "verify": "nfs", "check-jsonl": "nfs", "check-zst": "nfs",
                "files": "dst_root", "dirs": "dst_root", "ledger-live": "dst_root"}[args.cmd]
    mount = nfs_mount_of(getattr(args, nfs_side))
    lock = open(run_dir / "lanes" / "nfs_move.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # fail immediately if another nfs_move is running
    (run_dir / "lanes" / "nfs_move.pid").write_text(f"{os.getpid()}\n")
    ctx = Ctx(run_dir, mount, args.chunk_mib, args.max_mbps, args.stall)
    ctx.log(f"start {args.cmd} mount={mount} chunk={args.chunk_mib}MiB max_mbps={args.max_mbps} stall={args.stall}s")
    {"probe": cmd_probe, "verify": cmd_verify, "check-jsonl": cmd_check_jsonl, "check-zst": cmd_check_zst,
     "files": cmd_files, "dirs": cmd_dirs, "ledger-live": cmd_ledger_live}[args.cmd](ctx, args)
    ctx.log(f"end {args.cmd}")


if __name__ == "__main__":
    main()
