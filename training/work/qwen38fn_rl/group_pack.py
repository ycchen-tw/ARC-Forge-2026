"""Pack trained groups' episodes/ (arc3, run_arc3_level.py) or tasks/ (arc2, run_arc2.py: the workspaces with the codex
homes) and harness/ into one verified .tar.zst per group under --archive-dir (local only).

ARC3 round 2 on node2 (/data ~60 GiB free): a finished group still holds ~20-30 MB (run_arc3_level already drops each
episode's record/steps and recordings at its end, so the seed copy only costs disk while the episode runs), ~17 groups a
step; packed they are ~5x smaller and off /data. Once a group appears in a
batches.jsonl row (its step is done) or is recorded abandoned, nothing in the loop reads its episodes again (prepare reads
only pool groups; resume needs the group directory, routing_release needs tito_manifest.json -- both stay). Each pack:
tar | zstd --long to a temp file, tar --compare against the directory (contents, sizes, modes, mtimes; unix sockets are not
archived), rename, write <group>/ARCHIVED.txt, then delete the two directories. One group at a time, one stream. Restore:
    zstd -dc --long=27 <archive-dir>/<group>.tar.zst | tar -C <group> -xf -
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

#: a group holds one of the episode directories (arc3 episodes/, arc2 tasks/) and harness/
EPISODE_DIRS = ("episodes", "tasks")


def members_of(gdir):
    """The directories to pack, in the order they go into the tar: the group's episode directory and harness/."""
    heads = [m for m in EPISODE_DIRS if (gdir / m).is_dir()]
    if len(heads) != 1:
        raise RuntimeError(f"{gdir}: expected exactly one of {EPISODE_DIRS}, found {heads}")
    return (heads[0], "harness")


def du(path):
    return sum(p.lstat().st_blocks * 512 for p in path.rglob("*") if not p.is_symlink()) + path.lstat().st_blocks * 512


def rows(path):
    """Complete rows only: the writer's final row may still be partial."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]


def finished(root):
    """Groups whose episodes the loop never reads again: consumed by a recorded batch, or abandoned."""
    gids = [g["gid"] for row in rows(root / "batches.jsonl") for g in row["groups"]]
    gids += [r["gid"] for r in rows(root / "groups.jsonl") if r["event"] == "abandoned"]
    return list(dict.fromkeys(gids))


def verify(archive, gdir):
    with subprocess.Popen(["zstd", "-dcq", "--long=27", str(archive)], stdout=subprocess.PIPE) as z:
        cmp = subprocess.run(["tar", "--compare", "-C", str(gdir), "-f", "-"], stdin=z.stdout, capture_output=True, text=True)
        z.stdout.close()
    if z.returncode != 0 or cmp.returncode != 0:
        raise RuntimeError(f"{archive}: does not match {gdir} (zstd rc {z.returncode}, tar --compare rc {cmp.returncode}): "
                           f"{cmp.stdout[-2000:]}{cmp.stderr[-2000:]}")


def pack(gdir, archive_dir):
    archive = archive_dir / f"{gdir.name}.tar.zst"
    tmp = archive.with_name(archive.name + ".tmp")
    members = [m for m in (*EPISODE_DIRS, "harness") if (gdir / m).is_dir()]
    if archive.exists():
        if members:                          # killed between rename and delete: check again, then finish
            verify(archive, gdir)
            note(gdir, archive)
            for m in members:
                shutil.rmtree(gdir / m)
        return None
    want = members_of(gdir)
    if members != list(want):
        raise RuntimeError(f"{gdir}: expected {want} before packing, found {members} (and no {archive})")
    tmp.unlink(missing_ok=True)
    t0, before = time.time(), sum(du(gdir / m) for m in want)
    with tmp.open("wb") as out:
        tar = subprocess.Popen(["tar", "--create", "--warning=no-file-ignored", "-C", str(gdir), "-f", "-", *want],
                               stdout=subprocess.PIPE)
        z = subprocess.run(["zstd", "-q", "-T4", "-3", "--long=27", "-c"], stdin=tar.stdout, stdout=out)
        tar.stdout.close()
        if tar.wait() != 0 or z.returncode != 0:
            raise RuntimeError(f"{gdir}: tar rc {tar.returncode}, zstd rc {z.returncode}")
        out.flush(); os.fsync(out.fileno())
    verify(tmp, gdir)
    os.replace(tmp, archive)
    note(gdir, archive)
    for m in want:
        shutil.rmtree(gdir / m)
    return {"group": gdir.name, "bytes_before": before, "bytes_after": archive.stat().st_size,
            "seconds": round(time.time() - t0, 1), "utc": time.strftime("%FT%TZ", time.gmtime())}


def note(gdir, archive):
    (gdir / "ARCHIVED.txt").write_text(f"the episode directory and harness/ are in {archive}\n"
                                       f"restore: zstd -dc --long=27 {archive} | tar -C {gdir} -xf -\n")


def once(root, archive_dir):
    log = root / "group_pack.jsonl"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for gid in finished(root):
        gdir = root / "groups" / f"g{gid:06d}"
        if not gdir.is_dir():
            raise RuntimeError(f"g{gid:06d} is consumed or abandoned but {gdir} is not a directory")
        row = pack(gdir, archive_dir)
        if row is not None:
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0]); ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--archive-dir", type=Path, required=True, help="where <group>.tar.zst go (e.g. the run's tmpfs)")
    ap.add_argument("--once", action="store_true"); ap.add_argument("--poll-seconds", type=float, default=60)
    args = ap.parse_args()
    with (args.run_root / "group_pack.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            once(args.run_root, args.archive_dir)
            if args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
