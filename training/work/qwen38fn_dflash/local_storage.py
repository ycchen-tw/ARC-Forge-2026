"""Fail before touching a network mount, including through a symlink.

Read mountinfo and walk only local directory entries. A timeout around an NFS
syscall cannot guarantee SIGKILL works once a task is in uninterruptible I/O.
"""

import os
import re
import stat
from collections import deque
from pathlib import Path

LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "ramfs", "devtmpfs", "overlay", "zfs"}


def mounts():
    def unescape(s):
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
    result = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        result.append((Path(unescape(left.split()[4])), right.split()[0]))
    return sorted(result, key=lambda x: len(x[0].parts), reverse=True)


def require_local(path, table=None):
    table = mounts() if table is None else table
    prefixes = [(str(mount).rstrip("/") + "/", str(mount), kind) for mount, kind in table]
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    todo, current, links = deque(path.parts[1:]), Path("/"), 0
    while todo:
        part = todo.popleft()
        if part == "..":
            current = current.parent
            continue
        current /= part
        name = str(current)
        filesystem = next(kind for prefix, mount, kind in prefixes if name == mount or name.startswith(prefix))
        if filesystem not in LOCAL_FILESYSTEMS:
            raise ValueError(f"refusing {filesystem} path before filesystem access: {current}")
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            links += 1
            if links > 40:
                raise ValueError(f"too many symlinks: {path}")
            target = Path(os.readlink(current))
            if target.is_absolute():
                current = Path("/")
                parts = target.parts[1:]
            else:
                current = current.parent
                parts = target.parts
            todo.extendleft(reversed(parts))
        elif not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError(f"not a regular file or directory: {current}")
    return current


def local_tree(root):
    """Check every entry before descent; never follow a symlink into NFS."""
    table = mounts()
    prefixes = [(str(mount).rstrip("/") + "/", str(mount), kind) for mount, kind in table]
    root = require_local(root, table)
    seen, pending = set(), [root]
    while pending:
        directory = require_local(pending.pop(), table)
        if directory in seen:
            continue
        seen.add(directory)
        for child in directory.iterdir():
            name = str(child)
            filesystem = next(kind for prefix, mount, kind in prefixes if name == mount or name.startswith(prefix))
            if filesystem not in LOCAL_FILESYSTEMS:
                raise ValueError(f"refusing {filesystem} path before filesystem access: {child}")
            info = child.lstat()
            path = require_local(child, table) if stat.S_ISLNK(info.st_mode) else child
            if stat.S_ISDIR(info.st_mode) or (stat.S_ISLNK(info.st_mode) and path.is_dir()):
                pending.append(path)
    return root
