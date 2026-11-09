"""Carry a TCP service across the jail's network namespace.

The jail uses ``--unshare-net``, so inside it even the host's loopback is unreachable. The approach:
on the host side the TCP service is connected to a unix socket, that socket is bind-mounted into
the jail, and on the jail side it is connected back to ``127.0.0.1:<port>``. The agent therefore
always sees ``127.0.0.1:<env_port>`` -- **the address is identical in both modes**; this is a
design invariant: in competition what switches is the bridge's upstream, not the URL the agent sees.

stdlib only, because the jail side runs in the agent runtime (no aiohttp).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except socket.timeout:
        # Should not happen: serve() clears connect's timeout from the socket.
        # Reaching here means someone left a timeout on the socket again -- that defect silently
        # cuts the connection in the middle of a long generation (queueing + prefill TTFT alone can
        # take several minutes), and the except OSError below would swallow it (socket.timeout is
        # an OSError). So this must be loud, not silent.
        print(f"[bridge] BUG: socket timeout still set on {src.getsockname()!r}, "
              f"connection cut; serve() should already have done settimeout(None)",
              file=sys.stderr, flush=True)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


def _splice(a: socket.socket, b: socket.socket) -> None:
    threading.Thread(target=_pump, args=(a, b), daemon=True).start()
    threading.Thread(target=_pump, args=(b, a), daemon=True).start()


def serve(listen: socket.socket, connect) -> None:
    listen.listen(128)
    while True:
        client, _ = listen.accept()
        try:
            upstream = connect()
        except OSError as error:
            # An unreachable upstream is a real breakage, not the normal case. This used to close
            # quietly, and inside the jail all you saw was "connection refused" without knowing who refused.
            print(f"[bridge] upstream unreachable: {error}", file=sys.stderr, flush=True)
            client.close()
            continue
        # connect()'s timeout should only govern "can we connect". If left on the socket, every
        # later recv gets the same timeout -- the first recv waits for TTFT (queueing + prefill),
        # which long prompts easily exceed, so the whole connection is cut mid-generation (or before
        # it even starts) and both ends only see the other side close.
        for sock in (client, upstream):
            sock.settimeout(None)
        _splice(client, upstream)


@contextmanager
def _unix_address(path: Path):
    """Linux sun_path is only 108 bytes; a dirfd keeps the original location while shortening the bind/connect string.

    The jail mounts the same long socket directory too, so both ends need this. The fd stays open
    only for the syscall; cwd is not changed and no alias is created; the socket inode stays in the
    original run directory.
    """
    if len(os.fsencode(path)) < 108:
        yield str(path)
        return
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield f"/proc/self/fd/{directory}/{path.name}"
    finally:
        os.close(directory)


def listen_unix(path: Path) -> socket.socket:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with _unix_address(path) as address:
            sock.bind(address)
        os.chmod(path, 0o600)
    except BaseException:
        sock.close()
        raise
    return sock


def connect_unix(path: Path) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with _unix_address(path) as address:
            sock.connect(address)
    except BaseException:
        sock.close()
        raise
    return sock


def listen_tcp(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    return sock


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--listen-unix", type=Path)
    p.add_argument("--listen-tcp")
    p.add_argument("--connect-unix", type=Path)
    p.add_argument("--connect-tcp")
    a = p.parse_args(argv)

    if a.listen_unix:
        listener = listen_unix(a.listen_unix)
    else:
        host, _, port = a.listen_tcp.rpartition(":")
        listener = listen_tcp(host or "127.0.0.1", int(port))

    if a.connect_unix:
        def connect():
            return connect_unix(a.connect_unix)
    else:
        host, _, port = a.connect_tcp.rpartition(":")

        def connect():
            return socket.create_connection((host or "127.0.0.1", int(port)), 30)

    serve(listener, connect)


if __name__ == "__main__":
    main()
