#!/bin/bash
# The last stage of fence_arc2.sh: runs as this task's uid inside a user+net namespace it creates itself
# (unshare --user --map-root-user --net; uid 0 inside, still this task's uid outside, file permissions go by the
# outside uid). This namespace has only loopback: bridge.py connects the model port from the host's unix socket
# back to 127.0.0.1:<port>, so the URL codex sees is character-for-character the same as without the fence.
#
#   netns_inner.sh <driver argv...>
# Environment: ARC2_PY ARC2_BRIDGE ARC2_MODEL_PORT ARC2_MODEL_SOCK ARC2_NET_PROBE (host:port, must be unreachable)
set -euo pipefail
for v in ARC2_PY ARC2_BRIDGE ARC2_MODEL_PORT ARC2_MODEL_SOCK ARC2_NET_PROBE; do
    [ -n "${!v:-}" ] || { echo "netns_inner.sh needs $v" >&2; exit 64; }
done
/usr/sbin/ip link set lo up
"$ARC2_PY" -I "$ARC2_BRIDGE" --listen-tcp "127.0.0.1:$ARC2_MODEL_PORT" --connect-unix "$ARC2_MODEL_SOCK" \
    2>> "$HOME/.bridge.log" &
# "Something is listening" does not mean usable: the test is /health reaching sglang through the bridge; the
# internet must be unreachable.
"$ARC2_PY" -I - <<'PY' || exit $?
import os, socket, sys, time, urllib.request
port = os.environ["ARC2_MODEL_PORT"]
deadline = time.monotonic() + 20
last = None
while True:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            if r.status == 200:
                break
    except OSError as error:
        last = error
    if time.monotonic() > deadline:
        print(f"ARC2: model unreachable through the bridge: {last!r}", file=sys.stderr)
        sys.exit(71)
    time.sleep(0.2)
host, p = os.environ["ARC2_NET_PROBE"].rsplit(":", 1)
try:
    socket.create_connection((host, int(p)), timeout=5).close()
except OSError:
    sys.exit(0)
print(f"ARC2: {host}:{p} is reachable inside the namespace", file=sys.stderr)
sys.exit(70)
PY
echo "ARC2_NETNS lo up, bridge 127.0.0.1:$ARC2_MODEL_PORT, $ARC2_NET_PROBE unreachable" >&2
exec "$@"
