"""Smoke client for a QSA HiSparse server: wait for /health, run a few /generate calls one by one.

Usage: smoke_client.py <port> <server pid> <out.json> [prompt tokens ...]   (default 200 9000 33000)
Prompts are synthetic text of roughly the requested length; greedy, 64 new tokens.
Exit 1 on any HTTP error or a non-length finish.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma "
         "tau upsilon phi chi psi omega").split()


def call(port: int, path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode()
    return json.loads(text) if text else {}


def alive(pid: int) -> bool:
    """False once the process is gone or a zombie (its parent may not have reaped it yet)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_health(port: int, server_pid: int, limit_s: float = 2400) -> float:
    t0 = time.time()
    while True:
        try:
            call(port, "/health", timeout=10)
            return time.time() - t0
        except OSError:
            if not alive(server_pid):
                raise RuntimeError(f"server pid {server_pid} exited before /health answered")
            if time.time() - t0 > limit_s:
                raise
            time.sleep(5)


def prompt(tokens: int, seed: int) -> str:
    # ~1 token per word for these Greek letter names plus separators
    return " ".join(f"{WORDS[(seed + i * 7) % len(WORDS)]}{i % 97}" for i in range(tokens // 2))


def main() -> int:
    port, server_pid, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    lengths = [int(x) for x in sys.argv[4:]] or [200, 9000, 33000]
    results = {"health_wait_s": round(wait_health(port, server_pid), 1), "runs": []}
    ok = True
    for i, n in enumerate(lengths):
        t0 = time.time()
        r = call(port, "/generate", {"text": prompt(n, i),
                                      "sampling_params": {"max_new_tokens": 64, "temperature": 0}})
        meta = r["meta_info"]
        finish = meta["finish_reason"]["type"]
        ok &= finish == "length" and meta["completion_tokens"] == 64
        results["runs"].append({"target_tokens": n, "prompt_tokens": meta["prompt_tokens"],
                                "completion_tokens": meta["completion_tokens"], "finish": finish,
                                "latency_s": round(time.time() - t0, 2), "text_head": r["text"][:160]})
        print(json.dumps(results["runs"][-1]), flush=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
