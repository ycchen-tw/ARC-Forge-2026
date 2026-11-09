"""v16 serving gates against an already started L4x4 server (standard library only).

  gpu-free      per-card free device memory (nvidia-smi), fail below --min-free-gib
  vision-peak   concurrent chat requests that each carry many ARC-board PNGs, like codex's first message
                (every grid as an image, >=16 px per cell, padded to a multiple of 32: a 30x30 board is 480x480)
  long-prefill  one long random-token prefill per DP, so the first long prefill (and any kernel that tunes on it)
                happens here and not in the middle of the harness
  log-summary   DFlash accept/cycle, HiSparse and retract counters of a server.log (see ServerLogSummary)

Each subcommand writes <out>/result.json with status "passed" or raises (exit != 0).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import random
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

GIB = 1 << 30
#: ARC's ten colours (the palette the arc2 harness renders with is close enough for a memory gate)
PALETTE = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
           (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37)]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def gpu_free() -> list[dict]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        index, free, used, total = (int(x.strip()) for x in line.split(","))
        rows.append({"gpu": index, "free_mib": free, "used_mib": used, "total_mib": total,
                     "free_gib": round(free / 1024, 3)})
    if not rows:
        raise RuntimeError("nvidia-smi listed no GPUs")
    return rows


def require_free(rows: list[dict], minimum_gib: float, where: str) -> None:
    low = [row for row in rows if row["free_mib"] / 1024 < minimum_gib]
    if low:
        raise RuntimeError(f"{where}: GPU free memory below {minimum_gib} GiB: "
                           + ", ".join(f"GPU{r['gpu']} {r['free_gib']} GiB" for r in low))


def png(width: int, height: int, pixels: bytes) -> bytes:
    """RGB8 PNG, no filtering."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def board_png(rng: random.Random, cells: int, cell_px: int) -> bytes:
    side = cells * cell_px
    padded = -(-side // 32) * 32
    grid = [[rng.randrange(10) for _ in range(cells)] for _ in range(cells)]
    pixels = bytearray(padded * padded * 3)
    for y in range(side):
        row = grid[y // cell_px]
        for x in range(side):
            r, g, b = PALETTE[row[x // cell_px]]
            offset = (y * padded + x) * 3
            pixels[offset:offset + 3] = bytes((r, g, b))
    return png(padded, padded, bytes(pixels))


def post(url: str, body: dict, timeout: float) -> tuple[int, dict | str, float]:
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    elapsed = time.monotonic() - started
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, raw.decode(errors="replace")[:2000], elapsed


def vision_peak(args) -> dict:
    rng = random.Random(20261004)
    images = [board_png(rng, args.cells, args.cell_px) for _ in range(args.images)]
    content = [{"type": "text", "text": f"These are {args.images} ARC boards. Reply with the word OK."}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}
                for data in images]
    body = {"model": args.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": args.max_tokens, "temperature": 1.0, "top_p": 0.95}
    before = gpu_free()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(post, args.base_url.rstrip("/") + "/v1/chat/completions", body, args.timeout)
                   for _ in range(args.concurrency)]
        replies = [future.result() for future in futures]
    after = gpu_free()
    rows = []
    for status, reply, elapsed in replies:
        if status != 200 or not isinstance(reply, dict):
            raise RuntimeError(f"vision-peak request failed: HTTP {status}: {str(reply)[:600]}")
        prompt_tokens = reply["usage"]["prompt_tokens"]
        rows.append({"elapsed_s": round(elapsed, 2), "prompt_tokens": prompt_tokens,
                     "completion_tokens": reply["usage"]["completion_tokens"],
                     "finish_reason": reply["choices"][0]["finish_reason"]})
        # each padded board is (side/32)^2 merged patches; anything far below means the images were not encoded
        minimum = args.images * (-(-args.cells * args.cell_px // 32)) ** 2
        if prompt_tokens < minimum:
            raise RuntimeError(f"vision-peak prompt has {prompt_tokens} tokens, expected >= {minimum} image tokens")
    require_free(after, args.min_free_gib, "after vision-peak")
    return {"status": "passed", "images_per_request": args.images, "board": f"{args.cells}x{args.cells}",
            "cell_px": args.cell_px, "png_bytes": [len(x) for x in images[:3]], "concurrency": args.concurrency,
            "requests": rows, "gpu_free_before": before, "gpu_free_after": after}


def long_prefill(args) -> dict:
    rng = random.Random(7300)
    jobs = []
    for length in args.lengths:
        for _ in range(args.per_length):
            ids = [rng.randrange(1000, 100000) for _ in range(length)]
            jobs.append((length, {"input_ids": ids, "sampling_params": {"max_new_tokens": args.max_tokens,
                                                                         "temperature": 1.0, "top_p": 0.95}}))
    before = gpu_free()
    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        futures = [(length, pool.submit(post, args.base_url.rstrip("/") + "/generate", body, args.timeout))
                   for length, body in jobs]
        replies = [(length, future.result()) for length, future in futures]
    after = gpu_free()
    rows = []
    for length, (status, reply, elapsed) in replies:
        if status != 200 or not isinstance(reply, dict):
            raise RuntimeError(f"long-prefill {length} failed: HTTP {status}: {str(reply)[:600]}")
        meta = reply["meta_info"]
        if meta["prompt_tokens"] != length:
            raise RuntimeError(f"long-prefill sent {length} tokens, server counted {meta['prompt_tokens']}")
        rows.append({"input_tokens": length, "elapsed_s": round(elapsed, 2),
                     "completion_tokens": meta["completion_tokens"],
                     **{k: meta[k] for k in ("spec_verify_ct", "spec_accept_length") if k in meta}})
    require_free(after, args.min_free_gib, "after long-prefill")
    return {"status": "passed", "requests": rows, "gpu_free_before": before, "gpu_free_after": after}


class ServerLogSummary:
    """Incremental counters over an SGLang server.log (reads only the bytes appended since the last update)."""

    # "DP<d> TP<t> EP<e>]" with attention DP; pure TP4 (attn_dp 1) logs "TP<t> EP<e>]", counted as DP 0
    RANK = r"(?:\bDP(\d) )?TP\d+(?: EP\d+)?\]"
    DECODE = re.compile(RANK + r" Decode batch, #running-req: (\d+).*?accept len: ([0-9.]+).*?"
                        r"gen throughput \(token/s\): ([0-9.]+)")
    TIMING = re.compile(RANK + r" DFlash timing over (\d+) cycles: draft ([0-9.]+) ms mean, "
                        r"cycle ([0-9.]+) ms median.*batch ([0-9.]+) mean")
    CYCLE = re.compile(RANK + r" DFlash cycle (\d+): batch (\d+), live draft states (\d+) of (\d+) slots "
                       r"\(evicted (\d+) so far\)")
    RETRACT = re.compile(r"KV cache pool is full\. Retract requests\..*#retracted_reqs: (\d+)")
    IDLE = re.compile(RANK + r" QSA HiSparse session cache idle: (.*)$")

    def __init__(self, path: Path):
        self.path, self.offset, self.partial = Path(path), 0, b""
        self.decode = {}       # dp -> [lines, sum accept, sum running, last line]
        self.timing = {}       # dp -> last timing dict
        self.cycle = {}        # dp -> last cycle dict
        self.retract_events = self.retracted_requests = 0
        self.idle = {}         # dp -> last idle line tail
        self.errors = {"Traceback": 0, "out of memory": 0, "hostcomm: timeout": 0}

    def update(self) -> dict:
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
        self.offset += len(data)
        buffered = self.partial + data
        if b"\n" not in buffered:
            self.partial = buffered
            return self.snapshot()
        complete, self.partial = buffered.rsplit(b"\n", 1)
        for raw in complete.split(b"\n"):
            self._line(raw.decode(errors="replace"))
        return self.snapshot()

    def _line(self, line: str) -> None:
        for key in self.errors:
            if key in line:
                self.errors[key] += 1
        if "Decode batch" in line:
            m = self.DECODE.search(line)
            if m:
                row = self.decode.setdefault(m.group(1) or "0", [0, 0.0, 0, ""])
                row[0] += 1; row[1] += float(m.group(3)); row[2] += int(m.group(2)); row[3] = line[-260:]
            return
        m = self.TIMING.search(line)
        if m:
            self.timing[m.group(1) or "0"] = {"cycles": int(m.group(2)), "draft_ms": float(m.group(3)),
                                       "cycle_ms": float(m.group(4)), "batch": float(m.group(5))}
            return
        m = self.CYCLE.search(line)
        if m:
            self.cycle[m.group(1) or "0"] = {"cycle": int(m.group(2)), "batch": int(m.group(3)), "live_states": int(m.group(4)),
                                      "slots": int(m.group(5)), "evicted": int(m.group(6))}
            return
        m = self.RETRACT.search(line)
        if m:
            self.retract_events += 1
            self.retracted_requests += int(m.group(1))
            return
        m = self.IDLE.search(line)
        if m:
            self.idle[m.group(1) or "0"] = m.group(2)[-400:]

    def snapshot(self) -> dict:
        return {"decode": {dp: {"lines": n, "mean_accept_len": round(s / n, 3) if n else None,
                                "mean_running": round(r / n, 2) if n else None, "last": last}
                           for dp, (n, s, r, last) in sorted(self.decode.items())},
                "dflash_timing": self.timing, "dflash_cycle": self.cycle,
                "retract_events": self.retract_events, "retracted_requests": self.retracted_requests,
                "hisparse_idle": self.idle, "error_lines": dict(self.errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("gpu-free", "vision-peak", "long-prefill", "log-summary"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--min-free-gib", type=float, default=0.25)
        if name in ("vision-peak", "long-prefill"):
            p.add_argument("--base-url", required=True)
            p.add_argument("--timeout", type=float, default=900)
            p.add_argument("--max-tokens", type=int, default=16)
        if name == "vision-peak":
            p.add_argument("--model", required=True)
            p.add_argument("--images", type=int, default=20)
            p.add_argument("--cells", type=int, default=30)
            p.add_argument("--cell-px", type=int, default=16)
            p.add_argument("--concurrency", type=int, default=4)
        if name == "long-prefill":
            p.add_argument("--lengths", type=int, nargs="+", default=[8192, 65536])
            p.add_argument("--per-length", type=int, default=2, help="one per DP with the default round robin")
        if name == "log-summary":
            p.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gpu-free":
        rows = gpu_free()
        result = {"status": "pending", "gpu_free": rows}
        write_json(args.out / "result.json", result)
        require_free(rows, args.min_free_gib, "gpu-free")
        result["status"] = "passed"
    elif args.command == "vision-peak":
        result = vision_peak(args)
    elif args.command == "long-prefill":
        result = long_prefill(args)
    else:
        result = {"status": "passed", **ServerLogSummary(args.server_log).update()}
    write_json(args.out / "result.json", result)
    print(json.dumps(result)[:4000], flush=True)


if __name__ == "__main__":
    main()
