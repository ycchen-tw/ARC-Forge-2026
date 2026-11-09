"""Lossless streaming compression for large FP32 master/Adam checkpoints."""

import io
import subprocess
from pathlib import Path

import torch


def save_compressed(payload, path):
    path = Path(path)
    with path.open("xb") as output:
        process = subprocess.Popen(
            ["zstd", "-1", "-T4", "-q", "-c"], stdin=subprocess.PIPE, stdout=output, stderr=subprocess.PIPE
        )
        try:
            torch.save(payload, process.stdin)
            process.stdin.close()
            error = process.stderr.read().decode()
            if process.wait() != 0:
                raise RuntimeError(f"checkpoint compression failed: {error}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    return path.stat().st_size


def load_compressed(path):
    # A seekable buffer is required by torch's zip reader. Host RAM is checked
    # by the run preflight; no extra uncompressed disk copy is materialized.
    process = subprocess.Popen(["zstd", "-d", "-q", "-c", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        data = process.stdout.read()
        error = process.stderr.read().decode()
        if process.wait() != 0:
            raise RuntimeError(f"checkpoint decompression/checksum failed: {error}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
