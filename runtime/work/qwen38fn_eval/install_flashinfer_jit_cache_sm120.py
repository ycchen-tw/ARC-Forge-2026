"""Install the official FlashInfer 0.6.18 SM120 AOT modules as `flashinfer_jit_cache`.

This replaces install_sm120_aot.py for the sglang 0.5.20 line, and the change is
not just the version number.

install_sm120_aot.py drops one library into `flashinfer/data/aot/`, which is the
*fallback* branch of flashinfer's `_get_aot_dir()`. That branch resolves purely
by module name -- `FLASHINFER_AOT_DIR/<name>/<name>.so` -- with no version and no
source hash, and `JitSpec.try_load()` swallows a load failure into a warning and
silently JITs instead. So a library built against a different FlashInfer is
either used anyway or quietly recompiled; neither is visible from the outside.
That mattered here: 0.6.18's `gen_cutlass_fused_moe_module` adds
`-DCUTLASS_MIXED_GEMM_FP4_FP8_PREPROCESSED_SIGNS=1` and sixteen source files that
0.6.17 did not compile, so the 0.6.17 library is *not* a drop-in.

The `flashinfer_jit_cache` package is the other branch, and it is version
checked: `_get_aot_dir()` raises unless the package version starts with the
FlashInfer version. Installing the package is therefore the loud shape, and it
is what this does.

Only the SM120 modules are taken -- twelve of the wheel's 906, 84 MB of 1016 --
because every other module is JIT-built on this card today and installing the
package cannot make that worse: anything absent from the AOT directory falls
through to the same JIT it already used. Widening the set is a measurement, not
a guess: build the runtime, see what still compiles at startup, and add those.

    python3 install_flashinfer_jit_cache_sm120.py --runtime <package-root>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.6.18+cu130"
URL = ("https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/"
       "flashinfer_jit_cache-0.6.18%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl")
# Everything whose module name carries an SM120 target, plus the package itself.
# `120f` is the family-conditional variant of 12.0; flashinfer picks between them
# from the CUDA version it detects, so both have to be present or one of the two
# paths silently JITs.
WANTED = re.compile(r"/(cute_sm120_\w+|fp4_gemm_cutlass_sm120|fp4_quantization_120f?"
                    r"|fused_moe_120|gemm_sm120|mxfp8_gemm_cutlass_sm120"
                    r"|nvfp4_attention_sm120|sparse_mla_sm120)/")
PACKAGE_FILES = ("flashinfer_jit_cache/__init__.py", "flashinfer_jit_cache/_build_meta.py")


class RemoteZip(io.RawIOBase):
    """Read the wheel's members over HTTP ranges rather than fetching 1016 MB.

    Lifted in shape from probes/inspect_official_jit_wheel.py, which is how the
    0.6.17 library was taken. The size is asked of the server rather than pinned,
    because a pinned size that goes stale reads as a corrupt zip.
    """

    def __init__(self, url: str):
        self.url = url
        self.position = 0
        self.size = int(urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status != 206:
                raise RuntimeError(f"the server ignored the byte range: HTTP {response.status}")
            data = response.read()
        if len(data) != end - self.position:
            raise RuntimeError(f"short range read: {len(data)} of {end - self.position}")
        self.position = end
        return data


def recorded_hashes(archive: zipfile.ZipFile) -> dict[str, str]:
    """The wheel's own RECORD, so every extracted byte is checked against it."""
    name = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
    hashes = {}
    for row in csv.reader(io.StringIO(archive.read(name).decode())):
        if len(row) >= 2 and row[1].startswith("sha256="):
            hashes[row[0]] = row[1][len("sha256="):]
    if not hashes:
        raise RuntimeError(f"{name} carries no sha256 digests")
    return hashes


def urlsafe(digest: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def install(runtime: Path, url: str = URL, source: Path | None = None) -> dict:
    """Lay the package into the runtime's site-packages and report what landed."""
    site = runtime / "lib/python3.12/site-packages"
    if not site.is_dir():
        raise RuntimeError(f"no site-packages at {site}")
    handle = zipfile.ZipFile(str(source)) if source else zipfile.ZipFile(RemoteZip(url))
    with handle as archive:
        hashes = recorded_hashes(archive)
        # `<version>.data/purelib/` is where a wheel puts files destined for
        # site-packages; the prefix is stripped on install and has to be here too.
        prefix = f"flashinfer_jit_cache-{VERSION}.data/purelib/"
        members = [n for n in archive.namelist()
                   if n.startswith(prefix)
                   and (WANTED.search(n) or n[len(prefix):] in PACKAGE_FILES)]
        if not members:
            raise RuntimeError(f"no SM120 members under {prefix!r} in the wheel")
        for wanted in PACKAGE_FILES:
            if prefix + wanted not in members:
                raise RuntimeError(f"the wheel has no {wanted}; the package would not import")
        installed = {}
        for name in sorted(members):
            payload = archive.read(name)
            digest = hashlib.sha256(payload)
            expected = hashes.get(name)
            if expected is None:
                raise RuntimeError(f"{name} is not in the wheel's RECORD")
            if urlsafe(digest.digest()) != expected:
                raise RuntimeError(f"{name} does not match the wheel's RECORD")
            target = site / name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            installed[name[len(prefix):]] = {"bytes": len(payload),
                                             "sha256": digest.hexdigest()}
    # The fallback directory must stay empty. If both exist the package wins and
    # anything left in `flashinfer/data/aot` is dead weight that reads as live.
    stale = site / "flashinfer/data/aot"
    if stale.exists() and any(stale.iterdir()):
        raise RuntimeError(f"{stale} still holds modules; with the package installed "
                           "flashinfer never reads them, so they would be a lie")
    result = {"source_url": url if source is None else str(source),
              "package_version": VERSION, "flashinfer_version": "0.6.18",
              "modules": sorted(n for n in installed if n.endswith(".so")),
              "files": installed,
              "bytes": sum(f["bytes"] for f in installed.values())}
    (runtime / "flashinfer-jit-cache-sm120.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", type=Path, required=True,
                        help="the package root holding lib/python3.12/site-packages")
    parser.add_argument("--wheel", type=Path, default=None,
                        help="a local copy of the wheel; omitted, it is read over HTTP ranges")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args(argv)
    result = install(args.runtime, args.url, args.wheel)
    print(json.dumps({"modules": result["modules"], "bytes": result["bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
