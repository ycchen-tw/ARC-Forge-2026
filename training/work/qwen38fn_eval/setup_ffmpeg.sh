#!/bin/bash
# Install FFmpeg shared libraries for sglang's video input.
#
#   bash work/qwen38fn_eval/setup_ffmpeg.sh
#
# ## Why this is needed
#
# sglang decodes video through `srt/utils/video_decoder.py`: torchcodec first, decord as fallback.
# torchcodec 0.15.0 in `venv-0520` is **installed but fails to load** -- it ships its own
# `libtorchcodec_core{4..8}.so`, which link against the system `libav*`, and this machine has no FFmpeg:
#
#     RuntimeError: Could not load libtorchcodec. ... FFmpeg is not properly installed
#
# decord is not installed either. So until this is in place, **sglang cannot decode any video it receives**;
# the symptom is a request blowing up only once it reaches the decode stage, and the error does not say
# "FFmpeg missing".
#
# ## Why pin n8.1 and not latest
#
# torchcodec 0.15 looks for `libavutil.so.56`-`.60` (= FFmpeg 4-8). BtbN's
# **master build is `.so.61` (FFmpeg 9 dev), incompatible** -- all five versions fail to load in turn,
# and the error only says "version not found", never that it is too new. n8.1 provides
# `.so.60`, matching torchcodec's "FFmpeg version 8" branch.
#
# No sudo needed: unpack and use, relying only on LD_LIBRARY_PATH; touches neither the venv nor system paths.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env/env.sh

DEST=${DEST:-$HOME/work/dsv4/ffmpeg-n8.1}
VENV=${VENV:-$HOME/work/dsv4/venv-0520}
URL=${URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz}

say() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
die() { echo -e "\033[1;31m✗ $*\033[0m" >&2; exit 1; }

if [ ! -f "$DEST/lib/libavutil.so.60" ]; then
    say "downloading FFmpeg n8.1 → $DEST"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "$tmp/ffmpeg.tar.xz" "$URL" || die "download failed: $URL"
    tar -xJf "$tmp/ffmpeg.tar.xz" -C "$tmp"
    src=$(find "$tmp" -maxdepth 1 -type d -name 'ffmpeg-*' | head -1)
    [ -n "$src" ] || die "no ffmpeg-* directory after unpacking the tarball"
    mkdir -p "$(dirname "$DEST")"
    rm -rf "$DEST"
    mv "$src" "$DEST"
else
    say "$DEST already present"
fi

[ -f "$DEST/lib/libavutil.so.60" ] || die "$DEST/lib has no libavutil.so.60; wrong version?"

say "verifying that torchcodec loads"
[ -x "$VENV/bin/python" ] || die "no $VENV; run build_sglang_0520.sh first"
LD_LIBRARY_PATH="$DEST/lib:${LD_LIBRARY_PATH:-}" "$VENV/bin/python" - <<'PYCHK' || die "torchcodec still fails to load"
import sys
try:
    from torchcodec.decoders import VideoDecoder  # noqa: F401
except Exception as exc:
    print(f"✗ torchcodec failed to load: {type(exc).__name__}: {str(exc)[:300]}")
    sys.exit(1)
import torchcodec
print(f"· torchcodec {torchcodec.__version__} OK")
PYCHK

say "OK → $DEST"
echo "serve_qwen38fn_0520.sh / serve_qwen38fn_nvfp4_0520.sh pick it up automatically; no need to set LD_LIBRARY_PATH by hand."
