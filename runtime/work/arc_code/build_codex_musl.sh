#!/bin/bash
# Builds the codex used by arc-code: tag rust-v0.154.0 + patches/codex-0.154.0/, cross-compiled into a
# **musl static binary**.
#
#   bash work/arc_code/build_codex_musl.sh
#   OUT=/somewhere/codex bash work/arc_code/build_codex_musl.sh
#
# ## Why musl and not the local glibc
#
# This machine has glibc 2.39; the Kaggle competition image has **2.35** (docs/research/kaggle_env_specs.md).
# Built here for the gnu target, the binary dies on Kaggle with `GLIBC_2.38 not found` and never starts —
# `work/arc_agi3/arc_code_qwen/probes/build_kaggle_assets.py` already hit the same problem for
# bubblewrap. The official release is a musl static binary; doing the same means both sides run **the
# bit-identical binary**, so cross-machine comparisons don't gain a confounder.
#
# ## Not a single sudo needed
#
# The Linux job of the official release workflow installs only binutils / pkg-config / libcap-dev,
# and libcap-dev is for `--bin bwrap` — that binary cannot create namespaces on Kaggle
# (memory: kaggle-arc3-sandbox-capabilities); we don't build it, so that whole step goes away.
# rust comes through rustup (~/.rustup); zig and the musl toolchain are unpack-and-run tarballs.
#
# ## Disk
#
# source 105 MB + toolchain about 400 MB + target **about 8 GB**. When `/data` is tight, point
# CARGO_TARGET_DIR elsewhere; tmpfs works but is gone after a reboot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT=$PWD
source env/env.sh

ROOT=${ROOT:-$HOME/work/codex-0.154.0}
SRC=${SRC:-$ROOT/src}
TOOLS=${TOOLS:-$ROOT/tools}
TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/target}
OUT=${OUT:-$ROOT/codex}
PATCH_DIR=$REPO_ROOT/patches/codex-0.154.0
TARGET=x86_64-unknown-linux-musl
ZIG_VER=${ZIG_VER:-0.14.0}
# Unrelated to torchcodec/ffmpeg; the version here only affects how zig cc behaves as the C compiler.

TAG=$(sed -n 's/^TAG=//p' "$PATCH_DIR/BASE")
REF=$(sed -n 's/^REF=//p' "$PATCH_DIR/BASE")
URL=$(sed -n 's/^REPO=//p' "$PATCH_DIR/BASE")
[ -n "$TAG" ] && [ -n "$REF" ] && [ -n "$URL" ] || { echo "✗ $PATCH_DIR/BASE is missing fields" >&2; exit 1; }

say() { echo -e "\n\033[1;36m[$(date +%H:%M:%S)] $*\033[0m"; }
die() { echo -e "\033[1;31m✗ $*\033[0m" >&2; exit 1; }

# ── rust ────────────────────────────────────────────────────────────────────
say "rust toolchain"
export PATH="$HOME/.cargo/bin:$PATH"
command -v rustup >/dev/null || die "no rustup. Install it first: curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
# The version is read from codex's own rust-toolchain.toml, not copied here a second time.
RUST_VER=""
[ -f "$SRC/codex-rs/rust-toolchain.toml" ] &&
    RUST_VER=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$SRC/codex-rs/rust-toolchain.toml")
RUST_VER=${RUST_VER:-1.95.0}
rustup toolchain install "$RUST_VER" --profile minimal >/dev/null 2>&1 || true
rustup component add --toolchain "$RUST_VER" clippy rustfmt rust-src >/dev/null 2>&1 || true
rustup target add --toolchain "$RUST_VER" "$TARGET" >/dev/null 2>&1 || true

# ── zig (C/C++ compiler for the native deps) ─────────────────────────────────
say "zig $ZIG_VER"
mkdir -p "$TOOLS"
ZIG_DIR=$TOOLS/zig-linux-x86_64-$ZIG_VER
if [ ! -x "$ZIG_DIR/zig" ]; then
    curl -fsSL -o "$TOOLS/zig.tar.xz" \
        "https://ziglang.org/download/$ZIG_VER/zig-linux-x86_64-$ZIG_VER.tar.xz"
    tar -xJf "$TOOLS/zig.tar.xz" -C "$TOOLS"
fi
"$ZIG_DIR/zig" version >/dev/null || die "zig unpacked but does not run"

# ── musl toolchain (the real linker) ─────────────────────────────────────────
# The official script's comment is explicit: the linker must be the real musl gcc; **zig must not inject its own CRT**.
say "musl cross toolchain"
MUSL_DIR=$TOOLS/x86_64-linux-musl-cross
if [ ! -x "$MUSL_DIR/bin/x86_64-linux-musl-gcc" ]; then
    curl -fsSL -o "$TOOLS/musl-cross.tgz" https://musl.cc/x86_64-linux-musl-cross.tgz
    tar -xzf "$TOOLS/musl-cross.tgz" -C "$TOOLS"
fi
MUSL_GCC=$MUSL_DIR/bin/x86_64-linux-musl-gcc
printf '#include <stdio.h>\nint main(){return 0;}\n' > "$TOOLS/_probe.c"
"$MUSL_GCC" -static -o "$TOOLS/_probe" "$TOOLS/_probe.c" || die "musl gcc cannot build a static binary"
rm -f "$TOOLS/_probe" "$TOOLS/_probe.c"

# ── source ──────────────────────────────────────────────────────────────────
say "codex source @ $TAG"
if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$SRC"; git -C "$SRC" init -q; git -C "$SRC" remote add origin "$URL"
fi
git -C "$SRC" fetch --depth 1 origin tag "$TAG" --no-tags
# Patches applied by the previous build are still in the worktree **and the index** (apply --3way writes the index,
# and checkout -- . restores only from the index, i.e. restores nothing): reset --hard to clean up, then switch to the tag.
git -C "$SRC" reset -q --hard && git -C "$SRC" clean -qfd
git -C "$SRC" checkout -q "$TAG"
head_sha=$(git -C "$SRC" rev-parse HEAD)
# BASE records the **commit** sha. `git rev-parse <annotated-tag>` gives the tag object;
# the two differ, and with the wrong one this gate can never pass.
case "$head_sha" in "$REF"*) ;; *) die "$TAG resolves to $head_sha but BASE says $REF. Was the tag moved? Stopping.";; esac
git -C "$SRC" checkout -q -- . && git -C "$SRC" clean -qfd

say "apply patches"
for p in "$PATCH_DIR"/[0-9][0-9][0-9][0-9]-*.patch; do
    echo "  · $(basename "$p")"
    git -C "$SRC" apply --3way "$p" || die "$(basename "$p") does not apply"
done
grep -q "view_video" "$SRC/codex-rs/core/src/tools/handlers/mod.rs" || die "marker missing: view_video not registered"
grep -q "sglang_session_release" "$SRC/codex-rs/features/src/lib.rs" || die "marker missing: feature from 0002 not registered"
grep -q "compaction_reuse_prefix" "$SRC/codex-rs/features/src/lib.rs" || die "marker missing: feature from 0003 not registered"
grep -q "external_compact_request" "$SRC/codex-rs/features/src/lib.rs" || die "marker missing: feature from 0004 not registered"
grep -q "pre_compact_reminder" "$SRC/codex-rs/features/src/lib.rs" || die "marker missing: feature from 0005 not registered"
grep -q "is_pre_compact_reminder" "$SRC/codex-rs/core/src/compact.rs" || die "marker missing: 0006 not applied"
grep -q "install_tool_output_tokenizer" "$SRC/codex-rs/utils/output-truncation/src/lib.rs" || die "marker missing: 0007 not applied"

# ── cross-compile environment ────────────────────────────────────────────────
# Taken from the official .github/scripts/install-musl-build-tools.sh, minus the apt and libcap parts.
say "cross env"
WRAP=$TOOLS/wrappers; mkdir -p "$WRAP"
for mode in cc c++; do
    name=$([ "$mode" = cc ] && echo zigcc || echo zigcxx)
    cat > "$WRAP/$name" <<EOF
#!/usr/bin/env bash
set -euo pipefail
args=(); skip=0; pend=0
for arg in "\$@"; do
  if [[ "\$pend" -eq 1 ]]; then pend=0
    if [[ "\$arg" == /usr/include* ]]; then args+=("-idirafter" "\$arg"); else args+=("-I" "\$arg"); fi
    continue; fi
  if [[ "\$skip" -eq 1 ]]; then skip=0; continue; fi
  case "\$arg" in
    --target) skip=1; continue ;;
    --target=*|-target=*) continue ;;
    -target) skip=1; continue ;;
    -I) pend=1; continue ;;
    -I/usr/include|-I/usr/include/*) args+=("-idirafter" "\${arg#-I}"); continue ;;
    -Wp,-U_FORTIFY_SOURCE) args+=("-U_FORTIFY_SOURCE"); continue ;;
  esac
  args+=("\$arg")
done
exec "$ZIG_DIR/zig" $mode -target x86_64-linux-musl "\${args[@]}" -fno-sanitize=undefined
EOF
    chmod +x "$WRAP/$name"
done

export CARGO_TARGET_DIR="$TARGET_DIR"
export CC="$WRAP/zigcc" CXX="$WRAP/zigcxx"
export TARGET_CC="$CC" TARGET_CXX="$CXX"
export CC_x86_64_unknown_linux_musl="$CC" CXX_x86_64_unknown_linux_musl="$CXX"
# Host-side build scripts must stay on the real glibc; don't let zig take over.
export HOST_CC=/usr/bin/gcc HOST_CXX=/usr/bin/g++
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$MUSL_GCC"
export CFLAGS=-pthread CXXFLAGS=-pthread
# Official comment: aws-lc's jitter entropy breaks on the musl builder.
export AWS_LC_SYS_NO_JITTER_ENTROPY=1
export AWS_LC_SYS_NO_JITTER_ENTROPY_x86_64_unknown_linux_musl=1
export PKG_CONFIG_ALLOW_CROSS=1
mkdir -p "$TOOLS/empty-pkgconfig"
# Keep the host glibc's .pc files out of the musl cross-compile.
export PKG_CONFIG_LIBDIR_x86_64_unknown_linux_musl="$TOOLS/empty-pkgconfig"
export ZIG_GLOBAL_CACHE_DIR="$TOOLS/zig-cache"

say "cargo build (first build about 8 min @192 cores)"
cd "$SRC/codex-rs"
STABLE_GIT_COMMIT=$(git rev-parse HEAD) \
    cargo +"$RUST_VER" build --target "$TARGET" --release --bin codex

BIN=$TARGET_DIR/$TARGET/release/codex
[ -f "$BIN" ] || die "build finished but $BIN is missing"

say "strip + verify"
# The release profile carries debuginfo; unstripped it is 1.2 GB — always strip before uploading to Kaggle.
cp "$BIN" "$OUT"; strip --strip-debug --strip-unneeded "$OUT"
ldd "$OUT" 2>&1 | grep -q "statically linked" || die "$OUT is not statically linked; it will not start on Kaggle"
readelf -V "$OUT" 2>/dev/null | grep -q GLIBC && die "$OUT still has GLIBC version symbols"
ver=$("$OUT" --version)
echo "  $ver  ($(du -h "$OUT" | cut -f1))"
# manifest.CLI_PINS["codex"] is an EXACT match; if the version string doesn't match, the arm won't start.
case "$ver" in *0.154.0*) ;; *) die "version string is $ver; CLI_PINS expects 0.154.0";; esac
say "OK → $OUT"
