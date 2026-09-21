#!/usr/bin/env bash
# One-shot build environment for neural-doom.
#
# Fetches the pinned RISC-V toolchain, clones doomgeneric, copies the bare-
# metal port files, downloads the DOOM shareware WAD, builds the RV32I binary
# (code + appended WAD) and generates symbols.txt (guest global addresses the
# tracer/emulator probe). Safe to re-run.
#
#   ./setup_env.sh
#
# Output: doomgeneric/doomgeneric/doom_rv32i.bin  (+ .map, symbols.txt)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GCC_VER="15.2.0-1"
GCC_DIR="xpack-riscv-none-elf-gcc-${GCC_VER}"
GCC_URL="https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v${GCC_VER}/${GCC_DIR}-linux-x64.tar.gz"
TOOLCHAIN="$ROOT/toolchain"

# pinned for reproducibility; bump deliberately
DOOMGENERIC_REPO="https://github.com/ozkl/doomgeneric.git"
DOOMGENERIC_SHA="dcb7a8dbc7a16ce3dda29382ac9aae9d77d21284"

WAD_ZIP_URL="https://archive.org/download/DoomsharewareEpisode/doom.ZIP"
WAD_SIZE=4196020   # verified shareware doom1.wad (infotable ends here)

echo "==> RISC-V toolchain"
if [ ! -x "$TOOLCHAIN/bin/riscv-none-elf-gcc" ]; then
    echo "    downloading ${GCC_DIR} (~430 MB, one-time)..."
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    curl -fL --retry 3 -o "$TMP/gcc.tar.gz" "$GCC_URL"
    tar -xzf "$TMP/gcc.tar.gz" -C "$TMP"
    mv "$TMP/$GCC_DIR" "$TOOLCHAIN"
else
    echo "    already present: $("$TOOLCHAIN/bin/riscv-none-elf-gcc" --version | head -1)"
fi

echo "==> doomgeneric"
if [ ! -d "$ROOT/doomgeneric" ]; then
    git clone "$DOOMGENERIC_REPO" "$ROOT/doomgeneric"
fi
git -C "$ROOT/doomgeneric" checkout --quiet "$DOOMGENERIC_SHA"

BUILD_DIR="$ROOT/doomgeneric/doomgeneric"
echo "==> copying port files into $BUILD_DIR"
cp "$ROOT/port/start.s" "$ROOT/port/link.ld" "$ROOT/port/doomgeneric_rv32i.c" \
   "$ROOT/port/syscalls.c" "$ROOT/port/wadadd.py" "$ROOT/port/e1m1uv.lmp" "$BUILD_DIR/"
cp "$ROOT/port/Makefile.rv32i" "$BUILD_DIR/Makefile"

echo "==> DOOM shareware WAD"
if [ ! -f "$BUILD_DIR/doom1.wad" ]; then
    echo "    downloading doom.ZIP (archive.org) and extracting DOOM1.WAD..."
    TMP="${TMP:-$(mktemp -d)}"; trap 'rm -rf "$TMP"' EXIT
    curl -fL --retry 3 -o "$TMP/doom.ZIP" "$WAD_ZIP_URL"
    python3 - "$TMP/doom.ZIP" "$BUILD_DIR/doom1.wad" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    with z.open("DOOM1.WAD") as src, open(sys.argv[2], "wb") as dst:
        dst.write(src.read())
PY
fi
actual="$(stat -c%s "$BUILD_DIR/doom1.wad")"
if [ "$actual" != "$WAD_SIZE" ]; then
    echo "    WARNING: doom1.wad is $actual bytes, expected $WAD_SIZE." >&2
    echo "    A wrong WAD can fail at texture load (e.g. 'Missing patch in" >&2
    echo "    texture BIGDOOR1'). Drop a stock shareware doom1.wad in" >&2
    echo "    $BUILD_DIR and re-run." >&2
fi

echo "==> building bare-metal DOOM (rv32i, ilp32)"
make -C "$BUILD_DIR" clean
make -C "$BUILD_DIR" -j"$(nproc)"
make -C "$BUILD_DIR" wad      # truncate to 10M + append doom1.wad at 0x80A00000
make -C "$BUILD_DIR" symbols  # gamestate/gametic/... -> symbols.txt
make -C "$BUILD_DIR" demo     # first-level-completion build -> doom_rv32i_demo.bin

echo
echo "==> Build complete: $BUILD_DIR/doom_rv32i.bin ($(stat -c%s "$BUILD_DIR/doom_rv32i.bin") bytes)"
echo "==> Demo build:     $BUILD_DIR/doom_rv32i_demo.bin ($(stat -c%s "$BUILD_DIR/doom_rv32i_demo.bin") bytes)"
echo
echo "Next steps:"
echo "  uv sync"
echo "  # play at native speed in the browser:"
echo "  make -C emulator gui"
echo "  # watch the port complete E1M1 by itself (recorded vanilla playthrough):"
echo "  make -C emulator && emulator/emulator --bin $BUILD_DIR/doom_rv32i_demo.bin --no-input --steps 500000000"
echo "  # regenerate the trace dataset (see README 'Quickstart' for the full input schedule):"
echo "  emulator/emulator --bin $BUILD_DIR/doom_rv32i.bin --steps 105000000 --dataset data/dataset_dense.jsonl ..."
