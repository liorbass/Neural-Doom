#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/web"

if ! command -v emcc >/dev/null 2>&1; then
    if [ -f "/home/u/emsdk/emsdk_env.sh" ]; then
        # shellcheck disable=SC1091
        source "/home/u/emsdk/emsdk_env.sh" >/dev/null 2>&1
    fi
fi

if ! command -v emcc >/dev/null 2>&1; then
    echo "Error: emcc not found. Please run: cd /home/u/emsdk && source emsdk_env.sh" >&2
    exit 1
fi

echo "Compiling rv32i.c + mamba_nano.c to WebAssembly (web/rv32i.js + web/rv32i.wasm)..."
emcc -O3 -std=gnu11 -msimd128 \
    -s WASM=1 \
    -s INITIAL_MEMORY=33554432 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createRV32IModule" \
    -s EXPORTED_FUNCTIONS="['_rv32i_create','_rv32i_free','_rv32i_init','_rv32i_load_binary_data','_rv32i_run_steps','_rv32i_step','_rv32i_step_trace','_rv32i_decode_prompt','_rv32i_step_block','_rv32i_step_superblock','_rv32i_advance_branch','_rv32i_step_superblock_neural','_rv32i_step_superblock_neural_burst','_mamba_nano_reset_state','_mamba_nano_step','_mamba_nano_step_from_pc_regs','_mamba_nano_get_state_norm','_rv32i_get_write_reg','_rv32i_get_write_mem','_rv32i_get_ram_ptr','_rv32i_mem_read','_rv32i_mem_write','_rv32i_queue_key','_rv32i_push_key','_rv32i_clear_keys','_rv32i_get_vram_ptr','_rv32i_get_frame_rgba','_rv32i_get_pc','_rv32i_set_pc','_rv32i_get_steps','_rv32i_get_steps_double','_rv32i_set_steps','_rv32i_get_reg','_rv32i_set_reg','_rv32i_apply_prediction','_rv32i_get_timer_ms','_rv32i_set_timer_ms','_rv32i_get_keyboard','_rv32i_is_halted','_rv32i_probe_gs','_rv32i_probe_gametic','_malloc','_free']" \
    -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','setValue','getValue','HEAPU8','HEAP32','HEAPU32','UTF8ToString','stringToUTF8']" \
    -s ENVIRONMENT="web,node" \
    -I"$ROOT/emulator" \
    "$ROOT/emulator/rv32i.c" \
    "$ROOT/emulator/mamba_nano.c" \
    -lm \
    -o "$ROOT/web/rv32i.js"

echo "Build complete:"
ls -lh "$ROOT/web/rv32i.js" "$ROOT/web/rv32i.wasm"
