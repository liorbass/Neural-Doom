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

echo "Compiling arm32.c to WebAssembly (web/arm32.js + web/arm32.wasm)..."
emcc -O3 -std=gnu11 \
    -s WASM=1 \
    -s INITIAL_MEMORY=67108864 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="createARM32Module" \
    -s EXPORTED_FUNCTIONS="['_arm32_create','_arm32_free','_arm32_init','_arm32_load_binary_data','_arm32_run_steps','_arm32_step','_arm32_step_trace','_arm32_decode_prompt','_arm32_queue_key','_arm32_push_key','_arm32_clear_keys','_arm32_get_vram_ptr','_arm32_get_frame_rgba','_arm32_get_pc','_arm32_set_pc','_arm32_get_cpsr','_arm32_set_cpsr','_arm32_get_steps','_arm32_get_steps_double','_arm32_get_reg','_arm32_set_reg','_arm32_get_timer_ms','_arm32_get_keyboard','_arm32_is_halted','_arm32_probe_gs','_arm32_probe_gametic','_arm32_get_ram_ptr','_arm32_set_gamestate_addr','_arm32_set_gametic_addr','_arm32_step_superblock','_arm32_step_superblock_neural','_arm32_step_superblock_neural_burst','_mamba_nano_reset_state','_mamba_nano_get_state_norm','_mamba_nano_step_from_pc_regs','_malloc','_free']" \
    -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','setValue','getValue','HEAPU8','HEAPU32','HEAP32','UTF8ToString','stringToUTF8']" \
    -s ENVIRONMENT="web,node" \
    -I"$ROOT/emulator" \
    "$ROOT/emulator/arm32.c" \
    "$ROOT/emulator/mamba_nano.c" \
    -lm -o "$ROOT/web/arm32.js"

echo "ARM32 WebAssembly build complete:"
ls -lh "$ROOT/web/arm32.js" "$ROOT/web/arm32.wasm"
