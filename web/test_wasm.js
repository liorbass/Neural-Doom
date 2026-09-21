const createRV32IModule = require('./rv32i.js');

async function test() {
    console.log('[WASM TEST] Initializing RV32I WebAssembly Module...');
    const Module = await createRV32IModule();

    const cpu = Module._rv32i_create();
    if (!cpu) throw new Error('Failed to create CPU instance');

    const pc = Module._rv32i_get_pc(cpu) >>> 0;
    console.log(`[WASM TEST] CPU created successfully. Initial PC = 0x${pc.toString(16)}`);
    if (pc !== 0x80000000) throw new Error(`Unexpected PC: 0x${pc.toString(16)}`);

    // Test program:
    // addi x1, x0, 42   (0x02a00093)
    // addi x2, x1, 10   (0x00a08113)
    const code = new Uint8Array([
        0x93, 0x00, 0xa0, 0x02,
        0x13, 0x81, 0xa0, 0x00
    ]);

    const ptr = Module._malloc(code.length);
    Module.HEAPU8.set(code, ptr);
    Module._rv32i_load_binary_data(cpu, ptr, code.length);
    Module._free(ptr);

    console.log('[WASM TEST] Executing 2 instructions...');
    Module._rv32i_step(cpu);
    const x1 = Module._rv32i_get_reg(cpu, 1) >>> 0;
    console.log(`[WASM TEST] Step 1: x1 = ${x1} (expected 42)`);
    if (x1 !== 42) throw new Error(`Expected x1=42, got ${x1}`);

    Module._rv32i_step(cpu);
    const x2 = Module._rv32i_get_reg(cpu, 2) >>> 0;
    console.log(`[WASM TEST] Step 2: x2 = ${x2} (expected 52)`);
    if (x2 !== 52) throw new Error(`Expected x2=52, got ${x2}`);

    // Test rv32i_push_key
    console.log('[WASM TEST] Testing rv32i_push_key (Enter=13, Fire=163, Use=162)...');
    Module._rv32i_push_key(cpu, 13, 1);  // Enter down
    Module._rv32i_push_key(cpu, 13, 0);  // Enter up
    Module._rv32i_push_key(cpu, 163, 1); // KEY_FIRE down
    Module._rv32i_push_key(cpu, 163, 0); // KEY_FIRE up
    Module._rv32i_push_key(cpu, 162, 1); // KEY_USE down
    Module._rv32i_push_key(cpu, 162, 0); // KEY_USE up

    // Test rv32i_decode_prompt and UTF8ToString
    console.log('[WASM TEST] Testing rv32i_decode_prompt & UTF8ToString...');
    const promptPtr = Module._malloc(1024);
    const targetPtr = Module._malloc(256);
    const mnemPtr = Module._malloc(32);
    Module._rv32i_decode_prompt(cpu, promptPtr, 1024, mnemPtr, 32);
    const promptStr = Module.UTF8ToString(promptPtr);
    console.log('[WASM TEST] Decoded prompt:\n' + promptStr);
    if (!promptStr.includes('[CMD] STEP')) throw new Error('Prompt missing [CMD] STEP');

    Module._rv32i_step_trace(cpu, promptPtr, 1024, targetPtr, 256);
    const targetStr = Module.UTF8ToString(targetPtr);
    console.log('[WASM TEST] Decoded target:\n' + targetStr);
    if (!targetStr.includes('[NPC]')) throw new Error('Target missing [NPC]');

    Module._free(promptPtr);
    Module._free(targetPtr);
    Module._free(mnemPtr);
    Module._rv32i_free(cpu);

    console.log('\n[WASM BENCHMARK] Loading real DOOM binary into WebAssembly...');
    const fs = require('fs');
    const doomBin = fs.readFileSync('doomgeneric/doomgeneric/doom_rv32i.bin');
    const doomCpu = Module._rv32i_create();
    const binPtr = Module._malloc(doomBin.length);
    Module.HEAPU8.set(doomBin, binPtr);
    Module._rv32i_load_binary_data(doomCpu, binPtr, doomBin.length);
    Module._free(binPtr);

    console.log(`[WASM BENCHMARK] DOOM binary loaded (${(doomBin.length / 1024 / 1024).toFixed(2)} MB).`);
    console.log('[WASM BENCHMARK] Stepping 10,000,000 instructions in WASM...');
    const t0 = performance.now();
    Module._rv32i_run_steps(doomCpu, 10000000);
    const dt = (performance.now() - t0) / 1000;
    const rate = (10000000 / dt / 1e6).toFixed(2);
    const doomPc = Module._rv32i_get_pc(doomCpu) >>> 0;
    console.log(`[WASM BENCHMARK] 10,000,000 steps executed in ${dt.toFixed(3)}s = ${rate} Million steps/sec! (PC: 0x${doomPc.toString(16)})`);
    Module._rv32i_free(doomCpu);

    console.log('[WASM TEST] SUCCESS: All WebAssembly tests passed!');
}

test().catch(err => {
    console.error('[WASM TEST] ERROR:', err);
    process.exit(1);
});
