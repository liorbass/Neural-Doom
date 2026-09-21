const createARM32Module = require('./arm32.js');

async function test() {
    console.log('[ARM32 WASM TEST] Initializing ARM32 WebAssembly Module...');
    const Module = await createARM32Module();

    const cpu = Module._arm32_create();
    if (!cpu) throw new Error('Failed to create ARM32 CPU instance');

    const pc = Module._arm32_get_pc(cpu) >>> 0;
    console.log(`[ARM32 WASM TEST] CPU created successfully. Initial PC = 0x${pc.toString(16)}`);

    // Test program:
    // 1. mov r1, #10 (0xe3a0100a)
    // 2. mov r2, #20 (0xe3a02014)
    // 3. add r0, r1, r2, lsl #2 (0xe0810102) -> 10 + 80 = 90
    // 4. movs r3, #0 (0xe3b03000) -> Z=1
    // 5. addeq r4, r3, #42 (0x0283402a) -> executed because Z=1
    // 6. addne r5, r3, #99 (0x12835063) -> skipped because Z=1
    const code = new Uint32Array([
        0xe3a0100a,
        0xe3a02014,
        0xe0810102,
        0xe3b03000,
        0x0283402a,
        0x12835063
    ]);

    const bytes = new Uint8Array(code.buffer);
    const ptr = Module._malloc(bytes.length);
    Module.HEAPU8.set(bytes, ptr);
    Module._arm32_load_binary_data(cpu, ptr, bytes.length);
    Module._free(ptr);

    console.log('[ARM32 WASM TEST] Executing ALU & Barrel Shift instructions in WASM...');
    Module._arm32_step(cpu);
    const r1 = Module._arm32_get_reg(cpu, 1) >>> 0;
    if (r1 !== 10) throw new Error(`Expected r1=10, got ${r1}`);

    Module._arm32_step(cpu);
    const r2 = Module._arm32_get_reg(cpu, 2) >>> 0;
    if (r2 !== 20) throw new Error(`Expected r2=20, got ${r2}`);

    Module._arm32_step(cpu);
    const r0 = Module._arm32_get_reg(cpu, 0) >>> 0;
    console.log(`[ARM32 WASM TEST] r0 = ${r0} (expected 90)`);
    if (r0 !== 90) throw new Error(`Expected r0=90, got ${r0}`);

    console.log('[ARM32 WASM TEST] Executing Flag & Condition Code instructions in WASM...');
    Module._arm32_step(cpu);
    const cpsr = Module._arm32_get_cpsr(cpu) >>> 0;
    const z = (cpsr >> 30) & 1;
    if (z !== 1) throw new Error('Expected Z flag to be 1');

    Module._arm32_step(cpu); // addeq
    const r4 = Module._arm32_get_reg(cpu, 4) >>> 0;
    console.log(`[ARM32 WASM TEST] r4 (addeq) = ${r4} (expected 42)`);
    if (r4 !== 42) throw new Error(`Expected r4=42, got ${r4}`);

    Module._arm32_step(cpu); // addne (skipped)
    const r5 = Module._arm32_get_reg(cpu, 5) >>> 0;
    console.log(`[ARM32 WASM TEST] r5 (addne skipped) = ${r5} (expected 0)`);
    if (r5 !== 0) throw new Error(`Expected r5=0, got ${r5}`);

    // Test prompt trace & UTF8ToString
    console.log('[ARM32 WASM TEST] Testing prompt decoding & step tracing...');
    const promptPtr = Module._malloc(1024);
    const targetPtr = Module._malloc(256);
    const mnemPtr = Module._malloc(32);

    Module._arm32_set_pc(cpu, 8); // point back to add r0, r1, r2, lsl #2
    Module._arm32_step_trace(cpu, promptPtr, 1024, targetPtr, 256);

    const promptStr = Module.UTF8ToString(promptPtr);
    const targetStr = Module.UTF8ToString(targetPtr);
    console.log('[ARM32 WASM TEST] Prompt:\n' + promptStr);
    console.log('[ARM32 WASM TEST] Target:\n' + targetStr);

    if (!promptStr.includes('[CMD] STEP') || !promptStr.includes('[CPSR]')) {
        throw new Error('Malformed prompt string');
    }
    if (!targetStr.includes('[NPC]')) {
        throw new Error('Malformed target string');
    }

    // Test key push
    Module._arm32_push_key(cpu, 163, 1);
    Module._arm32_push_key(cpu, 163, 0);

    Module._free(promptPtr);
    Module._free(targetPtr);
    Module._free(mnemPtr);
    Module._arm32_free(cpu);

    console.log('\n[ARM32 WASM TEST] Benchmark: executing 10,000,000 ARM32 instructions in WASM...');
    const benchCpu = Module._arm32_create();
    const loopCode = new Uint32Array([
        0xe2800001, // add r0, r0, #1
        0xeafffffe  // b -4 (loop)
    ]);
    const loopBytes = new Uint8Array(loopCode.buffer);
    const loopPtr = Module._malloc(loopBytes.length);
    Module.HEAPU8.set(loopBytes, loopPtr);
    Module._arm32_load_binary_data(benchCpu, loopPtr, loopBytes.length);
    Module._free(loopPtr);

    const t0 = performance.now();
    Module._arm32_run_steps(benchCpu, 10000000);
    const dt = (performance.now() - t0) / 1000;
    const mips = (10000000 / dt / 1e6).toFixed(2);
    console.log(`[ARM32 WASM TEST] 10,000,000 ARM32 instructions executed in ${dt.toFixed(3)}s = ${mips} MIPS!`);

    Module._arm32_free(benchCpu);
    console.log('[ARM32 WASM TEST] SUCCESS: All ARM32 WebAssembly tests passed!');
}

test().catch(err => {
    console.error('[ARM32 WASM TEST] ERROR:', err);
    process.exit(1);
});
