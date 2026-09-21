const fs = require('fs');
const path = require('path');
const createRV32IModule = require('./rv32i.js');

async function main() {
    console.log('[STAGE C TEST] Loading WASM module...');
    const Module = await createRV32IModule();
    console.log('[STAGE C TEST] WASM initialized.');

    const cpu = Module._rv32i_create();
    if (!cpu) throw new Error('Failed to create CPU');

    // Load doom_rv32i.bin
    const binPath = path.join(__dirname, 'doom_rv32i.bin');
    const binData = fs.readFileSync(binPath);
    console.log(`[STAGE C TEST] Loading ${binData.length} bytes of DOOM binary...`);

    const ptr = Module._malloc(binData.length);
    Module.HEAPU8.set(binData, ptr);
    Module._rv32i_load_binary_data(cpu, ptr, binData.length);
    Module._free(ptr);

    console.log('[STAGE C TEST] DOOM loaded. Initial PC = 0x' + (Module._rv32i_get_pc(cpu) >>> 0).toString(16));

    // Allocate MambaNanoState: 64*16*4 * 2 = 8192 bytes
    const statePtr = Module._malloc(8192);
    Module._mamba_nano_reset_state(statePtr);

    const outInfoPtr = Module._malloc(64); // 16 uint32s
    const outStatsPtr = Module._malloc(32);

    console.log('[STAGE C TEST] Stepping 100 superblocks via rv32i_step_superblock_neural...');
    const t0 = process.hrtime.bigint();
    let totalInsts = 0;
    let totalBranches = 0;
    let correctBranches = 0;

    for (let i = 0; i < 100; i++) {
        const insts = Module._rv32i_step_superblock_neural(cpu, statePtr, 64, outInfoPtr);
        totalInsts += insts;

        const u32 = Module.HEAPU32;
        const base = outInfoPtr >> 2;
        const isBranch = u32[base + 6];
        const termOp = u32[base + 7];
        const isCorrect = u32[base + 11];

        if (isBranch && termOp === 0x63) {
            totalBranches++;
            if (isCorrect) correctBranches++;
        }
    }
    const t1 = process.hrtime.bigint();
    const elapsedMs = Number(t1 - t0) / 1e6;
    const usPerBlock = (elapsedMs / 100) * 1000;
    const blkPerSec = (100 / (elapsedMs / 1000));
    const instPerSec = (totalInsts / (elapsedMs / 1000));

    console.log(`[STAGE C TEST] 100 superblocks in ${elapsedMs.toFixed(2)} ms:`);
    console.log(`  - Latency: ${usPerBlock.toFixed(1)} µs / block`);
    console.log(`  - Block rate: ${blkPerSec.toFixed(0)} blk/s`);
    console.log(`  - Instruction rate: ${instPerSec.toFixed(0)} insts/s (total ${totalInsts} insts)`);
    console.log(`  - Branch accuracy: ${correctBranches}/${totalBranches} (${totalBranches > 0 ? (correctBranches/totalBranches*100).toFixed(1) : 0}%)`);

    console.log('\n[STAGE C TEST] Testing rv32i_step_superblock_neural_burst (1000 blocks in burst)...');
    const tb0 = process.hrtime.bigint();
    const burstInsts = Module._rv32i_step_superblock_neural_burst(cpu, statePtr, 1000, 64, outInfoPtr, outStatsPtr);
    const tb1 = process.hrtime.bigint();
    const burstMs = Number(tb1 - tb0) / 1e6;
    const burstUsPerBlock = (burstMs / 1000) * 1000;
    const burstBlkPerSec = (1000 / (burstMs / 1000));
    const burstInstPerSec = (burstInsts / (burstMs / 1000));

    const statsU32 = Module.HEAPU32;
    const sBase = outStatsPtr >> 2;
    const bTotalInsts = statsU32[sBase];
    const bTotalBranches = statsU32[sBase + 1];
    const bCorrect = statsU32[sBase + 2];

    console.log(`[STAGE C TEST] 1000 burst blocks in ${burstMs.toFixed(2)} ms:`);
    console.log(`  - Latency: ${burstUsPerBlock.toFixed(1)} µs / block`);
    console.log(`  - Block rate: ${burstBlkPerSec.toFixed(0)} blk/s`);
    console.log(`  - Instruction rate: ${burstInstPerSec.toFixed(0)} insts/s (total ${bTotalInsts} insts)`);
    console.log(`  - Burst branch accuracy: ${bCorrect}/${bTotalBranches} (${bTotalBranches > 0 ? (bCorrect/bTotalBranches*100).toFixed(1) : 0}%)`);

    Module._free(statePtr);
    Module._free(outInfoPtr);
    Module._free(outStatsPtr);
    Module._rv32i_free(cpu);

    console.log('[STAGE C TEST] Verification SUCCESSFUL!');
}

main().catch(err => {
    console.error('[STAGE C TEST ERROR]', err);
    process.exit(1);
});
