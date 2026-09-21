const fs = require('fs');
const path = require('path');
const createRV32IModule = require('./rv32i.js');

function loadCheckpointToCpu(Module, cpu, ckptPath) {
  const buf = fs.readFileSync(ckptPath);
  // Header: magic (4), version (4), steps (8), pc (4), timer_ms (4), keyboard (4), halted (4), regs (32*4=128), ram_size (4)
  // Total header size = 4 + 4 + 8 + 4 + 4 + 4 + 4 + 128 + 4 = 164 bytes
  const magic = buf.toString('utf8', 0, 4);
  if (magic !== 'RV32') {
    throw new Error('Invalid checkpoint magic: ' + magic);
  }
  const pc = buf.readUInt32LE(16);
  const timerMs = buf.readUInt32LE(20);
  const keyboard = buf.readUInt32LE(24);
  const halted = buf.readUInt32LE(28);

  Module._rv32i_set_pc(cpu, pc);
  Module._rv32i_set_timer_ms(cpu, timerMs);

  for (let r = 0; r < 32; r++) {
    const regVal = buf.readUInt32LE(32 + r * 4);
    Module._rv32i_set_reg(cpu, r, regVal);
  }

  const ramSize = buf.readUInt32LE(160);
  const ramOffset = 164;
  const ramPtr = Module._rv32i_get_ram_ptr(cpu);
  const ramSlice = buf.subarray(ramOffset, ramOffset + ramSize);
  Module.HEAPU8.set(ramSlice, ramPtr);

  console.log(`Successfully loaded checkpoint: PC=0x${pc.toString(16)}, RAM=${(ramSize / 1024 / 1024).toFixed(1)} MB`);
}

async function runBenchmark(name, cpu, Module, mambaStatePtr, burstSize, maxInsts) {
  const outInfoPtr = Module._malloc(64);
  const outStatsPtr = Module._malloc(32);

  // Warmup
  Module._rv32i_step_superblock_neural_burst(cpu, mambaStatePtr, 200, maxInsts, outInfoPtr, outStatsPtr);

  const t0 = process.hrtime.bigint();
  Module._rv32i_step_superblock_neural_burst(
    cpu,
    mambaStatePtr,
    burstSize,
    maxInsts,
    outInfoPtr,
    outStatsPtr
  );
  const t1 = process.hrtime.bigint();

  const elapsedMs = Number(t1 - t0) / 1e6;
  const elapsedSec = elapsedMs / 1000;

  const u32 = Module.HEAPU32;
  const statsIdx = outStatsPtr >> 2;
  const burstInsts       = u32[statsIdx + 0] >>> 0;
  const burstBranches    = u32[statsIdx + 1] >>> 0;
  const burstCorrect     = u32[statsIdx + 2] >>> 0;
  const burstTaken       = u32[statsIdx + 3] >>> 0;
  const burstFallthrough = u32[statsIdx + 4] >>> 0;

  const accPct = burstBranches > 0 ? (burstCorrect / burstBranches * 100).toFixed(2) : '100.00';
  const instsPerBlock = (burstInsts / burstSize).toFixed(1);
  const blocksPerSec = (burstSize / elapsedSec).toFixed(1);
  const instsPerSec = (burstInsts / elapsedSec).toFixed(1);
  const usPerBlock = (elapsedMs * 1000 / burstSize).toFixed(2);
  const mips = (burstInsts / elapsedSec / 1e6).toFixed(2);

  console.log(`\n--- [${name}] Benchmark Results (maxInsts=${maxInsts}) ---`);
  console.log(`Execution Time:          ${elapsedMs.toFixed(2)} ms (${elapsedSec.toFixed(3)} s)`);
  console.log(`Mega-Blocks Executed:    ${burstSize.toLocaleString()}`);
  console.log(`Total CPU Instructions:  ${burstInsts.toLocaleString()}`);
  console.log(`Mega-Block Density:      ${instsPerBlock} insts/block (vs ~5.8 previously)`);
  console.log(`Neural Branches:         ${burstBranches.toLocaleString()}`);
  console.log(`Branch Accuracy:         ${burstCorrect.toLocaleString()} / ${burstBranches.toLocaleString()} (${accPct}%)`);
  console.log(`Taken / Fallthrough:     ${burstTaken.toLocaleString()} / ${burstFallthrough.toLocaleString()}`);
  console.log(`Inference Latency:       ${usPerBlock} µs per Mega-Block`);
  console.log(`Throughput (Blocks):     ${Number(blocksPerSec).toLocaleString()} blocks/sec`);
  console.log(`Throughput (Insts):      ${Number(instsPerSec).toLocaleString()} insts/sec`);
  console.log(`Instruction Rate (MIPS): ${mips} MIPS`);

  Module._free(outInfoPtr);
  Module._free(outStatsPtr);
}

async function main() {
  console.log('=== DOOM Neural Execution Benchmark: Tier 2 & Tier 3 (SIMD128 + Mega-Block) ===');
  const Module = await createRV32IModule();
  console.log('WASM Module Loaded successfully with SIMD128 support.');

  const cpu = Module._rv32i_create();
  Module._rv32i_init(cpu);

  const ckptPath = path.resolve(__dirname, '../data/checkpoint.bin');
  if (fs.existsSync(ckptPath)) {
    loadCheckpointToCpu(Module, cpu, ckptPath);
  } else {
    console.error('Checkpoint not found: ' + ckptPath);
    return;
  }

  const mambaStatePtr = Module._malloc(8192);
  Module._mamba_nano_reset_state(mambaStatePtr);

  // Test 1: Baseline (Single Basic Block, maxInsts=1)
  await runBenchmark('Baseline Superblock (maxInsts=1)', cpu, Module, mambaStatePtr, 5000, 1);

  // Test 2: Tier 2 Mega-Block Chaining (maxInsts=32)
  await runBenchmark('Tier 2 Mega-Block (maxInsts=32)', cpu, Module, mambaStatePtr, 5000, 32);

  // Test 3: Tier 2 Mega-Block Chaining (maxInsts=64)
  await runBenchmark('Tier 2 Mega-Block (maxInsts=64)', cpu, Module, mambaStatePtr, 10000, 64);

  // Test 4: Tier 2 Mega-Block Chaining (maxInsts=128)
  await runBenchmark('Tier 2 Mega-Block (maxInsts=128)', cpu, Module, mambaStatePtr, 10000, 128);

  Module._free(mambaStatePtr);
  Module._rv32i_free(cpu);
}

main().catch(err => {
  console.error('Benchmark error:', err);
  process.exit(1);
});
