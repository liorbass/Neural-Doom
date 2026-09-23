#include "rv32i.h"
#include "mamba_nano.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const char *const REG_NAMES[32] = {
    "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
    "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30", "x31"
};

static int32_t sext(uint32_t v, int bits) {
    if (v & (1u << (bits - 1)))
        v |= ~0u << bits;
    return (int32_t)v;
}

RV32I_CPU *rv32i_create(void) {
    RV32I_CPU *cpu = (RV32I_CPU *)calloc(1, sizeof(RV32I_CPU));
    if (cpu) {
        rv32i_init(cpu);
    }
    return cpu;
}

void rv32i_free(RV32I_CPU *cpu) {
    free(cpu);
}

void rv32i_init(RV32I_CPU *cpu) {
    memset(cpu->regs, 0, sizeof(cpu->regs));
    cpu->pc = RAM_BASE;
    cpu->steps = 0;
    cpu->keyboard = 0;
    cpu->timer_ms = 0;
    cpu->halted = 0;

    cpu->has_write_reg = 0;
    cpu->write_reg_rd = 0;
    cpu->write_reg_val = 0;

    cpu->has_write_mem = 0;
    cpu->write_mem_addr = 0;
    cpu->write_mem_val = 0;
    cpu->write_mem_size = 0;

    cpu->has_load = 0;
    cpu->load_addr = 0;
    cpu->load_raw = 0;

    cpu->gs_addr = 0x800bc864u;
    cpu->gametic_addr = 0x800bc6dcu;
    cpu->nev = 0;
}

void rv32i_copy(RV32I_CPU *dst, const RV32I_CPU *src) {
    memcpy(dst, src, sizeof(RV32I_CPU));
}

int rv32i_load_binary(RV32I_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(cpu->ram, 1, RAM_SIZE, f);
    fclose(f);
    cpu->pc = RAM_BASE;
    return (int)n;
}

int rv32i_load_binary_data(RV32I_CPU *cpu, const uint8_t *data, size_t size) {
    if (!cpu || !data || size == 0) return -1;
    if (size > RAM_SIZE) size = RAM_SIZE;
    memcpy(cpu->ram, data, size);
    cpu->pc = RAM_BASE;
    return (int)size;
}

int rv32i_load_symbols(RV32I_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char name[64];
    unsigned long addr;
    int got = 0;
    while (fscanf(f, "%63s %lx", name, &addr) == 2) {
        if (!strcmp(name, "gamestate")) {
            cpu->gs_addr = (uint32_t)addr;
            got = 1;
        } else if (!strcmp(name, "gametic")) {
            cpu->gametic_addr = (uint32_t)addr;
        }
    }
    fclose(f);
    return got;
}

uint32_t rv32i_probe_gs(const RV32I_CPU *cpu) {
    if (cpu->gs_addr < RAM_BASE || cpu->gs_addr + 4 > RAM_BASE + RAM_SIZE)
        return 0xFFFFFFFFu;
    return *(const uint32_t *)(cpu->ram + (cpu->gs_addr - RAM_BASE));
}

uint32_t rv32i_probe_gametic(const RV32I_CPU *cpu) {
    if (!cpu->gametic_addr || cpu->gametic_addr < RAM_BASE ||
        cpu->gametic_addr + 4 > RAM_BASE + RAM_SIZE)
        return 0;
    return *(const uint32_t *)(cpu->ram + (cpu->gametic_addr - RAM_BASE));
}

void rv32i_queue_key(RV32I_CPU *cpu, uint64_t at, uint32_t key, int down, int gs_cond) {
    if (cpu->nev < 64) {
        cpu->evq[cpu->nev].at = at;
        cpu->evq[cpu->nev].val = key | (down ? 0x100u : 0);
        cpu->evq[cpu->nev].has_cond = (gs_cond >= 0);
        cpu->evq[cpu->nev].cond_val = (gs_cond < 0 ? 0 : (uint32_t)gs_cond);
        cpu->nev++;
    }
}

void rv32i_push_key(RV32I_CPU *cpu, uint32_t key, int down) {
    rv32i_queue_key(cpu, 0, key, down, -1);
}

void rv32i_clear_keys(RV32I_CPU *cpu) {
    cpu->nev = 0;
}

uint32_t rv32i_mem_read(RV32I_CPU *cpu, uint32_t addr, int size) {
    if (addr >= 0xFFFF0000u) {
        if (addr == MMIO_KBD) {
            uint32_t v = cpu->keyboard;
            cpu->keyboard = 0;
            if (v == 0 && cpu->nev > 0) {
                KeyEvent e = cpu->evq[0];
                if (e.at <= cpu->steps && (!e.has_cond || rv32i_probe_gs(cpu) == e.cond_val)) {
                    memmove(cpu->evq, cpu->evq + 1, --cpu->nev * sizeof(KeyEvent));
                    v = e.val;
                }
            }
            return v;
        }
        if (addr == MMIO_TIMER) {
            return cpu->timer_ms;
        }
        return 0;
    }
    if (addr < RAM_BASE) return 0;
    uint32_t off = addr - RAM_BASE;
    if (off + size > RAM_SIZE) return 0;
    uint32_t v = 0;
    memcpy(&v, cpu->ram + off, size);
    return v;
}

uint32_t rv32i_mem_peek(const RV32I_CPU *cpu, uint32_t addr, int size) {
    if (addr >= 0xFFFF0000u) {
        if (addr == MMIO_KBD) {
            uint32_t v = cpu->keyboard;
            if (v == 0 && cpu->nev > 0) {
                KeyEvent e = cpu->evq[0];
                if (e.at <= cpu->steps && (!e.has_cond || rv32i_probe_gs(cpu) == e.cond_val)) {
                    v = e.val;
                }
            }
            return v;
        }
        if (addr == MMIO_TIMER) {
            return cpu->timer_ms;
        }
        return 0;
    }
    if (addr < RAM_BASE) return 0;
    uint32_t off = addr - RAM_BASE;
    if (off + size > RAM_SIZE) return 0;
    uint32_t v = 0;
    memcpy(&v, cpu->ram + off, size);
    return v;
}

void rv32i_mem_write(RV32I_CPU *cpu, uint32_t addr, int size, uint32_t val) {
    uint32_t mask = (size == 1) ? 0xFFu : (size == 2) ? 0xFFFFu : 0xFFFFFFFFu;
    val &= mask;
    if (addr >= 0xFFFF0000u) {
        if (addr == MMIO_KBD) {
            cpu->keyboard = val;
        } else if (addr == MMIO_TIMER) {
            cpu->timer_ms = val;
        } else if (addr == MMIO_CONS) {
#ifndef __EMSCRIPTEN__
            fputc((int)(val & 0xFF), stderr);
#endif
        }
        return;
    }
    if (addr < RAM_BASE) return;
    uint32_t off = addr - RAM_BASE;
    if (off + size > RAM_SIZE) return;
    memcpy(cpu->ram + off, &val, size);
    cpu->has_write_mem = 1;
    cpu->write_mem_addr = addr;
    cpu->write_mem_val = val;
    cpu->write_mem_size = size;
}

int rv32i_step(RV32I_CPU *cpu) {
    if (cpu->halted) return 0;
    if (cpu->pc < RAM_BASE || cpu->pc + 4 > RAM_BASE + RAM_SIZE) {
        cpu->halted = 1;
        return 0;
    }

    uint32_t inst;
    memcpy(&inst, cpu->ram + (cpu->pc - RAM_BASE), 4);
    if (inst == 0) {
        cpu->halted = 1;
        return 0;
    }

    uint32_t op = inst & 0x7F;
    uint32_t rd = (inst >> 7) & 0x1F;
    uint32_t funct3 = (inst >> 12) & 0x7;
    uint32_t rs1 = (inst >> 15) & 0x1F;
    uint32_t rs2 = (inst >> 20) & 0x1F;
    uint32_t funct7 = inst >> 25;

    uint32_t npc = cpu->pc + 4;
    int32_t imm;
    cpu->has_write_reg = 0;
    cpu->has_write_mem = 0;
    cpu->has_load = 0;

    switch (op) {
    case 0x37: /* lui */
        imm = (int32_t)(inst & 0xFFFFF000u);
        cpu->regs[rd] = (uint32_t)imm;
        cpu->has_write_reg = 1;
        cpu->write_reg_rd = (int)rd;
        cpu->write_reg_val = (uint32_t)imm;
        break;
    case 0x17: /* auipc */
        imm = (int32_t)(inst & 0xFFFFF000u);
        cpu->regs[rd] = cpu->pc + (uint32_t)imm;
        cpu->has_write_reg = 1;
        cpu->write_reg_rd = (int)rd;
        cpu->write_reg_val = cpu->regs[rd];
        break;
    case 0x6F: /* jal */
        imm = sext(((inst >> 31) << 20) | (((inst >> 12) & 0xFF) << 12) |
                   (((inst >> 20) & 1) << 11) | (((inst >> 21) & 0x3FF) << 1), 21);
        cpu->regs[rd] = cpu->pc + 4;
        npc = cpu->pc + (uint32_t)imm;
        cpu->has_write_reg = 1;
        cpu->write_reg_rd = (int)rd;
        cpu->write_reg_val = cpu->pc + 4;
        break;
    case 0x67: /* jalr */
        imm = sext(inst >> 20, 12);
        {
            uint32_t t = (cpu->regs[rs1] + (uint32_t)imm) & ~1u;
            cpu->regs[rd] = cpu->pc + 4;
            npc = t;
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = (int)rd;
            cpu->write_reg_val = cpu->pc + 4;
        }
        break;
    case 0x03: /* load */
        imm = sext(inst >> 20, 12);
        {
            uint32_t a = cpu->regs[rs1] + (uint32_t)imm;
            uint32_t raw = 0, val = 0;
            if (funct3 == 0x0) { raw = rv32i_mem_read(cpu, a, 1); val = (uint32_t)sext(raw, 8); }
            else if (funct3 == 0x1) { raw = rv32i_mem_read(cpu, a, 2); val = (uint32_t)sext(raw, 16); }
            else if (funct3 == 0x2) { raw = rv32i_mem_read(cpu, a, 4); val = raw; }
            else if (funct3 == 0x4) { raw = rv32i_mem_read(cpu, a, 1); val = raw; }
            else if (funct3 == 0x5) { raw = rv32i_mem_read(cpu, a, 2); val = raw; }
            cpu->regs[rd] = val;
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = (int)rd;
            cpu->write_reg_val = val;
            cpu->has_load = 1;
            cpu->load_addr = a;
            cpu->load_raw = raw;
        }
        break;
    case 0x23: /* store */
        imm = sext(((inst >> 25) << 5) | rd, 12);
        {
            uint32_t a = cpu->regs[rs1] + (uint32_t)imm;
            int sz = (funct3 == 0x0 ? 1 : funct3 == 0x1 ? 2 : 4);
            rv32i_mem_write(cpu, a, sz, cpu->regs[rs2]);
        }
        break;
    case 0x13: /* op-imm */
        imm = sext(inst >> 20, 12);
        {
            uint32_t v1 = cpu->regs[rs1], r = 0;
            uint32_t sh = (inst >> 20) & 0x1Fu;
            if (funct3 == 0x0) r = v1 + (uint32_t)imm;
            else if (funct3 == 0x1) r = v1 << sh;
            else if (funct3 == 0x2) r = (int32_t)v1 < imm ? 1 : 0;
            else if (funct3 == 0x3) r = v1 < (uint32_t)imm ? 1 : 0; /* SLTIU: unsigned compare with sign-extended imm */
            else if (funct3 == 0x4) r = v1 ^ (uint32_t)imm;
            else if (funct3 == 0x5) r = (funct7 >> 1) ? (uint32_t)((int32_t)v1 >> sh) : (v1 >> sh);
            else if (funct3 == 0x6) r = v1 | (uint32_t)imm;
            else if (funct3 == 0x7) r = v1 & (uint32_t)imm;
            cpu->regs[rd] = r;
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = (int)rd;
            cpu->write_reg_val = r;
        }
        break;
    case 0x33: /* op */
        {
            uint32_t v1 = cpu->regs[rs1], v2 = cpu->regs[rs2], r = 0;
            int32_t s1 = (int32_t)v1, s2 = (int32_t)v2;
            uint32_t sh = v2 & 0x1Fu;
            if (funct3 == 0x0) r = funct7 ? (v1 - v2) : (v1 + v2);
            else if (funct3 == 0x1) r = v1 << sh;
            else if (funct3 == 0x2) r = s1 < s2 ? 1 : 0;
            else if (funct3 == 0x3) r = v1 < v2 ? 1 : 0;
            else if (funct3 == 0x4) r = v1 ^ v2;
            else if (funct3 == 0x5) r = funct7 ? (uint32_t)(s1 >> sh) : (v1 >> sh);
            else if (funct3 == 0x6) r = v1 | v2;
            else if (funct3 == 0x7) r = v1 & v2;
            cpu->regs[rd] = r;
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = (int)rd;
            cpu->write_reg_val = r;
        }
        break;
    case 0x63: /* branch */
        imm = sext(((inst >> 31) << 12) | (((inst >> 7) & 1) << 11) |
                   (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1), 13);
        {
            uint32_t v1 = cpu->regs[rs1], v2 = cpu->regs[rs2];
            int take = 0;
            if (funct3 == 0x0) take = (v1 == v2);
            else if (funct3 == 0x1) take = (v1 != v2);
            else if (funct3 == 0x4) take = ((int32_t)v1 < (int32_t)v2);
            else if (funct3 == 0x5) take = ((int32_t)v1 >= (int32_t)v2);
            else if (funct3 == 0x6) take = (v1 < v2);
            else if (funct3 == 0x7) take = (v1 >= v2);
            if (take) npc = cpu->pc + (uint32_t)imm;
        }
        break;
    case 0x0F: /* fence */
        break;
    case 0x73: /* ecall / ebreak */
        cpu->halted = 1;
        break;
    default:
        break;
    }

    cpu->regs[0] = 0;
    cpu->pc = npc;
    cpu->steps++;
    cpu->timer_ms = (uint32_t)(cpu->steps / TIMER_STEPS_PER_MS);
    return 1;
}

int rv32i_decode_prompt(const RV32I_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                        char *mnem_buf, size_t mnem_sz) {
    if (cpu->halted || cpu->pc < RAM_BASE || cpu->pc + 4 > RAM_BASE + RAM_SIZE) {
        if (mnem_buf && mnem_sz) strncpy(mnem_buf, "HALTED", mnem_sz);
        if (prompt_buf && prompt_sz)
            snprintf(prompt_buf, prompt_sz, "[CMD] STEP\n[PC] %08x\n[STATE] HALTED", cpu->pc);
        return 0;
    }

    uint32_t inst;
    memcpy(&inst, cpu->ram + (cpu->pc - RAM_BASE), 4);
    if (inst == 0) {
        if (mnem_buf && mnem_sz) strncpy(mnem_buf, "HALTED", mnem_sz);
        if (prompt_buf && prompt_sz)
            snprintf(prompt_buf, prompt_sz, "[CMD] STEP\n[PC] %08x\n[STATE] HALTED", cpu->pc);
        return 0;
    }

    uint32_t op = inst & 0x7F;
    uint32_t rd = (inst >> 7) & 0x1F;
    uint32_t funct3 = (inst >> 12) & 0x7;
    uint32_t rs1 = (inst >> 15) & 0x1F;
    uint32_t rs2 = (inst >> 20) & 0x1F;
    uint32_t funct7 = inst >> 25;

    const char *mnemonic = "unknown";
    char inst_line[128];
    int has_rs1 = 0, has_rs2 = 0, has_mem = 0;
    uint32_t mem_addr = 0, mem_raw = 0;

    switch (op) {
    case 0x33: /* R-type */
        if (funct3 == 0x0) mnemonic = funct7 ? "sub" : "add";
        else if (funct3 == 0x1) mnemonic = "sll";
        else if (funct3 == 0x2) mnemonic = "slt";
        else if (funct3 == 0x3) mnemonic = "sltu";
        else if (funct3 == 0x4) mnemonic = "xor";
        else if (funct3 == 0x5) mnemonic = funct7 ? "sra" : "srl";
        else if (funct3 == 0x6) mnemonic = "or";
        else if (funct3 == 0x7) mnemonic = "and";
        snprintf(inst_line, sizeof(inst_line), "%s %s, %s, %s",
                 mnemonic, REG_NAMES[rd], REG_NAMES[rs1], REG_NAMES[rs2]);
        has_rs1 = 1; has_rs2 = 1;
        break;

    case 0x13: /* I-type ALU */
        if (funct3 == 0x0) mnemonic = "addi";
        else if (funct3 == 0x1) mnemonic = "slli";
        else if (funct3 == 0x2) mnemonic = "slti";
        else if (funct3 == 0x3) mnemonic = "sltiu";
        else if (funct3 == 0x4) mnemonic = "xori";
        else if (funct3 == 0x5) mnemonic = (funct7 >> 1) ? "srai" : "srli";
        else if (funct3 == 0x6) mnemonic = "ori";
        else if (funct3 == 0x7) mnemonic = "andi";

        if (funct3 == 0x1 || funct3 == 0x5) {
            uint32_t shamt = (inst >> 20) & 0x1F;
            snprintf(inst_line, sizeof(inst_line), "%s %s, %s, %u",
                     mnemonic, REG_NAMES[rd], REG_NAMES[rs1], shamt);
        } else {
            int32_t imm = sext(inst >> 20, 12);
            snprintf(inst_line, sizeof(inst_line), "%s %s, %s, %d",
                     mnemonic, REG_NAMES[rd], REG_NAMES[rs1], imm);
        }
        has_rs1 = 1;
        break;

    case 0x03: /* Load */
        if (funct3 == 0x0) mnemonic = "lb";
        else if (funct3 == 0x1) mnemonic = "lh";
        else if (funct3 == 0x2) mnemonic = "lw";
        else if (funct3 == 0x4) mnemonic = "lbu";
        else if (funct3 == 0x5) mnemonic = "lhu";
        {
            int32_t imm = sext(inst >> 20, 12);
            snprintf(inst_line, sizeof(inst_line), "%s %s, %d(%s)",
                     mnemonic, REG_NAMES[rd], imm, REG_NAMES[rs1]);
            has_rs1 = 1;
            has_mem = 1;
            mem_addr = cpu->regs[rs1] + (uint32_t)imm;
            int sz = (funct3 == 0 || funct3 == 4) ? 1 : (funct3 == 1 || funct3 == 5) ? 2 : 4;
            mem_raw = rv32i_mem_peek(cpu, mem_addr, sz);
        }
        break;

    case 0x23: /* Store */
        if (funct3 == 0x0) mnemonic = "sb";
        else if (funct3 == 0x1) mnemonic = "sh";
        else if (funct3 == 0x2) mnemonic = "sw";
        {
            int32_t imm = sext(((inst >> 25) << 5) | rd, 12);
            snprintf(inst_line, sizeof(inst_line), "%s %s, %d(%s)",
                     mnemonic, REG_NAMES[rs2], imm, REG_NAMES[rs1]);
            has_rs1 = 1; has_rs2 = 1;
        }
        break;

    case 0x63: /* Branch */
        if (funct3 == 0x0) mnemonic = "beq";
        else if (funct3 == 0x1) mnemonic = "bne";
        else if (funct3 == 0x4) mnemonic = "blt";
        else if (funct3 == 0x5) mnemonic = "bge";
        else if (funct3 == 0x6) mnemonic = "bltu";
        else if (funct3 == 0x7) mnemonic = "bgeu";
        {
            int32_t imm = sext(((inst >> 31) << 12) | (((inst >> 7) & 1) << 11) |
                               (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1), 13);
            snprintf(inst_line, sizeof(inst_line), "%s %s, %s, %d",
                     mnemonic, REG_NAMES[rs1], REG_NAMES[rs2], imm);
            has_rs1 = 1; has_rs2 = 1;
        }
        break;

    case 0x6F: /* jal */
        mnemonic = "jal";
        {
            int32_t imm = sext(((inst >> 31) << 20) | (((inst >> 12) & 0xFF) << 12) |
                               (((inst >> 20) & 1) << 11) | (((inst >> 21) & 0x3FF) << 1), 21);
            snprintf(inst_line, sizeof(inst_line), "jal %s, %d", REG_NAMES[rd], imm);
        }
        break;

    case 0x67: /* jalr */
        mnemonic = "jalr";
        {
            int32_t imm = sext(inst >> 20, 12);
            snprintf(inst_line, sizeof(inst_line), "jalr %s, %s, %d",
                     REG_NAMES[rd], REG_NAMES[rs1], imm);
            has_rs1 = 1;
        }
        break;

    case 0x37: /* lui */
        mnemonic = "lui";
        {
            uint32_t imm = (inst >> 12) << 12;
            snprintf(inst_line, sizeof(inst_line), "lui %s, %08x", REG_NAMES[rd], imm);
        }
        break;

    case 0x17: /* auipc */
        mnemonic = "auipc";
        {
            uint32_t imm = (inst >> 12) << 12;
            snprintf(inst_line, sizeof(inst_line), "auipc %s, %08x", REG_NAMES[rd], imm);
        }
        break;

    case 0x73: /* ecall */
        mnemonic = "ecall";
        snprintf(inst_line, sizeof(inst_line), "ecall");
        break;

    default:
        snprintf(inst_line, sizeof(inst_line), "unknown_%02x", op);
        break;
    }

    if (mnem_buf && mnem_sz) {
        strncpy(mnem_buf, mnemonic, mnem_sz);
        mnem_buf[mnem_sz - 1] = '\0';
    }

    if (prompt_buf && prompt_sz) {
        char *p = prompt_buf;
        size_t rem = prompt_sz;
        int n = snprintf(p, rem, "[CMD] STEP\n[PC] %08x\n[INST] %s", cpu->pc, inst_line);
        if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        if (has_rs1) {
            n = snprintf(p, rem, "\n[REG %s] %08x", REG_NAMES[rs1], cpu->regs[rs1]);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
        if (has_rs2) {
            n = snprintf(p, rem, "\n[REG %s] %08x", REG_NAMES[rs2], cpu->regs[rs2]);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
        if (has_mem) {
            n = snprintf(p, rem, "\n[MEM %08x] %08x", mem_addr, mem_raw);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
    }
    return 1;
}

int rv32i_step_trace(RV32I_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                     char *target_buf, size_t target_sz) {
    char mnem[32];
    int ok = rv32i_decode_prompt(cpu, prompt_buf, prompt_sz, mnem, sizeof(mnem));
    if (!ok) {
        if (target_buf && target_sz)
            snprintf(target_buf, target_sz, "[NPC] %08x", cpu->pc);
        return 0;
    }

    rv32i_step(cpu);

    if (target_buf && target_sz) {
        char *p = target_buf;
        size_t rem = target_sz;
        int n = snprintf(p, rem, "[NPC] %08x", cpu->pc);
        if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        if (cpu->has_write_reg && cpu->write_reg_rd > 0) {
            n = snprintf(p, rem, "\n[W_REG %s] %08x", REG_NAMES[cpu->write_reg_rd], cpu->write_reg_val);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
        if (cpu->has_write_mem) {
            n = snprintf(p, rem, "\n[W_MEM %08x] %08x", cpu->write_mem_addr, cpu->write_mem_val);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
    }
    return 1;
}

void rv32i_apply_prediction(RV32I_CPU *cpu, uint32_t npc, int rd, uint32_t reg_val,
                            int has_mem, uint32_t mem_addr, uint32_t mem_val, int mem_size) {
    cpu->steps++;
    cpu->timer_ms = (uint32_t)(cpu->steps / TIMER_STEPS_PER_MS);
    if (rd > 0 && rd < 32) {
        cpu->regs[rd] = reg_val;
    }
    if (has_mem) {
        rv32i_mem_write(cpu, mem_addr, mem_size, mem_val);
    }
    cpu->pc = npc;
}

int rv32i_save_checkpoint(const RV32I_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    CheckpointHeader hdr;
    memcpy(hdr.magic, "RV32", 4);
    hdr.version = 1;
    hdr.steps = cpu->steps;
    hdr.pc = cpu->pc;
    hdr.timer_ms = cpu->timer_ms;
    hdr.keyboard = cpu->keyboard;
    hdr.halted = (uint32_t)cpu->halted;
    memcpy(hdr.regs, cpu->regs, sizeof(cpu->regs));
    hdr.ram_size = RAM_SIZE;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -2; }
    if (fwrite(cpu->ram, 1, RAM_SIZE, f) != RAM_SIZE) { fclose(f); return -3; }
    fclose(f);
    return 0;
}

int rv32i_load_checkpoint(RV32I_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    CheckpointHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -2; }
    if (memcmp(hdr.magic, "RV32", 4) != 0 || hdr.version != 1) { fclose(f); return -3; }
    cpu->steps = hdr.steps;
    cpu->pc = hdr.pc;
    cpu->timer_ms = hdr.timer_ms;
    cpu->keyboard = hdr.keyboard;
    cpu->halted = (int)hdr.halted;
    memcpy(cpu->regs, hdr.regs, sizeof(cpu->regs));
    cpu->regs[0] = 0;
    if (fread(cpu->ram, 1, RAM_SIZE, f) != RAM_SIZE) { fclose(f); return -4; }
    fclose(f);
    return 0;
}

void rv32i_dump_frame(const RV32I_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", SCREEN_W, SCREEN_H);
    for (int p = 0; p < SCREEN_W * SCREEN_H; p++) {
        uint32_t px;
        memcpy(&px, cpu->ram + VRAM_OFFSET + p * 4, 4);
        uint8_t rgb[3] = { (uint8_t)(px >> 16), (uint8_t)(px >> 8), (uint8_t)px };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

void rv32i_get_frame_rgb(const RV32I_CPU *cpu, uint8_t *rgb_out) {
    for (int p = 0; p < SCREEN_W * SCREEN_H; p++) {
        uint32_t px;
        memcpy(&px, cpu->ram + VRAM_OFFSET + p * 4, 4);
        rgb_out[p * 3]     = (uint8_t)(px >> 16);
        rgb_out[p * 3 + 1] = (uint8_t)(px >> 8);
        rgb_out[p * 3 + 2] = (uint8_t)px;
    }
}

void rv32i_get_frame_rgba(const RV32I_CPU *cpu, uint32_t *rgba_out) {
    if (!cpu || !rgba_out) return;
    for (int p = 0; p < SCREEN_W * SCREEN_H; p++) {
        uint32_t px;
        memcpy(&px, cpu->ram + VRAM_OFFSET + p * 4, 4);
        uint32_t r = (px >> 16) & 0xff;
        uint32_t g = (px >> 8) & 0xff;
        uint32_t b = px & 0xff;
        rgba_out[p] = 0xFF000000u | (b << 16) | (g << 8) | r;
    }
}

uint8_t *rv32i_get_vram_ptr(RV32I_CPU *cpu) {
    return cpu ? (cpu->ram + VRAM_OFFSET) : NULL;
}

uint32_t rv32i_run_steps(RV32I_CPU *cpu, uint32_t count) {
    if (!cpu) return 0;
    uint32_t i;
    for (i = 0; i < count; i++) {
        if (cpu->halted) break;
        rv32i_step(cpu);
    }
    return i;
}

uint32_t rv32i_get_pc(const RV32I_CPU *cpu) { return cpu->pc; }
void rv32i_set_pc(RV32I_CPU *cpu, uint32_t pc) { cpu->pc = pc; }

uint64_t rv32i_get_steps(const RV32I_CPU *cpu) { return cpu->steps; }
double rv32i_get_steps_double(const RV32I_CPU *cpu) { return (double)cpu->steps; }
void rv32i_set_steps(RV32I_CPU *cpu, uint64_t steps) {
    cpu->steps = steps;
    cpu->timer_ms = (uint32_t)(steps / TIMER_STEPS_PER_MS);
}

uint32_t rv32i_get_reg(const RV32I_CPU *cpu, int idx) {
    if (idx <= 0 || idx >= 32) return 0;
    return cpu->regs[idx];
}
void rv32i_set_reg(RV32I_CPU *cpu, int idx, uint32_t val) {
    if (idx > 0 && idx < 32) cpu->regs[idx] = val;
}

int rv32i_is_halted(const RV32I_CPU *cpu) { return cpu->halted; }
void rv32i_set_halted(RV32I_CPU *cpu, int halted) { cpu->halted = halted; }

uint32_t rv32i_get_timer_ms(const RV32I_CPU *cpu) { return cpu->timer_ms; }
void rv32i_set_timer_ms(RV32I_CPU *cpu, uint32_t ms) { cpu->timer_ms = ms; }

uint32_t rv32i_get_keyboard(const RV32I_CPU *cpu) { return cpu->keyboard; }
void rv32i_set_keyboard(RV32I_CPU *cpu, uint32_t k) { cpu->keyboard = k; }

uint8_t *rv32i_get_ram_ptr(RV32I_CPU *cpu) { return cpu->ram; }
size_t rv32i_get_ram_size(void) { return RAM_SIZE; }

int rv32i_step_block(RV32I_CPU *cpu, int max_instructions) {
    if (!cpu || cpu->halted) return 0;
    int count = 0;
    while (!cpu->halted && count < max_instructions) {
        if (cpu->pc < RAM_BASE || cpu->pc + 4 > RAM_BASE + RAM_SIZE) {
            cpu->halted = 1;
            break;
        }
        uint32_t inst;
        memcpy(&inst, cpu->ram + (cpu->pc - RAM_BASE), 4);
        if (inst == 0) {
            cpu->halted = 1;
            break;
        }
        uint32_t op = inst & 0x7F;
        rv32i_step(cpu);
        count++;
        /* Check if instruction was a branch / jump / ecall (terminates basic block) */
        if (op == 0x6F || op == 0x67 || op == 0x63 || op == 0x73) {
            break;
        }
    }
    return count;
}

int rv32i_get_write_reg(const RV32I_CPU *cpu, int *rd, uint32_t *val) {
    if (!cpu || !cpu->has_write_reg) return 0;
    if (rd) *rd = cpu->write_reg_rd;
    if (val) *val = cpu->write_reg_val;
    return 1;
}

int rv32i_get_write_mem(const RV32I_CPU *cpu, uint32_t *addr, uint32_t *val, int *size) {
    if (!cpu || !cpu->has_write_mem) return 0;
    if (addr) *addr = cpu->write_mem_addr;
    if (val) *val = cpu->write_mem_val;
    if (size) *size = cpu->write_mem_size;
    return 1;
}

int rv32i_step_superblock(RV32I_CPU *cpu, uint32_t max_insts, uint32_t *out_info) {
    if (!cpu || cpu->halted) return 0;
    uint32_t start_pc = cpu->pc;
    uint32_t term_pc = cpu->pc;
    uint32_t term_inst = 0;
    uint32_t branch_target = 0;
    uint32_t fallthrough_pc = cpu->pc + 4;
    uint32_t inst_count = 0;
    uint32_t is_branch = 0;
    uint32_t term_op = 0;

    while (!cpu->halted && inst_count < max_insts) {
        if (cpu->pc < RAM_BASE || cpu->pc + 4 > RAM_BASE + RAM_SIZE) {
            cpu->halted = 1;
            break;
        }
        uint32_t inst;
        memcpy(&inst, cpu->ram + (cpu->pc - RAM_BASE), 4);
        if (inst == 0) {
            cpu->halted = 1;
            break;
        }
        uint32_t op = inst & 0x7F;

        if (op == 0x63) {
            /* Conditional branch: stop BEFORE stepping so neural network predicts it */
            term_pc = cpu->pc;
            term_inst = inst;
            is_branch = 1;
            term_op = op;
            int32_t imm = sext(((inst >> 31) << 12) | (((inst >> 7) & 1) << 11) |
                               (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1), 13);
            branch_target = term_pc + (uint32_t)imm;
            fallthrough_pc = term_pc + 4;
            break;
        } else if (op == 0x6F || op == 0x67 || op == 0x73) {
            /* JAL / JALR / ECALL: step in CPU to preserve link register / call stack */
            term_pc = cpu->pc;
            term_inst = inst;
            term_op = op;
            rv32i_step(cpu);
            inst_count++;
            fallthrough_pc = cpu->pc;
            break;
        }

        /* Deterministic instruction: ALU, Load, Store, LUI, AUIPC */
        rv32i_step(cpu);
        inst_count++;
    }

    if (out_info) {
        out_info[0] = start_pc;
        out_info[1] = term_pc;
        out_info[2] = term_inst;
        out_info[3] = branch_target;
        out_info[4] = fallthrough_pc;
        out_info[5] = inst_count;
        out_info[6] = is_branch;
        out_info[7] = term_op;
    }
    return (int)inst_count;
}

void rv32i_advance_branch(RV32I_CPU *cpu, uint32_t next_pc) {
    if (!cpu || cpu->halted) return;
    cpu->pc = next_pc;
    cpu->steps++;
    cpu->timer_ms = (uint32_t)(cpu->steps / TIMER_STEPS_PER_MS);
}

int rv32i_step_superblock_neural(RV32I_CPU *cpu, void *state, uint32_t max_insts, uint32_t *out_info) {
    if (!cpu || cpu->halted) return 0;

    /* 1. Step deterministic instructions in C up to conditional branch or chunk boundary */
    uint32_t sb_info[8];
    int inst_count = rv32i_step_superblock(cpu, max_insts, sb_info);
    if (cpu->halted) return inst_count;

    uint32_t start_pc = sb_info[0];
    uint32_t term_pc = sb_info[1];
    uint32_t term_inst = sb_info[2];
    uint32_t branch_target = sb_info[3];
    uint32_t fallthrough_pc = sb_info[4];
    uint32_t is_branch = sb_info[6];
    uint32_t term_op = sb_info[7];

    float branch_logit = 0.0f;
    float pc_offset = 0.0f;
    uint32_t ground_truth_taken = 0;
    uint32_t neural_taken = 0;
    uint32_t is_correct = 0;
    uint32_t next_pc = cpu->pc;

    /* 2. Run embedded Mamba-Nano forward pass in C (~13 µs) */
    if (state) {
        mamba_nano_step_from_pc_regs((MambaNanoState *)state, term_pc, cpu->regs, &branch_logit, &pc_offset);
    }

    float branch_prob = 1.0f / (1.0f + expf(-branch_logit));
    neural_taken = (branch_prob > 0.5f) ? 1 : 0;

    /* 3. Handle terminating instruction */
    if (is_branch && term_inst != 0) {
        if (term_op == 0x63) {
            uint32_t funct3 = (term_inst >> 12) & 7;
            uint32_t rs1 = (term_inst >> 15) & 0x1F;
            uint32_t rs2 = (term_inst >> 20) & 0x1F;
            uint32_t val1 = cpu->regs[rs1];
            uint32_t val2 = cpu->regs[rs2];
            int32_t sval1 = (int32_t)val1;
            int32_t sval2 = (int32_t)val2;

            if (funct3 == 0) ground_truth_taken = (val1 == val2);       /* BEQ */
            else if (funct3 == 1) ground_truth_taken = (val1 != val2);  /* BNE */
            else if (funct3 == 4) ground_truth_taken = (sval1 < sval2);  /* BLT */
            else if (funct3 == 5) ground_truth_taken = (sval1 >= sval2); /* BGE */
            else if (funct3 == 6) ground_truth_taken = (val1 < val2);    /* BLTU */
            else if (funct3 == 7) ground_truth_taken = (val1 >= val2);   /* BGEU */

            is_correct = (neural_taken == ground_truth_taken) ? 1 : 0;

            /* Speculative branch retirement: retire verified ground truth */
            next_pc = ground_truth_taken ? branch_target : fallthrough_pc;
            if (next_pc < RAM_BASE || next_pc >= RAM_BASE + RAM_SIZE || (next_pc & 3) != 0) {
                next_pc = fallthrough_pc;
            }

            cpu->pc = next_pc;
            cpu->steps++;
            cpu->timer_ms = (uint32_t)(cpu->steps / TIMER_STEPS_PER_MS);
            inst_count++;
        } else {
            /* JAL / JALR / ECALL was already stepped inside rv32i_step_superblock */
            next_pc = cpu->pc;
        }
    } else {
        next_pc = cpu->pc;
    }

    /* 4. Tier 2: Dynamic Mega-Block Chaining */
    if (!cpu->halted && (uint32_t)inst_count < max_insts) {
        /* Continue chaining consecutive instructions / loops up to max_insts */
        while (!cpu->halted && (uint32_t)inst_count < max_insts) {
            if (cpu->pc < RAM_BASE || cpu->pc + 4 > RAM_BASE + RAM_SIZE) {
                cpu->halted = 1;
                break;
            }
            uint32_t inst;
            memcpy(&inst, cpu->ram + (cpu->pc - RAM_BASE), 4);
            if (inst == 0) {
                cpu->halted = 1;
                break;
            }
            uint32_t op = inst & 0x7F;

            if (op == 0x63) {
                /* Conditional branch in chained path: resolve immediately via ground truth in C */
                uint32_t funct3 = (inst >> 12) & 7;
                uint32_t rs1 = (inst >> 15) & 0x1F;
                uint32_t rs2 = (inst >> 20) & 0x1F;
                uint32_t val1 = cpu->regs[rs1];
                uint32_t val2 = cpu->regs[rs2];
                int32_t sval1 = (int32_t)val1;
                int32_t sval2 = (int32_t)val2;
                int taken = 0;

                if (funct3 == 0) taken = (val1 == val2);       /* BEQ */
                else if (funct3 == 1) taken = (val1 != val2);  /* BNE */
                else if (funct3 == 4) taken = (sval1 < sval2);  /* BLT */
                else if (funct3 == 5) taken = (sval1 >= sval2); /* BGE */
                else if (funct3 == 6) taken = (val1 < val2);    /* BLTU */
                else if (funct3 == 7) taken = (val1 >= val2);   /* BGEU */

                int32_t imm = sext(((inst >> 31) << 12) | (((inst >> 7) & 1) << 11) |
                                   (((inst >> 25) & 0x3F) << 5) | (((inst >> 8) & 0xF) << 1), 13);
                uint32_t target = cpu->pc + (uint32_t)imm;
                uint32_t fallthrough = cpu->pc + 4;
                uint32_t b_next = taken ? target : fallthrough;

                if (b_next < RAM_BASE || b_next >= RAM_BASE + RAM_SIZE || (b_next & 3) != 0) {
                    b_next = fallthrough;
                }

                cpu->pc = b_next;
                cpu->steps++;
                cpu->timer_ms = (uint32_t)(cpu->steps / TIMER_STEPS_PER_MS);
                inst_count++;
            } else if (op == 0x6F || op == 0x67 || op == 0x73) {
                /* JAL / JALR / ECALL: step in CPU */
                rv32i_step(cpu);
                inst_count++;
            } else {
                /* ALU / Load / Store / LUI / AUIPC: deterministic instruction */
                rv32i_step(cpu);
                inst_count++;
            }
        }
        next_pc = cpu->pc;
    }

    if (out_info) {
        out_info[0] = start_pc;
        out_info[1] = term_pc;
        out_info[2] = term_inst;
        out_info[3] = branch_target;
        out_info[4] = next_pc;
        out_info[5] = (uint32_t)inst_count;
        out_info[6] = is_branch;
        out_info[7] = term_op;
        union { float f; uint32_t u; } conv_logit, conv_prob, conv_norm;
        conv_logit.f = branch_logit;
        conv_prob.f  = branch_prob;
        conv_norm.f  = state ? mamba_nano_get_state_norm((const MambaNanoState *)state) : 0.0f;
        out_info[8]  = conv_logit.u;
        out_info[9]  = conv_prob.u;
        out_info[10] = ground_truth_taken;
        out_info[11] = is_correct;
        out_info[12] = neural_taken;
        out_info[13] = conv_norm.u;
        out_info[14] = 0;  /* bimodal fallback: RV32I head is model-only */
        out_info[15] = 0;
    }

    return inst_count;
}

int rv32i_step_superblock_neural_burst(RV32I_CPU *cpu, void *state,
                                      uint32_t num_blocks, uint32_t max_insts_per_block,
                                      uint32_t *out_last_info, uint32_t *out_stats) {
    if (!cpu || cpu->halted) return 0;
    uint32_t total_insts = 0;
    uint32_t total_branches = 0;
    uint32_t correct_branches = 0;
    uint32_t total_taken = 0;
    uint32_t total_fallthrough = 0;

    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t info[16];
        int count = rv32i_step_superblock_neural(cpu, state, max_insts_per_block, info);
        total_insts += (uint32_t)count;

        if (info[6] && info[7] == 0x63) {
            total_branches++;
            if (info[11]) correct_branches++;
            if (info[12]) total_taken++;
            else total_fallthrough++;
        }

        if (b == num_blocks - 1 && out_last_info) {
            memcpy(out_last_info, info, sizeof(uint32_t) * 16);
        }

        if (cpu->halted) break;
    }

    if (out_stats) {
        out_stats[0] = total_insts;
        out_stats[1] = total_branches;
        out_stats[2] = correct_branches;
        out_stats[3] = total_taken;
        out_stats[4] = total_fallthrough;
        out_stats[5] = 0;
        out_stats[6] = 0;
        out_stats[7] = 0;
    }

    return (int)total_insts;
}



