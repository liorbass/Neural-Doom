#include "arm32.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

const char *const ARM32_REG_NAMES[16] = {
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
};

const char *const ARM32_COND_NAMES[16] = {
    "eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc",
    "hi", "ls", "ge", "lt", "gt", "le", "", "nv"
};

static inline uint32_t ror32(uint32_t val, uint32_t amt) {
    amt &= 31;
    return amt ? ((val >> amt) | (val << (32 - amt))) : val;
}

static inline int check_cond(uint32_t cpsr, uint32_t cond) {
    int n = (cpsr & ARM32_FLAG_N) != 0;
    int z = (cpsr & ARM32_FLAG_Z) != 0;
    int c = (cpsr & ARM32_FLAG_C) != 0;
    int v = (cpsr & ARM32_FLAG_V) != 0;

    switch (cond) {
    case ARM32_COND_EQ: return z;
    case ARM32_COND_NE: return !z;
    case ARM32_COND_CS: return c;
    case ARM32_COND_CC: return !c;
    case ARM32_COND_MI: return n;
    case ARM32_COND_PL: return !n;
    case ARM32_COND_VS: return v;
    case ARM32_COND_VC: return !v;
    case ARM32_COND_HI: return c && !z;
    case ARM32_COND_LS: return !c || z;
    case ARM32_COND_GE: return (n == v);
    case ARM32_COND_LT: return (n != v);
    case ARM32_COND_GT: return !z && (n == v);
    case ARM32_COND_LE: return z || (n != v);
    case ARM32_COND_AL: return 1;
    case ARM32_COND_NV: return 0;
    default: return 0;
    }
}

ARM32_CPU *arm32_create(void) {
    ARM32_CPU *cpu = (ARM32_CPU *)calloc(1, sizeof(ARM32_CPU));
    if (!cpu) return NULL;
    cpu->ram = (uint8_t *)calloc(1, ARM32_RAM_SIZE);
    if (!cpu->ram) {
        free(cpu);
        return NULL;
    }
    arm32_init(cpu);
    return cpu;
}

void arm32_free(ARM32_CPU *cpu) {
    if (cpu) {
        if (cpu->ram) free(cpu->ram);
        free(cpu);
    }
}

void arm32_init(ARM32_CPU *cpu) {
    memset(cpu->r, 0, sizeof(cpu->r));
    cpu->r[13] = ARM32_RAM_BASE + ARM32_RAM_SIZE - 0x1000u; /* SP */
    cpu->r[15] = ARM32_RAM_BASE;                            /* PC */
    cpu->cpsr = 0x00000010u;                                /* User mode */
    cpu->steps = 0;
    cpu->timer_ms = 0;
    cpu->keyboard = 0;
    cpu->halted = 0;
    cpu->nev = 0;
    cpu->gametic_addr = 0;
    cpu->gamestate_addr = 0;
}

int arm32_load_binary(ARM32_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (size_t)sz > ARM32_RAM_SIZE) {
        fclose(f);
        return -1;
    }
    size_t rd = fread(cpu->ram, 1, sz, f);
    fclose(f);
    return (int)rd;
}

int arm32_load_binary_data(ARM32_CPU *cpu, const uint8_t *data, size_t size) {
    if (!cpu || !cpu->ram || size > ARM32_RAM_SIZE) return -1;
    memcpy(cpu->ram, data, size);
    return (int)size;
}

uint32_t arm32_mem_read(ARM32_CPU *cpu, uint32_t addr, int size) {
    if (addr >= 0xFFFF0000u) {
        if (addr == ARM32_MMIO_KBD) {
            uint32_t v = cpu->keyboard;
            cpu->keyboard = 0;
            if (v == 0 && cpu->nev > 0) {
                ARM32_KeyEvent e = cpu->evq[0];
                if (e.at <= cpu->steps && (!e.has_cond || arm32_probe_gs(cpu) == e.cond_val)) {
                    memmove(cpu->evq, cpu->evq + 1, --cpu->nev * sizeof(ARM32_KeyEvent));
                    v = e.val;
                }
            }
            return v;
        }
        if (addr == ARM32_MMIO_TIMER) {
            return cpu->timer_ms;
        }
        return 0;
    }

    uint32_t offset = addr;
    if (offset >= 0x80000000u) offset -= 0x80000000u;
    if (offset + (uint32_t)size > ARM32_RAM_SIZE) return 0;

    const uint8_t *p = cpu->ram + offset;
    switch (size) {
    case 1: return *p;
    case 2: {
        uint16_t v;
        memcpy(&v, p, 2);
        return v;
    }
    case 4: {
        uint32_t v;
        memcpy(&v, p, 4);
        return v;
    }
    default: return 0;
    }
}

uint32_t arm32_mem_peek(const ARM32_CPU *cpu, uint32_t addr, int size) {
    if (addr >= 0xFFFF0000u) {
        if (addr == ARM32_MMIO_KBD) return cpu->keyboard;
        if (addr == ARM32_MMIO_TIMER) return cpu->timer_ms;
        return 0;
    }
    uint32_t offset = addr;
    if (offset >= 0x80000000u) offset -= 0x80000000u;
    if (offset + (uint32_t)size > ARM32_RAM_SIZE) return 0;

    const uint8_t *p = cpu->ram + offset;
    switch (size) {
    case 1: return *p;
    case 2: {
        uint16_t v;
        memcpy(&v, p, 2);
        return v;
    }
    case 4: {
        uint32_t v;
        memcpy(&v, p, 4);
        return v;
    }
    default: return 0;
    }
}

void arm32_mem_write(ARM32_CPU *cpu, uint32_t addr, int size, uint32_t val) {
    cpu->has_write_mem = 1;
    cpu->write_mem_addr = addr;
    cpu->write_mem_val = val;
    cpu->write_mem_size = size;

    if (addr >= 0xFFFF0000u) {
        if (addr == ARM32_MMIO_KBD) {
            cpu->keyboard = val;
        }
        return;
    }

    uint32_t offset = addr;
    if (offset >= 0x80000000u) offset -= 0x80000000u;
    if (offset + (uint32_t)size > ARM32_RAM_SIZE) return;

    uint8_t *p = cpu->ram + offset;
    switch (size) {
    case 1: *p = (uint8_t)val; break;
    case 2: {
        uint16_t v = (uint16_t)val;
        memcpy(p, &v, 2);
        break;
    }
    case 4: {
        uint32_t v = val;
        memcpy(p, &v, 4);
        break;
    }
    }
}

void arm32_queue_key(ARM32_CPU *cpu, uint64_t at, uint32_t key, int down, int gs_cond) {
    if (cpu->nev < 64) {
        cpu->evq[cpu->nev].at = at;
        cpu->evq[cpu->nev].val = key | (down ? 0x100u : 0);
        cpu->evq[cpu->nev].has_cond = (gs_cond >= 0);
        cpu->evq[cpu->nev].cond_val = (gs_cond < 0 ? 0 : (uint32_t)gs_cond);
        cpu->nev++;
    }
}

void arm32_push_key(ARM32_CPU *cpu, uint32_t key, int down) {
    arm32_queue_key(cpu, 0, key, down, -1);
}

void arm32_clear_keys(ARM32_CPU *cpu) {
    cpu->nev = 0;
}

uint32_t arm32_probe_gs(const ARM32_CPU *cpu) {
    if (cpu->gamestate_addr == 0 || cpu->gamestate_addr >= ARM32_RAM_SIZE)
        return 0;
    return *(const uint32_t *)(cpu->ram + cpu->gamestate_addr);
}

uint32_t arm32_probe_gametic(const ARM32_CPU *cpu) {
    if (cpu->gametic_addr == 0 || cpu->gametic_addr >= ARM32_RAM_SIZE)
        return 0;
    return *(const uint32_t *)(cpu->ram + cpu->gametic_addr);
}

/* Barrel shifter helper */
static uint32_t shift_operand(ARM32_CPU *cpu, uint32_t inst, int *carry_out) {
    uint32_t type = (inst >> 5) & 3;
    uint32_t rm_idx = inst & 0xF;
    uint32_t rm = (rm_idx == 15) ? (cpu->r[15] + 8) : cpu->r[rm_idx];
    uint32_t shift_amt;

    if ((inst & (1 << 4)) == 0) {
        /* Shift by immediate */
        shift_amt = (inst >> 7) & 0x1F;
    } else {
        /* Shift by register */
        uint32_t rs_idx = (inst >> 8) & 0xF;
        shift_amt = cpu->r[rs_idx] & 0xFF;
    }

    uint32_t res = rm;
    int c = (cpu->cpsr & ARM32_FLAG_C) != 0;

    switch (type) {
    case 0: /* LSL */
        if (shift_amt == 0) {
            /* No shift, carry unchanged */
        } else if (shift_amt < 32) {
            c = (rm >> (32 - shift_amt)) & 1;
            res = rm << shift_amt;
        } else if (shift_amt == 32) {
            c = rm & 1;
            res = 0;
        } else {
            c = 0;
            res = 0;
        }
        break;

    case 1: /* LSR */
        if ((inst & (1 << 4)) == 0 && shift_amt == 0) shift_amt = 32; /* LSR #0 is LSR #32 */
        if (shift_amt == 0) {
            /* No shift */
        } else if (shift_amt < 32) {
            c = (rm >> (shift_amt - 1)) & 1;
            res = rm >> shift_amt;
        } else if (shift_amt == 32) {
            c = (rm >> 31) & 1;
            res = 0;
        } else {
            c = 0;
            res = 0;
        }
        break;

    case 2: /* ASR */
        if ((inst & (1 << 4)) == 0 && shift_amt == 0) shift_amt = 32; /* ASR #0 is ASR #32 */
        if (shift_amt == 0) {
            /* No shift */
        } else if (shift_amt < 32) {
            c = ((int32_t)rm >> (shift_amt - 1)) & 1;
            res = (uint32_t)((int32_t)rm >> shift_amt);
        } else {
            int sign = (rm >> 31) & 1;
            c = sign;
            res = sign ? 0xFFFFFFFFu : 0;
        }
        break;

    case 3: /* ROR / RRX */
        if ((inst & (1 << 4)) == 0 && shift_amt == 0) {
            /* RRX: Rotate right through carry */
            int old_c = c;
            c = rm & 1;
            res = (rm >> 1) | ((uint32_t)old_c << 31);
        } else {
            shift_amt &= 31;
            if (shift_amt > 0) {
                c = (rm >> (shift_amt - 1)) & 1;
                res = ror32(rm, shift_amt);
            }
        }
        break;
    }

    if (carry_out) *carry_out = c;
    return res;
}

int arm32_step(ARM32_CPU *cpu) {
    if (cpu->halted) return 0;

    uint32_t pc = cpu->r[15];
    uint32_t offset = pc;
    if (offset >= 0x80000000u) offset -= 0x80000000u;
    if (offset + 4 > ARM32_RAM_SIZE) {
        cpu->halted = 1;
        return 0;
    }

    uint32_t inst;
    memcpy(&inst, cpu->ram + offset, 4);

    cpu->has_write_reg = 0;
    cpu->has_write_mem = 0;
    cpu->has_write_cpsr = 0;
    cpu->cond_passed = 0;

    uint32_t cond = (inst >> 28) & 0xF;
    if (!check_cond(cpu->cpsr, cond)) {
        /* Predicated: skip instruction */
        cpu->r[15] += 4;
        cpu->steps++;
        return 1;
    }
    cpu->cond_passed = 1;

    uint32_t next_pc = pc + 4;
    int pc_written = 0;

    /* Branch & Branch with Link (B, BL) */
    if (((inst >> 25) & 7) == 5) {
        int link = (inst >> 24) & 1;
        int32_t imm24 = (int32_t)(inst & 0x00FFFFFF);
        if (imm24 & 0x00800000) imm24 |= 0xFF000000; /* Sign extend */
        int32_t target_offset = (imm24 << 2) + 8;    /* ARM pipeline offset: +8 */

        if (link) {
            cpu->r[14] = pc + 4;
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = 14;
            cpu->write_reg_val = pc + 4;
        }

        next_pc = pc + (uint32_t)target_offset;
        pc_written = 1;
    }
    /* Branch and Exchange (BX) */
    else if ((inst & 0x0FFFFFF0u) == 0x012FFF10u) {
        uint32_t rm = inst & 0xF;
        uint32_t target = cpu->r[rm];
        if (target & 1) {
            cpu->cpsr |= ARM32_FLAG_T; /* Switch to Thumb */
            target &= ~1u;
        }
        next_pc = target;
        pc_written = 1;
    }
    /* Multiply and Multiply-Accumulate */
    else if (((inst >> 22) & 0x3F) == 0 && ((inst >> 4) & 0xF) == 9) {
        uint32_t rd = (inst >> 16) & 0xF;
        uint32_t rn = (inst >> 12) & 0xF;
        uint32_t rs = (inst >> 8) & 0xF;
        uint32_t rm = inst & 0xF;
        int a = (inst >> 21) & 1;
        int s = (inst >> 20) & 1;

        uint32_t res = cpu->r[rm] * cpu->r[rs];
        if (a) res += cpu->r[rn];

        cpu->r[rd] = res;
        cpu->has_write_reg = 1;
        cpu->write_reg_rd = rd;
        cpu->write_reg_val = res;

        if (s) {
            uint32_t flags = cpu->cpsr & ~(ARM32_FLAG_N | ARM32_FLAG_Z);
            if (res & 0x80000000u) flags |= ARM32_FLAG_N;
            if (res == 0) flags |= ARM32_FLAG_Z;
            cpu->cpsr = flags;
            cpu->has_write_cpsr = 1;
            cpu->write_cpsr_val = flags;
        }
    }
    /* Block Data Transfer (LDM / STM) */
    else if (((inst >> 25) & 7) == 4) {
        uint32_t rn_idx = (inst >> 16) & 0xF;
        uint32_t rn = cpu->r[rn_idx];
        int p = (inst >> 24) & 1;
        int u = (inst >> 23) & 1;
        int w = (inst >> 21) & 1;
        int l = (inst >> 20) & 1;
        uint32_t reg_list = inst & 0xFFFF;

        int count = 0;
        for (int i = 0; i < 16; i++) {
            if (reg_list & (1 << i)) count++;
        }

        uint32_t addr = rn;
        if (!u) addr -= (uint32_t)(count * 4);
        uint32_t writeback_val = u ? (rn + (uint32_t)(count * 4)) : (rn - (uint32_t)(count * 4));

        uint32_t cur = addr;
        for (int i = 0; i < 16; i++) {
            if (reg_list & (1 << i)) {
                if (p && u) cur += 4;
                if (!p && !u) cur += 4;

                if (l) {
                    uint32_t val = arm32_mem_read(cpu, cur, 4);
                    cpu->r[i] = val;
                    if (i == 15) {
                        next_pc = val;
                        pc_written = 1;
                    }
                    cpu->has_write_reg = 1;
                    cpu->write_reg_rd = i;
                    cpu->write_reg_val = val;
                } else {
                    uint32_t val = (i == 15) ? (pc + 12) : cpu->r[i];
                    arm32_mem_write(cpu, cur, 4, val);
                }

                if (!p && u) cur += 4;
                if (p && !u) cur += 4;
            }
        }

        if (w && !(l && (reg_list & (1 << rn_idx)))) {
            cpu->r[rn_idx] = writeback_val;
        }
    }
    /* Single Data Transfer (LDR / STR) */
    else if (((inst >> 26) & 3) == 1) {
        int i_bit = (inst >> 25) & 1;
        int p_bit = (inst >> 24) & 1;
        int u_bit = (inst >> 23) & 1;
        int b_bit = (inst >> 22) & 1;
        int w_bit = (inst >> 21) & 1;
        int l_bit = (inst >> 20) & 1;
        uint32_t rn_idx = (inst >> 16) & 0xF;
        uint32_t rd_idx = (inst >> 12) & 0xF;
        uint32_t rn = (rn_idx == 15) ? (pc + 8) : cpu->r[rn_idx];

        uint32_t offset_val;
        if (!i_bit) {
            offset_val = inst & 0xFFF;
        } else {
            offset_val = shift_operand(cpu, inst, NULL);
        }

        uint32_t target_addr = p_bit ? (u_bit ? (rn + offset_val) : (rn - offset_val)) : rn;

        if (l_bit) {
            /* LDR / LDRB */
            uint32_t val = arm32_mem_read(cpu, target_addr, b_bit ? 1 : 4);
            cpu->r[rd_idx] = val;
            if (rd_idx == 15) {
                next_pc = val;
                pc_written = 1;
            }
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = rd_idx;
            cpu->write_reg_val = val;
        } else {
            /* STR / STRB */
            uint32_t val = (rd_idx == 15) ? (pc + 12) : cpu->r[rd_idx];
            arm32_mem_write(cpu, target_addr, b_bit ? 1 : 4, val);
        }

        if (!p_bit) {
            cpu->r[rn_idx] = u_bit ? (rn + offset_val) : (rn - offset_val);
        } else if (w_bit) {
            cpu->r[rn_idx] = target_addr;
        }
    }
    /* Halfword / Signed Data Transfer (LDRH, STRH, LDRSB, LDRSH) */
    else if (((inst >> 25) & 7) == 0 && (inst & 0x90) == 0x90 && ((inst >> 4) & 0xF) != 9) {
        int p = (inst >> 24) & 1;
        int u = (inst >> 23) & 1;
        int imm_flag = (inst >> 22) & 1;
        int w = (inst >> 21) & 1;
        int l = (inst >> 20) & 1;
        uint32_t rn_idx = (inst >> 16) & 0xF;
        uint32_t rd_idx = (inst >> 12) & 0xF;
        uint32_t rn = (rn_idx == 15) ? (pc + 8) : cpu->r[rn_idx];
        uint32_t op = (inst >> 5) & 3;

        uint32_t offset_val = imm_flag ? (((inst >> 4) & 0xF0) | (inst & 0x0F)) : cpu->r[inst & 0xF];
        uint32_t target_addr = p ? (u ? (rn + offset_val) : (rn - offset_val)) : rn;

        if (l) {
            uint32_t val = 0;
            if (op == 1) val = arm32_mem_read(cpu, target_addr, 2); /* LDRH */
            else if (op == 2) { /* LDRSB */
                val = arm32_mem_read(cpu, target_addr, 1);
                if (val & 0x80) val |= 0xFFFFFF00u;
            } else if (op == 3) { /* LDRSH */
                val = arm32_mem_read(cpu, target_addr, 2);
                if (val & 0x8000) val |= 0xFFFF0000u;
            }
            cpu->r[rd_idx] = val;
            if (rd_idx == 15) { next_pc = val; pc_written = 1; }
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = rd_idx;
            cpu->write_reg_val = val;
        } else {
            if (op == 1) { /* STRH */
                uint32_t val = (rd_idx == 15) ? (pc + 12) : cpu->r[rd_idx];
                arm32_mem_write(cpu, target_addr, 2, val & 0xFFFF);
            }
        }

        if (!p) cpu->r[rn_idx] = u ? (rn + offset_val) : (rn - offset_val);
        else if (w) cpu->r[rn_idx] = target_addr;
    }
    /* Data Processing (ALU) */
    else if (((inst >> 26) & 3) == 0) {
        int i_bit = (inst >> 25) & 1;
        uint32_t opcode = (inst >> 21) & 0xF;
        int s_bit = (inst >> 20) & 1;
        uint32_t rn_idx = (inst >> 16) & 0xF;
        uint32_t rd_idx = (inst >> 12) & 0xF;
        uint32_t rn = (rn_idx == 15) ? (pc + 8) : cpu->r[rn_idx];

        uint32_t op2;
        int shifter_carry = (cpu->cpsr & ARM32_FLAG_C) != 0;

        if (i_bit) {
            uint32_t imm8 = inst & 0xFF;
            uint32_t rot = (inst >> 8) & 0xF;
            op2 = ror32(imm8, rot * 2);
            if (rot != 0) shifter_carry = (op2 >> 31) & 1;
        } else {
            op2 = shift_operand(cpu, inst, &shifter_carry);
        }

        uint32_t res = 0;
        int write_rd = 1;
        int alu_carry = shifter_carry;
        int alu_overflow = (cpu->cpsr & ARM32_FLAG_V) != 0;

        int carry_in = (cpu->cpsr & ARM32_FLAG_C) != 0;

        switch (opcode) {
        case 0x0: /* AND */ res = rn & op2; break;
        case 0x1: /* EOR */ res = rn ^ op2; break;
        case 0x2: /* SUB */ {
            uint64_t sub = (uint64_t)rn - (uint64_t)op2;
            res = (uint32_t)sub;
            alu_carry = (rn >= op2);
            alu_overflow = ((rn ^ op2) & (rn ^ res)) >> 31;
            break;
        }
        case 0x3: /* RSB */ {
            uint64_t sub = (uint64_t)op2 - (uint64_t)rn;
            res = (uint32_t)sub;
            alu_carry = (op2 >= rn);
            alu_overflow = ((op2 ^ rn) & (op2 ^ res)) >> 31;
            break;
        }
        case 0x4: /* ADD */ {
            uint64_t sum = (uint64_t)rn + (uint64_t)op2;
            res = (uint32_t)sum;
            alu_carry = (sum >> 32) & 1;
            alu_overflow = (~(rn ^ op2) & (rn ^ res)) >> 31;
            break;
        }
        case 0x5: /* ADC */ {
            uint64_t sum = (uint64_t)rn + (uint64_t)op2 + (uint64_t)carry_in;
            res = (uint32_t)sum;
            alu_carry = (sum >> 32) & 1;
            alu_overflow = (~(rn ^ op2) & (rn ^ res)) >> 31;
            break;
        }
        case 0x6: /* SBC */ {
            uint64_t sub = (uint64_t)rn - (uint64_t)op2 - (uint64_t)(!carry_in);
            res = (uint32_t)sub;
            alu_carry = ((uint64_t)rn >= (uint64_t)op2 + (uint64_t)(!carry_in));
            alu_overflow = ((rn ^ op2) & (rn ^ res)) >> 31;
            break;
        }
        case 0x7: /* RSC */ {
            uint64_t sub = (uint64_t)op2 - (uint64_t)rn - (uint64_t)(!carry_in);
            res = (uint32_t)sub;
            alu_carry = ((uint64_t)op2 >= (uint64_t)rn + (uint64_t)(!carry_in));
            alu_overflow = ((op2 ^ rn) & (op2 ^ res)) >> 31;
            break;
        }
        case 0x8: /* TST */ res = rn & op2; write_rd = 0; break;
        case 0x9: /* TEQ */ res = rn ^ op2; write_rd = 0; break;
        case 0xA: /* CMP */ {
            uint64_t sub = (uint64_t)rn - (uint64_t)op2;
            res = (uint32_t)sub;
            alu_carry = (rn >= op2);
            alu_overflow = ((rn ^ op2) & (rn ^ res)) >> 31;
            write_rd = 0;
            break;
        }
        case 0xB: /* CMN */ {
            uint64_t sum = (uint64_t)rn + (uint64_t)op2;
            res = (uint32_t)sum;
            alu_carry = (sum >> 32) & 1;
            alu_overflow = (~(rn ^ op2) & (rn ^ res)) >> 31;
            write_rd = 0;
            break;
        }
        case 0xC: /* ORR */ res = rn | op2; break;
        case 0xD: /* MOV */ res = op2; break;
        case 0xE: /* BIC */ res = rn & ~op2; break;
        case 0xF: /* MVN */ res = ~op2; break;
        }

        if (write_rd) {
            cpu->r[rd_idx] = res;
            if (rd_idx == 15) {
                next_pc = res;
                pc_written = 1;
            }
            cpu->has_write_reg = 1;
            cpu->write_reg_rd = rd_idx;
            cpu->write_reg_val = res;
        }

        if (s_bit) {
            uint32_t flags = cpu->cpsr & ~(ARM32_FLAG_N | ARM32_FLAG_Z | ARM32_FLAG_C | ARM32_FLAG_V);
            if (res & 0x80000000u) flags |= ARM32_FLAG_N;
            if (res == 0) flags |= ARM32_FLAG_Z;
            if (alu_carry) flags |= ARM32_FLAG_C;
            if (alu_overflow) flags |= ARM32_FLAG_V;

            cpu->cpsr = flags;
            cpu->has_write_cpsr = 1;
            cpu->write_cpsr_val = flags;
        }
    }

    if (!pc_written) cpu->r[15] = next_pc;
    else cpu->r[15] = next_pc;

    cpu->steps++;
    return 1;
}

int arm32_run_steps(ARM32_CPU *cpu, int n) {
    int done = 0;
    for (int i = 0; i < n; i++) {
        if (!arm32_step(cpu)) break;
        done++;
    }
    return done;
}

int arm32_decode_prompt(const ARM32_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                        char *mnem_buf, size_t mnem_sz) {
    if (cpu->halted) {
        if (mnem_buf && mnem_sz) strncpy(mnem_buf, "HALTED", mnem_sz);
        if (prompt_buf && prompt_sz)
            snprintf(prompt_buf, prompt_sz, "[CMD] STEP\n[PC] %08x\n[STATE] HALTED", cpu->r[15]);
        return 0;
    }

    uint32_t pc = cpu->r[15];
    uint32_t offset = pc;
    if (offset >= 0x80000000u) offset -= 0x80000000u;
    if (offset + 4 > ARM32_RAM_SIZE) {
        if (mnem_buf && mnem_sz) strncpy(mnem_buf, "HALTED", mnem_sz);
        if (prompt_buf && prompt_sz)
            snprintf(prompt_buf, prompt_sz, "[CMD] STEP\n[PC] %08x\n[STATE] HALTED", pc);
        return 0;
    }

    uint32_t inst;
    memcpy(&inst, cpu->ram + offset, 4);

    uint32_t cond = (inst >> 28) & 0xF;
    const char *cond_str = ARM32_COND_NAMES[cond];

    char inst_line[128] = "unknown";
    char mnem[32] = "unknown";

    /* Disassemble basic instruction classes */
    if (((inst >> 25) & 7) == 5) {
        int link = (inst >> 24) & 1;
        int32_t imm24 = (int32_t)(inst & 0x00FFFFFF);
        if (imm24 & 0x00800000) imm24 |= 0xFF000000;
        uint32_t target = pc + (uint32_t)((imm24 << 2) + 8);
        snprintf(mnem, sizeof(mnem), "%s%s", link ? "bl" : "b", cond_str);
        snprintf(inst_line, sizeof(inst_line), "%s 0x%08x", mnem, target);
    } else if (((inst >> 26) & 3) == 1) {
        int l = (inst >> 20) & 1;
        int b = (inst >> 22) & 1;
        uint32_t rd = (inst >> 12) & 0xF;
        uint32_t rn = (inst >> 16) & 0xF;
        snprintf(mnem, sizeof(mnem), "%s%s%s", l ? "ldr" : "str", b ? "b" : "", cond_str);
        snprintf(inst_line, sizeof(inst_line), "%s %s, [%s]", mnem, ARM32_REG_NAMES[rd], ARM32_REG_NAMES[rn]);
    } else if (((inst >> 26) & 3) == 0) {
        static const char *const ops[16] = {
            "and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc",
            "tst", "teq", "cmp", "cmn", "orr", "mov", "bic", "mvn"
        };
        uint32_t opcode = (inst >> 21) & 0xF;
        int s = (inst >> 20) & 1;
        uint32_t rd = (inst >> 12) & 0xF;
        uint32_t rn = (inst >> 16) & 0xF;
        snprintf(mnem, sizeof(mnem), "%s%s%s", ops[opcode], cond_str, s ? "s" : "");
        if (opcode == 0xD || opcode == 0xF) { /* MOV / MVN */
            snprintf(inst_line, sizeof(inst_line), "%s %s, op2", mnem, ARM32_REG_NAMES[rd]);
        } else if (opcode >= 0x8 && opcode <= 0xB) { /* TST / TEQ / CMP / CMN */
            snprintf(inst_line, sizeof(inst_line), "%s %s, op2", mnem, ARM32_REG_NAMES[rn]);
        } else {
            snprintf(inst_line, sizeof(inst_line), "%s %s, %s, op2", mnem, ARM32_REG_NAMES[rd], ARM32_REG_NAMES[rn]);
        }
    } else {
        snprintf(mnem, sizeof(mnem), "inst_%08x", inst);
        snprintf(inst_line, sizeof(inst_line), "word 0x%08x", inst);
    }

    if (mnem_buf && mnem_sz) strncpy(mnem_buf, mnem, mnem_sz);

    if (prompt_buf && prompt_sz) {
        char *p = prompt_buf;
        size_t rem = prompt_sz;
        int n = snprintf(p, rem, "[CMD] STEP\n[PC] %08x\n[CPSR] %08x (N=%d Z=%d C=%d V=%d)\n[INST] %s",
                         pc, cpu->cpsr,
                         (cpu->cpsr & ARM32_FLAG_N) != 0,
                         (cpu->cpsr & ARM32_FLAG_Z) != 0,
                         (cpu->cpsr & ARM32_FLAG_C) != 0,
                         (cpu->cpsr & ARM32_FLAG_V) != 0,
                         inst_line);
        if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }

        /* Print relevant input registers */
        uint32_t rn = (inst >> 16) & 0xF;
        uint32_t rm = inst & 0xF;
        if (rn < 16) {
            n = snprintf(p, rem, "\n[REG %s] %08x", ARM32_REG_NAMES[rn], cpu->r[rn]);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
        if (rm < 16 && rm != rn) {
            n = snprintf(p, rem, "\n[REG %s] %08x", ARM32_REG_NAMES[rm], cpu->r[rm]);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
    }

    return 1;
}

int arm32_step_trace(ARM32_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                     char *target_buf, size_t target_sz) {
    char mnem[32];
    int ok = arm32_decode_prompt(cpu, prompt_buf, prompt_sz, mnem, sizeof(mnem));
    if (!ok) {
        if (target_buf && target_sz) snprintf(target_buf, target_sz, "[NPC] %08x", cpu->r[15]);
        return 0;
    }

    arm32_step(cpu);

    if (target_buf && target_sz) {
        char *p = target_buf;
        size_t rem = target_sz;
        int n = snprintf(p, rem, "[NPC] %08x", cpu->r[15]);
        if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }

        if (cpu->has_write_reg) {
            n = snprintf(p, rem, "\n[W_REG %s] %08x", ARM32_REG_NAMES[cpu->write_reg_rd], cpu->write_reg_val);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
        if (cpu->has_write_cpsr) {
            n = snprintf(p, rem, "\n[W_CPSR] %08x", cpu->write_cpsr_val);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
        if (cpu->has_write_mem) {
            n = snprintf(p, rem, "\n[W_MEM %08x] %08x", cpu->write_mem_addr, cpu->write_mem_val);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; }
        }
    }

    return 1;
}

uint8_t *arm32_get_vram_ptr(ARM32_CPU *cpu) {
    uint32_t offset = ARM32_MMIO_VRAM;
    if (offset >= 0x80000000u) offset -= 0x80000000u;
    return cpu->ram + offset;
}

void arm32_get_frame_rgba(ARM32_CPU *cpu, uint32_t *rgba_out) {
    const uint8_t *vram = arm32_get_vram_ptr(cpu);
    memcpy(rgba_out, vram, ARM32_MMIO_VRAM_SIZE);
}

uint32_t arm32_get_pc(const ARM32_CPU *cpu) { return cpu->r[15]; }
void arm32_set_pc(ARM32_CPU *cpu, uint32_t pc) { cpu->r[15] = pc; }
uint32_t arm32_get_cpsr(const ARM32_CPU *cpu) { return cpu->cpsr; }
void arm32_set_cpsr(ARM32_CPU *cpu, uint32_t cpsr) { cpu->cpsr = cpsr; }
uint64_t arm32_get_steps(const ARM32_CPU *cpu) { return cpu->steps; }
double arm32_get_steps_double(const ARM32_CPU *cpu) { return (double)cpu->steps; }
uint32_t arm32_get_reg(const ARM32_CPU *cpu, int idx) {
    if (idx >= 0 && idx < 16) return cpu->r[idx];
    return 0;
}
void arm32_set_reg(ARM32_CPU *cpu, int idx, uint32_t val) {
    if (idx >= 0 && idx < 16) cpu->r[idx] = val;
}
uint32_t arm32_get_timer_ms(const ARM32_CPU *cpu) { return cpu->timer_ms; }
uint32_t arm32_get_keyboard(const ARM32_CPU *cpu) { return cpu->keyboard; }
int arm32_is_halted(const ARM32_CPU *cpu) { return cpu->halted; }
uint8_t *arm32_get_ram_ptr(ARM32_CPU *cpu) { return cpu->ram; }
size_t arm32_get_ram_size(void) { return ARM32_RAM_SIZE; }

#define ARM32_CKPT_MAGIC 0x4D524133u /* '3ARM' */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t steps;
    uint32_t timer_ms;
    uint32_t keyboard;
    uint32_t cpsr;
    uint32_t r[16];
    uint32_t ram_size;
} ARM32_CkptHeader;

int arm32_save_checkpoint(const ARM32_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    ARM32_CkptHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = ARM32_CKPT_MAGIC;
    hdr.version = 1;
    hdr.steps = cpu->steps;
    hdr.timer_ms = cpu->timer_ms;
    hdr.keyboard = cpu->keyboard;
    hdr.cpsr = cpu->cpsr;
    memcpy(hdr.r, cpu->r, sizeof(hdr.r));
    hdr.ram_size = ARM32_RAM_SIZE;

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    if (fwrite(cpu->ram, 1, ARM32_RAM_SIZE, f) != ARM32_RAM_SIZE) { fclose(f); return -1; }

    fclose(f);
    return 0;
}

int arm32_load_checkpoint(ARM32_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    ARM32_CkptHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    if (hdr.magic != ARM32_CKPT_MAGIC) { fclose(f); return -2; }

    cpu->steps = hdr.steps;
    cpu->timer_ms = hdr.timer_ms;
    cpu->keyboard = hdr.keyboard;
    cpu->cpsr = hdr.cpsr;
    memcpy(cpu->r, hdr.r, sizeof(hdr.r));

    if (fread(cpu->ram, 1, ARM32_RAM_SIZE, f) != ARM32_RAM_SIZE) { fclose(f); return -3; }

    fclose(f);
    return 0;
}
