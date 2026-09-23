#ifndef __ARM32_H__
#define __ARM32_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARM32_RAM_BASE 0x00000000u
#define ARM32_RAM_SIZE 0x02000000u  /* 32 MiB RAM */

/* MMIO register addresses */
#define ARM32_MMIO_KBD       0xFFFF0000u
#define ARM32_MMIO_TIMER     0xFFFF0004u
#define ARM32_MMIO_CONS      0xFFFF0008u
#define ARM32_MMIO_VRAM      0x81000000u
#define ARM32_MMIO_VRAM_SIZE (320u * 200u * 4u) /* 256 KB */

/* Virtual clock: 10 000 guest instructions per emulated millisecond,
 * identical to the RV32I engine so both ports share DOOM's timing model. */
#define ARM32_TIMER_STEPS_PER_MS 10000u

/* CPSR condition flag masks */
#define ARM32_FLAG_N (1u << 31)  /* Negative */
#define ARM32_FLAG_Z (1u << 30)  /* Zero */
#define ARM32_FLAG_C (1u << 29)  /* Carry */
#define ARM32_FLAG_V (1u << 28)  /* Overflow */
#define ARM32_FLAG_I (1u << 7)   /* IRQ disable */
#define ARM32_FLAG_F (1u << 6)   /* FIQ disable */
#define ARM32_FLAG_T (1u << 5)   /* Thumb state (0=ARM, 1=Thumb) */

/* Condition codes (bits 31:28 of ARM instruction) */
enum {
    ARM32_COND_EQ = 0,  /* Z == 1 */
    ARM32_COND_NE = 1,  /* Z == 0 */
    ARM32_COND_CS = 2,  /* C == 1 (Carry Set / Unsigned Higher or Same) */
    ARM32_COND_CC = 3,  /* C == 0 (Carry Clear / Unsigned Lower) */
    ARM32_COND_MI = 4,  /* N == 1 (Minus / Negative) */
    ARM32_COND_PL = 5,  /* N == 0 (Plus / Positive or Zero) */
    ARM32_COND_VS = 6,  /* V == 1 (Overflow) */
    ARM32_COND_VC = 7,  /* V == 0 (No Overflow) */
    ARM32_COND_HI = 8,  /* C == 1 && Z == 0 (Unsigned Higher) */
    ARM32_COND_LS = 9,  /* C == 0 || Z == 1 (Unsigned Lower or Same) */
    ARM32_COND_GE = 10, /* N == V (Signed Greater Than or Equal) */
    ARM32_COND_LT = 11, /* N != V (Signed Less Than) */
    ARM32_COND_GT = 12, /* Z == 0 && N == V (Signed Greater Than) */
    ARM32_COND_LE = 13, /* Z == 1 || N != V (Signed Less Than or Equal) */
    ARM32_COND_AL = 14, /* Always (unconditional) */
    ARM32_COND_NV = 15  /* Never / Unpredictable */
};

typedef struct {
    uint64_t at;
    uint32_t val;
    int has_cond;
    uint32_t cond_val;
} ARM32_KeyEvent;

typedef struct {
    uint32_t r[16];   /* r0..r15 (r13=SP, r14=LR, r15=PC) */
    uint32_t cpsr;    /* Current Program Status Register */
    uint64_t steps;
    uint32_t timer_ms;
    uint32_t keyboard;
    int halted;

    uint8_t *ram;

    /* Single-step tracking for neural model dataset/verification */
    int has_write_reg;
    int write_reg_rd;
    uint32_t write_reg_val;

    int has_write_mem;
    uint32_t write_mem_addr;
    uint32_t write_mem_val;
    int write_mem_size;

    int has_write_cpsr;
    uint32_t write_cpsr_val;

    int cond_passed;

    /* Key schedule */
    ARM32_KeyEvent evq[64];
    int nev;

    /* Guest probes */
    uint32_t gametic_addr;
    uint32_t gamestate_addr;
} ARM32_CPU;

/* Lifecycle */
ARM32_CPU *arm32_create(void);
void arm32_free(ARM32_CPU *cpu);
void arm32_init(ARM32_CPU *cpu);
int arm32_load_binary(ARM32_CPU *cpu, const char *path);
int arm32_load_binary_data(ARM32_CPU *cpu, const uint8_t *data, size_t size);

/* Execution */
int arm32_step(ARM32_CPU *cpu);
int arm32_step_trace(ARM32_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                     char *target_buf, size_t target_sz);
int arm32_run_steps(ARM32_CPU *cpu, int n);

/* Prompt formatting without execution */
int arm32_decode_prompt(const ARM32_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                        char *mnem_buf, size_t mnem_sz);

/* Memory and I/O */
uint32_t arm32_mem_read(ARM32_CPU *cpu, uint32_t addr, int size);
uint32_t arm32_mem_peek(const ARM32_CPU *cpu, uint32_t addr, int size);
void arm32_mem_write(ARM32_CPU *cpu, uint32_t addr, int size, uint32_t val);

/* Key handling */
void arm32_queue_key(ARM32_CPU *cpu, uint64_t at, uint32_t key, int down, int gs_cond);
void arm32_push_key(ARM32_CPU *cpu, uint32_t key, int down);
void arm32_clear_keys(ARM32_CPU *cpu);

/* Checkpointing */
int arm32_save_checkpoint(const ARM32_CPU *cpu, const char *path);
int arm32_load_checkpoint(ARM32_CPU *cpu, const char *path);

/* Framebuffer & Getters for ABI/WASM bindings */
uint8_t *arm32_get_vram_ptr(ARM32_CPU *cpu);
void arm32_get_frame_rgba(ARM32_CPU *cpu, uint32_t *rgba_out);
uint32_t arm32_get_pc(const ARM32_CPU *cpu);
void arm32_set_pc(ARM32_CPU *cpu, uint32_t pc);
uint32_t arm32_get_cpsr(const ARM32_CPU *cpu);
void arm32_set_cpsr(ARM32_CPU *cpu, uint32_t cpsr);
uint64_t arm32_get_steps(const ARM32_CPU *cpu);
double arm32_get_steps_double(const ARM32_CPU *cpu);
uint32_t arm32_get_reg(const ARM32_CPU *cpu, int idx);
void arm32_set_reg(ARM32_CPU *cpu, int idx, uint32_t val);
uint32_t arm32_get_timer_ms(const ARM32_CPU *cpu);
uint32_t arm32_get_keyboard(const ARM32_CPU *cpu);
int arm32_is_halted(const ARM32_CPU *cpu);
uint32_t arm32_probe_gs(const ARM32_CPU *cpu);
uint32_t arm32_probe_gametic(const ARM32_CPU *cpu);
void arm32_set_gamestate_addr(ARM32_CPU *cpu, uint32_t addr);
void arm32_set_gametic_addr(ARM32_CPU *cpu, uint32_t addr);
uint8_t *arm32_get_ram_ptr(ARM32_CPU *cpu);
size_t arm32_get_ram_size(void);

/* Basic-block trace record (mirrors model/block_schema.BlockTransition) */
typedef struct {
    uint32_t start_pc, end_pc, next_pc;
    uint32_t inst_count;
    uint32_t regs_in[16];
    uint32_t regs_out[16];
    uint32_t mem_addr[256];
    uint32_t mem_val[256];
    uint32_t mem_size[256];
    int n_mem;
    int halted;
    int terminated;
} ARM32_BlockTrace;

int arm32_step_block_trace(ARM32_CPU *cpu, uint32_t max_insts, ARM32_BlockTrace *t);

/* Superblock stepping with the Mamba-Nano observer (same contract as the
 * RV32I engine: the predictor is consulted, ground truth is retired). */
int arm32_step_superblock(ARM32_CPU *cpu, uint32_t max_insts, uint32_t *out_info);
int arm32_step_superblock_neural(ARM32_CPU *cpu, void *state, uint32_t max_insts,
                                 uint32_t *out_info);
int arm32_step_superblock_neural_burst(ARM32_CPU *cpu, void *state,
                                       uint32_t num_blocks, uint32_t max_insts_per_block,
                                       uint32_t *out_last_info, uint32_t *out_stats);

extern const char *const ARM32_REG_NAMES[16];
extern const char *const ARM32_COND_NAMES[16];

#ifdef __cplusplus
}
#endif

#endif /* __ARM32_H__ */
