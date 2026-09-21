#ifndef RV32I_H
#define RV32I_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAM_BASE       0x80000000u
#define RAM_SIZE       (17 * 1024 * 1024)   /* 17 MiB */
#define MMIO_KBD       0xFFFF0000u
#define MMIO_TIMER     0xFFFF0004u
#define MMIO_CONS      0xFFFF0008u
#define VRAM_ADDR      0x81000000u
#define VRAM_OFFSET    (VRAM_ADDR - RAM_BASE)
#define SCREEN_W       320
#define SCREEN_H       200

#define TIMER_STEPS_PER_MS 10000

extern const char *const REG_NAMES[32];

typedef struct {
    uint64_t at;
    uint32_t val;
    int has_cond;
    uint32_t cond_val;
} KeyEvent;

typedef struct {
    uint8_t ram[RAM_SIZE];
    uint32_t regs[32];
    uint32_t pc;
    uint64_t steps;
    uint32_t keyboard;
    uint32_t timer_ms;
    int halted;

    /* Writeback tracking for tracing & state inspection */
    int has_write_reg;
    int write_reg_rd;
    uint32_t write_reg_val;

    int has_write_mem;
    uint32_t write_mem_addr;
    uint32_t write_mem_val;
    int write_mem_size;

    /* Load tracking for prompt generation */
    int has_load;
    uint32_t load_addr;
    uint32_t load_raw;

    /* Probe addresses from symbols.txt */
    uint32_t gs_addr;
    uint32_t gametic_addr;

    /* Scheduled key event queue */
    KeyEvent evq[64];
    int nev;
} RV32I_CPU;

#pragma pack(push, 1)
typedef struct {
    char magic[4];          /* "RV32" */
    uint32_t version;       /* 1 */
    uint64_t steps;
    uint32_t pc;
    uint32_t timer_ms;
    uint32_t keyboard;
    uint32_t halted;
    uint32_t regs[32];
    uint32_t ram_size;      /* 17825792 = 17 MiB */
} CheckpointHeader;
#pragma pack(pop)

/* Memory & CPU lifecycle */
RV32I_CPU *rv32i_create(void);
void rv32i_free(RV32I_CPU *cpu);
void rv32i_init(RV32I_CPU *cpu);
void rv32i_copy(RV32I_CPU *dst, const RV32I_CPU *src);

int rv32i_load_binary(RV32I_CPU *cpu, const char *path);
int rv32i_load_binary_data(RV32I_CPU *cpu, const uint8_t *data, size_t size);
int rv32i_load_symbols(RV32I_CPU *cpu, const char *path);

uint32_t rv32i_probe_gs(const RV32I_CPU *cpu);
uint32_t rv32i_probe_gametic(const RV32I_CPU *cpu);

/* Direct memory access */
uint32_t rv32i_mem_read(RV32I_CPU *cpu, uint32_t addr, int size);
uint32_t rv32i_mem_peek(const RV32I_CPU *cpu, uint32_t addr, int size);
void rv32i_mem_write(RV32I_CPU *cpu, uint32_t addr, int size, uint32_t val);

/* Key schedule queue */
void rv32i_queue_key(RV32I_CPU *cpu, uint64_t at, uint32_t key, int down, int gs_cond);
void rv32i_push_key(RV32I_CPU *cpu, uint32_t key, int down);
void rv32i_clear_keys(RV32I_CPU *cpu);

/* Execution */
int rv32i_step(RV32I_CPU *cpu);
int rv32i_step_trace(RV32I_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                     char *target_buf, size_t target_sz);

/* Prompt formatting without execution (for model driver) */
int rv32i_decode_prompt(const RV32I_CPU *cpu, char *prompt_buf, size_t prompt_sz,
                        char *mnem_buf, size_t mnem_sz);

/* Apply model prediction to CPU state */
void rv32i_apply_prediction(RV32I_CPU *cpu, uint32_t npc, int rd, uint32_t reg_val,
                            int has_mem, uint32_t mem_addr, uint32_t mem_val, int mem_size);

/* Checkpoints */
int rv32i_save_checkpoint(const RV32I_CPU *cpu, const char *path);
int rv32i_load_checkpoint(RV32I_CPU *cpu, const char *path);

/* Frame extraction */
void rv32i_dump_frame(const RV32I_CPU *cpu, const char *path);
void rv32i_get_frame_rgb(const RV32I_CPU *cpu, uint8_t *rgb_out);
void rv32i_get_frame_rgba(const RV32I_CPU *cpu, uint32_t *rgba_out);
uint8_t *rv32i_get_vram_ptr(RV32I_CPU *cpu);
uint32_t rv32i_run_steps(RV32I_CPU *cpu, uint32_t count);

/* State accessors (for ctypes / interop) */
uint32_t rv32i_get_pc(const RV32I_CPU *cpu);
void rv32i_set_pc(RV32I_CPU *cpu, uint32_t pc);
uint64_t rv32i_get_steps(const RV32I_CPU *cpu);
double rv32i_get_steps_double(const RV32I_CPU *cpu);
void rv32i_set_steps(RV32I_CPU *cpu, uint64_t steps);
uint32_t rv32i_get_reg(const RV32I_CPU *cpu, int idx);
void rv32i_set_reg(RV32I_CPU *cpu, int idx, uint32_t val);
int rv32i_is_halted(const RV32I_CPU *cpu);
void rv32i_set_halted(RV32I_CPU *cpu, int halted);
uint32_t rv32i_get_timer_ms(const RV32I_CPU *cpu);
void rv32i_set_timer_ms(RV32I_CPU *cpu, uint32_t ms);
uint32_t rv32i_get_keyboard(const RV32I_CPU *cpu);
void rv32i_set_keyboard(RV32I_CPU *cpu, uint32_t k);
uint8_t *rv32i_get_ram_ptr(RV32I_CPU *cpu);
size_t rv32i_get_ram_size(void);

/* Basic block stepping and writeback inspection */
int rv32i_step_block(RV32I_CPU *cpu, int max_instructions);
int rv32i_step_superblock(RV32I_CPU *cpu, uint32_t max_insts, uint32_t *out_info);
void rv32i_advance_branch(RV32I_CPU *cpu, uint32_t next_pc);
int rv32i_get_write_reg(const RV32I_CPU *cpu, int *rd, uint32_t *val);
int rv32i_get_write_mem(const RV32I_CPU *cpu, uint32_t *addr, uint32_t *val, int *size);

/* Neural Superblock Execution (Stage C: Embedded C/WASM Mamba-Nano) */
struct MambaNanoState;
int rv32i_step_superblock_neural(RV32I_CPU *cpu, void *state, uint32_t max_insts, uint32_t *out_info);
int rv32i_step_superblock_neural_burst(RV32I_CPU *cpu, void *state,
                                      uint32_t num_blocks, uint32_t max_insts_per_block,
                                      uint32_t *out_last_info, uint32_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* RV32I_H */

