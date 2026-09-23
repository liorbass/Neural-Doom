/*
 * Headless harness for the bare-metal ARM32 (ARMv4T) DOOM port.
 * Mirrors the RV32I emulator's probe/progress reporting so the ARM port can
 * be validated without a browser: boots the guest, optionally drives the
 * menu into E1M1, prints gamestate/gametic progress and dumps frames.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arm32.h"

#define SCREEN_W 320
#define SCREEN_H 200

static ARM32_CPU *g_cpu;
static uint64_t g_trace_n = 0;

static void step_cpu(ARM32_CPU *cpu) {
    arm32_step(cpu);
    if (g_trace_n && cpu->steps <= g_trace_n) {
        uint32_t pc = arm32_get_pc(cpu);
        uint32_t off = pc >= 0x80000000u ? pc - 0x80000000u : pc;
        uint32_t inst = 0;
        if (off + 4 <= arm32_get_ram_size())
            memcpy(&inst, arm32_get_ram_ptr(cpu) + off, 4);
        fprintf(stderr, "T %llu pc=%08x inst=%08x sp=%08x lr=%08x cpsr=%08x\n",
                (unsigned long long)cpu->steps, pc, inst,
                arm32_get_reg(cpu, 13), arm32_get_reg(cpu, 14),
                arm32_get_cpsr(cpu));
    }
}

static int load_symbols(ARM32_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char name[64];
    unsigned long addr;
    int n = 0;
    while (fscanf(f, "%63s %lx", name, &addr) == 2) {
        if (!strcmp(name, "gamestate")) arm32_set_gamestate_addr(cpu, (uint32_t)addr);
        else if (!strcmp(name, "gametic")) arm32_set_gametic_addr(cpu, (uint32_t)addr);
        n++;
    }
    fclose(f);
    return n;
}

static void dump_frame(const ARM32_CPU *cpu, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", SCREEN_W, SCREEN_H);
    const uint8_t *vram = arm32_get_vram_ptr((ARM32_CPU *)cpu);
    for (int p = 0; p < SCREEN_W * SCREEN_H; p++) {
        uint32_t px;
        memcpy(&px, vram + p * 4, 4);
        uint8_t rgb[3] = { (uint8_t)(px >> 16), (uint8_t)(px >> 8), (uint8_t)px };
        fwrite(rgb, 1, 3, f);
    }
    fclose(f);
}

static void push(ARM32_CPU *cpu, uint32_t key) {
    arm32_push_key(cpu, key, 1);
    arm32_run_steps(cpu, 200000);
    arm32_push_key(cpu, key, 0);
    arm32_run_steps(cpu, 200000);
}


static void write_block_json(FILE *f, const ARM32_BlockTrace *t) {
    fprintf(f, "{\"start_pc\":%u,\"end_pc\":%u,\"next_pc\":%u,\"instruction_count\":%u,\"regs_in\":[",
            t->start_pc, t->end_pc, t->next_pc, t->inst_count);
    for (int i = 0; i < 16; i++) fprintf(f, "%s%u", i ? "," : "", t->regs_in[i]);
    fprintf(f, "],\"reg_writes\":{");
    int first = 1;
    for (int i = 1; i < 16; i++) {
        if (t->regs_out[i] != t->regs_in[i]) {
            fprintf(f, "%s\"%d\":%u", first ? "" : ",", i, t->regs_out[i]);
            first = 0;
        }
    }
    fprintf(f, "},\"mem_writes\":[");
    for (int i = 0; i < t->n_mem; i++)
        fprintf(f, "%s[%u,%u,%u]", i ? "," : "", t->mem_addr[i], t->mem_val[i], t->mem_size[i]);
    fprintf(f, "],\"halted\":%s}\n", t->halted ? "true" : "false");
}

static uint64_t trace_blocks(ARM32_CPU *cpu, const char *path, uint64_t max_blocks,
                             uint64_t skip_steps, uint64_t skip_every) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 0; }
    ARM32_BlockTrace t;
    uint64_t n = 0;
    while (n < max_blocks && !cpu->halted) {
        if (arm32_step_block_trace(cpu, 32, &t)) {
            write_block_json(f, &t);
            n++;
            /* Stride sampling: jump forward periodically so the corpus spans
             * many gameplay phases instead of one contiguous window. */
            if (skip_steps && skip_every && (n % skip_every) == 0)
                arm32_run_steps(cpu, (int)skip_steps);
        }
    }
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    const char *bin = "doomgeneric/doomgeneric/doom_arm32.bin";
    const char *symbols = NULL;
    const char *frame_path = NULL;
    uint64_t max_steps = 200000000ULL;
    int dump_every_ms = 0;
    int boot_menu = 0;
    const char *blocks_path = NULL;
    uint64_t max_blocks = 250000;
    uint64_t skip_steps = 0;
    uint64_t skip_every = 2000;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bin") && i + 1 < argc) bin = argv[++i];
        else if (!strcmp(argv[i], "--symbols") && i + 1 < argc) symbols = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) max_steps = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--frame") && i + 1 < argc) frame_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-every") && i + 1 < argc) dump_every_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--boot-menu")) boot_menu = 1;
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) g_trace_n = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) blocks_path = argv[++i];
        else if (!strcmp(argv[i], "--max-blocks") && i + 1 < argc) max_blocks = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--trace-skip-steps") && i + 1 < argc) skip_steps = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--trace-skip-every") && i + 1 < argc) skip_every = strtoull(argv[++i], NULL, 0);
    }

    g_cpu = arm32_create();
    if (!g_cpu) { fprintf(stderr, "alloc failed\n"); return 1; }

    if (symbols) {
        int n = load_symbols(g_cpu, symbols);
        fprintf(stderr, "[SYMBOLS] %d probe symbols from %s\n", n, symbols);
    }
    if (arm32_load_binary(g_cpu, bin) <= 0) { perror(bin); return 1; }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t menu_budget = boot_menu ? 60000000ULL : 0;
    uint64_t target = menu_budget ? menu_budget : max_steps;
    uint32_t last_gs = 0xFFFFFFFFu;
    uint64_t last_dump_ms = 0;

    while (g_cpu->steps < target && !g_cpu->halted) {
        step_cpu(g_cpu);

        uint32_t gs = arm32_probe_gs(g_cpu);
        if (gs != last_gs) {
            last_gs = gs;
            fprintf(stderr, "[GS] gamestate=%u gametic=%u at step %llu (%llu ms)\n",
                    gs, arm32_probe_gametic(g_cpu),
                    (unsigned long long)g_cpu->steps,
                    (unsigned long long)arm32_get_timer_ms(g_cpu));
        }
        if (g_cpu->steps % 1000000 == 0 && g_cpu->steps) {
            fprintf(stderr, "step %llums=%llu gametic=%u gs=%u pc=%08x\n",
                    (unsigned long long)g_cpu->steps,
                    (unsigned long long)arm32_get_timer_ms(g_cpu),
                    arm32_probe_gametic(g_cpu), gs, arm32_get_pc(g_cpu));
        }
        if (dump_every_ms && g_cpu->steps % 10000 == 0) {
            uint64_t ms = arm32_get_timer_ms(g_cpu);
            if (ms - last_dump_ms >= (uint64_t)dump_every_ms) {
                last_dump_ms = ms;
                if (frame_path) dump_frame(g_cpu, frame_path);
            }
        }
    }

    if (boot_menu) {
        fprintf(stderr, "[MENU] driving title -> E1M1\n");
        push(g_cpu, 27);            /* ESC  */
        arm32_run_steps(g_cpu, 1000000);
        push(g_cpu, 13);            /* ENTER: new game */
        arm32_run_steps(g_cpu, 1000000);
        push(g_cpu, 13);            /* ENTER: episode 1 */
        arm32_run_steps(g_cpu, 1000000);
        push(g_cpu, 13);            /* ENTER: skill */
        arm32_run_steps(g_cpu, 5000000);
        last_gs = 0xFFFFFFFFu;
        while (g_cpu->steps < max_steps && !g_cpu->halted) {
            step_cpu(g_cpu);
            uint32_t gs = arm32_probe_gs(g_cpu);
            if (gs != last_gs) {
                last_gs = gs;
                fprintf(stderr, "[GS] gamestate=%u gametic=%u at step %llu (%llu ms)\n",
                        gs, arm32_probe_gametic(g_cpu),
                        (unsigned long long)g_cpu->steps,
                        (unsigned long long)arm32_get_timer_ms(g_cpu));
            }
            if (g_cpu->steps % 1000000 == 0 && g_cpu->steps) {
                fprintf(stderr, "step %llums=%llu gametic=%u gs=%u pc=%08x\n",
                        (unsigned long long)g_cpu->steps,
                        (unsigned long long)arm32_get_timer_ms(g_cpu),
                        arm32_probe_gametic(g_cpu), gs, arm32_get_pc(g_cpu));
            }
            if (dump_every_ms && g_cpu->steps % 10000 == 0) {
                uint64_t ms = arm32_get_timer_ms(g_cpu);
                if (ms - last_dump_ms >= (uint64_t)dump_every_ms) {
                    last_dump_ms = ms;
                    if (frame_path) dump_frame(g_cpu, frame_path);
                }
            }
        }
    }

    if (blocks_path) {
        fprintf(stderr, "[TRACE] collecting up to %llu basic blocks -> %s\n",
                (unsigned long long)max_blocks, blocks_path);
        uint64_t n = trace_blocks(g_cpu, blocks_path, max_blocks, skip_steps, skip_every);
        fprintf(stderr, "[TRACE] wrote %llu blocks\n", (unsigned long long)n);
    }

    if (frame_path) dump_frame(g_cpu, frame_path);

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "[ARM32] %llu steps in %.2fs = %.0f steps/s (game %llu ms), halted=%d\n",
            (unsigned long long)g_cpu->steps, dt,
            dt > 0 ? (double)g_cpu->steps / dt : 0.0,
            (unsigned long long)arm32_get_timer_ms(g_cpu), g_cpu->halted);
    return 0;
}
