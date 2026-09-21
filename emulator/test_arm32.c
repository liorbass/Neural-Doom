#include "arm32.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

void test_basic_alu() {
    printf("[ARM32 TEST] Testing basic ALU & Barrel Shifter...\n");
    ARM32_CPU *cpu = arm32_create();
    assert(cpu != NULL);

    /* Test 1: mov r1, #10 (0xe3a0100a) */
    /* Test 2: mov r2, #20 (0xe3a02014) */
    /* Test 3: add r0, r1, r2, lsl #2 (0xe0810102) -> 10 + (20 << 2) = 10 + 80 = 90 */
    uint32_t code[] = {
        0xe3a0100au, /* mov r1, #10 */
        0xe3a02014u, /* mov r2, #20 */
        0xe0810102u, /* add r0, r1, r2, lsl #2 */
    };
    arm32_load_binary_data(cpu, (const uint8_t *)code, sizeof(code));

    arm32_step(cpu);
    assert(cpu->r[1] == 10);

    arm32_step(cpu);
    assert(cpu->r[2] == 20);

    arm32_step(cpu);
    assert(cpu->r[0] == 90);
    printf("  -> MOV and ADD with LSL #2 passed: r0 = %u (expected 90)\n", cpu->r[0]);

    arm32_free(cpu);
}

void test_flags_and_predication() {
    printf("[ARM32 TEST] Testing CPSR Flags & Condition Codes...\n");
    ARM32_CPU *cpu = arm32_create();
    assert(cpu != NULL);

    /*
     * 1. movs r0, #0 (0xe3b00000) -> Z=1
     * 2. addeq r1, r0, #42 (0x0280102a) -> executed because Z=1 -> r1 = 42
     * 3. addne r2, r0, #99 (0x12802063) -> SKIPPED because Z=1 (not NE) -> r2 remains 0
     * 4. subs r3, r1, #42 (0xe251302a) -> 42 - 42 = 0 -> Z=1, C=1 (no borrow)
     */
    uint32_t code[] = {
        0xe3b00000u, /* movs r0, #0 */
        0x0280102au, /* addeq r1, r0, #42 */
        0x12802063u, /* addne r2, r0, #99 */
        0xe251302au, /* subs r3, r1, #42 */
    };
    arm32_load_binary_data(cpu, (const uint8_t *)code, sizeof(code));

    arm32_step(cpu);
    assert((cpu->cpsr & ARM32_FLAG_Z) != 0);

    arm32_step(cpu);
    assert(cpu->r[1] == 42); /* addeq executed */

    arm32_step(cpu);
    assert(cpu->r[2] == 0);  /* addne was skipped! */

    arm32_step(cpu);
    assert(cpu->r[3] == 0);
    assert((cpu->cpsr & ARM32_FLAG_Z) != 0);
    assert((cpu->cpsr & ARM32_FLAG_C) != 0);

    printf("  -> Condition codes EQ (executed) and NE (skipped) verified successfully!\n");
    arm32_free(cpu);
}

void test_branch_and_link() {
    printf("[ARM32 TEST] Testing Branch with Link (BL)...\n");
    ARM32_CPU *cpu = arm32_create();
    assert(cpu != NULL);

    /*
     * PC=0: bl +8 (to PC=16) (0xeb000002) -> offset = (2<<2)+8 = 16
     * PC=4: mov r0, #1
     * ...
     * PC=16: mov r1, #77
     */
    uint32_t code[5] = {0};
    code[0] = 0xeb000002u; /* bl to offset 16 */
    code[1] = 0xe3a00001u; /* mov r0, #1 (skipped) */
    code[4] = 0xe3a0104du; /* mov r1, #77 */

    arm32_load_binary_data(cpu, (const uint8_t *)code, sizeof(code));

    arm32_step(cpu);
    assert(cpu->r[14] == 4);  /* LR = return address PC+4 */
    assert(cpu->r[15] == 16); /* Target PC = 16 */

    arm32_step(cpu);
    assert(cpu->r[1] == 77);
    assert(cpu->r[0] == 0);   /* Skipped instruction was not run */

    printf("  -> BL branch jumped to PC=16 and saved LR=4 correctly!\n");
    arm32_free(cpu);
}

void test_memory_and_block_transfer() {
    printf("[ARM32 TEST] Testing LDR/STR and LDM/STM...\n");
    ARM32_CPU *cpu = arm32_create();
    assert(cpu != NULL);

    /*
     * 1. mov r0, #0x1000
     * 2. mov r1, #111
     * 3. mov r2, #222
     * 4. mov r3, #255
     * 5. stmia r0!, {r1, r2, r3} (0xe8a0000e) -> stores r1, r2, r3 at 0x1000, 0x1004, 0x1008
     * 6. mov r4, #0
     * 7. mov r5, #0
     * 8. mov r6, #0
     * 9. sub r0, r0, #12
     * 10. ldmia r0, {r4, r5, r6} (0xe8900070) -> loads into r4, r5, r6
     */
    uint32_t code[] = {
        0xe3a00a01u, /* mov r0, #0x1000 */
        0xe3a0106fu, /* mov r1, #111 */
        0xe3a020deu, /* mov r2, #222 */
        0xe3a030ffu, /* mov r3, #255 */
        0xe8a0000eu, /* stmia r0!, {r1, r2, r3} */
        0xe240000cu, /* sub r0, r0, #12 */
        0xe8900070u, /* ldmia r0, {r4, r5, r6} */
    };
    arm32_load_binary_data(cpu, (const uint8_t *)code, sizeof(code));

    for (int i = 0; i < 7; i++) {
        arm32_step(cpu);
    }

    assert(cpu->r[4] == 111);
    assert(cpu->r[5] == 222);
    assert(cpu->r[6] == 255);
    printf("  -> Block transfer STM/LDM multi-register load/store verified!\n");

    arm32_free(cpu);
}

void test_prompt_trace() {
    printf("[ARM32 TEST] Testing prompt decoding & step tracing for neural model...\n");
    ARM32_CPU *cpu = arm32_create();
    assert(cpu != NULL);

    uint32_t code[] = {
        0xe0810102u, /* add r0, r1, r2, lsl #2 */
    };
    arm32_load_binary_data(cpu, (const uint8_t *)code, sizeof(code));
    cpu->r[1] = 5;
    cpu->r[2] = 10;

    char prompt[512] = {0};
    char target[256] = {0};

    arm32_step_trace(cpu, prompt, sizeof(prompt), target, sizeof(target));
    printf("Prompt:\n%s\nTarget:\n%s\n", prompt, target);

    assert(strstr(prompt, "[CMD] STEP") != NULL);
    assert(strstr(prompt, "[CPSR]") != NULL);
    assert(strstr(target, "[NPC] 00000004") != NULL);
    assert(strstr(target, "[W_REG r0] 0000002d") != NULL); /* 5 + 40 = 45 = 0x2d */

    printf("  -> Prompt and target generation format verified!\n");
    arm32_free(cpu);
}

int main() {
    printf("=========================================\n");
    printf("     Running ARM32 Core Verification     \n");
    printf("=========================================\n");
    test_basic_alu();
    test_flags_and_predication();
    test_branch_and_link();
    test_memory_and_block_transfer();
    test_prompt_trace();
    printf("\n[SUCCESS] ALL ARM32 CORE TESTS PASSED!\n");
    return 0;
}
