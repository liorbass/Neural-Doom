    .section .text.init
    .global _start
    .arm

_start:
    /* Stack pointer: top of the stack area defined in link_arm.ld */
    ldr     sp, =__stack_top

    /* Zero out BSS */
    mov     r2, #0
    ldr     r0, =__bss_start
    ldr     r1, =__bss_end
1:
    cmp     r0, r1
    bhs     2f
    str     r2, [r0], #4
    b       1b
2:
    bl      main

    /* If main returns, spin forever */
3:
    b       3b
