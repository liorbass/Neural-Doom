.section .text.init
.global _start

_start:
    # Set global pointer
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop

    # Set stack pointer (top of our 16MB RAM, leaving space for heap)
    la sp, __stack_top

    # Zero out BSS section
    la t0, __bss_start
    la t1, __bss_end
1:
    bge t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b
2:

    # Call main
    call main

    # If main returns, halt the CPU
halt_loop:
    ecall
    j halt_loop
