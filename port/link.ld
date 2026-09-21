OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    /* 16MB of RAM starting at 0x80000000 */
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M
}

SECTIONS
{
    . = ORIGIN(RAM);

    .text : {
        KEEP(*(.text.init))
        *(.text .text.*)
    } > RAM

    .rodata : {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > RAM

    .data : {
        __global_pointer$ = . + 0x800;
        *(.data .data.*)
        *(.sdata .sdata.*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    _end = .;
    __heap_start = .;
    /* Heap runs from end of bss up to the WAD (which lives at 0x80A00000) */
    __heap_end = 0x80A00000;

    /* Stack below the WAD-adjacent free area, growing down.
     * VRAM is at 0x81000000; keep the stack well away from it. */
    __stack_top = 0x80F00000;

    /DISCARD/ : {
        *(.comment)
        *(.note.*)
        *(.eh_frame*)
    }
}
