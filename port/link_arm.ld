OUTPUT_ARCH("arm")
ENTRY(_start)

MEMORY
{
    /* 32MB of RAM starting at 0x00000000.
     * The emulator also mirrors this RAM at 0x80000000, which is where the
     * port's MMIO constants live (WAD at 0x80A00000, VRAM at 0x81000000). */
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 32M
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
    } > RAM

    .data : {
        *(.data .data.*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    _end = .;
    __heap_start = .;
    /* Heap runs from end of bss up to the WAD (which lives at 0x00A00000,
     * visible to the guest as 0x80A00000 through the RAM mirror). */
    __heap_end = 0x00A00000;

    /* Stack below the VRAM mirror (VRAM is at 0x81000000 = offset 0x1000000). */
    __stack_top = 0x00F00000;

    /DISCARD/ : {
        *(.comment)
        *(.note.*)
        *(.eh_frame*)
        *(.ARM.exidx*)
        *(.ARM.extab*)
    }
}
