/*
 * doomgeneric platform port for neural-doom bare-metal RV32I.
 * All libc comes from newlib (see syscalls.c); this file only
 * implements the doomgeneric platform interface + main.
 */

#include <stdint.h>
#include <stddef.h>
#include "doomgeneric.h"

/* MMIO map (must match emulator) */
#define IO_KEYBOARD_PORT   (*(volatile uint32_t *)0xFFFF0000)
#define IO_TIMER_PORT      (*(volatile uint32_t *)0xFFFF0004)
#define IO_VRAM_START      ((uint32_t *)0x81000000)  /* 320*200*4 = 256KB */

void DG_Init()
{
    /* Point DOOM's framebuffer straight at our VRAM region.
     * doomgeneric_Create malloc'd a buffer; redirect to MMIO VRAM. */
    DG_ScreenBuffer = IO_VRAM_START;
}

void DG_DrawFrame()
{
    /* The emulator reads VRAM directly from the memory array.
     * Nothing to blit. A real port would trigger a swap/vblank here. */
}

void DG_SleepMs(uint32_t ms)
{
    /* No-op: the emulator advances the virtual timer externally. */
}

uint32_t DG_GetTicksMs()
{
    return IO_TIMER_PORT;
}

int DG_GetKey(int *pressed, unsigned char *doomKey)
{
    uint32_t key_event = IO_KEYBOARD_PORT;

    if (key_event == 0)
        return 0; /* no event */

    /* Clear port after reading (read-to-clear semantics) */
    IO_KEYBOARD_PORT = 0;

    /* Encoding: bits[8] = pressed(1)/released(0), bits[7:0] = doomkey */
    *pressed = (key_event >> 8) & 1;
    *doomKey = (unsigned char)(key_event & 0xFF);
    return 1;
}

void DG_SetWindowTitle(const char *title)
{
    /* no window in bare metal */
}

int main(int argc, char **argv)
{
#ifdef PLAY_DEMO
    /* Boots straight into the E1M1DEMO lump (see `make demo`): a recorded
     * vanilla 1.9 playthrough that exits E1M1. -playdemo first tries to open
     * e1m1demo.lmp as a file; our single-fd filesystem already has the WAD
     * open, so that fails and DOOM falls back to the WAD lump (vanilla
     * "-playdemo demo1" trick). singledemo=true makes key events unable to
     * skip the demo and quits the guest when it ends. */
    static char *args[] = {"doom",     "-iwad", "doom1.wad", "-playdemo",
                           "e1m1demo", "-nofullscreen", NULL};

    doomgeneric_Create(6, args);
#else
    static char *args[] = {"doom", "-iwad", "doom1.wad", "-nofullscreen", NULL};

    doomgeneric_Create(4, args);
#endif

    while (1) {
        doomgeneric_Tick();
    }

    return 0;
}
