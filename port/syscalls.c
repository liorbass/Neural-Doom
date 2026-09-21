/*
 * Newlib syscall layer for neural-doom bare-metal RV32I.
 * 
 * These functions let us link against newlib's full libc (printf, malloc,
 * fopen, fread...) while running without an OS.
 *
 * MMIO map (must match emulator):
 *   0xFFFF0000 - keyboard input port
 *   0xFFFF0004 - millisecond timer port
 *   0xFFFF0008 - debug console output (writes chars here)
 */

#include <stdint.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <errno.h>

#define IO_DEBUG_CONSOLE (*(volatile uint32_t *)0xFFFF0008)
#define IO_TIMER_PORT    (*(volatile uint32_t *)0xFFFF0004)

/* WAD file: loaded into RAM by the loader (appended to binary) */
#define WAD_BASE_ADDR ((uint8_t *)0x80A00000)
/* doom1.wad's infotable ends at 4196020 bytes, just past 4MB — allow 6MB */
#define WAD_MAX_SIZE  (6 * 1024 * 1024)

static int wad_fd_open = 0;
static long wad_pos = 0;
static long wad_size = 0;

/* ------------------------------------------------------------------ */
/* Heap: _sbrk for newlib malloc                                      */
/* ------------------------------------------------------------------ */

extern char __heap_start;   /* from linker script */
extern char __heap_end;

void *_sbrk(int incr)
{
    static char *heap_ptr = 0;
    char *prev;

    if (heap_ptr == 0)
        heap_ptr = &__heap_start;

    if (heap_ptr + incr > &__heap_end) {
        errno = ENOMEM;
        return (void *)-1;
    }

    prev = heap_ptr;
    heap_ptr += incr;
    return prev;
}

/* ------------------------------------------------------------------ */
/* Console: printf / fprintf route here via _write                    */
/* ------------------------------------------------------------------ */

int _write(int fd, const char *buf, int len)
{
    /* Pack chars into the MMIO debug console port.
     * The emulator collects these but DOOM doesn't depend on them. */
    for (int i = 0; i < len; i++) {
        IO_DEBUG_CONSOLE = (uint32_t)(unsigned char)buf[i];
    }
    return len;
}

int _read(int fd, char *buf, int len)
{
    if (fd == 3) {
        /* WAD file read */
        if (wad_pos >= wad_size) return 0;
        if (wad_pos + len > wad_size) len = (int)(wad_size - wad_pos);

        uint8_t *src = WAD_BASE_ADDR + wad_pos;
        for (int i = 0; i < len; i++) buf[i] = (char)src[i];
        wad_pos += len;
        return len;
    }
    return 0; /* stdin: no input */
}

/* ------------------------------------------------------------------ */
/* Virtual filesystem: DOOM's fopen/fread on the WAD                  */
/* ------------------------------------------------------------------ */

static long detect_wad_size(void)
{
    /* WAD header: first 4 bytes "IWAD"/"PWAD", then lump count,
     * then infotable offset. Use infotable to find real size. */
    uint8_t *h = WAD_BASE_ADDR;
    if ((h[0] == 'I' || h[0] == 'P') && h[1] == 'W' && h[2] == 'A' && h[3] == 'D') {
        uint32_t infotable_ofs = *(uint32_t *)(h + 12);
        uint32_t numlumps = *(uint32_t *)(h + 4);
        long end = (long)infotable_ofs + (long)numlumps * 16;
        if (end > 0 && end <= WAD_MAX_SIZE)
            return end;
    }
    return WAD_MAX_SIZE;
}

int _open(const char *name, int flags, int mode)
{
    /* Any file DOOM opens (the WAD, config) maps to our WAD blob */
    if (!wad_fd_open) {
        wad_fd_open = 1;
        wad_pos = 0;
        wad_size = detect_wad_size();
        return 3; /* fd 3 */
    }
    errno = EMFILE;
    return -1;
}

int _close(int fd)
{
    if (fd == 3) {
        wad_fd_open = 0;
        return 0;
    }
    return -1;
}

int _lseek(int fd, long offset, int whence)
{
    if (fd != 3) return -1;
    if (whence == 0)      wad_pos = offset;            /* SEEK_SET */
    else if (whence == 1) wad_pos += offset;           /* SEEK_CUR */
    else if (whence == 2) wad_pos = wad_size + offset; /* SEEK_END */
    if (wad_pos < 0) wad_pos = 0;
    return (int)wad_pos;
}

int _fstat(int fd, struct stat *st)
{
    st->st_mode = S_IFCHR;
    if (fd == 3) {
        st->st_mode = S_IFREG;
        st->st_size = wad_size;
    }
    return 0;
}

int _isatty(int fd) { return (fd <= 2) ? 1 : 0; }

/* ------------------------------------------------------------------ */
/* Process stubs                                                       */
/* ------------------------------------------------------------------ */

void _exit(int status)
{
    while (1) {
        __asm__ volatile ("ecall");
    }
}

void _kill(int pid, int sig) { }
int  _getpid(void) { return 1; }

/* Remaining newlib syscall stubs */
int _link(const char *oldpath, const char *newpath) { errno = EMLINK; return -1; }
int _unlink(const char *path) { errno = ENOENT; return -1; }
int _stat(const char *path, struct stat *st) { errno = ENOENT; return -1; }
int _fork(void) { errno = EAGAIN; return -1; }
int _wait(int *status) { errno = ECHILD; return -1; }
int _execve(char *path, char *const argv[], char *const envp[]) { errno = ENOMEM; return -1; }
int mkdir(const char *path, mode_t mode) { return 0; } /* pretend success */

struct tms { long tms_utime, tms_utime2, tms_cutime, tms_cstime; };
long _times(struct tms *buf)
{
    if (buf) { buf->tms_utime = buf->tms_utime2 = buf->tms_cutime = buf->tms_cstime = 0; }
    return (long)IO_TIMER_PORT;
}

int _gettimeofday(struct timeval *tv, void *tz)
{
    if (tv) {
        uint32_t ms = IO_TIMER_PORT;
        tv->tv_sec = ms / 1000;
        tv->tv_usec = (ms % 1000) * 1000;
    }
    return 0;
}
