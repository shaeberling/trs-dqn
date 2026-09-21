
#include <z80.c>
#include <unistd.h>
#include <strings.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>

// Model I specs
#define TIMER_HZ_1 40
#define CLOCK_MHZ_1 1.77408
#define CYCLES_PER_TIMER ((unsigned int) (CLOCK_MHZ_1 * 1000000 / TIMER_HZ_1))

typedef unsigned char (*Z80_MEM_READ_FUNC)(unsigned long, ushort);
typedef void (*Z80_MEM_WRITE_FUNC)(int, ushort, byte);
typedef unsigned char (*Z80_IO_READ_FUNC)(unsigned long, ushort);
typedef void (*Z80_IO_WRITE_FUNC)(int, ushort, byte);

static Z80_MEM_READ_FUNC z80_mem_read_func = NULL;
static Z80_MEM_WRITE_FUNC z80_mem_write_func = NULL;
static Z80_IO_READ_FUNC z80_io_read_func = NULL;
static Z80_IO_WRITE_FUNC z80_io_write_func = NULL;

static volatile int z80_is_running = 1;

static Z80Context ctx;

volatile unsigned char ram[64 * 1024];

/* Optional debugger stop on visible screen text. Stops before a held key can
 * dismiss an episode-ending message. Disabled for ordinary interactive play. */
static ushort video_stop_address = 0;
static unsigned char video_stop_text[64];
static int video_stop_length = 0;
static int video_stop_normalize_text = 0;

void z80_set_video_stop(ushort address, const unsigned char *text, int length)
{
    video_stop_length = 0;
    video_stop_normalize_text = 0;
    if (length <= 0 || length > 64 || address < 0x3c00 ||
        (int)address + length > 0x4000) return;
    video_stop_address = address;
    memcpy(video_stop_text, text, length);
    video_stop_length = length;
}

/* Match the same visible ASCII text as Python's screen reader. The game
 * leaves existing graphics in spaces between printed words, so a text-space
 * can be any non-ASCII-text cell, not just the empty graphics glyph 0x80. */
void z80_set_video_text_stop(ushort address, const unsigned char *text, int length)
{
    z80_set_video_stop(address, text, length);
    video_stop_normalize_text = 1;
}

static int video_stop_matches(void)
{
    if (!video_stop_normalize_text)
        return memcmp((const void *)(ram + video_stop_address),
                      video_stop_text, video_stop_length) == 0;
    for (int i = 0; i < video_stop_length; i++) {
        unsigned char ch = ram[video_stop_address + i];
        if (ch < 32 || ch >= 127) ch = ' ';
        if (ch != video_stop_text[i]) return 0;
    }
    return 1;
}

float screenshot[(2 * 64) * (3 * 16)];

void take_screenshot(int left, int top, int width, int height)
{
    bzero(screenshot, sizeof(screenshot));
    for (int x = left; x < (left + width); x++) {
        for (int y = top; y < (top + height); y++) {
            byte ch = ram[0x3c00 + y * 64 + x];
            if (ch < 0x80 || ch > 0xbf) continue;
            ch -= 0x80;
            int base = (y * 3 * 128) + (x * 2);
            for (int i = 0; i < 6; i++) {
                if (ch & 1) {
                    screenshot[base + (i & 1) + (i >> 1) * 128] = 1.0f;
                }
                ch = ch >> 1;
            }
        }
    }
}

static unsigned char z80_mem_read(unsigned long param, ushort address)
{
    if (z80_mem_read_func != NULL) {
        return (*z80_mem_read_func)(param, address);
    }
    return ram[address];
}

static void z80_mem_write(unsigned long param, ushort address, unsigned char data)
{
    if (z80_mem_write_func != NULL) {
        (*z80_mem_write_func)(param, address, data);
    }
    ram[address] = data;
}

static unsigned char z80_io_read(unsigned long param, ushort address)
{
    if (z80_io_read_func != NULL) {
        return (*z80_io_read_func)(param, address);
    }
    return 255;
}

static void z80_io_write(unsigned long param, ushort address, unsigned char data)
{
    if (z80_io_write_func != NULL) {
        (*z80_io_write_func)(param, address, data);
    }
}

void z80_init_callbacks(Z80_MEM_READ_FUNC mem_read_func,
                        Z80_MEM_WRITE_FUNC mem_write_func,
                        Z80_IO_READ_FUNC io_read_func,
                        Z80_IO_WRITE_FUNC io_write_func)
{
    z80_mem_read_func = mem_read_func;
    z80_mem_write_func = mem_write_func;
    z80_io_read_func = io_read_func;
    z80_io_write_func = io_write_func;
}

static int get_ticks()
{
    static struct timeval start_tv, now;
    static int init = 0;

    if (!init) {
       gettimeofday(&start_tv, NULL);
       init = 1;
    }

    gettimeofday(&now, NULL);
    return (now.tv_sec - start_tv.tv_sec) * 1000 +
                 (now.tv_usec - start_tv.tv_usec) / 1000;
}

static void delay(int ms)
{
    int was_error;
    struct timespec elapsed, tv;

    elapsed.tv_sec = ms / 1000;
    elapsed.tv_nsec = (ms % 1000) * 1000000;
    do {
        errno = 0;

        tv.tv_sec = elapsed.tv_sec;
        tv.tv_nsec = elapsed.tv_nsec;
        was_error = nanosleep(&tv, &elapsed);
    } while (was_error && (errno == EINTR));
}

static void sync_time_with_host()
{
	int curtime;
	int deltatime;
    static int lasttime = 0;

    deltatime = 1000 / TIMER_HZ_1;

	curtime = get_ticks();

	if (lasttime + deltatime > curtime) {
		delay(lasttime + deltatime - curtime);
    }
	curtime = get_ticks();

	lasttime += deltatime;
	if ((lasttime + deltatime) < curtime) {
		lasttime = curtime;
	}
}

void z80_reset(ushort entryAddr)
{
    bzero(&ctx, sizeof(Z80Context));
    Z80RESET(&ctx);
    ctx.PC = entryAddr;
    ctx.memRead = z80_mem_read;
    ctx.memWrite = z80_mem_write;
    ctx.ioRead = z80_io_read;
    ctx.ioWrite = z80_io_write;
}

void z80_run(ushort entryAddr)
{
    z80_reset(entryAddr);

    while (z80_is_running) {
        Z80Execute(&ctx);
        if (ctx.tstates >= CYCLES_PER_TIMER) {
		    sync_time_with_host();
		    ctx.tstates -=  CYCLES_PER_TIMER;
		}
    }
}

void z80_set_running(int is_running)
{
    z80_is_running = is_running;
}

int z80_run_for_tstates(int tstates, int original_speed)
{
    if (!original_speed) {
        ctx.tstates = 0;
    }
    int threshold_tstates = ctx.tstates + tstates;
    while (ctx.tstates <= threshold_tstates) {
        Z80Execute(&ctx);
        if (video_stop_length &&
            ram[video_stop_address] == video_stop_text[0] &&
            video_stop_matches()) break;
        if (original_speed && (ctx.tstates >= CYCLES_PER_TIMER)) {
            sync_time_with_host();
            ctx.tstates -=  CYCLES_PER_TIMER;
            threshold_tstates -= CYCLES_PER_TIMER;
        }
    }
    return ctx.tstates - threshold_tstates;
}

/* Opaque, same-build snapshots for headless RL environment resets. These are
 * never policy inputs. Host pointers are neither exported nor trusted on load.
 * Custom callbacks and real-time/UI execution are not supported by this API. */
typedef struct {
    uint32_t magic, version, context_size;
    Z80Context cpu;
    unsigned char memory[64 * 1024];
    ushort stop_address;
    unsigned char stop_text[64];
    int stop_length, stop_normalize;
} TRSSnapshot;

size_t z80_snapshot_size(void)
{
    return sizeof(TRSSnapshot);
}

static int snapshot_callbacks_supported(void)
{
    return !z80_mem_read_func && !z80_mem_write_func &&
           !z80_io_read_func && !z80_io_write_func;
}

int z80_save_snapshot(void *destination, size_t length)
{
    if (!destination || length != sizeof(TRSSnapshot) ||
        !snapshot_callbacks_supported()) return 0;
    TRSSnapshot saved;
    memset(&saved, 0, sizeof(saved));
    saved.magic = 0x54525353;
    saved.version = 1;
    saved.context_size = sizeof(Z80Context);
    saved.cpu = ctx;
    saved.cpu.memRead = NULL;
    saved.cpu.memWrite = NULL;
    saved.cpu.ioRead = NULL;
    saved.cpu.ioWrite = NULL;
    saved.cpu.memParam = saved.cpu.ioParam = 0;
    memcpy(saved.memory, (const void *)ram, sizeof(saved.memory));
    saved.stop_address = video_stop_address;
    memcpy(saved.stop_text, video_stop_text, sizeof(saved.stop_text));
    saved.stop_length = video_stop_length;
    saved.stop_normalize = video_stop_normalize_text;
    memcpy(destination, &saved, sizeof(saved));
    return 1;
}

int z80_restore_snapshot(const void *source, size_t length)
{
    if (!source || length != sizeof(TRSSnapshot) ||
        !snapshot_callbacks_supported()) return 0;
    TRSSnapshot saved;
    memcpy(&saved, source, sizeof(saved));
    if (saved.magic != 0x54525353 || saved.version != 1 ||
        saved.context_size != sizeof(Z80Context) ||
        saved.stop_length < 0 || saved.stop_length > 64 ||
        (saved.stop_normalize != 0 && saved.stop_normalize != 1) ||
        (saved.stop_length && (saved.stop_address < 0x3c00 ||
         (int)saved.stop_address + saved.stop_length > 0x4000))) return 0;
    ctx = saved.cpu;
    ctx.memRead = z80_mem_read;
    ctx.memWrite = z80_mem_write;
    ctx.ioRead = z80_io_read;
    ctx.ioWrite = z80_io_write;
    ctx.memParam = ctx.ioParam = 0;
    memcpy((void *)ram, saved.memory, sizeof(saved.memory));
    video_stop_address = saved.stop_address;
    memcpy(video_stop_text, saved.stop_text, sizeof(video_stop_text));
    video_stop_length = saved.stop_length;
    video_stop_normalize_text = saved.stop_normalize;
    return 1;
}
