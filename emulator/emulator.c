/* emulator.c - native RV32I DOOM emulator with tracing, checkpointing, and GUI */
#include "rv32i.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

static RV32I_CPU *g_cpu;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int gui_mode = 0;
static const char *frame_path = "/tmp/doom_frame.ppm";
static double g_speed = 1.0;
static int g_speed_changed = 0;

/* ---------------- live GUI: tiny HTTP server (frames out, keys in, speed) -------- */
static const char HTML[] =
"<!doctype html><html><head><meta charset='utf-8'><title>DOOM rv32i native</title>\n"
"<style>\n"
"body{background:#111;color:#ccc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,monospace;text-align:center;margin:0;padding:16px}\n"
"#view{display:inline-block;box-shadow:0 4px 24px rgba(0,0,0,0.9);border:2px solid #333;border-radius:4px;background:#000}\n"
"canvas{width:960px;height:600px;image-rendering:pixelated;display:block;cursor:pointer}\n"
".panel{width:960px;margin:12px auto 0;background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:12px 16px;box-sizing:border-box;display:flex;flex-direction:column;gap:10px}\n"
".row{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:13px}\n"
".grp{display:flex;align-items:center;gap:8px}\n"
"input[type=range]{width:200px;accent-color:#e22;cursor:pointer}\n"
".btn{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:6px 12px;font-size:12px;border-radius:4px;cursor:pointer;transition:all 0.15s;font-family:inherit}\n"
".btn:hover{background:#3a3a3a;border-color:#666;color:#fff}\n"
".btn-reset{background:#911;border-color:#b22;color:#fff;font-weight:bold}\n"
".btn-reset:hover{background:#c22;border-color:#e33}\n"
".badge{background:#222;padding:4px 10px;border-radius:4px;border:1px solid #333;font-family:monospace;font-size:13px;color:#4af;min-width:125px;display:inline-block;text-align:center}\n"
".help{color:#777;font-size:11px}\n"
"</style></head>\n"
"<body>\n"
"<div id='view'><canvas id='c' width='320' height='200'></canvas></div>\n"
"<div class='panel'>\n"
"  <div class='row'>\n"
"    <div class='grp'>\n"
"      <label for='sp'><strong>Speed:</strong></label>\n"
"      <input type='range' id='sp' min='0.1' max='5.0' step='0.05' value='1.0'>\n"
"      <span class='badge' id='sp-val'>1.00x (Native)</span>\n"
"    </div>\n"
"    <div class='grp'>\n"
"      <button class='btn' onclick='setSp(0.25)'>0.25x</button>\n"
"      <button class='btn' onclick='setSp(0.5)'>0.5x</button>\n"
"      <button class='btn btn-reset' id='reset-btn' onclick='setSp(1.0)'>Reset to Native Speed (1.0x)</button>\n"
"      <button class='btn' onclick='setSp(2.0)'>2.0x</button>\n"
"      <button class='btn' onclick='setSp(0.0)'>Max (Uncapped)</button>\n"
"    </div>\n"
"  </div>\n"
"  <div class='help'>ARROWS move &middot; CTRL fire &middot; SPACE use &middot; SHIFT run &middot; ENTER/ESC menus &middot; Click canvas to play</div>\n"
"</div>\n"
"<script>\n"
"const cv=document.getElementById('c'),cx=cv.getContext('2d'),img=cx.createImageData(320,200);\n"
"const sl=document.getElementById('sp'),lbl=document.getElementById('sp-val');\n"
"function updLbl(v){\n"
"  if(v<=0.001){lbl.textContent='Max (Uncapped)';lbl.style.color='#f55';}\n"
"  else if(Math.abs(v-1.0)<0.01){lbl.textContent='1.00x (Native)';lbl.style.color='#4af';}\n"
"  else{lbl.textContent=v.toFixed(2)+'x';lbl.style.color=v>1.0?'#fa4':'#6df';}\n"
"}\n"
"async function setSp(v){\n"
"  sl.value=v<=0.001?0.1:Math.min(v,5.0);\n"
"  updLbl(v);\n"
"  sl.blur();\n"
"  try{await fetch('/speed?v='+v);}catch(e){}\n"
"}\n"
"sl.addEventListener('input',e=>{const v=parseFloat(e.target.value);updLbl(v);fetch('/speed?v='+v);});\n"
"sl.addEventListener('change',e=>{const v=parseFloat(e.target.value);setSp(v);});\n"
"(async()=>{\n"
"  for(;;){\n"
"    try{\n"
"      const b=await fetch('/frame',{cache:'no-store'});\n"
"      const a=new Uint8ClampedArray(await b.arrayBuffer());\n"
"      for(let i=0,p=0;i<a.length;i+=3,p+=4){img.data[p]=a[i];img.data[p+1]=a[i+1];img.data[p+2]=a[i+2];img.data[p+3]=255;}\n"
"      cx.putImageData(img,0,0);\n"
"    }catch(e){}\n"
"    await new Promise(r=>setTimeout(r,16));\n"
"  }\n"
"})();\n"
"const K={Enter:13,Escape:27,Space:162,KeyE:162,ArrowUp:173,KeyW:173,ArrowDown:175,KeyS:175,ArrowLeft:172,KeyA:172,ArrowRight:174,KeyD:174,\n"
"ControlLeft:163,ControlRight:163,KeyF:163,ShiftLeft:182,ShiftRight:182,AltLeft:184,AltRight:184};\n"
"addEventListener('keydown',e=>{\n"
"  if((e.target===sl||e.target.tagName==='BUTTON')&&['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space','Enter'].includes(e.code)){\n"
"    e.target.blur();\n"
"  }\n"
"  const k=K[e.code];\n"
"  if(k!==undefined){\n"
"    e.preventDefault();\n"
"    if(!e.repeat)fetch('/key?c='+k+'&d=1');\n"
"  }\n"
"});\n"
"addEventListener('keyup',e=>{\n"
"  const k=K[e.code];\n"
"  if(k!==undefined){\n"
"    e.preventDefault();\n"
"    fetch('/key?c='+k+'&d=0');\n"
"  }\n"
"});\n"
"</script></body></html>";

static void send_all(int c, const void *b, size_t n) {
    const uint8_t *p = (const uint8_t *)b;
    while (n) {
        ssize_t r = send(c, p, n, MSG_NOSIGNAL);
        if (r <= 0) break;
        p += r; n -= r;
    }
}

static void *server_thread(void *arg) {
    int port = (int)(intptr_t)arg;
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa; memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_ANY); sa.sin_port = htons((uint16_t)port);
    if (bind(ls, (struct sockaddr *)&sa, sizeof(sa)) || listen(ls, 16)) { perror("bind"); exit(1); }
    fprintf(stderr, "[GUI] open http://localhost:%d/\n", port);
    char cmd[256]; snprintf(cmd, sizeof(cmd), "xdg-open http://localhost:%d/ >/dev/null 2>&1 &", port);
    int sys_res = system(cmd); (void)sys_res;
    for (;;) {
        int c = accept(ls, NULL, NULL);
        if (c < 0) continue;
        char req[1024]; ssize_t r = recv(c, req, sizeof(req) - 1, 0);
        if (r <= 0) { close(c); continue; }
        req[r] = 0;
        char head[256];
        if (strncmp(req, "GET /frame", 10) == 0) {
            static uint8_t rgb[SCREEN_W * SCREEN_H * 3];
            pthread_mutex_lock(&g_mu);
            rv32i_get_frame_rgb(g_cpu, rgb);
            pthread_mutex_unlock(&g_mu);
            snprintf(head, sizeof(head),
                     "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: %d\r\n"
                     "Cache-Control: no-store\r\nConnection: close\r\n\r\n", SCREEN_W * SCREEN_H * 3);
            send_all(c, head, strlen(head)); send_all(c, rgb, sizeof(rgb));
        } else if (strncmp(req, "GET /key", 8) == 0) {
            int code = 0, d = 0; char *q = strchr(req, '?');
            if (q) {
                char *pc = strstr(q, "c="), *pd = strstr(q, "&d=");
                if (pc) code = atoi(pc + 2);
                if (pd) d = atoi(pd + 3);
            }
            pthread_mutex_lock(&g_mu);
            rv32i_queue_key(g_cpu, 0, (uint32_t)code, d, -1);
            pthread_mutex_unlock(&g_mu);
            snprintf(head, sizeof(head), "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nCache-Control: no-store\r\n"
                     "Connection: close\r\n\r\nok");
            send_all(c, head, strlen(head));
        } else if (strncmp(req, "GET /speed", 10) == 0) {
            char *q = strchr(req, '?');
            if (q) {
                char *pv = strstr(q, "v=");
                if (pv) {
                    double val = atof(pv + 2);
                    if (val < 0.0) val = 0.0;
                    if (val > 20.0) val = 20.0;
                    pthread_mutex_lock(&g_mu);
                    g_speed = val;
                    g_speed_changed = 1;
                    pthread_mutex_unlock(&g_mu);
                }
            }
            char body[128];
            pthread_mutex_lock(&g_mu);
            double cur_sp = g_speed;
            pthread_mutex_unlock(&g_mu);
            int blen = snprintf(body, sizeof(body), "{\"speed\":%.2f}\n", cur_sp);
            snprintf(head, sizeof(head),
                     "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n"
                     "Cache-Control: no-store\r\nConnection: close\r\n\r\n", blen);
            send_all(c, head, strlen(head)); send_all(c, body, blen);
        } else {
            snprintf(head, sizeof(head), "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %zu\r\n"
                     "Connection: close\r\n\r\n", strlen(HTML));
            send_all(c, head, strlen(head)); send_all(c, HTML, strlen(HTML));
        }
        close(c);
    }
    return NULL;
}

static void escape_json_str(FILE *f, const char *s) {
    while (*s) {
        if (*s == '\n') fputs("\\n", f);
        else if (*s == '\r') fputs("\\r", f);
        else if (*s == '\t') fputs("\\t", f);
        else if (*s == '\"') fputs("\\\"", f);
        else if (*s == '\\') fputs("\\\\", f);
        else fputc(*s, f);
        s++;
    }
}

static void write_json_pair(FILE *f, const char *prompt, const char *target) {
    fputs("{\"state\": \"", f);
    escape_json_str(f, prompt);
    fputs("\", \"next\": \"", f);
    escape_json_str(f, target);
    fputs("\"}\n", f);
}

static int parse_key_code(const char *name) {
    if (!strcasecmp(name, "ENTER")) return 13;
    if (!strcasecmp(name, "ESC")) return 27;
    if (!strcasecmp(name, "TAB")) return 9;
    if (!strcasecmp(name, "SPACE")) return 0x20;
    if (!strcasecmp(name, "UP")) return 0xAD;
    if (!strcasecmp(name, "DOWN")) return 0xAF;
    if (!strcasecmp(name, "LEFT")) return 0xAC;
    if (!strcasecmp(name, "RIGHT")) return 0xAE;
    if (!strcasecmp(name, "FIRE")) return 0xA3;
    if (!strcasecmp(name, "USE")) return 0xA2;
    if (!strcasecmp(name, "SHIFT")) return 0xB6;
    if (!strcasecmp(name, "CTRL")) return 0x9D;
    if (!strcasecmp(name, "ALT")) return 0xB8;
    return (int)strtol(name, NULL, 0);
}

static void parse_custom_schedule(RV32I_CPU *cpu, const char *spec, uint64_t start, uint64_t every) {
    char *buf = strdup(spec);
    if (!buf) return;
    char *saveptr1 = NULL;
    char *token = strtok_r(buf, ",", &saveptr1);
    uint64_t cur_step = start;
    while (token) {
        while (*token == ' ') token++;
        char key_s[32] = {0}, state_s[32] = {0}, cond_s[32] = {0};
        int parts = sscanf(token, "%31[^:]:%31[^:]:%31s", key_s, state_s, cond_s);
        if (parts >= 2) {
            int code = parse_key_code(key_s);
            int down = (!strcasecmp(state_s, "down") || !strcasecmp(state_s, "press") || !strcasecmp(state_s, "pressed"));
            int gs_cond = -1;
            if (parts >= 3 && cond_s[0] == 'g' && cond_s[1] == 's') {
                gs_cond = atoi(cond_s + 2);
            }
            rv32i_queue_key(cpu, cur_step, (uint32_t)code, down, gs_cond);
            cur_step += every;
        }
        token = strtok_r(NULL, ",", &saveptr1);
    }
    free(buf);
}

int main(int argc, char **argv) {
    const char *bin = "doomgeneric/doomgeneric/doom_rv32i.bin";
    const char *symbols = NULL;
    uint64_t max_steps = 105000000ULL;
    int realtime = 0, dump_every_ms = 0, no_input = 0;
    const char *dataset_path = NULL;
    uint64_t sample_every = 100000ULL;
    uint64_t ds_lo = 0, ds_hi = ~0ULL;
    uint64_t checkpoint_at = 0;
    const char *checkpoint_file = "data/checkpoint.bin";
    const char *input_spec = NULL;
    uint64_t input_start = 60000000ULL;
    uint64_t input_every = 3000000ULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bin") && i + 1 < argc) bin = argv[++i];
        else if (!strcmp(argv[i], "--symbols") && i + 1 < argc) symbols = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) max_steps = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--realtime")) realtime = 1;
        else if (!strcmp(argv[i], "--frame") && i + 1 < argc) frame_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-every") && i + 1 < argc) dump_every_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-input")) no_input = 1;
        else if (!strcmp(argv[i], "--input") && i + 1 < argc) input_spec = argv[++i];
        else if (!strcmp(argv[i], "--input-start") && i + 1 < argc) input_start = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--input-every") && i + 1 < argc) input_every = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--dataset") && i + 1 < argc) dataset_path = argv[++i];
        else if (!strcmp(argv[i], "--sample-every") && i + 1 < argc) sample_every = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--dataset-range") && i + 1 < argc) {
            sscanf(argv[++i], "%llu-%llu", (unsigned long long *)&ds_lo, (unsigned long long *)&ds_hi);
        }
        else if (!strcmp(argv[i], "--checkpoint-at") && i + 1 < argc) checkpoint_at = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--checkpoint-file") && i + 1 < argc) checkpoint_file = argv[++i];
        else if (!strcmp(argv[i], "--gui")) { gui_mode = 1; realtime = 1; max_steps = 1ULL << 62; }
        else if (!strcmp(argv[i], "--speed") && i + 1 < argc) {
            g_speed = atof(argv[++i]);
            if (g_speed < 0.0) g_speed = 0.0;
            realtime = 1;
        }
    }

    g_cpu = rv32i_create();
    if (!g_cpu) {
        fprintf(stderr, "Failed to allocate CPU state\n");
        return 1;
    }

    if (!symbols) {
        static char spath[1024];
        const char *slash = strrchr(bin, '/');
        size_t dirlen = slash ? (size_t)(slash - bin + 1) : 0;
        if (dirlen + sizeof("symbols.txt") <= sizeof(spath)) {
            memcpy(spath, bin, dirlen);
            strcpy(spath + dirlen, "symbols.txt");
            symbols = spath;
        }
    }
    if (symbols && rv32i_load_symbols(g_cpu, symbols)) {
        fprintf(stderr, "[SYMBOLS] gamestate from %s\n", symbols);
    } else if (getenv("EMULATOR_QUIET") == NULL) {
        fprintf(stderr, "[SYMBOLS] %s not found; using reference-build defaults\n", symbols ? symbols : "symbols.txt");
    }

    if (rv32i_load_binary(g_cpu, bin) <= 0) {
        perror(bin);
        return 1;
    }

    int gui_port = 8000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gui") && i + 1 < argc) {
            int p = atoi(argv[i + 1]);
            if (p > 0) gui_port = p;
        }
    }

    if (gui_mode) {
        pthread_t th;
        pthread_create(&th, NULL, server_thread, (void *)(intptr_t)gui_port);
        pthread_detach(th);
    } else if (!no_input) {
        if (input_spec) {
            parse_custom_schedule(g_cpu, input_spec, input_start, input_every);
            fprintf(stderr, "[INPUT] scheduled custom events from --input\n");
        } else {
            /* Default scripted schedule */
            struct { uint64_t at; uint32_t key; int down; int gs; } sched[] = {
                {60000000,13,1,-1},{60200000,13,0,-1},{63000000,13,1,-1},{63200000,13,0,-1},
                {66000000,13,1,-1},{66200000,13,0,-1},{69000000,13,1,-1},{69200000,13,0,-1},
                {74000000,0xad,1,0},{80000000,0xa3,1,-1},{81000000,0xa3,0,-1},{95000000,0xad,0,0},
            };
            for (size_t i = 0; i < sizeof(sched)/sizeof(*sched); i++) {
                rv32i_queue_key(g_cpu, sched[i].at, sched[i].key, sched[i].down, sched[i].gs);
            }
        }
    }

    FILE *ds_file = NULL;
    uint64_t ds_count = 0;
    if (dataset_path) {
        ds_file = fopen(dataset_path, "w");
        if (!ds_file) { perror(dataset_path); return 1; }
        fprintf(stderr, "[DATASET] recording pairs every %llu steps in [%llu, %llu) -> %s\n",
                (unsigned long long)sample_every, (unsigned long long)ds_lo,
                (unsigned long long)ds_hi, dataset_path);
    }

    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t ms_anchor_steps = 0;
    struct timespec wall_anchor; clock_gettime(CLOCK_MONOTONIC, &wall_anchor);
    uint64_t last_dump_ms = 0;
    uint32_t last_gs = 0xFFFFFFFFu;

    static char p_buf[1024];
    static char t_buf[256];

    while (g_cpu->steps < max_steps && !g_cpu->halted) {
        int do_trace = (ds_file && g_cpu->steps >= ds_lo && g_cpu->steps < ds_hi &&
                        (g_cpu->steps % sample_every == 0));

        if (do_trace) {
            if (gui_mode) pthread_mutex_lock(&g_mu);
            int ok = rv32i_step_trace(g_cpu, p_buf, sizeof(p_buf), t_buf, sizeof(t_buf));
            if (gui_mode) pthread_mutex_unlock(&g_mu);
            if (ok) {
                write_json_pair(ds_file, p_buf, t_buf);
                ds_count++;
            }
        } else {
            if (gui_mode) pthread_mutex_lock(&g_mu);
            rv32i_step(g_cpu);
            if (gui_mode) pthread_mutex_unlock(&g_mu);
        }

        if (checkpoint_at && g_cpu->steps >= checkpoint_at) {
            if (rv32i_save_checkpoint(g_cpu, checkpoint_file) == 0) {
                fprintf(stderr, "[CHECKPOINT] saved %s at step %llu, pc=%08x\n",
                        checkpoint_file, (unsigned long long)g_cpu->steps, g_cpu->pc);
            } else {
                fprintf(stderr, "[CHECKPOINT] failed to save %s\n", checkpoint_file);
            }
            break;
        }

        uint32_t gs = rv32i_probe_gs(g_cpu);
        if (gs != last_gs) {
            last_gs = gs;
            fprintf(stderr, "[GS] gamestate=%u gametic=%u at step %llu (%llu ms)\n",
                    gs, rv32i_probe_gametic(g_cpu), (unsigned long long)g_cpu->steps,
                    (unsigned long long)(g_cpu->steps / TIMER_STEPS_PER_MS));
        }

        if (g_cpu->steps % 1000000 == 0 && g_cpu->steps) {
            fprintf(stderr, "step %llums=%llu gametic=%u gs=%u\n",
                    (unsigned long long)g_cpu->steps, (unsigned long long)(g_cpu->steps / 10000),
                    rv32i_probe_gametic(g_cpu), gs);
        }

        if (g_cpu->steps % 10000 == 0) {
            uint64_t ms = g_cpu->steps / 10000;
            if (dump_every_ms && ms - last_dump_ms >= (uint64_t)dump_every_ms) {
                last_dump_ms = ms;
                rv32i_dump_frame(g_cpu, frame_path);
            }
            if (realtime) {
                pthread_mutex_lock(&g_mu);
                double sp = g_speed;
                int changed = g_speed_changed;
                if (changed) g_speed_changed = 0;
                pthread_mutex_unlock(&g_mu);

                if (changed) {
                    ms_anchor_steps = g_cpu->steps;
                    clock_gettime(CLOCK_MONOTONIC, &wall_anchor);
                }

                if (sp > 0.001) {
                    struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
                    uint64_t elapsed_us = (uint64_t)((now.tv_sec - wall_anchor.tv_sec) * 1000000ULL
                                        + (now.tv_nsec - wall_anchor.tv_nsec) / 1000);
                    uint64_t steps_done = g_cpu->steps - ms_anchor_steps;
                    uint64_t target_us = (uint64_t)((double)steps_done / (10.0 * sp));
                    if (target_us > elapsed_us) {
                        uint64_t diff_us = target_us - elapsed_us;
                        struct timespec sl = { (time_t)(diff_us / 1000000ULL), (long)((diff_us % 1000000ULL) * 1000) };
                        nanosleep(&sl, NULL);
                    } else if (elapsed_us > target_us + 100000ULL) {
                        /* Lag recovery: if emulator falls behind wall time by >100ms, re-anchor */
                        ms_anchor_steps = g_cpu->steps;
                        clock_gettime(CLOCK_MONOTONIC, &wall_anchor);
                    }
                }
            }
        }
    }

    if (ds_file) {
        fclose(ds_file);
        fprintf(stderr, "[DATASET] wrote %llu pairs to %s\n", (unsigned long long)ds_count, dataset_path);
    }

    rv32i_dump_frame(g_cpu, frame_path);
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "[NATIVE] %llu steps in %.2fs = %.0f steps/s (game %llums)\n",
            (unsigned long long)g_cpu->steps, dt, g_cpu->steps / (dt > 0 ? dt : 1e-9),
            (unsigned long long)(g_cpu->steps / 10000));

    rv32i_free(g_cpu);
    return 0;
}
