CC     ?= cc
CFLAGS ?= -O3 -std=gnu11 -Wall -mavx2 -mfma
BIN    ?= ../doomgeneric/doomgeneric/doom_rv32i.bin

all: emulator libemulator.so libemulator_arm32.so test_arm32 emulator_arm32

emulator: emulator.c rv32i.c mamba_nano.c rv32i.h mamba_nano.h
	$(CC) $(CFLAGS) -pthread -o $@ emulator.c rv32i.c mamba_nano.c -lm

emulator_arm32: emulator_arm32.c arm32.c mamba_nano.c arm32.h mamba_nano.h
	$(CC) $(CFLAGS) -o $@ emulator_arm32.c arm32.c mamba_nano.c -lm

libemulator.so: rv32i.c rv32i.h mamba_nano.c mamba_nano.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ rv32i.c mamba_nano.c -lm

libemulator_arm32.so: arm32.c mamba_nano.c arm32.h mamba_nano.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ arm32.c mamba_nano.c -lm

test_arm32: test_arm32.c arm32.c mamba_nano.c arm32.h mamba_nano.h
	$(CC) $(CFLAGS) -o $@ test_arm32.c arm32.c mamba_nano.c -lm

# play in your browser at native speed: http://localhost:8000
gui: emulator
	./emulator --bin $(BIN) --gui

# build WebAssembly module (web/rv32i.js + web/rv32i.wasm)
wasm:
	./build_wasm.sh

# play 100% inside your browser via WebAssembly: http://localhost:8080
web: wasm
	install -m 644 $(BIN) ../web/doom_rv32i.bin
	python3 -m http.server 8080 --directory ../web

# scripted E1M1 gameplay
play: emulator
	./emulator --bin $(BIN) --realtime --steps 105000000

clean:
	rm -f emulator libemulator.so

.PHONY: all gui play wasm web clean
