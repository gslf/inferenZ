# inferenZ - single-model CPU inference engine for Qwen3.8-Flash-Next (Linux only)
CC      ?= gcc
ARCH    ?= -march=native
CFLAGS  ?= -O3 $(ARCH) -std=gnu23 -fopenmp -Wall -Wextra -Wno-unused-parameter -fno-strict-aliasing -g
LDLIBS  = -lm -lpthread -luring

SRC = src/main.c src/engine.c src/model_fwd.c src/server.c src/http.c src/chat.c src/json.c src/gguf.c \
      src/quant.c src/tokenizer.c src/sampler.c src/threadpool.c src/expert_cache.c src/diag.c \
      src/safetensors.c src/convert.c src/util.c
OBJ = $(SRC:src/%.c=build/%.o)

all: inz

inz: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDLIBS)

build/%.o: src/%.c src/*.h
	@mkdir -p build
	$(CC) $(CFLAGS) -c $< -o $@

build/server.o: web/index.html

build/tokenizer.o: src/unicode_tables.inc

tools/hwprobe: tools/hwprobe.c
	$(CC) -O3 $(ARCH) -pthread -o $@ $< -luring

test: inz tests/test_quant tests/test_tokenizer tests/test_chat tests/test_chat_parse tests/test_forward tests/test_kernels tests/test_kernels_avx2 tests/test_threadpool
	./tests/test_quant 1024 2048
	./tests/test_chat_parse
	./tests/test_forward
	./tests/test_kernels
	./tests/test_kernels_avx2
	./tests/test_threadpool
	python3 tests/test_regressions.py
	python3 tests/code_eval.py --selftest

tests/test_chat: tests/test_chat.c src/chat.c src/json.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_chat_parse: tests/test_chat_parse.c src/chat.c src/json.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_quant: tests/test_quant.c src/quant.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_kernels: tests/test_kernels.c src/quant.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_kernels_avx2: tests/test_kernels.c src/quant.c src/util.c
	$(CC) $(CFLAGS) -mno-avxvnni -mno-avx512vnni -o $@ $^ -lm

tests/test_threadpool: tests/test_threadpool.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ $^ -lpthread

tests/test_forward: tests/test_forward.c src/model_fwd.c src/quant.c src/threadpool.c src/util.c
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -Wl,--gc-sections -o $@ tests/test_forward.c src/quant.c src/threadpool.c src/util.c -lm -lpthread

tests/test_tokenizer: tests/test_tokenizer.c src/tokenizer.c src/json.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -rf build inz tests/test_quant tests/test_tokenizer tests/test_chat tests/test_chat_parse tests/test_forward tests/test_kernels tests/test_kernels_avx2 tests/test_threadpool tools/hwprobe

.PHONY: all test clean
