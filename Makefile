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

test: inz tests/test_quant tests/test_tokenizer tests/test_chat tests/test_chat_parse
	./tests/test_quant 1024 2048
	./tests/test_chat_parse
	python3 tests/code_eval.py --selftest

tests/test_chat: tests/test_chat.c src/chat.c src/json.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_chat_parse: tests/test_chat_parse.c src/chat.c src/json.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_quant: tests/test_quant.c src/quant.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

tests/test_tokenizer: tests/test_tokenizer.c src/tokenizer.c src/json.c src/util.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -rf build inz tests/test_quant tests/test_tokenizer tests/test_chat tests/test_chat_parse tools/hwprobe

.PHONY: all test clean
