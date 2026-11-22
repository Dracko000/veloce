# Veloce — build & test
#
#   make            compile the example modules to C
#   make demo       build & run the tiny-LM inference demo binary
#   make test       run the compiler + end-to-end test suites (auto venv)
#   make examples   run the transformer and train->deploy end-to-end examples
#   make train      train the tiny ternary classifier, pack weights, mmap-deploy
#   make bench      measure real kernel throughput vs the PRD target
#   make bench-midi end-to-end 275.6M ternary decoder, measured tok/s
#   make model-module  re-render examples/midi_lm.vel from veloce/lmmodule.py
#   make bench-midi-05b  same bench at d_model=2048 (0.5B-shaped config)
#   make model-1b   1B-class ternary decoder (990M trits + 66M fp16 embed), eager + cached
#   make plan       print materialisation savings for the transformer layer
#   make corpus     regenerate data/corpus.txt from this repo's docs
#   make venv       create .venv and install test dependencies
#   make doctor     check the local toolchain
#   make clean
#
# The compiler itself (velocec + veloce/) is pure Python stdlib and needs no
# virtualenv. Test/example targets use NumPy as an independent reference, so
# they run inside .venv, which is created and populated on first use.

CC       ?= cc
CFLAGS   ?= -std=c11 -O3 -march=native -ffast-math -pthread
INC      := -Iruntime
VDIR     := build
PYTHON   ?= python3
VENV     := .venv
VENV_PY  := $(VENV)/bin/python
VENV_REQ := requirements-test.txt
VENV_STAMP := $(VENV)/.requirements.stamp

.PHONY: all demo test examples kv train train-native bigram corpus bench bench-midi bench-midi-05b bench-roofline model-1b \
        model-module plan bench-throughput venv doctor clean

all: $(VDIR)/transformer_block.c $(VDIR)/tiny_lm.c $(VDIR)/tiny_lm_cls.c $(VDIR)/hello.c

$(VDIR):
	@mkdir -p $(VDIR)

$(VDIR)/%.c: examples/%.vel | $(VDIR)
	@./velocec emit $< -o $@

# ---------------------------------------------------------------------------
# Test environment: created lazily, re-installed only when requirements change
# ---------------------------------------------------------------------------
$(VENV_STAMP): $(VENV_REQ)
	@echo "==> creating test venv in $(VENV)"
	@$(PYTHON) -m venv $(VENV)
	@$(VENV_PY) -m pip install --quiet --upgrade pip
	@$(VENV_PY) -m pip install --quiet -r $(VENV_REQ)
	@touch $@

venv: $(VENV_STAMP)

demo: $(VDIR)/tiny_lm.c
	@$(CC) $(CFLAGS) $(INC) -c $(VDIR)/tiny_lm.c -o $(VDIR)/tiny_lm.o
	@$(CC) $(CFLAGS) $(INC) demo/tiny_lm.c $(VDIR)/tiny_lm.o -lm -o $(VDIR)/tiny_lm
	@echo "--- running demo ---"
	@./$(VDIR)/tiny_lm

test: $(VENV_STAMP)
	@echo "=== unit + end-to-end suite ==="
	@$(VENV_PY) tests/test_veloce.py
	@echo "=== compiler suite (unittest) ==="
	@$(VENV_PY) tests/test_compiler.py

examples: $(VENV_STAMP)
	@echo "=== transformer block vs NumPy reference ==="
	@$(VENV_PY) examples/run_transformer.py

kv: $(VENV_STAMP)
	@echo "=== KV-cache decoding vs NumPy reference ==="
	@$(VENV_PY) examples/run_kv_decode.py

train: $(VENV_STAMP)
	@$(VENV_PY) examples/train_tiny_lm.py

bench: $(VENV_STAMP)
	@$(VENV_PY) benchmarks/bench_throughput.py --json benchmarks/results.json

plan:
	@./velocec check examples/transformer_block.vel

bench-roofline: $(VENV_STAMP)
	@$(VENV_PY) benchmarks/bench_gemv_roofline.py

bench-throughput: bench

# --cached is part of the committed run: the KV-cache block is how the cache
# path is judged, and an artefact with "cached": null cannot answer what
# cached decode costs (which is exactly how it went missing before).
bench-midi: $(VENV_STAMP)
	@$(VENV_PY) benchmarks/bench_midi_lm.py --verify --cached \
		--cached-chunk 2 --json benchmarks/midi_lm_results.json

# Same network, the 0.5B-shaped config.  This is a --config invocation, not a
# second hand-maintained .vel file: veloce/lmmodule.py renders the module for
# whatever shape the config names.
bench-midi-05b: $(VENV_STAMP)
	@$(VENV_PY) benchmarks/bench_midi_lm.py --config D=2048 --config F=5632 \
		--config layers=7 --config name=midi_lm_05b --verify

model-module: $(VENV_STAMP)
	@$(VENV_PY) -c 'from veloce.lmmodule import decoder_module; \
	              open("examples/midi_lm.vel", "w").write(decoder_module({}))'
	@echo "re-rendered examples/midi_lm.vel"

doctor:
	@echo "python:  $$($(PYTHON) --version 2>&1)"
	@echo "cc:      $$($(CC) --version 2>&1 | head -1)"
	@echo "venv:    $$(test -x $(VENV_PY) && echo present || echo 'missing (run: make venv)')"
	@echo "cores:   $$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
	@echo "numpy:   $$($(PYTHON) -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'not on system python (venv will provide it)')"

clean:
	@rm -rf $(VDIR)

# 1B-class decoder: V=32000 D=2048 F=5632 layers=18 -> 990,380,032 ternary params
# + 65,536,000 fp16 embedding = 1.056B total. Weights ~329 MB (1.6 bits/trit).
MODEL_1B = --config name=veloce_1b --config V=32000 --config D=2048 --config F=5632 --config layers=18

model-1b: $(VENV_STAMP)
	$(VENV)/python benchmarks/bench_midi_lm.py $(MODEL_1B) --model-dir models/veloce-1b \
		--iters 5 --json benchmarks/veloce_1b_results.json
	$(VENV)/python benchmarks/bench_midi_lm.py $(MODEL_1B) --model-dir models/veloce-1b \
		--cached --iters 3 --json benchmarks/veloce_1b_cached.json

# --- training without Python ----------------------------------------------
# byte-LM trained entirely in C: forward, backward, Adam, deterministic sampling.
train-native:
	@mkdir -p build
	$(CC) -O3 -march=native -o build/train_native runtime/train_native.c -lm
	./build/train_native data/corpus.txt 3000

# bigram = closed form: count byte pairs once, no gradient iterations.
bigram:
	@mkdir -p build
	$(CC) -O2 -march=native -o build/bigram runtime/bigram.c -lm
	./build/bigram data/corpus.txt

# regenerates data/corpus.txt from this repository's own documentation
corpus:
	$(PYTHON) tools/make_corpus.py
