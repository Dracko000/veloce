# Veloce — build & test
#
#   make            compile the example modules to C
#   make demo       build & run the tiny-LM inference demo binary
#   make test       run the compiler + end-to-end test suites (auto venv)
#   make examples   run the transformer and train->deploy end-to-end examples
#   make train      train the tiny ternary classifier, pack weights, mmap-deploy
#   make bench      measure real kernel throughput vs the PRD target
#   make bench-midi end-to-end 275.6M ternary decoder, measured tok/s
#   make plan       print materialisation savings for the transformer layer
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

.PHONY: all demo test examples train bench bench-midi plan bench-throughput venv doctor clean

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

train: $(VENV_STAMP)
	@$(VENV_PY) examples/train_tiny_lm.py

bench: $(VENV_STAMP)
	@$(VENV_PY) benchmarks/bench_throughput.py --json benchmarks/results.json

plan:
	@./velocec check examples/transformer_block.vel

bench-throughput: bench

bench-midi: $(VENV_STAMP)
	@$(VENV_PY) benchmarks/bench_midi_lm.py --verify --json benchmarks/midi_lm_results.json

doctor:
	@echo "python:  $$($(PYTHON) --version 2>&1)"
	@echo "cc:      $$($(CC) --version 2>&1 | head -1)"
	@echo "venv:    $$(test -x $(VENV_PY) && echo present || echo 'missing (run: make venv)')"
	@echo "cores:   $$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
	@echo "numpy:   $$($(PYTHON) -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'not on system python (venv will provide it)')"

clean:
	@rm -rf $(VDIR)
