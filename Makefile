#!/usr/bin/env make

DATEANDTIME=$(shell date +%Y-%m-%dT%T%z)

# External commands
BLACK ?= black
CTAGS ?= ctags
FIND ?= find
PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
SPHINX ?= $(PYTHON) -m sphinx
BENCHMARK ?= $(PYTEST)
RM_R ?= rm -fr
TOX ?= tox
# Detect macOS to customize how we query the cpu count.
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo unknown')
ifeq ($(uname_S),Darwin)
    NPROC ?= sysctl -n hw.activecpu
else
    NPROC ?= nproc
endif

# Options
flags ?=

# Capture extra arguments for the benchmark target
EXTRA_BENCH_ARGS ?= 

# Default job count -- this is used if "-j" is not present in MAKEFLAGS.
nproc := $(shell sh -c '$(NPROC) 2>/dev/null || echo 4')

# Extract the "-j#" flags in $(MAKEFLAGS) so that we can forward the value to
# other commands.  This can be empty.
JOB_FLAGS := $(shell echo -- $(MAKEFLAGS) | grep -o -e '-j[0-9]\+' | head -n 1)
# Extract just the number from "-j#".
JOB_COUNT := $(shell printf '%s' "$(JOB_FLAGS)" | sed -e 's/-j//')
# We have "-jX" from MAKEFLAGS but tox wants "--parallel X"
DASH_J := $(shell echo -- $(JOB_FLAGS) -j$(nproc) | grep -o -e '-j[0-9]\+' | head -n 1)
NUM_JOBS := $(shell printf %s "$(DASH_J)" | sed -e 's/-j//')

TESTCMD ?= $(PYTEST)
BENCHMARKCMD ?= $(BENCHMARK)
TOXCMD ?= $(TOX) run-parallel --parallel-live
ifdef V
    TESTCMD += --verbose
    TOXCMD += -v
    BENCHMARKCMD += --benchmark-verbose
endif

# Data
ARTIFACTS := build
ARTIFACTS += dist
ARTIFACTS += __pycache__
ARTIFACTS += tags
ARTIFACTS += *.egg-info
ARTIFACTS += fuzz_*.pkg.spec # Files created by OSS-Fuzz when running locally

PYTHON_DIRS := tests
PYTHON_DIRS += jsonpickle
PYTHON_DIRS += fuzzing

# The default target of this makefile is....
all:: help

help::
	@echo "---- Makefile Targets ----"
	@echo "make help           - print this message"
	@echo "make test           - run unit tests"
	@echo "make tox            - run unit tests using tox"
	@echo "make clean          - remove cruft"
	@echo "make benchmark      - run pytest benchmarking"
	@echo "make doc            - generate documentation using sphinx"
.PHONY: help

test::
	$(TESTCMD) jsonpickle tests $(flags)
.PHONY: test

tox::
	$(TOXCMD) $(flags)
.PHONY: tox

benchmark::
	@if [ "$(uname_S)" = "Linux" ]; then \
		echo "Operating System detected: Linux"; \
		\
		if [ -f /etc/os-release ]; then \
			. /etc/os-release; \
			if [ "$$ID" = "ubuntu" ]; then \
				echo "Ubuntu detected."; \
				\
				if ! command -v taskset >/dev/null 2>&1; then \
					echo "'taskset' not found. We need this to reduce noise. Installing util-linux..."; \
					\
					sudo apt-get update && \
					\
					sudo apt-get install -y util-linux && \
					\
					if ! command -v taskset >/dev/null 2>&1; then \
						echo "Failed to install util-linux. Please install it manually."; \
						exit 1; \
					else \
						echo "Successfully installed util-linux."; \
					fi; \
				else \
					echo "'taskset' is already installed."; \
				fi; \
			else \
				echo "Non-Ubuntu Linux distribution detected. Skipping util-linux installation."; \
			fi; \
		else \
			echo "Cannot detect OS. Skipping util-linux installation."; \
		fi; \
		\
		NUM_CORES=`$(NPROC)`; \
		\
		# use #cores / 2 - 1 because cores 0, 1, and last get most noise \
		# and we divide by 2 to get physical cores for systems with SMT (most x86) \
		# so this works well for 4+ core machines \
		if [ $$NUM_CORES -gt 16 ]; then \
			TARGET_CORE=7; \
		else \
			TARGET_CORE=$$((NUM_CORES / 2 - 1)); \
		fi; \
		\
		echo "Running '$(BENCHMARKCMD)' on core $$TARGET_CORE of $$NUM_CORES cores."; \
		\
		# execute the benchmark command with core affinity set to the target core \
		# disable gc for reproducibility, warmup on for the experimental python 3.13 JIT \
		taskset -c $$TARGET_CORE $(BENCHMARKCMD) --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-min-rounds=10000 --benchmark-histogram=./images/benchmark-$(DATEANDTIME) $(EXTRA_BENCH_ARGS) ./jsonpickle_benchmarks.py; \
	else \
	    $(BENCHMARKCMD) --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-min-rounds=10000 --benchmark-histogram=./images/benchmark-$(DATEANDTIME) $(EXTRA_BENCH_ARGS) ./jsonpickle_benchmarks.py; \
		echo "The 'benchmark' target has much less noise on Linux, try running it on there!"; \
	fi
.PHONY: benchmark

doc::
	$(SPHINX) docs build/html

tags::
	$(FIND) $(PYTHON_DIRS) -name '*.py' -print0 | xargs -0 $(CTAGS) -f tags

clean::
	$(FIND) $(PYTHON_DIRS) -name '*.py[cod]' -print0 | xargs -0 rm -f
	$(FIND) $(PYTHON_DIRS) -name '__pycache__' -print0 | xargs -0 rm -fr
	$(RM_R) $(ARTIFACTS)
.PHONY: clean

format::
	$(BLACK) --skip-string-normalization --target-version py310 $(PYTHON_DIRS)
.PHONY: format
