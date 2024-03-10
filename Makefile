#!/usr/bin/env make

DATEANDTIME=$(shell date +%Y-%m-%dT%T%z)

# External commands
BLACK ?= black
CTAGS ?= ctags
FIND ?= find
PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
SPHINX ?= $(PYTHON) -m sphinx
BENCHMARK ?= py.test
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

TESTCMD ?= $(PYTEST) -p no:cacheprovier
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

PYTHON_DIRS := tests
PYTHON_DIRS += jsonpickle

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
	$(TESTCMD) $(flags)
.PHONY: test

tox::
	$(TOXCMD) $(flags)
.PHONY: tox

benchmark::
	$(BENCHMARKCMD) --benchmark-only --benchmark-histogram=./images/benchmark-$(DATEANDTIME) ./jsonpickle_benchmarks.py
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
