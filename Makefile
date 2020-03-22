#!/usr/bin/env make

# External commands
CTAGS ?= ctags
FIND ?= find
NPROC ?= nproc
PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest
RM_R ?= rm -fr
SH ?= sh
TOX ?= tox

# Options
flags ?=
timeout ?= 600

# Default job count -- this is used if "-j" is not present in MAKEFLAGS.
nproc := $(shell $(NPROC) 2>/dev/null || echo 4)
# Extract the "-j#" flags in $(MAKEFLAGS) so that we can forward the value to
# other commands.  This can be empty.
JOB_FLAGS := $(shell echo -- $(MAKEFLAGS) | grep -o -e '-j[0-9]\+' | head -n 1)
# Extract just the number from "-j#".
JOB_COUNT := $(shell printf %s "$(JOB_FLAGS)" | sed -e 's/-j//')
# We have "-jX" from MAKEFLAGS but tox wants "-j X"
DASH_J := $(shell echo -- $(JOB_FLAGS) -j$(nproc) | grep -o -e '-j[0-9]\+' | head -n 1)
NUM_JOBS := $(shell printf %s "$(DASH_J)" | sed -e 's/-j//')

TESTCMD ?= $(PYTEST)
TOXCMD ?= $(TOX)
TOXCMD += --parallel $(NUM_JOBS)
TOXCMD += --develop --skip-missing-interpreters
ifdef multi
    TOXENV ?= 'py{26,27,32,33,34,35,36,37,38},py{27,37}-sa{10,11,12,13},py{27,37}-libs'
    TOXCMD += -e $(TOXENV)
endif
ifdef V
    TESTCMD += --verbose
    TOXCMD += -v
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

help:
	@echo "---- Makefile Targets ----"
	@echo "make help    - print this message"
	@echo "make test    - run unit tests"
	@echo "make tox     - run unit tests on multiple pythons with tox"
	@echo "make clean   - remove cruft"
.PHONY: help

test:
	$(TESTCMD) $(flags)
.PHONY: test

tox:
	$(TOXCMD) $(flags)
.PHONY: tox

check:
	$(TOXCMD) -e flake8 $(flags)
.PHONY: check

tags:
	$(FIND) $(PYTHON_DIRS) -name '*.py' -print0 | xargs -0 $(CTAGS) -f tags

clean:
	$(FIND) $(PYTHON_DIRS) -name '*.py[cod]' -print0 | xargs -0 rm -f
	$(FIND) $(PYTHON_DIRS) -name '__pycache__' -print0 | xargs -0 rm -fr
	$(RM_R) $(ARTIFACTS)
.PHONY: clean
