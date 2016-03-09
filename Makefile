#!/usr/bin/env make

# External commands
CTAGS ?= ctags
FIND ?= find
NOSE ?= nosetests
PYTHON ?= python
RM_R ?= rm -fr
SH ?= sh
TOX ?= tox

# Options
flags ?=
timeout ?= 600
TESTCMD ?= $(NOSE) --with-doctest $(flags)
TOXCMD ?= $(TOX) --develop --skip-missing-interpreters

# Data
ARTIFACTS := build
ARTIFACTS += dist
ARTIFACTS += tags
ARTIFACTS += *.egg-info

PYTHON_DIRS := tests
PYTHON_DIRS += jsonpickle

# The default target of this makefile is....
all:: help

help:
	@echo "================"
	@echo "Makefile Targets"
	@echo "================"
	@echo "make help - print this message"
	@echo "make test - run unit tests"
	@echo "make clean - remove cruft"
.PHONY: help

test:
	$(TESTCMD) $(PYTHON_DIRS)
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
	$(RM_R) $(ARTIFACTS)
.PHONY: clean
