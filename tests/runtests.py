#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import os
import sys

testdir = os.path.dirname(os.path.abspath(__file__))  # noqa: E402
sys.path.insert(1, os.path.dirname(testdir))  # noqa: E402

import unittest

import backend_test
import datetime_test
import document_test
import handler_test
import jsonpickle_test
import object_test
import stdlib_test
import util_test
import feedparser_test
import bson_test
import numpy_test


def suite():
    suite = unittest.TestSuite()
    suite.addTest(util_test.suite())
    suite.addTest(handler_test.suite())
    suite.addTest(backend_test.suite())
    suite.addTest(jsonpickle_test.suite())
    suite.addTest(datetime_test.suite())
    suite.addTest(document_test.suite())
    suite.addTest(object_test.suite())
    suite.addTest(stdlib_test.suite())
    suite.addTest(feedparser_test.suite())
    suite.addTest(numpy_test.suite())
    suite.addTest(bson_test.suite())
    return suite


def main():
    return unittest.TextTestRunner(verbosity=2).run(suite())


if __name__ == '__main__':
    sys.exit(not main().wasSuccessful())
