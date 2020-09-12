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
import unittest

testdir = os.path.dirname(os.path.abspath(__file__))  # noqa: E402
sys.path.insert(1, os.path.dirname(testdir))  # noqa: E402

import backend_test  # noqa: E402
import datetime_test  # noqa: E402
import document_test  # noqa: E402
import handler_test  # noqa: E402
import jsonpickle_test  # noqa: E402
import object_test  # noqa: E402
import stdlib_test  # noqa: E402
import util_test  # noqa: E402
import feedparser_test  # noqa: E402
import bson_test  # noqa: E402
import numpy_test  # noqa: E402
import pandas_test  # noqa: E402


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
    suite.addTest(pandas_test.suite())
    return suite


def main():
    return unittest.TextTestRunner(verbosity=2).run(suite())


if __name__ == '__main__':
    sys.exit(not main().wasSuccessful())
