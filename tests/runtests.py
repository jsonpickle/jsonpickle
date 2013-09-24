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

testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(testdir))

import unittest

import util_tests
import jsonpickle_test
import thirdparty_tests
import backends_tests
import document_test
import datetime_tests
import handler_tests

def suite():
    suite = unittest.TestSuite()
    suite.addTest(util_tests.suite())
    suite.addTest(util_tests.suite())
    suite.addTest(jsonpickle_test.suite())
    suite.addTest(document_test.suite())
    suite.addTest(thirdparty_tests.suite())
    suite.addTest(backends_tests.suite())
    suite.addTest(datetime_tests.suite())
    suite.addTest(handler_tests.suite())
    return suite

def main():
    #unittest.main(defaultTest='suite')
    return unittest.TextTestRunner(verbosity=2).run(suite())

if __name__ == '__main__':
    sys.exit(not main().wasSuccessful())
