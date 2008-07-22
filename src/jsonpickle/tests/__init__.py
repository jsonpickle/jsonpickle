# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import unittest

import jsonpickle.tests.util
import jsonpickle.tests.jsonpickle_test
import jsonpickle.tests.thirdparty_tests

def suite():
    suite = unittest.TestSuite()    
    suite.addTest(util.suite())
    suite.addTest(jsonpickle_test.suite())
    suite.addTest(thirdparty_tests.suite())
    return suite

if __name__ == '__main__':
    #unittest.main()
    #unittest.TextTestRunner(verbosity=2).run(suite())
    unittest.main(defaultTest='suite')