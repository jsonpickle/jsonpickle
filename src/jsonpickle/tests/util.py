# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import unittest
import doctest
import jsonpickle

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(jsonpickle.util))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')