# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import unittest
import doctest
import time

import jsonpickle.util
from jsonpickle.util import *
from jsonpickle.tests.classes import Thing, ListSubclass, DictSubclass

class IsPrimitiveTestCase(unittest.TestCase):
    def test_int(self):
        self.assertTrue(isprimitive(0))
        self.assertTrue(isprimitive(3))
        self.assertTrue(isprimitive(-3))

    def test_float(self):
        self.assertTrue(isprimitive(0))
        self.assertTrue(isprimitive(3.5))
        self.assertTrue(isprimitive(-3.5))
        self.assertTrue(isprimitive(float(3)))

    def test_long(self):
        self.assertTrue(isprimitive(long(3)))

    def test_bool(self):
        self.assertTrue(isprimitive(True))
        self.assertTrue(isprimitive(False))

    def test_None(self):
        self.assertTrue(isprimitive(None))

    def test_str(self):
        self.assertTrue(isprimitive('hello'))
        self.assertTrue(isprimitive(''))

    def test_unicode(self):
        self.assertTrue(isprimitive(u'hello'))
        self.assertTrue(isprimitive(u''))
        self.assertTrue(isprimitive(unicode('hello')))

    def test_list(self):
        self.assertFalse(isprimitive([]))
        self.assertFalse(isprimitive([4, 4]))

    def test_dict(self):
        self.assertFalse(isprimitive({'key':'value'}))
        self.assertFalse(isprimitive({}))

    def test_tuple(self):
        self.assertFalse(isprimitive((1, 3)))
        self.assertFalse(isprimitive((1,)))

    def test_set(self):
        self.assertFalse(isprimitive(set([1, 3])))

    def test_object(self):
        self.assertFalse(isprimitive(Thing('test')))

class IsCollection(unittest.TestCase):
    def test_list(self):
        self.assertTrue(iscollection([1, 2]))
    
    def test_set(self):
        self.assertTrue(iscollection(set([1, 2])))
        
    def test_tuple(self):
        self.assertTrue(iscollection((1, 2)))
        
    def test_dict(self):
        self.assertFalse(iscollection({'key':'value'}))
    
    def test_other(self):
        self.assertFalse(iscollection(1))

class IsDictionary(unittest.TestCase):
    def test_dict(self):
        self.assertTrue(isdictionary({'key':'value'}))
    
    def test_list(self):
        self.assertFalse(isdictionary([1, 2]))

class IsDictionarySubclass(unittest.TestCase):
    def test_subclass(self):
        self.assertTrue(is_dictionary_subclass(DictSubclass()))
    
    def test_dict(self):
        self.assertFalse(is_dictionary_subclass({'key':'value'}))

class IsCollectionSubclass(unittest.TestCase):
    def test_subclass(self):
        self.assertTrue(is_collection_subclass(ListSubclass()))
    
    def test_list(self):
        self.assertFalse(is_collection_subclass([]))

class IsNonComplex(unittest.TestCase):
    def setUp(self):
        self.time = time.struct_time('123456789')
        
    def test_time_struct(self):
        self.assertTrue(is_noncomplex(self.time))

    def test_other(self):
        self.assertFalse(is_noncomplex('a'))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(IsPrimitiveTestCase))
    suite.addTest(unittest.makeSuite(IsCollection))
    suite.addTest(unittest.makeSuite(IsDictionary))
    suite.addTest(unittest.makeSuite(IsDictionarySubclass))
    suite.addTest(unittest.makeSuite(IsCollectionSubclass))
    suite.addTest(unittest.makeSuite(IsNonComplex))
    suite.addTest(doctest.DocTestSuite(jsonpickle.util))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')