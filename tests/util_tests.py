# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import unittest
import doctest
import time

import jsonpickle.util
from jsonpickle.compat import unicode
from jsonpickle.util import is_dictionary
from jsonpickle.util import is_dictionary_subclass
from jsonpickle.util import is_list
from jsonpickle.util import is_noncomplex
from jsonpickle.util import is_primitive
from jsonpickle.util import is_set
from jsonpickle.util import is_sequence
from jsonpickle.util import is_sequence_subclass
from jsonpickle.util import is_tuple
from jsonpickle._samples import Thing, ListSubclass, DictSubclass


class PrimitiveTestCase(unittest.TestCase):

    def test_int(self):
        self.assertTrue(is_primitive(0))
        self.assertTrue(is_primitive(3))
        self.assertTrue(is_primitive(-3))

    def test_float(self):
        self.assertTrue(is_primitive(0))
        self.assertTrue(is_primitive(3.5))
        self.assertTrue(is_primitive(-3.5))
        self.assertTrue(is_primitive(float(3)))

    def test_long(self):
        self.assertTrue(is_primitive(long(3)))

    def test_bool(self):
        self.assertTrue(is_primitive(True))
        self.assertTrue(is_primitive(False))

    def test_None(self):
        self.assertTrue(is_primitive(None))

    def test_str(self):
        self.assertTrue(is_primitive('hello'))
        self.assertTrue(is_primitive(''))

    def test_unicode(self):
        self.assertTrue(is_primitive(unicode('hello')))
        self.assertTrue(is_primitive(unicode('')))
        self.assertTrue(is_primitive(unicode('hello')))

    def test_list(self):
        self.assertFalse(is_primitive([]))
        self.assertFalse(is_primitive([4, 4]))

    def test_dict(self):
        self.assertFalse(is_primitive({'key':'value'}))
        self.assertFalse(is_primitive({}))

    def test_tuple(self):
        self.assertFalse(is_primitive((1, 3)))
        self.assertFalse(is_primitive((1,)))

    def test_set(self):
        self.assertFalse(is_primitive(set([1, 3])))

    def test_object(self):
        self.assertFalse(is_primitive(Thing('test')))


class SequenceTestCase(unittest.TestCase):

    def test_list(self):
        self.assertTrue(is_list([1, 2]))

    def test_set(self):
        self.assertTrue(is_set(set([1, 2])))

    def test_tuple(self):
        self.assertTrue(is_tuple((1, 2)))

    def test_dict(self):
        self.assertFalse(is_list({'key':'value'}))
        self.assertFalse(is_set({'key':'value'}))
        self.assertFalse(is_tuple({'key':'value'}))

    def test_other(self):
        self.assertFalse(is_list(1))
        self.assertFalse(is_set(1))
        self.assertFalse(is_tuple(1))

    def test_is_sequence(self):
        self.assertTrue(is_sequence([]))
        self.assertTrue(is_sequence(tuple()))
        self.assertTrue(is_sequence(set()))


class DictionaryTestCase(unittest.TestCase):

    def test_dict(self):
        self.assertTrue(is_dictionary({'key':'value'}))

    def test_list(self):
        self.assertFalse(is_dictionary([1, 2]))


class DictionarySubclassTestCase(unittest.TestCase):

    def test_subclass(self):
        self.assertTrue(is_dictionary_subclass(DictSubclass()))

    def test_dict(self):
        self.assertFalse(is_dictionary_subclass({'key':'value'}))


class SequenceSubclassTestCase(unittest.TestCase):

    def test_subclass(self):
        self.assertTrue(is_sequence_subclass(ListSubclass()))

    def test_list(self):
        self.assertFalse(is_sequence_subclass([]))


class NonComplexTestCase(unittest.TestCase):
    def setUp(self):
        self.time = time.struct_time('123456789')

    def test_time_struct(self):
        self.assertTrue(is_noncomplex(self.time))

    def test_other(self):
        self.assertFalse(is_noncomplex('a'))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DictionaryTestCase))
    suite.addTest(unittest.makeSuite(DictionarySubclassTestCase))
    suite.addTest(unittest.makeSuite(NonComplexTestCase))
    suite.addTest(unittest.makeSuite(PrimitiveTestCase))
    suite.addTest(unittest.makeSuite(SequenceTestCase))
    suite.addTest(unittest.makeSuite(SequenceSubclassTestCase))
    suite.addTest(doctest.DocTestSuite(jsonpickle.util))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
