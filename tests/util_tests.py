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
from jsonpickle.compat import long
from jsonpickle.util import is_dictionary
from jsonpickle.util import is_dictionary_subclass
from jsonpickle.util import is_function
from jsonpickle.util import is_list
from jsonpickle.util import is_noncomplex
from jsonpickle.util import is_primitive
from jsonpickle.util import is_set
from jsonpickle.util import is_sequence
from jsonpickle.util import is_sequence_subclass
from jsonpickle.util import is_tuple
from jsonpickle.util import itemgetter
from jsonpickle._samples import Thing, ListSubclass, DictSubclass


class UtilTestCase(unittest.TestCase):

    def test_is_primitive_int(self):
        self.assertTrue(is_primitive(0))
        self.assertTrue(is_primitive(3))
        self.assertTrue(is_primitive(-3))

    def test_is_primitive_float(self):
        self.assertTrue(is_primitive(0))
        self.assertTrue(is_primitive(3.5))
        self.assertTrue(is_primitive(-3.5))
        self.assertTrue(is_primitive(float(3)))

    def test_is_primitive_long(self):
        self.assertTrue(is_primitive(long(3)))

    def test_is_primitive_bool(self):
        self.assertTrue(is_primitive(True))
        self.assertTrue(is_primitive(False))

    def test_is_primitive_None(self):
        self.assertTrue(is_primitive(None))

    def test_is_primitive_str(self):
        self.assertTrue(is_primitive('hello'))
        self.assertTrue(is_primitive(''))

    def test_is_primitive_unicode(self):
        self.assertTrue(is_primitive(unicode('hello')))
        self.assertTrue(is_primitive(unicode('')))
        self.assertTrue(is_primitive(unicode('hello')))

    def test_is_primitive_list(self):
        self.assertFalse(is_primitive([]))
        self.assertFalse(is_primitive([4, 4]))

    def test_is_primitive_dict(self):
        self.assertFalse(is_primitive({'key':'value'}))
        self.assertFalse(is_primitive({}))

    def test_is_primitive_tuple(self):
        self.assertFalse(is_primitive((1, 3)))
        self.assertFalse(is_primitive((1,)))

    def test_is_primitive_set(self):
        self.assertFalse(is_primitive(set([1, 3])))

    def test_is_primitive_object(self):
        self.assertFalse(is_primitive(Thing('test')))

    def test_is_list_list(self):
        self.assertTrue(is_list([1, 2]))

    def test_is_list_set(self):
        self.assertTrue(is_set(set([1, 2])))

    def test_is_list_tuple(self):
        self.assertTrue(is_tuple((1, 2)))

    def test_is_list_dict(self):
        self.assertFalse(is_list({'key':'value'}))
        self.assertFalse(is_set({'key':'value'}))
        self.assertFalse(is_tuple({'key':'value'}))

    def test_is_list_other(self):
        self.assertFalse(is_list(1))
        self.assertFalse(is_set(1))
        self.assertFalse(is_tuple(1))

    def test_is_sequence_various(self):
        self.assertTrue(is_sequence([]))
        self.assertTrue(is_sequence(tuple()))
        self.assertTrue(is_sequence(set()))

    def test_is_dictionary_dict(self):
        self.assertTrue(is_dictionary({}))

    def test_is_dicitonary_sequences(self):
        self.assertFalse(is_dictionary([]))
        self.assertFalse(is_dictionary(set()))

    def test_is_dictionary_tuple(self):
        self.assertFalse(is_dictionary(tuple()))

    def test_is_dictionary_primitive(self):
        self.assertFalse(is_dictionary(int()))
        self.assertFalse(is_dictionary(None))
        self.assertFalse(is_dictionary(str()))

    def test_is_dictionary_subclass_dict(self):
        self.assertFalse(is_dictionary_subclass({}))

    def test_is_dictionary_subclass_subclass(self):
        self.assertTrue(is_dictionary_subclass(DictSubclass()))

    def test_is_sequence_subclass_subclass(self):
        self.assertTrue(is_sequence_subclass(ListSubclass()))

    def test_is_sequence_subclass_list(self):
        self.assertFalse(is_sequence_subclass([]))

    def test_is_noncomplex_time_struct(self):
        t = time.struct_time('123456789')
        self.assertTrue(is_noncomplex(t))

    def test_is_noncomplex_other(self):
        self.assertFalse(is_noncomplex('a'))

    def test_is_function_builtins(self):
        self.assertTrue(is_function(globals))

    def test_is_function_lambda(self):
        self.assertTrue(is_function(lambda: False))

    def test_is_function_instance_method(self):
        class Foo(object):
            def method(self):
                pass
            @staticmethod
            def staticmethod():
                pass
            @classmethod
            def classmethod(cls):
                pass
        f = Foo()
        self.assertTrue(is_function(f.method))
        self.assertTrue(is_function(f.staticmethod))
        self.assertTrue(is_function(f.classmethod))

    def test_itemgetter(self):
        expect = '0'
        actual = itemgetter((0, 'zero'))
        self.assertEqual(expect, actual)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(UtilTestCase))
    suite.addTest(doctest.DocTestSuite(jsonpickle.util))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
