# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009-2018 David Aguilar (davvid -at- gmail.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import io
import time

from jsonpickle import util


class Thing:
    def __init__(self, name):
        self.name = name
        self.child = None


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


class MethodTestClass:
    variable = None

    @staticmethod
    def static_method():
        pass

    @classmethod
    def class_method(cls):
        pass

    def bound_method(self):
        pass


class MethodTestSubclass(MethodTestClass):
    pass


class MethodTestOldStyle:
    def bound_method(self):
        pass


def test_is_primitive_int():
    assert util.is_primitive(0)
    assert util.is_primitive(3)
    assert util.is_primitive(-3)


def test_is_primitive_float():
    assert util.is_primitive(0)
    assert util.is_primitive(3.5)
    assert util.is_primitive(-3.5)
    assert util.is_primitive(float(3))


def test_is_primitive_long():
    assert util.is_primitive(2**64)


def test_is_primitive_bool():
    assert util.is_primitive(True)
    assert util.is_primitive(False)


def test_is_primitive_None():
    assert util.is_primitive(None)


def test_is_primitive_bytes():
    assert not util.is_primitive(b'hello')
    assert not util.is_primitive(b'foo')
    assert util.is_primitive('foo')


def test_is_primitive_unicode():
    assert util.is_primitive('')
    assert util.is_primitive('hello')


def test_is_primitive_list():
    assert not util.is_primitive([])
    assert not util.is_primitive([4, 4])


def test_is_primitive_dict():
    assert not util.is_primitive({'key': 'value'})
    assert not util.is_primitive({})


def test_is_primitive_tuple():
    assert not util.is_primitive((1, 3))
    assert not util.is_primitive((1,))


def test_is_primitive_set():
    assert not util.is_primitive({1, 3})


def test_is_primitive_object():
    assert not util.is_primitive(Thing('test'))


def test_is_list_list():
    assert util.is_list([1, 2])


def test_is_list_set():
    assert util.is_set({1, 2})


def test_is_list_tuple():
    assert util.is_tuple((1, 2))


def test_is_list_dict():
    assert not util.is_list({'key': 'value'})
    assert not util.is_set({'key': 'value'})
    assert not util.is_tuple({'key': 'value'})


def test_is_list_other():
    assert not util.is_list(1)
    assert not util.is_set(1)
    assert not util.is_tuple(1)


def test_is_sequence_various():
    assert util.is_sequence([])
    assert util.is_sequence(tuple())
    assert util.is_sequence(set())


def test_is_dictionary_dict():
    assert util.is_dictionary({})


def test_is_dicitonary_sequences():
    assert not util.is_dictionary([])
    assert not util.is_dictionary(set())


def test_is_dictionary_tuple():
    assert not util.is_dictionary(tuple())


def test_is_dictionary_primitive():
    assert not util.is_dictionary(int())
    assert not util.is_dictionary(None)
    assert not util.is_dictionary('')


def test_is_dictionary_subclass_dict():
    assert not util.is_dictionary_subclass({})


def test_is_dictionary_subclass_subclass():
    assert util.is_dictionary_subclass(DictSubclass())


def test_is_sequence_subclass_subclass():
    assert util.is_sequence_subclass(ListSubclass())


def test_is_sequence_subclass_list():
    assert not util.is_sequence_subclass([])


def test_is_noncomplex_time_struct():
    t = time.struct_time('123456789')
    assert util.is_noncomplex(t)


def test_is_noncomplex_other():
    assert not util.is_noncomplex('a')


def test_is_function_builtins():
    assert util.is_function(globals)


def test_is_function_lambda():
    assert util.is_function(lambda: False)


def test_is_function_instance_method():
    class Foo:
        def method(self):
            pass

        @staticmethod
        def staticmethod():
            pass

        @classmethod
        def classmethod(cls):
            pass

    f = Foo()
    assert util.is_function(f.method)
    assert util.is_function(f.staticmethod)
    assert util.is_function(f.classmethod)


def test_itemgetter():
    expect = '0'
    actual = util.itemgetter((0, 'zero'))
    assert expect == actual


def test_has_method():
    instance = MethodTestClass()
    x = 1
    has_method = util.has_method
    # no attribute
    assert not has_method(instance, 'foo')
    # builtin method type
    assert not has_method(int, '__getnewargs__')
    assert has_method(x, '__getnewargs__')
    # staticmethod
    assert has_method(instance, 'static_method')
    assert has_method(MethodTestClass, 'static_method')
    # not a method
    assert not has_method(instance, 'variable')
    assert not has_method(MethodTestClass, 'variable')
    # classmethod
    assert has_method(instance, 'class_method')
    assert has_method(MethodTestClass, 'class_method')
    # bound method
    assert has_method(instance, 'bound_method')
    assert not has_method(MethodTestClass, 'bound_method')
    # subclass bound method
    sub_instance = MethodTestSubclass()
    assert has_method(sub_instance, 'bound_method')
    assert not has_method(MethodTestSubclass, 'bound_method')
    # old style object
    old_instance = MethodTestOldStyle()
    assert has_method(old_instance, 'bound_method')
    assert not has_method(MethodTestOldStyle, 'bound_method')


def test_importable_name():
    func_being_tested_obj = util.importable_name
    io_method_obj = io.BytesIO(b'bytes').readline
    assert (
        util.importable_name(func_being_tested_obj) == 'jsonpickle.util.importable_name'
    )
    assert util.importable_name(io_method_obj) == '_io.BytesIO.readline'
    assert util.importable_name(dict) == 'builtins.dict'
