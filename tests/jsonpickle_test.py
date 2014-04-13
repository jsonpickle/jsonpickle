# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009, 2011, 2013 David Aguilar
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import collections
import datetime
import doctest
import os
import time
import unittest
import sys
import decimal

import jsonpickle

from jsonpickle import handlers
from jsonpickle import tags
from jsonpickle.compat import unicode
from jsonpickle.compat import unichr
from jsonpickle.compat import PY32

from jsonpickle._samples import (
        BrokenReprThing,
        DictSubclass,
        GetstateDict,
        GetstateReturnsList,
        ListSubclass,
        ListSubclassWithInit,
        NamedTuple,
        ObjWithJsonPickleRepr,
        OldStyleClass,
        SetSubclass,
        Thing,
        ThingWithSlots,
        ThingWithProps,
        )


class PicklingTestCase(unittest.TestCase):
    def setUp(self):
        self.pickler = jsonpickle.pickler.Pickler()
        self.unpickler = jsonpickle.unpickler.Unpickler()

    def tearDown(self):
        self.pickler.reset()
        self.unpickler.reset()

    def test_string(self):
        self.assertEqual('a string', self.pickler.flatten('a string'))
        self.assertEqual('a string', self.unpickler.restore('a string'))

    def test_unicode(self):
        self.assertEqual(unicode('a string'), self.pickler.flatten('a string'))
        self.assertEqual(unicode('a string'), self.unpickler.restore('a string'))

    def test_int(self):
        self.assertEqual(3, self.pickler.flatten(3))
        self.assertEqual(3, self.unpickler.restore(3))

    def test_float(self):
        self.assertEqual(3.5, self.pickler.flatten(3.5))
        self.assertEqual(3.5, self.unpickler.restore(3.5))

    def test_boolean(self):
        self.assertTrue(self.pickler.flatten(True))
        self.assertFalse(self.pickler.flatten(False))
        self.assertTrue(self.unpickler.restore(True))
        self.assertFalse(self.unpickler.restore(False))

    def test_none(self):
        self.assertTrue(self.pickler.flatten(None) is None)
        self.assertTrue(self.unpickler.restore(None) is None)

    def test_list(self):
        # multiple types of values
        listA = [1, 35.0, 'value']
        self.assertEqual(listA, self.pickler.flatten(listA))
        self.assertEqual(listA, self.unpickler.restore(listA))
        # nested list
        listB = [40, 40, listA, 6]
        self.assertEqual(listB, self.pickler.flatten(listB))
        self.assertEqual(listB, self.unpickler.restore(listB))
        # 2D list
        listC = [[1, 2], [3, 4]]
        self.assertEqual(listC, self.pickler.flatten(listC))
        self.assertEqual(listC, self.unpickler.restore(listC))
        # empty list
        listD = []
        self.assertEqual(listD, self.pickler.flatten(listD))
        self.assertEqual(listD, self.unpickler.restore(listD))

    def test_list_subclass(self):
        obj = ListSubclass()
        obj.extend([1, 2, 3])
        flattened = self.pickler.flatten(obj)
        self.assertTrue(tags.OBJECT in flattened)
        self.assertTrue(tags.SEQ in flattened)
        self.assertEqual(len(flattened[tags.SEQ]), 3)
        for v in obj:
            self.assertTrue(v in flattened[tags.SEQ])
        restored = self.unpickler.restore(flattened)
        self.assertEqual(type(restored), ListSubclass)
        self.assertEqual(restored, obj)

    def test_list_subclass_with_data(self):
        obj = ListSubclass()
        obj.extend([1, 2, 3])
        data = SetSubclass([1, 2, 3])
        obj.data = data
        flattened = self.pickler.flatten(obj)
        restored = self.unpickler.restore(flattened)
        self.assertEqual(restored, obj)
        self.assertEqual(type(restored.data), SetSubclass)
        self.assertEqual(restored.data, data)

    def test_set(self):
        setlist = ['orange', 'apple', 'grape']
        setA = set(setlist)

        flattened = self.pickler.flatten(setA)
        for s in setlist:
            self.assertTrue(s in flattened[tags.SET])

        setA_pickled = {tags.SET: setlist}
        self.assertEqual(setA, self.unpickler.restore(setA_pickled))

    def test_set_subclass(self):
        obj = SetSubclass([1, 2, 3])
        flattened = self.pickler.flatten(obj)
        self.assertTrue(tags.OBJECT in flattened)
        self.assertTrue(tags.SEQ in flattened)
        self.assertEqual(len(flattened[tags.SEQ]), 3)
        for v in obj:
            self.assertTrue(v in flattened[tags.SEQ])
        restored = self.unpickler.restore(flattened)
        self.assertEqual(type(restored), SetSubclass)
        self.assertEqual(restored, obj)

    def test_set_subclass_with_data(self):
        obj = SetSubclass([1, 2, 3])
        data = ListSubclass()
        data.extend([1, 2, 3])
        obj.data = data
        flattened = self.pickler.flatten(obj)
        restored = self.unpickler.restore(flattened)
        self.assertEqual(type(restored.data), ListSubclass)
        self.assertEqual(restored.data, data)

    def test_dict(self):
        dictA = {'key1': 1.0, 'key2': 20, 'key3': 'thirty'}
        self.assertEqual(dictA, self.pickler.flatten(dictA))
        self.assertEqual(dictA, self.unpickler.restore(dictA))
        dictB = {}
        self.assertEqual(dictB, self.pickler.flatten(dictB))
        self.assertEqual(dictB, self.unpickler.restore(dictB))

    def test_tuple(self):
        # currently all collections are converted to lists
        tupleA = (4, 16, 32)
        tupleA_pickled = {tags.TUPLE: [4, 16, 32]}
        self.assertEqual(tupleA_pickled, self.pickler.flatten(tupleA))
        self.assertEqual(tupleA, self.unpickler.restore(tupleA_pickled))
        tupleB = (4,)
        tupleB_pickled = {tags.TUPLE: [4]}
        self.assertEqual(tupleB_pickled, self.pickler.flatten(tupleB))
        self.assertEqual(tupleB, self.unpickler.restore(tupleB_pickled))

    def test_tuple_roundtrip(self):
        data = (1,2,3)
        newdata = jsonpickle.decode(jsonpickle.encode(data))
        self.assertEqual(data, newdata)

    def test_set_roundtrip(self):
        data = set([1,2,3])
        newdata = jsonpickle.decode(jsonpickle.encode(data))
        self.assertEqual(data, newdata)

    def test_list_roundtrip(self):
        data = [1,2,3]
        newdata = jsonpickle.decode(jsonpickle.encode(data))
        self.assertEqual(data, newdata)

    def test_defaultdict_roundtrip(self):
        """Make sure we can handle collections.defaultdict(list)"""
        # setup
        defaultdict = collections.defaultdict(list)
        defaultdict['a'] = 1
        defaultdict['b'].append(2)
        defaultdict['c'] = collections.defaultdict(dict)
        # jsonpickle work your magic
        encoded = jsonpickle.encode(defaultdict)
        newdefaultdict = jsonpickle.decode(encoded)
        # jsonpickle never fails
        self.assertEqual(newdefaultdict['a'], 1)
        self.assertEqual(newdefaultdict['b'], [2])
        self.assertEqual(type(newdefaultdict['c']), collections.defaultdict)
        self.assertEqual(defaultdict.default_factory, list)
        self.assertEqual(newdefaultdict.default_factory, list)

    def test_deque_roundtrip(self):
        """Make sure we can handle collections.deque"""
        old_deque = collections.deque([0, 1, 2])
        encoded = jsonpickle.encode(old_deque)
        new_deque = jsonpickle.decode(encoded)
        self.assertNotEqual(encoded, 'nil')
        self.assertEqual(old_deque[0], 0)
        self.assertEqual(new_deque[0], 0)
        self.assertEqual(old_deque[1], 1)
        self.assertEqual(new_deque[1], 1)
        self.assertEqual(old_deque[2], 2)
        self.assertEqual(new_deque[2], 2)

    def test_namedtuple_roundtrip(self):
        old_nt = NamedTuple(0, 1, 2)
        encoded = jsonpickle.encode(old_nt)
        new_nt = jsonpickle.decode(encoded)
        self.assertEqual(type(old_nt), type(new_nt))
        self.assertTrue(old_nt is not new_nt)
        self.assertEqual(old_nt.a, new_nt.a)
        self.assertEqual(old_nt.b, new_nt.b)
        self.assertEqual(old_nt.c, new_nt.c)
        self.assertEqual(old_nt[0], new_nt[0])
        self.assertEqual(old_nt[1], new_nt[1])
        self.assertEqual(old_nt[2], new_nt[2])

    def test_counter_roundtrip(self):
        if sys.version_info < (2, 7):
            # collections.Counter was introduced in Python 2.7
            return
        counter = collections.Counter({1: 2})
        encoded = jsonpickle.encode(counter)
        decoded = jsonpickle.decode(encoded)
        self.assertTrue(type(decoded) is collections.Counter)
        # the integer key becomes a string when keys=False
        self.assertEqual(decoded.get('1'), 2)

    def test_counter_roundtrip_with_keys(self):
        if sys.version_info < (2, 7):
            # collections.Counter was introduced in Python 2.7
            return
        counter = collections.Counter({1: 2})
        encoded = jsonpickle.encode(counter, keys=True)
        decoded = jsonpickle.decode(encoded, keys=True)
        self.assertTrue(type(decoded) is collections.Counter)
        self.assertEqual(decoded.get(1), 2)

    def test_class(self):
        inst = Thing('test name')
        inst.child = Thing('child name')

        flattened = self.pickler.flatten(inst)
        self.assertEqual('test name', flattened['name'])
        child = flattened['child']
        self.assertEqual('child name', child['name'])

        inflated = self.unpickler.restore(flattened)
        self.assertEqual('test name', inflated.name)
        self.assertTrue(type(inflated) is Thing)
        self.assertEqual('child name', inflated.child.name)
        self.assertTrue(type(inflated.child) is Thing)

    def test_classlist(self):
        array = [Thing('one'), Thing('two'), 'a string']

        flattened = self.pickler.flatten(array)
        self.assertEqual('one', flattened[0]['name'])
        self.assertEqual('two', flattened[1]['name'])
        self.assertEqual('a string', flattened[2])

        inflated = self.unpickler.restore(flattened)
        self.assertEqual('one', inflated[0].name)
        self.assertTrue(type(inflated[0]) is Thing)
        self.assertEqual('two', inflated[1].name)
        self.assertTrue(type(inflated[1]) is Thing)
        self.assertEqual('a string', inflated[2])

    def test_classdict(self):
        dict = {'k1':Thing('one'), 'k2':Thing('two'), 'k3':3}

        flattened = self.pickler.flatten(dict)
        self.assertEqual('one', flattened['k1']['name'])
        self.assertEqual('two', flattened['k2']['name'])
        self.assertEqual(3, flattened['k3'])

        inflated = self.unpickler.restore(flattened)
        self.assertEqual('one', inflated['k1'].name)
        self.assertTrue(type(inflated['k1']) is Thing)
        self.assertEqual('two', inflated['k2'].name)
        self.assertTrue(type(inflated['k2']) is Thing)
        self.assertEqual(3, inflated['k3'])

        #TODO show that non string keys fail

    def test_recursive(self):
        """create a recursive structure and test that we can handle it
        """
        parent = Thing('parent')
        child = Thing('child')
        child.sibling = Thing('sibling')

        parent.self = parent
        parent.child = child
        parent.child.twin = child
        parent.child.parent = parent
        parent.child.sibling.parent = parent

        cloned = jsonpickle.decode(jsonpickle.encode(parent))

        self.assertEqual(parent.name,
                         cloned.name)
        self.assertEqual(parent.child.name,
                         cloned.child.name)
        self.assertEqual(parent.child.sibling.name,
                         cloned.child.sibling.name)
        self.assertEqual(cloned,
                         cloned.child.parent)
        self.assertEqual(cloned,
                         cloned.child.sibling.parent)
        self.assertEqual(cloned,
                         cloned.child.twin.parent)
        self.assertEqual(cloned.child,
                         cloned.child.twin)

    def test_newstyleslots(self):
        obj = ThingWithSlots(True, False)
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertTrue(newobj.a)
        self.assertFalse(newobj.b)

    def test_newstyleslots_with_children(self):
        obj = ThingWithSlots(Thing('a'), Thing('b'))
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(newobj.a.name, 'a')
        self.assertEqual(newobj.b.name, 'b')

    def test_oldstyleclass(self):
        obj = OldStyleClass()
        obj.value = 1234

        flattened = self.pickler.flatten(obj)
        self.assertEqual(1234, flattened['value'])

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1234, inflated.value)

    def test_struct_time(self):
        expect = time.struct_time('123456789')

        flattened = self.pickler.flatten(expect)
        actual = self.unpickler.restore(flattened)
        self.assertEqual(expect, actual)

    def test_dictsubclass(self):
        obj = DictSubclass()
        obj['key1'] = 1

        flattened = self.pickler.flatten(obj)
        self.assertEqual({'key1': 1,
                          tags.OBJECT:
                            'jsonpickle._samples.DictSubclass'
                         },
                         flattened)
        self.assertEqual(flattened[tags.OBJECT],
                         'jsonpickle._samples.DictSubclass')

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1, inflated['key1'])
        self.assertEqual(inflated.name, 'Test')

    def test_dictsubclass_notunpickable(self):
        self.pickler.unpicklable = False

        obj = DictSubclass()
        obj['key1'] = 1

        flattened = self.pickler.flatten(obj)
        self.assertEqual(1, flattened['key1'])
        self.assertFalse(tags.OBJECT in flattened)

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1, inflated['key1'])

    def test_getstate_dict_subclass_structure(self):
        obj = GetstateDict('test')
        obj['key1'] = 1

        flattened = self.pickler.flatten(obj)
        self.assertTrue(tags.OBJECT in flattened)
        self.assertEqual('jsonpickle._samples.GetstateDict',
                         flattened[tags.OBJECT])
        self.assertTrue(tags.STATE in flattened)
        self.assertTrue(tags.TUPLE in flattened[tags.STATE])
        self.assertEqual(['test', {'key1': 1}],
                         flattened[tags.STATE][tags.TUPLE])

    def test_getstate_dict_subclass_roundtrip_simple(self):
        obj = GetstateDict('test')
        obj['key1'] = 1

        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)

        self.assertEqual(1, inflated['key1'])
        self.assertEqual(inflated.name, 'test')

    def test_getstate_dict_subclass_roundtrip_cyclical(self):
        obj = GetstateDict('test')
        obj['key1'] = 1

        # The "name" field of obj2 points to obj (reference)
        obj2 = GetstateDict(obj)
        # The "obj2" key in obj points to obj2 (cyclical reference)
        obj['obj2'] = obj2

        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)

        # The dict must be preserved
        self.assertEqual(1, inflated['key1'])

        # __getstate__/__setstate__ must have been run
        self.assertEqual(inflated.name, 'test')
        self.assertEqual(inflated.active, True)
        self.assertEqual(inflated['obj2'].active, True)

        # The reference must be preserved
        self.assertTrue(inflated is inflated['obj2'].name)

    def test_getstate_list_simple(self):
        obj = GetstateReturnsList(1, 2)
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.x, 1)
        self.assertEqual(inflated.y, 2)

    def test_getstate_list_inside_list(self):
        obj1 = GetstateReturnsList(1, 2)
        obj2 = GetstateReturnsList(3, 4)
        obj = [obj1, obj2]
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated[0].x, 1)
        self.assertEqual(inflated[0].y, 2)
        self.assertEqual(inflated[1].x, 3)
        self.assertEqual(inflated[1].y, 4)

    def test_tuple_notunpicklable(self):
        self.pickler.unpicklable = False

        flattened = self.pickler.flatten(('one', 2, 3))
        self.assertEqual(flattened, ['one', 2, 3])

    def test_set_not_unpicklable(self):
        self.pickler.unpicklable = False

        flattened = self.pickler.flatten(set(['one', 2, 3]))
        self.assertTrue('one' in flattened)
        self.assertTrue(2 in flattened)
        self.assertTrue(3 in flattened)
        self.assertTrue(isinstance(flattened, list))

    def test_datetime(self):
        obj = datetime.datetime.now()

        flattened = self.pickler.flatten(obj)
        self.assertTrue(tags.OBJECT in flattened)
        self.assertTrue('__reduce__' in flattened)

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(obj, inflated)

    def test_datetime_inside_int_keys_defaults(self):
        t = datetime.time(hour=10)
        s = jsonpickle.encode({1:t, 2:t})
        d = jsonpickle.decode(s)
        self.assertEqual(d["1"], d["2"])
        self.assertTrue(d["1"] is d["2"])
        self.assertTrue(isinstance(d["1"], datetime.time))

    def test_datetime_inside_int_keys_with_keys_enabled(self):
        t = datetime.time(hour=10)
        s = jsonpickle.encode({1:t, 2:t}, keys=True)
        d = jsonpickle.decode(s, keys=True)
        self.assertEqual(d[1], d[2])
        self.assertTrue(d[1] is d[2])
        self.assertTrue(isinstance(d[1], datetime.time))

    def test_broken_repr_dict_key(self):
        """Tests that we can pickle dictionaries with keys that have
        broken __repr__ implementations.
        """
        br = BrokenReprThing('test')
        obj = { br: True }
        pickler = jsonpickle.pickler.Pickler()
        flattened = pickler.flatten(obj)
        self.assertTrue('<BrokenReprThing "test">' in flattened)
        self.assertTrue(flattened['<BrokenReprThing "test">'])

    def test_repr_not_unpickable(self):
        obj = datetime.datetime.now()
        pickler = jsonpickle.pickler.Pickler(unpicklable=False)
        flattened = pickler.flatten(obj)
        self.assertFalse(tags.REPR in flattened)
        self.assertFalse(tags.OBJECT in flattened)
        self.assertEqual(str(obj), flattened)

    def test_thing_with_module(self):
        obj = Thing('with-module')
        obj.themodule = os

        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.themodule, os)

    def test_thing_with_module_safe(self):
        obj = Thing('with-module')
        obj.themodule = os
        flattened = self.pickler.flatten(obj)
        self.unpickler.safe = True
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.themodule, None)

    def test_thing_with_submodule(self):
        from distutils import sysconfig

        obj = Thing('with-submodule')
        obj.submodule = sysconfig

        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.submodule, sysconfig)

    def test_type_reference(self):
        """This test ensures that users can store references to types.
        """
        obj = Thing('object-with-type-reference')

        # reference the built-in 'object' type
        obj.typeref = object

        flattened = self.pickler.flatten(obj)
        self.assertEqual(flattened['typeref'], {
                            tags.TYPE: '__builtin__.object',
                         })

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.typeref, object)

    def test_class_reference(self):
        """This test ensures that users can store references to classes.
        """
        obj = Thing('object-with-class-reference')

        # reference the 'Thing' class (not an instance of the class)
        obj.classref = Thing

        flattened = self.pickler.flatten(obj)
        self.assertEqual(flattened['classref'], {
                            tags.TYPE: 'jsonpickle._samples.Thing',
                         })

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.classref, Thing)

    def test_supports_getstate_setstate(self):
        obj = ThingWithProps('object-which-defines-getstate-setstate')
        flattened = self.pickler.flatten(obj)
        self.assertTrue(flattened[tags.STATE].get('__identity__'))
        self.assertTrue(flattened[tags.STATE].get('nom'))
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(obj, inflated)

    def test_references(self):
        obj_a = Thing('foo')
        obj_b = Thing('bar')
        coll = [obj_a, obj_b, obj_b]
        flattened = self.pickler.flatten(coll)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(len(inflated), len(coll))
        for x in range(len(coll)):
            self.assertEqual(repr(coll[x]), repr(inflated[x]))

    def test_references_in_number_keyed_dict(self):
        """
        Make sure a dictionary with numbers as keys and objects as values
        can make the round trip.

        Because JSON must coerce integers to strings in dict keys, the sort
        order may have a tendency to change between pickling and unpickling,
        and this could affect the object references.
        """
        one = Thing('one')
        two = Thing('two')
        twelve = Thing('twelve')
        two.child = twelve
        obj = {
            1: one,
            2: two,
            12: twelve,
        }
        self.assertNotEqual(list(sorted(obj.keys())),
                            list(map(int, sorted(map(str, obj.keys())))))
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(len(inflated), 3)
        self.assertEqual(inflated['12'].name, 'twelve')

    def test_list_subclass_with_init(self):
        obj = ListSubclassWithInit('foo')
        self.assertEqual(obj.attr, 'foo')
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(type(inflated), ListSubclassWithInit)

    def test_builtin_error(self):
        expect = AssertionError
        json = jsonpickle.encode(expect)
        actual = jsonpickle.decode(json)
        self.assertEqual(expect, actual)
        self.assertTrue(expect is actual)

    def test_decimal(self):
        obj = decimal.Decimal(1)
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(type(inflated), decimal.Decimal)


class JSONPickleTestCase(unittest.TestCase):

    def setUp(self):
        self.obj = Thing('A name')
        self.expected_json = (
                '{"'+tags.OBJECT+'": "jsonpickle._samples.Thing",'
                ' "name": "A name", "child": null}')

    def test_encode(self):
        expect = self.obj
        pickled = jsonpickle.encode(self.obj)
        actual = jsonpickle.decode(pickled)
        self.assertEqual(expect.name, actual.name)
        self.assertEqual(expect.child, actual.child)

    def test_encode_notunpicklable(self):
        expect = {'name': 'A name', 'child': None}
        pickled = jsonpickle.encode(self.obj, unpicklable=False)
        actual = jsonpickle.decode(pickled)
        self.assertEqual(expect['name'], actual['name'])

    def test_decode(self):
        unpickled = jsonpickle.decode(self.expected_json)
        self.assertEqual(self.obj.name, unpickled.name)
        self.assertEqual(type(self.obj), type(unpickled))

    def test_json(self):
        expect = self.obj
        pickled = jsonpickle.encode(self.obj)
        actual = jsonpickle.decode(pickled)
        self.assertEqual(actual.name, expect.name)
        self.assertEqual(actual.child, expect.child)

        unpickled = jsonpickle.decode(self.expected_json)
        self.assertEqual(self.obj.name, unpickled.name)
        self.assertEqual(type(self.obj), type(unpickled))

    def test_unicode_dict_keys(self):
        uni = unichr(0x1234)
        pickled = jsonpickle.encode({uni: uni})
        unpickled = jsonpickle.decode(pickled)
        self.assertTrue(uni in unpickled)
        self.assertEqual(unpickled[uni], uni)

    def test_tuple_dict_keys_default(self):
        """Test that we handle dictionaries with tuples as keys."""
        tuple_dict = {(1, 2): 3, (4, 5): { (7, 8): 9 }}
        pickled = jsonpickle.encode(tuple_dict)
        expect = {'(1, 2)': 3, '(4, 5)': {'(7, 8)': 9}}
        actual = jsonpickle.decode(pickled)
        self.assertEqual(expect, actual)

        tuple_dict = {(1, 2): [1, 2]}
        pickled = jsonpickle.encode(tuple_dict)
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(unpickled['(1, 2)'], [1, 2])

    def test_tuple_dict_keys_with_keys_enabled(self):
        """Test that we handle dictionaries with tuples as keys."""
        tuple_dict = {(1, 2): 3, (4, 5): { (7, 8): 9 }}
        pickled = jsonpickle.encode(tuple_dict, keys=True)
        expect = tuple_dict
        actual = jsonpickle.decode(pickled, keys=True)
        self.assertEqual(expect, actual)

        tuple_dict = {(1, 2): [1, 2]}
        pickled = jsonpickle.encode(tuple_dict, keys=True)
        unpickled = jsonpickle.decode(pickled, keys=True)
        self.assertEqual(unpickled[(1, 2)], [1, 2])

    def test_datetime_dict_keys_defaults(self):
        """Test that we handle datetime objects as keys."""
        datetime_dict = {datetime.datetime(2008, 12, 31): True}
        pickled = jsonpickle.encode(datetime_dict)
        expect = {'datetime.datetime(2008, 12, 31, 0, 0)': True}
        actual = jsonpickle.decode(pickled)
        self.assertEqual(expect, actual)

    def test_datetime_dict_keys_with_keys_enabled(self):
        """Test that we handle datetime objects as keys."""
        datetime_dict = {datetime.datetime(2008, 12, 31): True}
        pickled = jsonpickle.encode(datetime_dict, keys=True)
        expect = datetime_dict
        actual = jsonpickle.decode(pickled, keys=True)
        self.assertEqual(expect, actual)

    def test_object_dict_keys(self):
        """Test that we handle random objects as keys.

        """
        thing = Thing('random')
        pickled = jsonpickle.encode({thing: True})
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(unpickled, {unicode('Thing("random")'): True})

    def test_int_dict_keys_defaults(self):
        int_dict = {1000: [1, 2]}
        pickled = jsonpickle.encode(int_dict)
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(unpickled['1000'], [1, 2])

    def test_int_dict_keys_with_keys_enabled(self):
        int_dict = {1000: [1, 2]}
        pickled = jsonpickle.encode(int_dict, keys=True)
        unpickled = jsonpickle.decode(pickled, keys=True)
        self.assertEqual(unpickled[1000], [1, 2])

    def test_list_of_objects(self):
        """Test that objects in lists are referenced correctly"""
        a = Thing('a')
        b = Thing('b')
        pickled = jsonpickle.encode([a, b, b])
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(unpickled[1], unpickled[2])
        self.assertEqual(type(unpickled[0]), Thing)
        self.assertEqual(unpickled[0].name, 'a')
        self.assertEqual(unpickled[1].name, 'b')
        self.assertEqual(unpickled[2].name, 'b')

    def test_refs_keys_values(self):
        """Test that objects in dict keys are referenced correctly
        """
        j = Thing('random')
        object_dict = {j: j}
        pickled = jsonpickle.encode(object_dict, keys=True)
        unpickled = jsonpickle.decode(pickled, keys=True)
        self.assertEqual(list(unpickled.keys()), list(unpickled.values()))

    def test_object_keys_to_list(self):
        """Test that objects in dict values are referenced correctly
        """
        j = Thing('random')
        object_dict = {j: [j, j]}
        pickled = jsonpickle.encode(object_dict, keys=True)
        unpickled = jsonpickle.decode(pickled, keys=True)
        obj = list(unpickled.keys())[0]
        self.assertEqual(j.name, obj.name)
        self.assertTrue(obj is unpickled[obj][0])
        self.assertTrue(obj is unpickled[obj][1])

    def test_refs_in_objects(self):
        """Test that objects in lists are referenced correctly"""
        a = Thing('a')
        b = Thing('b')
        pickled = jsonpickle.encode([a, b, b])
        unpickled = jsonpickle.decode(pickled)
        self.assertNotEqual(unpickled[0], unpickled[1])
        self.assertEqual(unpickled[1], unpickled[2])
        self.assertTrue(unpickled[1] is unpickled[2])

    def test_refs_recursive(self):
        """Test that complicated recursive refs work"""

        a = Thing('a')
        a.self_list = [Thing('0'), Thing('1'), Thing('2')]
        a.first = a.self_list[0]
        a.stuff = {a.first: a.first}
        a.morestuff = {a.self_list[1]: a.stuff}

        pickle = jsonpickle.encode(a, keys=True)
        b = jsonpickle.decode(pickle, keys=True)

        item = b.self_list[0]
        self.assertEqual(b.first, item)
        self.assertEqual(b.stuff[b.first], item)
        self.assertEqual(b.morestuff[b.self_list[1]][b.first], item)

    def test_load_backend(self):
        """Test that we can call jsonpickle.load_backend()

        """
        if PY32:
            self.skipTest('no simplejson for python 3.2')
            return
        jsonpickle.load_backend('simplejson', 'dumps', 'loads', ValueError)
        self.assertTrue(True)

    def test_set_preferred_backend_allows_magic(self):
        """Tests that we can use the pluggable backends magically
        """
        backend = 'os.path'
        jsonpickle.load_backend(backend, 'split', 'join', AttributeError)
        jsonpickle.set_preferred_backend(backend)

        slash_hello, world = jsonpickle.encode('/hello/world')
        jsonpickle.remove_backend(backend)

        self.assertEqual(slash_hello, '/hello')
        self.assertEqual(world, 'world')

    def test_load_backend_submodule(self):
        """Test that we can load a submodule as a backend

        """
        jsonpickle.load_backend('os.path', 'split', 'join', AttributeError)
        self.assertTrue('os.path' in jsonpickle.json._backend_names and
                        'os.path' in jsonpickle.json._encoders and
                        'os.path' in jsonpickle.json._decoders and
                        'os.path' in jsonpickle.json._encoder_options and
                        'os.path' in jsonpickle.json._decoder_exceptions)

    def _backend_is_partially_loaded(self, backend):
        """Return True if the specified backend is incomplete"""
        return (backend in jsonpickle.json._backend_names or
                backend in jsonpickle.json._encoders or
                backend in jsonpickle.json._decoders or
                backend in jsonpickle.json._encoder_options or
                backend in jsonpickle.json._decoder_exceptions)

    def test_load_backend_skips_bad_encode(self):
        """Test that we ignore bad encoders"""

        jsonpickle.load_backend('os.path', 'bad!', 'split', AttributeError)
        self.failIf(self._backend_is_partially_loaded('os.path'))

    def test_load_backend_skips_bad_decode(self):
        """Test that we ignore bad decoders"""

        jsonpickle.load_backend('os.path', 'join', 'bad!', AttributeError)
        self.failIf(self._backend_is_partially_loaded('os.path'))

    def test_load_backend_skips_bad_decoder_exceptions(self):
        """Test that we ignore bad decoder exceptions"""

        jsonpickle.load_backend('os.path', 'join', 'split', 'bad!')
        self.failIf(self._backend_is_partially_loaded('os.path'))

    def test_list_item_reference(self):
        thing = Thing('parent')
        thing.child = Thing('child')
        thing.child.refs = [thing]

        encoded = jsonpickle.encode(thing)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(id(decoded.child.refs[0]), id(decoded))

    def test_reference_to_list(self):
        thing = Thing('parent')
        thing.a = [1]
        thing.b = thing.a
        thing.b.append(thing.a)
        thing.b.append([thing.a])

        encoded = jsonpickle.encode(thing)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(decoded.a[0], 1)
        self.assertEqual(decoded.b[0], 1)
        self.assertEqual(id(decoded.a), id(decoded.b))
        self.assertEqual(id(decoded.a), id(decoded.a[1]))
        self.assertEqual(id(decoded.a), id(decoded.a[2][0]))

    def test_repr_using_jsonpickle(self):
        thing = ObjWithJsonPickleRepr()
        thing.child = ObjWithJsonPickleRepr()
        thing.child.parent = thing

        encoded = jsonpickle.encode(thing)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(id(decoded), id(decoded.child.parent))

    def test_ordered_dict(self):
        if sys.version_info < (2, 7):
            return

        d = collections.OrderedDict()
        d.update(c=3)
        d.update(a=1)
        d.update(b=2)

        encoded = jsonpickle.encode(d)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(d, decoded)

    def test_ordered_dict_int_keys(self):
        if sys.version_info < (2, 7):
            return
        d = {
            1: collections.OrderedDict([(2, -2), (3, -3)]),
            4: collections.OrderedDict([(5, -5), (6, -6)]),
        }
        encoded = jsonpickle.encode(d, keys=True)
        decoded = jsonpickle.decode(encoded, keys=True)

        self.assertEqual(collections.OrderedDict, type(decoded[1]))
        self.assertEqual(collections.OrderedDict, type(decoded[4]))
        self.assertEqual(-2, decoded[1][2])
        self.assertEqual(-3, decoded[1][3])
        self.assertEqual(-5, decoded[4][5])
        self.assertEqual(-6, decoded[4][6])
        self.assertEqual(d, decoded)

    def test_make_refs_disabled_list(self):
        obj_a = Thing('foo')
        obj_b = Thing('bar')
        coll = [obj_a, obj_b, obj_b]
        encoded = jsonpickle.encode(coll, make_refs=False)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(len(decoded), 3)
        self.assertTrue(decoded[0] is not decoded[1])
        self.assertTrue(decoded[1] is not decoded[2])

    def test_make_refs_disabled_reference_to_list(self):
        thing = Thing('parent')
        thing.a = [1]
        thing.b = thing.a
        thing.b.append(thing.a)
        thing.b.append([thing.a])

        encoded = jsonpickle.encode(thing, make_refs=False)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(decoded.a[0], 1)
        self.assertEqual(decoded.b[0:3], '[1,')
        self.assertEqual(decoded.a[1][0:3], '[1,')
        self.assertEqual(decoded.a[2][0][0:3], '[1,')

    def test_posix_stat_result(self):
        try:
            import posix
        except ImportError:
            return
        expect = posix.stat(__file__)
        encoded = jsonpickle.encode(expect)
        actual = jsonpickle.decode(encoded)
        self.assertEqual(expect, actual)


# Test classes for ExternalHandlerTestCase
class Mixin(object):
    def ok(self):
        return True


class UnicodeMixin(unicode, Mixin):
    def __add__(self, rhs):
        obj = super(UnicodeMixin, self).__add__(rhs)
        return UnicodeMixin(obj)


class UnicodeMixinHandler(handlers.BaseHandler):

    def flatten(self, obj, data):
        data['value'] = obj
        return data

    def restore(self, obj):
        return UnicodeMixin(obj['value'])

handlers.register(UnicodeMixin, UnicodeMixinHandler)


class ExternalHandlerTestCase(unittest.TestCase):
    def test_unicode_mixin(self):
        obj = UnicodeMixin('test')
        self.assertEqual(type(obj), UnicodeMixin)
        self.assertEqual(unicode(obj), unicode('test'))

        # Encode into JSON
        content = jsonpickle.encode(obj)

        # Resurrect from JSON
        new_obj = jsonpickle.decode(content)
        new_obj += ' passed'

        self.assertEqual(unicode(new_obj), unicode('test passed'))
        self.assertEqual(type(new_obj), UnicodeMixin)
        self.assertTrue(new_obj.ok())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(PicklingTestCase))
    suite.addTest(unittest.makeSuite(JSONPickleTestCase))
    suite.addTest(unittest.makeSuite(ExternalHandlerTestCase))
    suite.addTest(doctest.DocTestSuite(jsonpickle.pickler))
    suite.addTest(doctest.DocTestSuite(jsonpickle.unpickler))
    suite.addTest(doctest.DocTestSuite(jsonpickle))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
