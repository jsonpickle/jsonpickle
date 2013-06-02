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

from six import u

import jsonpickle

from jsonpickle import handlers
from jsonpickle import tags
from jsonpickle.compat import unicode

from jsonpickle._samples import (
        BrokenReprThing,
        DictSubclass,
        ListSubclass,
        ListSubclassWithInit,
        NamedTuple,
        ObjWithJsonPickleRepr,
        SetSubclass,
        Thing,
        ThingWithSlots,
        ThingWithProps,
        )


class PicklingTestCase(unittest.TestCase):
    def setUp(self):
        self.pickler = jsonpickle.pickler.Pickler()
        self.unpickler = jsonpickle.unpickler.Unpickler()

    def test_string(self):
        self.assertEqual('a string', self.pickler.flatten('a string'))
        self.assertEqual('a string', self.unpickler.restore('a string'))

    def test_unicode(self):
        self.assertEqual(u('a string'), self.pickler.flatten('a string'))
        self.assertEqual(u('a string'), self.unpickler.restore('a string'))

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
        from pickle import _EmptyClass

        obj = _EmptyClass()
        obj.value = 1234

        flattened = self.pickler.flatten(obj)
        self.assertEqual(1234, flattened['value'])

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1234, inflated.value)

    def test_struct_time(self):
        t = time.struct_time('123456789')

        flattened = self.pickler.flatten(t)
        self.assertEqual(['1', '2', '3', '4', '5', '6', '7', '8', '9'], flattened)

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

    def test_tuple_notunpicklable(self):
        self.pickler.unpicklable = False

        flattened = self.pickler.flatten(('one', 2, 3))
        self.assertEqual(flattened, ['one', 2, 3])

    def test_set_notunpicklable(self):
        self.pickler.unpicklable = False

        flattened = self.pickler.flatten(set(['one', 2, 3]))
        self.assertEqual(sorted(flattened), sorted(['one', 2, 3]))

    def test_datetime(self):
        obj = datetime.datetime.now()

        flattened = self.pickler.flatten(obj)
        self.assertTrue(tags.OBJECT in flattened)
        self.assertTrue('__reduce__' in flattened)

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(obj, inflated)

    def test_datetime_inside_int_keys(self):
        t = datetime.time(hour=10)
        s = jsonpickle.encode({1:t, 2:t})
        d = jsonpickle.decode(s)
        self.assertEqual(d["1"], d["2"])
        self.assertTrue(d["1"] is d["2"])
        self.assertTrue(isinstance(d["1"], datetime.time))

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

    def test_thing_with_module(self):
        obj = Thing('with-module')
        obj.themodule = os

        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(inflated.themodule, os)

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

    def test_list_subclass_with_init(self):
        obj = ListSubclassWithInit('foo')
        self.assertEqual(obj.attr, 'foo')
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(type(inflated), ListSubclassWithInit)

class JSONPickleTestCase(unittest.TestCase):
    def setUp(self):
        self.obj = Thing('A name')
        self.expected_json = (
                '{"'+tags.OBJECT+'": "jsonpickle._samples.Thing",'
                ' "name": "A name", "child": null}')

    def test_encode(self):
        pickled = jsonpickle.encode(self.obj)
        self.assertEqual(self.expected_json, pickled)

    def test_encode_notunpicklable(self):
        pickled = jsonpickle.encode(self.obj, unpicklable=False)
        self.assertEqual('{"name": "A name", "child": null}', pickled)

    def test_decode(self):
        unpickled = jsonpickle.decode(self.expected_json)
        self.assertEqual(self.obj.name, unpickled.name)
        self.assertEqual(type(self.obj), type(unpickled))

    def test_json(self):
        pickled = jsonpickle.encode(self.obj)
        self.assertEqual(self.expected_json, pickled)

        unpickled = jsonpickle.decode(self.expected_json)
        self.assertEqual(self.obj.name, unpickled.name)
        self.assertEqual(type(self.obj), type(unpickled))

    def test_unicode_dict_keys(self):
        pickled = jsonpickle.encode({'é'.decode('utf-8'): 'é'.decode('utf-8')})
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(unpickled['é'.decode('utf-8')], 'é'.decode('utf-8'))
        self.assertTrue('é'.decode('utf-8') in unpickled)

    def test_tuple_dict_keys(self):
        """Test that we handle dictionaries with tuples as keys.
        We do not model this presently, so ensure that we at
        least convert those tuples to repr strings.

        TODO: handle dictionaries with non-stringy keys.
        """
        pickled = jsonpickle.encode({(1, 2): 3,
                                     (4, 5): { (7, 8): 9 }})
        unpickled = jsonpickle.decode(pickled)
        subdict = unpickled['(4, 5)']

        self.assertEqual(unpickled['(1, 2)'], 3)
        self.assertEqual(subdict['(7, 8)'], 9)

    def test_datetime_dict_keys(self):
        """Test that we handle datetime objects as keys.
        We do not model this presently, so ensure that we at
        least convert those tuples into repr strings.

        """
        pickled = jsonpickle.encode({datetime.datetime(2008, 12, 31): True})
        unpickled = jsonpickle.decode(pickled)
        self.assertTrue(unpickled['datetime.datetime(2008, 12, 31, 0, 0)'])

    def test_object_dict_keys(self):
        """Test that we handle random objects as keys.

        """
        thing = Thing('random')
        pickled = jsonpickle.encode({thing: True})
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(unpickled,
                {u('Thing("random")'): True})

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

    def test_load_backend(self):
        """Test that we can call jsonpickle.load_backend()

        """
        jsonpickle.load_backend('simplejson', 'dumps', 'loads', ValueError)

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


# Test classes for ExternalHandlerTestCase
class Mixin(object):
    def ok(self):
        return True


class UnicodeMixin(unicode, Mixin):
    def __add__(self, rhs):
        obj = super(UnicodeMixin, self).__add__(rhs)
        return UnicodeMixin(obj)


class UnicodeMixinHandler(handlers.BaseHandler):
    _handles = UnicodeMixin,
    def flatten(self, obj, data):
        data['value'] = obj
        return data

    def restore(self, obj):
        return UnicodeMixin(obj['value'])


class ExternalHandlerTestCase(unittest.TestCase):
    def test_unicode_mixin(self):
        obj = UnicodeMixin('test')
        self.assertEqual(type(obj), UnicodeMixin)
        self.assertEqual(unicode(obj), u('test'))

        # Encode into JSON
        content = jsonpickle.encode(obj)

        # Resurrect from JSON
        new_obj = jsonpickle.decode(content)
        new_obj += ' passed'

        self.assertEqual(unicode(new_obj), u('test passed'))
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
