# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009, 2011, 2013 David Aguilar
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import doctest
import os
import unittest
import collections

import jsonpickle

from jsonpickle import tags
from jsonpickle.compat import unicode
from jsonpickle.compat import unichr
from jsonpickle.compat import PY32


class Thing(object):

    def __init__(self, name):
        self.name = name
        self.child = None

    def __repr__(self):
        return 'Thing("%s")' % self.name


class ThingWithProps(object):

    def __init__(self, name='', dogs='reliable', monkies='tricksy'):
        self.name = name
        self._critters = (('dogs', dogs), ('monkies', monkies))

    def _get_identity(self):
        keys = [self.dogs, self.monkies, self.name]
        return hash('-'.join([str(key) for key in keys]))

    identity = property(_get_identity)

    def _get_dogs(self):
        return self._critters[0][1]

    dogs = property(_get_dogs)

    def _get_monkies(self):
        return self._critters[1][1]

    monkies = property(_get_monkies)

    def __getstate__(self):
        out = dict(
            __identity__=self.identity,
            nom=self.name,
            dogs=self.dogs,
            monkies=self.monkies,
        )
        return out

    def __setstate__(self, state_dict):
        self._critters = (('dogs', state_dict.get('dogs')),
                          ('monkies', state_dict.get('monkies')))
        self.name = state_dict.get('nom', '')
        ident = state_dict.get('__identity__')
        if ident != self.identity:
            raise ValueError('expanded object does not match originial state!')

    def __eq__(self, other):
        return self.identity == other.identity



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

    def test_set(self):
        setlist = ['orange', 'apple', 'grape']
        setA = set(setlist)

        flattened = self.pickler.flatten(setA)
        for s in setlist:
            self.assertTrue(s in flattened[tags.SET])

        setA_pickled = {tags.SET: setlist}
        self.assertEqual(setA, self.unpickler.restore(setA_pickled))

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
                            tags.TYPE: 'jsonpickle_test.Thing',
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

    def test_builtin_error(self):
        expect = AssertionError
        json = jsonpickle.encode(expect)
        actual = jsonpickle.decode(json)
        self.assertEqual(expect, actual)
        self.assertTrue(expect is actual)


class JSONPickleTestCase(unittest.TestCase):

    def setUp(self):
        self.obj = Thing('A name')
        self.expected_json = (
                '{"'+tags.OBJECT+'": "jsonpickle_test.Thing",'
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


class PicklableNamedTuple(object):
    """
    A picklable namedtuple wrapper, to demonstrate the need
    for protocol 2 compatibility. Yes, this is contrived in
    its use of new, but it demonstrates the issue.
    """

    def __new__(cls, propnames, vals):
        # it's necessary to use the correct class name for class resolution
        # classes that fake their own names may never be unpicklable
        ntuple = collections.namedtuple(cls.__name__, propnames)
        ntuple.__getnewargs__ = (lambda self: (propnames, vals))
        instance = ntuple.__new__(ntuple, *vals)
        return instance


class PickleProtocol2Thing(object):

    def __init__(self, *args):
        self.args = args

    def __getnewargs__(self):
        return self.args

    def __eq__(self, other):
        """
        Make PickleProtocol2Thing('slotmagic') ==
             PickleProtocol2Thing('slotmagic')
        """
        if self.__dict__ == other.__dict__ and dir(self) == dir(other):
            for prop in dir(self):
                selfprop = getattr(self, prop)
                if not callable(selfprop) and prop[0] != '_':
                    if selfprop != getattr(other, prop):
                        return False
            return True
        else:
            return False


# these two instances are used below and in tests
slotmagic = PickleProtocol2Thing('slotmagic')
dictmagic = PickleProtocol2Thing('dictmagic')

class PickleProtocol2GetState(PickleProtocol2Thing):
    def __new__(cls, *args):
        instance = super(PickleProtocol2GetState, cls).__new__(cls)
        instance.newargs = args
        return instance

    def __getstate__(self):
        return 'I am magic'

class PickleProtocol2GetStateDict(PickleProtocol2Thing):
    def __getstate__(self):
        return {'magic': True}

class PickleProtocol2GetStateSlots(PickleProtocol2Thing):
    def __getstate__(self):
        return (None, {'slotmagic': slotmagic})

class PickleProtocol2GetStateSlotsDict(PickleProtocol2Thing):
    def __getstate__(self):
        return ({'dictmagic': dictmagic}, {'slotmagic': slotmagic})


class PickleProtocol2GetSetState(PickleProtocol2GetState):
    def __setstate__(self, state):
        """
        Contrived example, easy to test
        """
        if state == "I am magic":
            self.magic = True
        else:
            self.magic = False


class PickleProtocol2ChildThing(object):

    def __init__(self, child):
        self.child = child

    def __getnewargs__(self):
        return ([self.child],)


class PicklingProtocol2TestCase(unittest.TestCase):

    def test_pickle_newargs(self):
        """
        Ensure we can pickle and unpickle an object whose class needs arguments
        to __new__ and get back the same typle
        """
        instance = PicklableNamedTuple(('a', 'b'), (1, 2))
        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(instance, decoded)

    def test_validate_reconstruct_by_newargs(self):
        """
        Ensure that the exemplar tuple's __getnewargs__ works
        This is necessary to know whether the breakage exists
        in jsonpickle or not
        """
        instance = PicklableNamedTuple(('a', 'b'), (1, 2))
        newinstance = PicklableNamedTuple.__new__(PicklableNamedTuple,
                                                 *(instance.__getnewargs__()))
        self.assertEqual(instance, newinstance)

    def test_getnewargs_priority(self):
        """
        Ensure newargs are used before py/state when decoding
        (As per PEP 307, classes are not supposed to implement
        all three magic methods)
        """
        instance = PickleProtocol2GetState('whatevs')
        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(decoded.newargs, ('whatevs',))

    def test_restore_dict_state(self):
        """
        Ensure that if getstate returns a dict, and there is no custom
        __setstate__, the dict is used as a source of variables to restore
        """
        instance = PickleProtocol2GetStateDict('whatevs')
        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)
        self.assertTrue(decoded.magic)

    def test_restore_slots_state(self):
        """
        Ensure that if getstate returns a 2-tuple with a dict in the second
        position, and there is no custom __setstate__, the dict is used as a
        source of variables to restore
        """
        instance = PickleProtocol2GetStateSlots('whatevs')
        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(decoded.slotmagic.__dict__, slotmagic.__dict__)
        self.assertEqual(decoded.slotmagic, slotmagic)

    def test_restore_slots_dict_state(self):
        """
        Ensure that if getstate returns a 2-tuple with a dict in both positions,
        and there is no custom __setstate__, the dicts are used as a source of
        variables to restore
        """
        instance = PickleProtocol2GetStateSlotsDict('whatevs')
        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(PickleProtocol2Thing('slotmagic'),
                         PickleProtocol2Thing('slotmagic'))
        self.assertEqual(decoded.slotmagic.__dict__, slotmagic.__dict__)
        self.assertEqual(decoded.slotmagic, slotmagic)
        self.assertEqual(decoded.dictmagic, dictmagic)

    def test_setstate(self):
        """
        Ensure output of getstate is passed to setstate
        """
        instance = PickleProtocol2GetSetState('whatevs')
        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)
        self.assertTrue(decoded.magic)

    def test_handles_nested_objects(self):
        child = PickleProtocol2Thing(None)
        instance = PickleProtocol2Thing(child, child)

        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(PickleProtocol2Thing, decoded.__class__)
        self.assertEqual(PickleProtocol2Thing, decoded.args[0].__class__)
        self.assertEqual(PickleProtocol2Thing, decoded.args[1].__class__)
        self.assertTrue(decoded.args[0] is decoded.args[1])

    def test_handles_cyclical_objects(self):
        child = PickleProtocol2Thing(None)
        instance = PickleProtocol2Thing(child, child)
        child.args = (instance,) # create a cycle
        # TODO we do not properly restore references inside of lists.
        # Change the above tuple into a list to show the breakage.

        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)

        # Ensure the right objects were constructed
        self.assertEqual(PickleProtocol2Thing, decoded.__class__)
        self.assertEqual(PickleProtocol2Thing, decoded.args[0].__class__)
        self.assertEqual(PickleProtocol2Thing, decoded.args[1].__class__)
        self.assertEqual(PickleProtocol2Thing, decoded.args[0].args[0].__class__)
        self.assertEqual(PickleProtocol2Thing, decoded.args[1].args[0].__class__)

        # It's turtles all the way down
        self.assertEqual(PickleProtocol2Thing, decoded.args[0].args[0]
                                                      .args[0].args[0]
                                                      .args[0].args[0]
                                                      .args[0].args[0]
                                                      .args[0].args[0]
                                                      .args[0].args[0]
                                                      .args[0].args[0]
                                                      .args[0].__class__)
        # Ensure that references are properly constructed
        self.assertTrue(decoded.args[0] is decoded.args[1])
        self.assertTrue(decoded is decoded.args[0].args[0])
        self.assertTrue(decoded is decoded.args[1].args[0])
        self.assertTrue(decoded.args[0] is decoded.args[0].args[0].args[0])
        self.assertTrue(decoded.args[0] is decoded.args[0].args[1].args[0])

    def test_handles_cyclical_objects_in_lists(self):
        child = PickleProtocol2ChildThing(None)
        instance = PickleProtocol2ChildThing([child, child])
        child.child = instance # create a cycle

        encoded = jsonpickle.encode(instance)
        decoded = jsonpickle.decode(encoded)

        self.assertTrue(decoded is decoded.child[0].child)
        self.assertTrue(decoded is decoded.child[1].child)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(PicklingTestCase))
    suite.addTest(unittest.makeSuite(PicklingProtocol2TestCase))
    suite.addTest(unittest.makeSuite(JSONPickleTestCase))
    suite.addTest(doctest.DocTestSuite(jsonpickle.pickler))
    suite.addTest(doctest.DocTestSuite(jsonpickle.unpickler))
    suite.addTest(doctest.DocTestSuite(jsonpickle))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
