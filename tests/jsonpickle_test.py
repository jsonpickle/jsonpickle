# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009-2024 David Aguilar (davvid -at- gmail.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import collections
import os
import warnings

import pytest

import jsonpickle
import jsonpickle.backend
import jsonpickle.handlers
from jsonpickle import tags, util


class ListLike:
    def __init__(self):
        self.internal_list = []

    def append(self, item):
        self.internal_list.append(item)

    def __getitem__(self, i):
        return self.internal_list[i]


class Thing:
    def __init__(self, name):
        self.name = name
        self.child = None

    def __iter__(self):
        for attr in [
            x for x in getattr(self.__class__, '__dict__') if not x.startswith('__')
        ]:
            yield attr, getattr(self, attr)

    def __repr__(self):
        return 'Thing("%s")' % self.name


class Capture:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return object.__repr__(self) + (f'({self.args!r}, {self.kwargs!r})')


class ThingWithProps:
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

    def __init__(self, name='', dogs='reliable', monkies='tricksy'):
        self.name = name
        self._critters = (('dogs', dogs), ('monkies', monkies))

    def __eq__(self, other):
        return self.identity == other.identity

    def __getstate__(self):
        out = dict(
            __identity__=self.identity,
            nom=self.name,
            dogs=self.dogs,
            monkies=self.monkies,
        )
        return out

    def __setstate__(self, state_dict):
        self._critters = (
            ('dogs', state_dict.get('dogs')),
            ('monkies', state_dict.get('monkies')),
        )
        self.name = state_dict.get('nom', '')
        ident = state_dict.get('__identity__')
        if ident != self.identity:
            raise ValueError('expanded object does not match original state!')


class UserDict(dict):
    """A user class that inherits from :class:`dict`"""

    def __init__(self, **kwargs):
        dict.__init__(self, **kwargs)
        self.valid = False


class Outer:
    class Middle:
        class Inner:
            pass


class MySlots:
    __slots__ = ('alpha', '__beta')

    def __init__(self):
        self.alpha = 1
        self.__beta = 1


class MyPropertiesSlots:
    __slots__ = ('alpha', 'arr')

    def __init__(self):
        self.alpha = 1
        self.arr = [1, 2, 3]

    @property
    def arr_len(self):
        return len(self.arr)

    @property
    def other_object(self):
        return Thing('Test')

    def __eq__(self, other):
        return (
            self.alpha == other.alpha
            and self.arr == other.arr
            and self.arr_len == other.arr_len
            and list(self.other_object) == list(other.other_object)
        )


class MyPropertiesDict:
    def __init__(self):
        self.alpha = 1
        self.arr = [1, 2, 3]

    @property
    def arr_len(self):
        return len(self.arr)

    @property
    def other_object(self):
        return Thing('Test')

    def __eq__(self, other):
        return (
            self.alpha == other.alpha
            and self.arr == other.arr
            and self.arr_len == other.arr_len
            and list(self.other_object) == list(other.other_object)
        )


# see issue #478
class SafeData:
    __slots__ = ()


class SafeString(str, SafeData):
    __slots__ = ()


def on_missing_callback(class_name):
    # not actually a runtime problem but it doesn't matter
    warnings.warn('The unpickler could not find %s' % class_name, RuntimeWarning)


@pytest.fixture
def pickler():
    """Returns a default-constructed pickler"""
    return jsonpickle.pickler.Pickler()


@pytest.fixture
def unpickler():
    """Returns a default-constructed unpickler"""
    return jsonpickle.unpickler.Unpickler()


@pytest.fixture
def b85_pickler():
    """Returns a pickler setup for base85 encoding"""
    return jsonpickle.pickler.Pickler(use_base85=True)


def test_bytes_base85(b85_pickler):
    """base85 is emitted when the pickler is setup to do so"""
    data = os.urandom(16)
    encoded = util.b85encode(data)
    assert b85_pickler.flatten(data) == {tags.B85: encoded}


def test_bytes_base64_default(pickler):
    """base64 must be used by default"""
    data = os.urandom(16)
    encoded = util.b64encode(data)
    assert pickler.flatten(data) == {tags.B64: encoded}


def test_decode_base85(unpickler):
    """base85 data must be restored"""
    expected = 'Pÿthöñ 3!'.encode()
    pickled = {tags.B85: util.b85encode(expected)}
    assert unpickler.restore(pickled) == expected


@pytest.mark.parametrize('value', ['', '/', '\udc00', 1, True, False, None, [], {}])
def test_decode_invalid_b85(value, unpickler):
    """Invalid base85 data restores to an empty string"""
    expected = b''
    pickled = {tags.B85: value}
    assert unpickler.restore(pickled) == expected


def test_base85_still_handles_base64(unpickler):
    """base64 must be restored even though base85 is the default"""
    expected = 'Pÿthöñ 3!'.encode()
    pickled = {tags.B64: util.b64encode(expected)}
    assert unpickler.restore(pickled) == expected


@pytest.mark.parametrize(
    'value', ['', 'x', '!', '\udc00', 0, 1, True, False, None, [], {}]
)
def test_decode_invalid_b64(value, unpickler):
    """Invalid base85 data restores to an empty string"""
    expected = b''
    pickled = {tags.B64: value}
    assert unpickler.restore(pickled) == expected


def test_string(pickler, unpickler):
    """Strings must roundtrip"""
    assert pickler.flatten('a string') == 'a string'
    assert unpickler.restore('a string') == 'a string'


def test_int(pickler, unpickler):
    """Ints must roundtrip"""
    assert pickler.flatten(3) == 3
    assert unpickler.restore(3) == 3


def test_float(pickler, unpickler):
    """Floats must roundtrip"""
    assert pickler.flatten(3.5) == 3.5
    assert unpickler.restore(3.5) == 3.5


def test_boolean(pickler, unpickler):
    """Booleans must roundtrip"""
    assert pickler.flatten(True) is True
    assert pickler.flatten(False) is False
    assert unpickler.restore(True) is True
    assert unpickler.restore(False) is False


def test_none(pickler, unpickler):
    """None must roundtrip"""
    assert pickler.flatten(None) is None
    assert unpickler.restore(None) is None


def test_list(pickler, unpickler):
    """Nested and mixed lists must roundtrip"""
    # multiple types of values
    list_a = [1, 35.0, 'value']
    assert pickler.flatten(list_a) == list_a
    assert unpickler.restore(list_a) == list_a
    # nested list
    list_b = [40, 40, list_a, 6]
    assert pickler.flatten(list_b) == list_b
    assert unpickler.restore(list_b) == list_b
    # 2D list
    list_c = [[1, 2], [3, 4]]
    assert pickler.flatten(list_c) == list_c
    assert unpickler.restore(list_c) == list_c
    # empty list
    list_d = []
    assert pickler.flatten(list_d) == list_d
    assert unpickler.restore(list_d) == list_d


def test_nonetype():
    """NoneType must roundtrip"""
    typ = type(None)
    typ_pickled = jsonpickle.encode(typ)
    typ_unpickled = jsonpickle.decode(typ_pickled)
    assert typ_unpickled == typ


def test_set(pickler, unpickler):
    """Validate the internal representation for set() objects"""
    set_list = ['orange', 'apple', 'grape']
    set_obj = set(set_list)
    flattened = pickler.flatten(set_obj)
    for s in set_list:
        assert s in flattened[tags.SET]
    set_pickle = {tags.SET: set_list}
    assert unpickler.restore(set_pickle) == set_obj


@pytest.mark.parametrize('value', ['', 0, 1, True, False, None, [], {}])
def test_set_with_invalid_data(value, unpickler):
    """Invalid serialized set data results in an empty set"""
    data = {tags.SET: value}
    result = unpickler.restore(data)
    assert result == set()


@pytest.mark.parametrize('value', ['', 0, 1, True, False, None, [], {}])
def test_tuple_with_invalid_data(value, unpickler):
    """Invalid serialized tuple data results in an empty tuple"""
    data = {tags.TUPLE: value}
    result = unpickler.restore(data)
    assert result == tuple()


def test_iterator_with_invalid_data(unpickler):
    """Invalid serialized iterator data results in an empty iterator"""
    data = {tags.ITERATOR: set()}
    result = unpickler.restore(data)
    with pytest.raises(StopIteration):
        next(result)
        assert False


@pytest.mark.parametrize(
    'value', ['', 0, True, False, None, [], {}, ('x',), {'x': True}]
)
def test_reduce_with_invalid_data(value, unpickler):
    """Invalid serialized reduce data results in an empty list"""
    data = {tags.REDUCE: value}
    result = unpickler.restore(data)
    assert result == []


@pytest.mark.parametrize('value', ['', 'x', 1, True, [], {}])
def test_restore_id_with_invalid_data(value, unpickler):
    """Invalid serialized ID data results in None"""
    result = unpickler.restore({'ref': {tags.ID: value}})
    assert result['ref'] is None


def test_dict(pickler, unpickler):
    """Our custom keys are preserved when user dicts contain them"""
    dict_a = {'key1': 1.0, 'key2': 20, 'key3': 'thirty', tags.JSON_KEY + '6': 6}
    assert pickler.flatten(dict_a) == dict_a
    assert unpickler.restore(dict_a) == dict_a
    dict_b = {}
    assert pickler.flatten(dict_b) == dict_b
    assert unpickler.restore(dict_b) == dict_b


def test_tuple(pickler, unpickler):
    """Validate the internal represntation for tuples"""
    # currently all collections are converted to lists
    tuple_a = (4, 16, 32)
    tuple_a_pickle = {tags.TUPLE: [4, 16, 32]}
    assert pickler.flatten(tuple_a) == tuple_a_pickle
    assert unpickler.restore(tuple_a_pickle) == tuple_a
    tuple_b = (4,)
    tuple_b_pickle = {tags.TUPLE: [4]}
    assert pickler.flatten(tuple_b) == tuple_b_pickle
    assert unpickler.restore(tuple_b_pickle) == tuple_b


def _roundtrip(data):
    """Encode and decode and object through jsonpickle"""
    return jsonpickle.decode(jsonpickle.encode(data))


def test_tuple_roundtrip():
    """Tuples can roundtrip"""
    data = (1, 2, 3)
    assert _roundtrip(data) == data


def test_set_roundtrip():
    """Sets can roundtrip"""
    data = {1, 2, 3}
    assert _roundtrip(data) == data


def test_list_roundtrip():
    """Lists can roundtrip"""
    data = [1, 2, 3]
    assert _roundtrip(data) == data


def test_class(pickler, unpickler):
    """Nested objects can roundtrip"""
    inst = Thing('test name')
    inst.child = Thing('child name')
    flattened = pickler.flatten(inst)
    assert flattened['name'] == 'test name'
    child = flattened['child']
    assert child['name'] == 'child name'
    inflated = unpickler.restore(flattened)
    assert inflated.name == 'test name'
    assert type(inflated) is Thing
    assert inflated.child.name == 'child name'
    assert type(inflated.child) is Thing


def test_classlist(pickler, unpickler):
    """Lists with mixed object values can roundtrip"""
    array = [Thing('one'), Thing('two'), 'a string']
    flattened = pickler.flatten(array)
    assert 'one' == flattened[0]['name']
    assert 'two' == flattened[1]['name']
    assert 'a string' == flattened[2]
    inflated = unpickler.restore(flattened)
    assert 'one' == inflated[0].name
    assert type(inflated[0]) is Thing
    assert 'two' == inflated[1].name
    assert type(inflated[1]) is Thing
    assert 'a string' == inflated[2]


def test_classdict(pickler, unpickler):
    """Dicts with object values can roundtrip"""
    obj_dict = {'k1': Thing('one'), 'k2': Thing('two'), 'k3': 3}
    flattened = pickler.flatten(obj_dict)
    assert 'one' == flattened['k1']['name']
    assert 'two' == flattened['k2']['name']
    assert 3 == flattened['k3']
    inflated = unpickler.restore(flattened)
    assert 'one' == inflated['k1'].name
    assert type(inflated['k1']) is Thing
    assert 'two' == inflated['k2'].name
    assert type(inflated['k2']) is Thing
    assert 3 == inflated['k3']


def test_recursive():
    """create a recursive structure and test that we can handle it"""
    parent = Thing('parent')
    child = Thing('child')
    child.sibling = Thing('sibling')
    parent.self = parent
    parent.child = child
    parent.child.twin = child
    parent.child.parent = parent
    parent.child.sibling.parent = parent
    cloned = jsonpickle.decode(jsonpickle.encode(parent))
    assert parent.name == cloned.name
    assert parent.child.name == cloned.child.name
    assert parent.child.sibling.name == cloned.child.sibling.name
    assert cloned == cloned.child.parent
    assert cloned == cloned.child.sibling.parent
    assert cloned == cloned.child.twin.parent
    assert cloned.child == cloned.child.twin


def test_tuple_notunpicklable(pickler):
    """Tuples become lists when unpicklable is False"""
    pickler.unpicklable = False
    flattened = pickler.flatten(('one', 2, 3))
    assert flattened == ['one', 2, 3]


def test_set_not_unpicklable(pickler):
    """Sets become lists when unpicklable is False"""
    pickler.unpicklable = False
    flattened = pickler.flatten({'one', 2, 3})
    assert 'one' in flattened
    assert 2 in flattened
    assert 3 in flattened
    assert isinstance(flattened, list)


def test_thing_with_module(pickler, unpickler):
    """Objects with references to modules can roundtrip"""
    obj = Thing('with-module')
    obj.themodule = os
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert inflated.themodule == os


def test_thing_with_module_safe(pickler, unpickler):
    """Objects with inner references to modules can roundtrip in safe mode"""
    obj = Thing('with-module')
    obj.themodule = os
    flattened = pickler.flatten(obj)
    unpickler.safe = True
    inflated = unpickler.restore(flattened)
    assert inflated.themodule is os
    # Unsafe mode
    unpickler.safe = False
    inflated = unpickler.restore(flattened)
    assert inflated.themodule is os


def test_thing_with_submodule(pickler, unpickler):
    """Objects with inner references to modules can roundtrip"""
    obj = Thing('with-submodule')
    obj.submodule = collections
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert inflated.submodule == collections


def test_type_reference(pickler, unpickler):
    """This test ensures that users can store references to types."""
    obj = Thing('object-with-type-reference')
    # reference the built-in 'object' type
    obj.typeref = object
    flattened = pickler.flatten(obj)
    assert flattened['typeref'] == {tags.TYPE: 'builtins.object'}
    inflated = unpickler.restore(flattened)
    assert inflated.typeref == object


def test_class_reference(pickler, unpickler):
    """This test ensures that users can store references to classes."""
    obj = Thing('object-with-class-reference')
    # reference the 'Thing' class (not an instance of the class)
    obj.classref = Thing
    flattened = pickler.flatten(obj)
    assert flattened['classref'] == {tags.TYPE: 'jsonpickle_test.Thing'}
    inflated = unpickler.restore(flattened)
    assert inflated.classref is Thing


def test_supports_getstate_setstate(pickler, unpickler):
    """Objects with getstate and setstate can roundtrip"""
    obj = ThingWithProps('object-which-defines-getstate-setstate')
    flattened = pickler.flatten(obj)
    assert flattened[tags.STATE].get('__identity__')
    assert flattened[tags.STATE].get('nom')
    inflated = unpickler.restore(flattened)
    assert obj == inflated


def test_references(pickler, unpickler):
    """References in lists can roundtrip"""
    obj_a = Thing('foo')
    obj_b = Thing('bar')
    coll = [obj_a, obj_b, obj_b]
    flattened = pickler.flatten(coll)
    inflated = unpickler.restore(flattened)
    assert len(inflated) == len(coll)
    for x in range(len(coll)):
        assert repr(coll[x]) == repr(inflated[x])


def test_references_in_number_keyed_dict(pickler, unpickler):
    """Dicts with numbers as keys and objects as values can roundtrip

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
    assert list(sorted(obj.keys())) != list(map(int, sorted(map(str, obj.keys()))))
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert len(inflated) == 3
    assert inflated['12'].name == 'twelve'


def test_builtin_error():
    """Builtin errors can roundtrip"""
    expect = AssertionError
    json = jsonpickle.encode(expect)
    actual = jsonpickle.decode(json)
    assert expect == actual
    assert expect is actual


def test_builtin_function():
    """Builtin functions can roundtrip"""
    expect = dir
    json = jsonpickle.encode(expect)
    actual = jsonpickle.decode(json)
    assert expect == actual
    assert expect is actual


def test_restore_legacy_builtins():
    """Decoding is backwards compatible.

    jsonpickle 0.9.6 and earlier used the Python 2 `__builtin__`
    naming for builtins. Ensure those can be loaded until they're
    no longer supported.
    """
    ae = jsonpickle.decode('{"py/type": "__builtin__.AssertionError"}')
    assert ae is AssertionError
    ae = jsonpickle.decode('{"py/type": "exceptions.AssertionError"}')
    assert ae is AssertionError
    cls = jsonpickle.decode('{"py/type": "__builtin__.int"}')
    assert cls is int


@pytest.mark.parametrize(
    'value,expect',
    [
        ('module_does_not_exist/ignored', None),
        ('builtins/int', None),
        ('builtins/invalid.int', None),
        ('builtins/builtinsx.int', None),
    ],
)
def test_restore_invalid_repr(value, expect, unpickler):
    """Test restoring invalid repr tags"""
    result = unpickler.restore({tags.REPR: value})
    assert result is expect


def test_unpickler_on_missing():
    """Emit warnings when decoding objects whose classes are missing"""
    encoded = jsonpickle.encode(Outer.Middle.Inner())
    assert isinstance(jsonpickle.decode(encoded), Outer.Middle.Inner)
    # Alter the encoded string to create cases where the class is missing
    # at multiple levels.
    assert encoded == '{"py/object": "jsonpickle_test.Outer.Middle.Inner"}'
    missing_cases = [
        '{"py/object": "MISSING.Outer.Middle.Inner"}',
        '{"py/object": "jsonpickle_test.MISSING.Middle.Inner"}',
        '{"py/object": "jsonpickle_test.Outer.MISSING.Inner"}',
        '{"py/object": "jsonpickle_test.Outer.Middle.MISSING"}',
    ]
    for case in missing_cases:
        # https://docs.python.org/3/library/warnings.html#testing-warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jsonpickle.decode(case, on_missing='warn')
            assert issubclass(w[-1].category, UserWarning)
            assert 'Unpickler._restore_object could not find' in str(w[-1].message)
            jsonpickle.decode(case, on_missing=on_missing_callback)
            assert issubclass(w[-1].category, RuntimeWarning)
            assert 'The unpickler could not find' in str(w[-1].message)

        assert jsonpickle.decode(
            case, on_missing='ignore'
        ) == jsonpickle.backend.json.loads(case)
        try:
            jsonpickle.decode(case, on_missing='error')
        except jsonpickle.errors.ClassNotFoundError:
            # it's supposed to error
            assert True
        else:
            assert False


def test_private_slot_members():
    """Objects with private slot members can roundtrip"""
    obj = jsonpickle.loads(jsonpickle.dumps(MySlots()))
    alpha = getattr(obj, 'alpha', '(missing alpha)')
    beta = getattr(obj, '_' + obj.__class__.__name__ + '__beta', '(missing beta)')
    assert alpha == beta


def test_include_properties_slots():
    """Properties of slot-using objects are encoded with include_properties=True"""
    obj = MyPropertiesSlots()
    dumped = jsonpickle.dumps(obj, include_properties=True)
    assert 'py/property' in dumped
    assert jsonpickle.loads(dumped) == obj


def test_include_properties_dict():
    """Dicts with properties are encoded with include_properties=True"""
    obj = MyPropertiesDict()
    dumped = jsonpickle.dumps(obj, include_properties=True)
    assert 'py/property' in dumped
    assert jsonpickle.loads(dumped) == obj


def test_load_non_fully_qualified_classes():
    """Custom classes can be specified using string names when decoding"""
    # reuse MyPropertiesSlots because it has a nice eq method
    obj = MyPropertiesSlots()
    encoded = jsonpickle.encode(obj)
    # MyPropertiesSlots and MyPropertiesDict have compatible eq methods
    decoded = jsonpickle.decode(
        encoded, classes={'MyPropertiesSlots': MyPropertiesDict}
    )
    assert isinstance(decoded, MyPropertiesDict)
    assert obj == decoded


def test_classes_dict():
    """Custom classes can be used when decoding"""
    # reuse MyPropertiesSlots because it has a nice eq method
    obj = MyPropertiesSlots()
    encoded = jsonpickle.encode(obj)
    # MyPropertiesSlots and MyPropertiesDict have compatible eq methods
    decoded = jsonpickle.decode(encoded, classes={MyPropertiesSlots: MyPropertiesDict})
    assert isinstance(decoded, MyPropertiesDict)
    assert obj == decoded


def test_warnings():
    """Emit warnings when pickling file descriptors"""
    data = os.fdopen(os.pipe()[0])
    with warnings.catch_warnings(record=True) as w:
        jsonpickle.encode(data, warn=True)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "replaced with None" in str(w[-1].message)


def test_API_names():
    """Enforce expected names in main module"""
    names = list(vars(jsonpickle))
    assert 'pickler' in names
    assert 'unpickler' in names
    assert 'JSONBackend' in names
    assert '__version__' in names
    assert 'register' in names
    assert 'unregister' in names
    assert 'Pickler' in names
    assert 'Unpickler' in names
    assert 'encode' in names
    assert 'decode' in names


def test_encode():
    """Encode an object with default settings"""
    expect = Thing('A name')
    pickle = jsonpickle.encode(expect)
    actual = jsonpickle.decode(pickle)
    assert expect.name == actual.name
    assert expect.child == actual.child


def test_encode_notunpicklable():
    """Encode an object with unpicklable=False"""
    obj = Thing('A name')
    expect = {'name': 'A name', 'child': None}
    pickle = jsonpickle.encode(obj, unpicklable=False)
    actual = jsonpickle.decode(pickle)
    assert expect['name'] == actual['name']


def test_decode():
    """Validate that we can decode a known json payload"""
    expect = Thing('A name')
    expected_json = (
        '{"%s": "jsonpickle_test.Thing", "name": "A name", "child": null}' % tags.OBJECT
    )
    actual = jsonpickle.decode(expected_json)
    assert expect.name == actual.name
    assert type(expect) is type(actual)


def test_json():
    """Validate the json representation"""
    expect = Thing('A name')
    expected_json = (
        '{"%s": "jsonpickle_test.Thing", "name": "A name", "child": null}' % tags.OBJECT
    )
    pickle = jsonpickle.encode(expect)
    actual = jsonpickle.decode(pickle)
    assert actual.name == expect.name
    assert actual.child == expect.child
    actual = jsonpickle.decode(expected_json)
    assert expect.name == actual.name
    assert type(expect) is type(actual)


def test_unicode_dict_keys():
    """Non-ascii unicode data in dict keys is preserved"""
    uni = chr(0x1234)
    pickle = jsonpickle.encode({uni: uni})
    actual = jsonpickle.decode(pickle)
    assert uni in actual
    assert actual[uni] == uni


def test_tuple_dict_keys_default():
    """Test that we handle dictionaries with tuples as keys."""
    tuple_dict = {(1, 2): 3, (4, 5): {(7, 8): 9}}
    pickle = jsonpickle.encode(tuple_dict)
    expect = {'(1, 2)': 3, '(4, 5)': {'(7, 8)': 9}}
    actual = jsonpickle.decode(pickle)
    assert expect == actual
    tuple_dict = {(1, 2): [1, 2]}
    pickle = jsonpickle.encode(tuple_dict)
    actual = jsonpickle.decode(pickle)
    assert actual['(1, 2)'] == [1, 2]


def test_tuple_dict_keys_with_keys_enabled():
    """Test that we handle dictionaries with tuples as keys."""
    tuple_dict = {(1, 2): 3, (4, 5): {(7, 8): 9}}
    pickle = jsonpickle.encode(tuple_dict, keys=True)
    expect = tuple_dict
    actual = jsonpickle.decode(pickle, keys=True)
    assert expect == actual
    tuple_dict = {(1, 2): [1, 2]}
    pickle = jsonpickle.encode(tuple_dict, keys=True)
    actual = jsonpickle.decode(pickle, keys=True)
    assert actual[(1, 2)] == [1, 2]


def test_None_dict_key_default():
    """None is stringified by default when used as a dict key"""
    expect = {'null': None}
    obj = {None: None}
    pickle = jsonpickle.encode(obj)
    actual = jsonpickle.decode(pickle)
    assert expect == actual


def test_None_dict_key_with_keys_enabled():
    """None can be used as a dict key when keys=True"""
    expect = {None: None}
    obj = {None: None}
    pickle = jsonpickle.encode(obj, keys=True)
    actual = jsonpickle.decode(pickle, keys=True)
    assert expect == actual


def test_object_dict_keys():
    """Test that we handle random objects as keys."""
    thing = Thing('random')
    pickle = jsonpickle.encode({thing: True})
    actual = jsonpickle.decode(pickle)
    assert actual == {'Thing("random")': True}


def test_int_dict_keys_defaults():
    """Int keys are stringified by default"""
    int_dict = {1000: [1, 2]}
    pickle = jsonpickle.encode(int_dict)
    actual = jsonpickle.decode(pickle)
    assert actual['1000'] == [1, 2]


def test_int_dict_keys_with_keys_enabled():
    """Int keys are supported with keys=True"""
    int_dict = {1000: [1, 2]}
    pickle = jsonpickle.encode(int_dict, keys=True)
    actual = jsonpickle.decode(pickle, keys=True)
    assert actual[1000] == [1, 2]


def test_string_key_requiring_escape_dict_keys_with_keys_enabled():
    """Dict keys that require escaping are handled"""
    json_key_dict = {tags.JSON_KEY + '6': [1, 2]}
    pickled = jsonpickle.encode(json_key_dict, keys=True)
    unpickled = jsonpickle.decode(pickled, keys=True)
    assert unpickled[tags.JSON_KEY + '6'] == [1, 2]


def test_string_key_not_requiring_escape_dict_keys_with_keys_enabled():
    """String keys that do not require escaping are not escaped"""
    str_dict = {'name': [1, 2]}
    pickled = jsonpickle.encode(str_dict, keys=True)
    unpickled = jsonpickle.decode(pickled)
    assert 'name' in unpickled


def test_dict_subclass():
    """Dict subclasses can roundtrip"""
    obj = UserDict()
    obj.valid = True
    obj.s = 'string'
    obj.d = 'd_string'
    obj['d'] = {}
    obj['s'] = 'test'
    pickle = jsonpickle.encode(obj)
    actual = jsonpickle.decode(pickle)
    assert type(actual) is UserDict
    assert 'd' in actual
    assert 's' in actual
    assert hasattr(actual, 'd')
    assert hasattr(actual, 's')
    assert hasattr(actual, 'valid')
    assert obj['d'] == actual['d']
    assert obj['s'] == actual['s']
    assert obj.d == actual.d
    assert obj.s == actual.s
    assert obj.valid == actual.valid


def test_dict_subclass_with_references():
    """Dict subclasses with references can roundtrip"""
    d = UserDict()
    d.s = 'string'
    d['s'] = 'test'
    obj = [d, d, d.__dict__]
    pickle = jsonpickle.encode(obj)
    actual = jsonpickle.decode(pickle)
    assert type(actual) is list
    assert type(actual[0]) is UserDict
    assert actual[0].s == 'string'
    assert actual[0]['s'] == 'test'
    assert actual[0] == actual[1]
    assert actual[0].__dict__ == actual[2]


def test_list_of_objects():
    """Using the same object in a list is preserved and not duplicated"""
    a = Thing('a')
    b = Thing('b')
    pickle = jsonpickle.encode([a, b, b])
    actual = jsonpickle.decode(pickle)
    assert actual[1] == actual[2]
    assert type(actual[0]) is Thing
    assert actual[0].name == 'a'
    assert actual[1].name == 'b'
    assert actual[2].name == 'b'


def test_refs_keys_values():
    """Objects in dict keys can roundtrip"""
    j = Thing('random')
    object_dict = {j: j}
    pickle = jsonpickle.encode(object_dict, keys=True)
    actual = jsonpickle.decode(pickle, keys=True)
    assert list(actual.values()) == list(actual.keys())


def test_object_keys_to_list():
    """Objects in dict values have their references preserved"""
    j = Thing('random')
    object_dict = {j: [j, j]}
    pickle = jsonpickle.encode(object_dict, keys=True)
    actual = jsonpickle.decode(pickle, keys=True)
    obj = list(actual.keys())[0]
    assert obj.name == j.name
    assert obj is actual[obj][0]
    assert obj is actual[obj][1]


def test_refs_in_objects():
    """Objects in lists have their references preserved"""
    a = Thing('a')
    b = Thing('b')
    pickle = jsonpickle.encode([a, b, b])
    actual = jsonpickle.decode(pickle)
    assert actual[0] != actual[1]
    assert actual[1] == actual[2]
    assert actual[1] is actual[2]


def test_refs_recursive():
    """Complicated recursive refs can roundtrip"""
    a = Thing('a')
    a.self_list = [Thing('0'), Thing('1'), Thing('2')]
    a.first = a.self_list[0]
    a.stuff = {a.first: a.first}
    a.morestuff = {a.self_list[1]: a.stuff}
    pickle = jsonpickle.encode(a, keys=True)
    b = jsonpickle.decode(pickle, keys=True)
    item = b.self_list[0]
    assert item == b.first
    assert item == b.stuff[b.first]
    assert item == b.morestuff[b.self_list[1]][b.first]


def test_load_backend():
    """Ensures that load_backend() is working"""
    assert jsonpickle.load_backend('simplejson', 'dumps', 'loads', ValueError)


def test_set_preferred_backend_allows_magic():
    """Pluggable backends can be used"""
    backend = 'os.path'
    jsonpickle.load_backend(backend, 'split', 'join', AttributeError)
    jsonpickle.set_preferred_backend(backend)
    slash_hello, world = jsonpickle.encode('/hello/world')
    jsonpickle.remove_backend(backend)

    assert slash_hello == '/hello'
    assert world == 'world'


def test_load_backend_submodule():
    """Inner modules can be used as a backend"""
    jsonpickle.load_backend('os.path', 'split', 'join', AttributeError)
    assert (
        'os.path' in jsonpickle.json._backend_names
        and 'os.path' in jsonpickle.json._encoders
        and 'os.path' in jsonpickle.json._decoders
        and 'os.path' in jsonpickle.json._encoder_options
        and 'os.path' in jsonpickle.json._decoder_exceptions
    )
    jsonpickle.remove_backend('os.path')


def _backend_is_partially_loaded(backend):
    """Return True if the specified backend is incomplete"""
    return (
        backend in jsonpickle.json._backend_names
        or backend in jsonpickle.json._encoders
        or backend in jsonpickle.json._decoders
        or backend in jsonpickle.json._encoder_options
        or backend in jsonpickle.json._decoder_exceptions
    )


def test_load_backend_handles_bad_encode():
    """Ignore bad encoders"""
    load_backend = jsonpickle.load_backend
    assert not load_backend('os.path', 'bad!', 'split', AttributeError)
    assert not _backend_is_partially_loaded('os.path')


def test_load_backend_raises_on_bad_decode():
    """Ignore bad decoders"""
    load_backend = jsonpickle.load_backend
    assert not load_backend('os.path', 'join', 'bad!', AttributeError)
    assert not _backend_is_partially_loaded('os.path')


def test_load_backend_handles_bad_loads_exc():
    """Ignore bad decoder exceptions"""
    load_backend = jsonpickle.load_backend
    assert not load_backend('os.path', 'join', 'split', 'bad!')
    assert not _backend_is_partially_loaded('os.path')


def test_list_item_reference():
    """References to objects inside lists are preserved"""
    thing = Thing('parent')
    thing.child = Thing('child')
    thing.child.refs = [thing]
    encoded = jsonpickle.encode(thing)
    decoded = jsonpickle.decode(encoded)
    assert id(decoded) == id(decoded.child.refs[0])


def test_reference_to_list():
    """References to lists are preserved"""
    thing = Thing('parent')
    thing.a = [1]
    thing.b = thing.a
    thing.b.append(thing.a)
    thing.b.append([thing.a])
    encoded = jsonpickle.encode(thing)
    decoded = jsonpickle.decode(encoded)
    assert 1 == decoded.a[0]
    assert 1 == decoded.b[0]
    assert id(decoded.b) == id(decoded.a)
    assert id(decoded.a[1]) == id(decoded.a)
    assert id(decoded.a[2][0]) == id(decoded.a)


def test_make_refs_disabled_list():
    """Lists are duplicated when make_refs is False"""
    obj_a = Thing('foo')
    obj_b = Thing('bar')
    coll = [obj_a, obj_b, obj_b]
    encoded = jsonpickle.encode(coll, make_refs=False)
    decoded = jsonpickle.decode(encoded)

    assert len(decoded) == 3
    assert decoded[0] is not decoded[1]
    assert decoded[1] is not decoded[2]


def test_make_refs_disabled_reference_to_list():
    """References are lost when maek_refs is False"""
    thing = Thing('parent')
    thing.a = [1]
    thing.b = thing.a
    thing.b.append(thing.a)
    thing.b.append([thing.a])
    encoded = jsonpickle.encode(thing, make_refs=False)
    decoded = jsonpickle.decode(encoded)
    assert decoded.a[0] == 1
    assert decoded.b[0] == 1
    assert decoded.a[1][0:3] == '[1,'  # ]
    assert decoded.a[2][0][0:3] == '[1,'  # ]


def test_can_serialize_inner_classes():
    """Specify hidden classes when decoding"""

    class InnerScope:
        """Private class visible to this method only"""

        def __init__(self, name):
            self.name = name

    obj = InnerScope('test')
    encoded = jsonpickle.encode(obj)
    # Single class
    decoded = jsonpickle.decode(encoded, classes=InnerScope)
    _test_inner_class(InnerScope, obj, decoded)
    # List of classes
    decoded = jsonpickle.decode(encoded, classes=[InnerScope])
    _test_inner_class(InnerScope, obj, decoded)
    # Tuple of classes
    decoded = jsonpickle.decode(encoded, classes=(InnerScope,))
    _test_inner_class(InnerScope, obj, decoded)
    # Set of classes
    decoded = jsonpickle.decode(encoded, classes={InnerScope})
    _test_inner_class(InnerScope, obj, decoded)


def _test_inner_class(cls, obj, decoded):
    assert isinstance(obj, cls)
    assert decoded.name == obj.name


def test_can_serialize_nested_classes():
    """Instances of nested classes can roundtrip"""
    middle = Outer.Middle
    inner = Outer.Middle.Inner
    encoded_middle = jsonpickle.encode(middle)
    encoded_inner = jsonpickle.encode(inner)
    decoded_middle = jsonpickle.decode(encoded_middle)
    decoded_inner = jsonpickle.decode(encoded_inner)

    assert isinstance(decoded_middle, type)
    assert isinstance(decoded_inner, type)
    assert decoded_middle == middle
    assert decoded_inner == inner


def test_can_serialize_nested_class_objects():
    """Class objects in an inner scope can roundtrip"""
    middle_obj = Outer.Middle()
    middle_obj.attribute = 5
    inner_obj = Outer.Middle.Inner()
    inner_obj.attribute = 6
    encoded_middle_obj = jsonpickle.encode(middle_obj)
    encoded_inner_obj = jsonpickle.encode(inner_obj)
    decoded_middle_obj = jsonpickle.decode(encoded_middle_obj)
    decoded_inner_obj = jsonpickle.decode(encoded_inner_obj)
    assert isinstance(decoded_middle_obj, Outer.Middle)
    assert isinstance(decoded_inner_obj, Outer.Middle.Inner)
    assert middle_obj.attribute == decoded_middle_obj.attribute
    assert inner_obj.attribute == decoded_inner_obj.attribute


def test_listlike():
    """List-like objects can roundtrip"""
    # https://github.com/jsonpickle/jsonpickle/issues/362
    ll = ListLike()
    ll.internal_list.append(1)
    roundtrip_ll = jsonpickle.decode(jsonpickle.encode(ll))
    assert len(roundtrip_ll.internal_list) == len(ll.internal_list)


def test_v1_decode():
    # TODO: Find a simple example that reproduces #364
    assert True


def test_depth_tracking(pickler):
    liszt = []
    pickler.flatten([liszt, liszt, liszt, liszt, liszt])
    assert pickler._depth == -1


def test_readonly_attrs():
    """Objects with readonly attributes can roundtrip"""
    safe_str = SafeString('test')
    pickled = jsonpickle.encode(safe_str, handle_readonly=True)
    pickled_json_dict = jsonpickle.backend.json.loads(pickled)
    # make sure it's giving concise output by erroring if it includes
    # a default method which is unnecessary
    assert 'join' not in pickled_json_dict
    unpickled = jsonpickle.decode(pickled)
    assert SafeString == unpickled.__class__
    assert safe_str == unpickled


def test_readonly_str_attrs():
    """Objects with readonly string attributes can roundtrip"""
    safe_str = SafeString('test')
    # We'll first try setting handle_readonly=True when encoding.
    encoded = jsonpickle.encode(safe_str, handle_readonly=True)
    actual = jsonpickle.decode(encoded, handle_readonly=True)
    assert safe_str == actual
    # Next we'll ensure that we can decode a payload that contains readonly attributes
    # by omitting the handle_readonly option when pickling.
    encoded = jsonpickle.encode(safe_str)
    actual = jsonpickle.decode(encoded, handle_readonly=True)
    assert safe_str == actual


class PicklableNamedTuple:
    """A namedtuple wrapper that uses ``__getnewargs__``

    Demonstrates the need for protocol 2 compatibility.
    This is contrived in its use of new but it demonstrates the issue.
    """

    def __new__(cls, propnames, vals):
        # it's necessary to use the correct class name for class resolution
        # classes that fake their own names may never be unpicklable
        ntuple = collections.namedtuple(cls.__name__, propnames)
        ntuple.__getnewargs__ = lambda self: (propnames, vals)
        instance = ntuple.__new__(ntuple, *vals)
        return instance


class PicklableNamedTupleEx:
    """A namedtuple wrapper that uses ``__getnewargs__`` and ``__getnewargs_ex__``

    Demonstrates the need for protocol 4 compatibility.
    This is contrived in its use of new but it demonstrates the issue.
    """

    def __getnewargs__(self):
        raise NotImplementedError("This class needs __getnewargs_ex__")

    def __new__(cls, newargs=__getnewargs__, **kwargs):
        # it's necessary to use the correct class name for class resolution
        # classes that fake their own names may never be unpicklable
        ntuple = collections.namedtuple(cls.__name__, sorted(kwargs.keys()))
        ntuple.__getnewargs_ex__ = lambda self: ((), kwargs)
        ntuple.__getnewargs__ = newargs
        instance = ntuple.__new__(ntuple, *[b for a, b in sorted(kwargs.items())])
        return instance


class PickleProtocol2Thing:
    """An object that implements ``__getnewargs__`` for pickle protocol v2"""

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
    """An object whose ``__getstate__`` returns a string"""

    def __new__(cls, *args):
        instance = super().__new__(cls)
        instance.newargs = args
        return instance

    def __getstate__(self):
        return 'I am magic'


class PickleProtocol2GetStateDict(PickleProtocol2Thing):
    """An object whose ``__getstate__`` returns a regular dict"""

    def __getstate__(self):
        return {'magic': True}


class PickleProtocol2GetStateNestedDict(PickleProtocol2Thing):
    """An object whose ``__getstate__`` returns a nested dict"""

    def __getstate__(self):
        return {'nested': {'magic': True}}


class PickleProtocol2GetStateSlots(PickleProtocol2Thing):
    """An object whose ``__getstate__`` return a tuple containing None and slot data"""

    def __getstate__(self):
        return (None, {'slotmagic': slotmagic})


class PickleProtocol2GetStateSlotsDict(PickleProtocol2Thing):
    """An object whose ``__getstate__`` return a tuple containing dict and slot data"""

    def __getstate__(self):
        return ({'dictmagic': dictmagic}, {'slotmagic': slotmagic})


class PickleProtocol2GetSetState(PickleProtocol2GetState):
    """An object whose ``__setstate__`` detects specific strings"""

    def __setstate__(self, state):
        """Contrived example, easy to test"""
        if state == 'I am magic':
            self.magic = True
        else:
            self.magic = False


class PickleProtocol2ChildThing:
    """Provides ``__getnewargs__`` for pickle protocol v2"""

    def __init__(self, child):
        self.child = child

    def __getnewargs__(self):
        return ([self.child],)


class PickleProtocol2ReduceString:
    """A reducible object that implements ``__reduce__`` only"""

    def __reduce__(self):
        return __name__ + '.slotmagic'


class PickleProtocol2ReduceExString:
    """A reducible object that implements both ``__reduce_ex__`` and ``__reduce__``"""

    def __reduce_ex__(self, n):
        return __name__ + '.slotmagic'

    def __reduce__(self):
        """This method is ignored"""
        assert False, 'Should not be here'


class PickleProtocol2ReduceTuple:
    """A reducible object that returns the PickleProtocol2ReduceTuple callable"""

    def __init__(self, argval, optional=None):
        self.argval = argval
        self.optional = optional

    def __reduce__(self):
        return (
            PickleProtocol2ReduceTuple,  # callable
            ('yam', 1),  # args
            None,  # state
            iter([]),  # listitems
            iter([]),  # dictitems
        )


class ReducibleIterator:
    """A reducible iterator"""

    def __next__(self):
        raise StopIteration()

    def __iter__(self):
        return self

    def __reduce__(self):
        return ReducibleIterator, ()


def protocol_2_reduce_tuple_func(*args):
    """Return a pickle protocol v2 reducer"""
    return PickleProtocol2ReduceTupleFunc(*args)


class PickleProtocol2ReduceTupleFunc:
    """A reducible object with args and optional state"""

    def __init__(self, argval, optional=None):
        self.argval = argval
        self.optional = optional

    def __reduce__(self):
        return (
            protocol_2_reduce_tuple_func,  # callable
            ('yam', 1),  # args
            None,  # state
            iter([]),  # listitems
            iter([]),  # dictitems
        )


def __newobj__(lol, fail):
    """newobj is special-cased so that it is not actually called"""


class PickleProtocol2ReduceNewobj(PickleProtocol2ReduceTupleFunc):
    """Reducible object that return the ``__newobj__`` callable and args"""

    def __new__(cls, *args):
        inst = super(cls, cls).__new__(cls)
        inst.newargs = args
        return inst

    def __reduce__(self):
        return (
            __newobj__,  # callable
            (PickleProtocol2ReduceNewobj, 'yam', 1),  # args
            None,  # state
            iter([]),  # listitems
            iter([]),  # dictitems
        )


class PickleProtocol2ReduceTupleState(PickleProtocol2ReduceTuple):
    """Reducible object with args and state"""

    def __reduce__(self):
        return (
            PickleProtocol2ReduceTuple,  # callable
            ('yam', 1),  # args
            {'foo': 1},  # state
            iter([]),  # listitems
            iter([]),  # dictitems
        )


class PickleProtocol2ReduceTupleSetState(PickleProtocol2ReduceTuple):
    """Reducible object with args and state that implements ``__setstate__``"""

    def __reduce__(self):
        return (
            type(self),  # callable
            ('yam', 1),  # args
            {'foo': 1},  # state
            iter([]),  # listitems
            iter([]),  # dictitems
        )

    def __setstate__(self, state):
        self.bar = state['foo']


class PickleProtocol2ReduceTupleStateSlots:
    """Reducible object with tuple ``__slots__``"""

    __slots__ = ('argval', 'optional', 'foo')

    def __init__(self, argval, optional=None):
        self.argval = argval
        self.optional = optional

    def __reduce__(self):
        return (
            PickleProtocol2ReduceTuple,  # callable
            ('yam', 1),  # args
            {'foo': 1},  # state
            iter([]),  # listitems
            iter([]),  # dictitems
        )


class PickleProtocol2ReduceListitemsAppend:
    """A reducible object that only implements append()"""

    def __init__(self):
        self.inner = []

    def append(self, item):
        self.inner.append(item)

    def __reduce__(self):
        return (
            PickleProtocol2ReduceListitemsAppend,  # callable
            (),  # args
            {},  # state
            iter(['foo', 'bar']),  # listitems
            iter([]),  # dictitems
        )


class PickleProtocol2ReduceListitemsExtend:
    """A reducible object that only implements extend()"""

    def __init__(self):
        self.inner = []

    def extend(self, items):
        self.inner.exend(items)

    def __reduce__(self):
        return (
            PickleProtocol2ReduceListitemsAppend,  # callable
            (),  # args
            {},  # state
            iter(['foo', 'bar']),  # listitems
            iter([]),  # dictitems
        )


class PickleProtocol2ReduceDictitems:
    """A reducible object with dictitems that implements ``__setitem__``"""

    def __init__(self):
        self.inner = {}

    def __setitem__(self, k, v):
        return self.inner.__setitem__(k, v)

    def __reduce__(self):
        return (
            PickleProtocol2ReduceDictitems,  # callable
            (),  # args
            {},  # state
            [],  # listitems
            iter(zip(['foo', 'bar'], ['foo', 'bar'])),  # dictitems
        )


def test_pickle_newargs_ex():
    """
    Ensure we can pickle and unpickle an object whose class needs arguments
    to __new__ and get back the same type
    """
    instance = PicklableNamedTupleEx(**{'a': 'b', 'n': 2})
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert instance == decoded


def test_validate_reconstruct_by_newargs_ex():
    """
    Ensure that the exemplar tuple's __getnewargs_ex__ works
    This is necessary to know whether the breakage exists
    in jsonpickle or not
    """
    instance = PicklableNamedTupleEx(**{'a': 'b', 'n': 2})
    args, kwargs = instance.__getnewargs_ex__()
    newinstance = PicklableNamedTupleEx.__new__(PicklableNamedTupleEx, *args, **kwargs)
    assert instance == newinstance


def test_references_named_tuple():
    """Object references and identities are preserved inside a named tuple"""
    shared = Thing('shared')
    instance = PicklableNamedTupleEx(**{'a': shared, 'n': shared})
    child = Thing('child')
    shared.child = child
    child.child = instance
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded[0] == decoded[1]
    assert decoded[0] is decoded[1]
    assert decoded.a is decoded.n
    assert decoded.a.name == 'shared'
    assert decoded.a.child.name == 'child'
    assert decoded.a.child.child is decoded
    assert decoded.n.child.child is decoded
    assert decoded.a.child is decoded.n.child
    assert decoded.__class__.__name__ == PicklableNamedTupleEx.__name__
    # TODO the class itself looks just like the real class, but it's
    # actually a reconstruction; PicklableNamedTupleEx is not type(decoded).
    assert decoded.__class__ is not PicklableNamedTupleEx


def test_reduce_complex_num():
    """Test complex/imaginary numbers"""
    instance = 5j
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded == instance


def test_reduce_complex_zero():
    """Test complex/imaginary zero"""
    instance = 0j
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded == instance


def test_reduce_dictitems():
    """Test reduce with dictitems set (as a generator)"""
    instance = PickleProtocol2ReduceDictitems()
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.inner == {'foo': 'foo', 'bar': 'bar'}


def test_reduce_listitems_extend():
    """Test reduce with listitems set (as a generator), yielding single items"""
    instance = PickleProtocol2ReduceListitemsExtend()
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.inner == ['foo', 'bar']


def test_reduce_listitems_append():
    """Test reduce with listitems set (as a generator), yielding single items"""
    instance = PickleProtocol2ReduceListitemsAppend()
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.inner == ['foo', 'bar']


def test_reduce_state_setstate():
    """Objects with ``__setstate__`` and optional state arguments can roundtrip"""
    instance = PickleProtocol2ReduceTupleSetState(5)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.argval == 'yam'
    assert decoded.optional == 1
    assert decoded.bar == 1
    assert not hasattr(decoded, 'foo')


def test_reduce_state_no_dict():
    """Objects with state but without `__dict__`` and ``__setstate__`` can roundtrip"""
    instance = PickleProtocol2ReduceTupleStateSlots(5)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.argval == 'yam'
    assert decoded.optional == 1
    assert decoded.foo == 1


def test_reduce_state_dict():
    """Reduce an object with __dict__ and no __setstate__"""
    instance = PickleProtocol2ReduceTupleState(5)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.argval == 'yam'
    assert decoded.optional == 1
    assert decoded.foo == 1


def test_reduce_basic():
    """Reduce an object with callable and args only"""
    instance = PickleProtocol2ReduceTuple(5)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.argval == 'yam'
    assert decoded.optional == 1


def test_reduce_basic_func():
    """Reduce an object with args and a module-level callable"""
    instance = PickleProtocol2ReduceTupleFunc(5)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.argval == 'yam'
    assert decoded.optional == 1


def test_reduce_newobj():
    """Reduce an object with a callable __newobj__"""
    instance = PickleProtocol2ReduceNewobj(5)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.newargs == ('yam', 1)


def test_reduce_iter():
    """Iterators with ``__reduce__`` can roundtrip"""
    instance = iter('123')
    assert util.is_iterator(instance)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert next(decoded) == '1'
    assert next(decoded) == '2'
    assert next(decoded) == '3'


def test_reduce_iterable():
    """Reducible objects that are iterable should also pickle"""
    instance = ReducibleIterator()
    assert util.is_iterator(instance)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert isinstance(decoded, ReducibleIterator)


def test_reduce_string():
    """Handle redirection to another object when ``__reduce__`` returns a string"""
    instance = PickleProtocol2ReduceString()
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded == slotmagic


def test_reduce_ex_string():
    """Handle redirection to another object when ``__reduce_ex__`` returns a string

    Ensure that ``__reduce_ex__`` has higher priority than ``__reduce__``.
    """
    instance = PickleProtocol2ReduceExString()
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded == slotmagic


def test_pickle_newargs():
    """Objects whose class needs arguments to ``__new__`` can roundtrip"""
    instance = PicklableNamedTuple(('a', 'b'), (1, 2))
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert instance == decoded


def test_validate_reconstruct_by_newargs():
    """Validate PicklableNamedTuple's Python semantics

    Ensure that the exemplar tuple's __getnewargs__ works This is necessary to
    know whether potential breakage exists in jsonpickle or in the test class.
    """
    instance = PicklableNamedTuple(('a', 'b'), (1, 2))
    newinstance = PicklableNamedTuple.__new__(
        PicklableNamedTuple, *(instance.__getnewargs__())
    )
    assert instance == newinstance


def test_getnewargs_priority():
    """newargs must be used before py/state when decoding

    As per PEP 307, classes are not supposed to implement
    all three magic methods.
    """
    instance = PickleProtocol2GetState('whatevs')
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.newargs == ('whatevs',)


def test_restore_dict_state():
    """getstate without setstate can roundtrip

    If ``__getstate__`` returns a dict and there is no custom ``__setstate__`` then the
    dict is used as a source of variables to restore.
    """
    instance = PickleProtocol2GetStateDict('whatevs')
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.magic


def test_restore_nested_dict_state_with_references_preserved():
    """Nested dicts in "py/state" retain references and are not duplicated"""
    instance1 = PickleProtocol2GetStateNestedDict('whatevs')
    instance2 = PickleProtocol2GetStateNestedDict('different')
    encoded = jsonpickle.encode([instance1, instance1, instance2, instance2])
    decoded = jsonpickle.decode(encoded)
    assert decoded[0].nested['magic']
    assert decoded[1] is decoded[0]
    assert decoded[2].nested['magic']
    assert decoded[3] is decoded[2]


def test_restore_slots_state():
    """Serialize objects with __slots__

    Ensure that if getstate returns a 2-tuple with a dict in the second
    position, and there is no custom __setstate__, the dict is used as a
    source of variables to restore.
    """
    instance = PickleProtocol2GetStateSlots('whatevs')
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.slotmagic.__dict__ == slotmagic.__dict__
    assert decoded.slotmagic == slotmagic


def test_restore_slots_dict_state():
    """getstate with dicts in both positions can roundtrip

    If ``__getstate__`` returns a 2-tuple with a dict in both positions,
    and there is no custom ``__setstate__``, the dicts are used as a source of
    variables to restore.
    """
    instance = PickleProtocol2GetStateSlotsDict('whatevs')
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert PickleProtocol2Thing('slotmagic') == PickleProtocol2Thing('slotmagic')
    assert decoded.slotmagic.__dict__ == slotmagic.__dict__
    assert decoded.slotmagic == slotmagic
    assert decoded.dictmagic == dictmagic


def test_setstate():
    """Ensure output of getstate is passed to setstate"""
    instance = PickleProtocol2GetSetState('whatevs')
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded.magic


def test_handles_nested_objects():
    """Nested objects can roundtrip"""
    child = PickleProtocol2Thing(None)
    instance = PickleProtocol2Thing(child, child)
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert PickleProtocol2Thing == decoded.__class__
    assert PickleProtocol2Thing == decoded.args[0].__class__
    assert PickleProtocol2Thing == decoded.args[1].__class__
    assert decoded.args[0] is decoded.args[1]


def test_cyclical_objects():
    """Cyclical objects with inner tuples and lists can roundtrip"""
    _test_cyclical_objects(True)
    _test_cyclical_objects(False)


def _test_cyclical_objects(use_tuple):
    child = Capture(None)
    instance = Capture(child, child)
    # create a cycle
    if use_tuple:
        child.args = (instance,)
    else:
        child.args = [instance]
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    # Ensure the right objects were constructed
    assert Capture == decoded.__class__
    assert Capture == decoded.args[0].__class__
    assert Capture == decoded.args[1].__class__
    assert Capture == decoded.args[0].args[0].__class__
    assert Capture == decoded.args[1].args[0].__class__
    # It's turtles all the way down
    assert (
        Capture
        == decoded.args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .args[0]
        .__class__
    )
    # Ensure that references are properly constructed
    assert decoded.args[0] is decoded.args[1]
    assert decoded is decoded.args[0].args[0]
    assert decoded is decoded.args[1].args[0]
    assert decoded.args[0] is decoded.args[0].args[0].args[0]
    assert decoded.args[0] is decoded.args[1].args[0].args[0]


def test_handles_cyclical_objects_in_lists():
    """Cyclical objects in lists can roundtrip"""
    child = PickleProtocol2ChildThing(None)
    instance = PickleProtocol2ChildThing([child, child])
    child.child = instance  # create a cycle
    encoded = jsonpickle.encode(instance)
    decoded = jsonpickle.decode(encoded)
    assert decoded is decoded.child[0].child
    assert decoded is decoded.child[1].child


def test_cyclical_objects_unpickleable():
    """Cyclical objects can roundtrip with their references retained"""
    _test_cyclical_objects_unpickleable(True)
    _test_cyclical_objects_unpickleable(False)


def _test_cyclical_objects_unpickleable(use_tuple):
    child = Capture(None)
    instance = Capture(child, child)
    # create a cycle
    if use_tuple:
        child.args = (instance,)
    else:
        child.args = [instance]
    encoded = jsonpickle.encode(instance, unpicklable=False)
    decoded = jsonpickle.decode(encoded)
    assert isinstance(decoded, dict)
    assert 'args' in decoded
    assert 'kwargs' in decoded
    # Tuple is lost via json
    args = decoded['args']
    assert isinstance(args, list)
    # Get the children
    assert len(args) == 2
    decoded_child0 = args[0]
    decoded_child1 = args[1]
    # Circular references become None
    assert decoded_child0 == {'args': [None], 'kwargs': {}}
    assert decoded_child1 == {'args': [None], 'kwargs': {}}


def test_dict_references_are_preserved():
    """References and object identities are preserved in dicts"""
    data = {}
    actual = jsonpickle.decode(jsonpickle.encode([data, data]))
    assert isinstance(actual, list)
    assert isinstance(actual[0], dict)
    assert isinstance(actual[1], dict)
    assert actual[0] is actual[1]


def test_repeat_objects_are_expanded():
    """All objects are present in the json output"""
    # When references are disabled we should create expanded copies
    # of any object that appears more than once in the object stream.
    alice = Thing('alice')
    bob = Thing('bob')
    alice.child = bob
    car = Thing('car')
    car.driver = alice
    car.owner = alice
    car.passengers = [alice, bob]
    pickler = jsonpickle.Pickler(make_refs=False)
    flattened = pickler.flatten(car)
    assert flattened['name'] == 'car'
    assert flattened['driver']['name'] == 'alice'
    assert flattened['owner']['name'] == 'alice'
    assert flattened['passengers'][0]['name'] == 'alice'
    assert flattened['passengers'][1]['name'] == 'bob'
    assert flattened['driver']['child']['name'] == 'bob'
    assert flattened['passengers'][0]['child']['name'] == 'bob'
