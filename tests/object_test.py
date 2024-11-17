import array
import collections
import datetime
import decimal
import enum
import queue
import re
import threading

import pytest
from helper import SkippableTest

import jsonpickle
from jsonpickle import handlers, tags, util


@pytest.fixture
def pickler():
    """Returns a default-constructed pickler"""
    return jsonpickle.pickler.Pickler()


@pytest.fixture
def unpickler():
    """Returns a default-constructed unpickler"""
    return jsonpickle.unpickler.Unpickler()


class Thing:
    def __init__(self, name):
        self.name = name
        self.child = None


class DictSubclass(dict):
    name = 'Test'


class ListSubclass(list):
    pass


class BrokenReprThing(Thing):
    def __repr__(self):
        raise Exception('%s has a broken repr' % self.name)

    def __str__(self):
        return '<BrokenReprThing "%s">' % self.name


class GetstateDict(dict):
    def __init__(self, name, **kwargs):
        dict.__init__(self, **kwargs)
        self.name = name
        self.active = False

    def __getstate__(self):
        return (self.name, dict(self.items()))

    def __setstate__(self, state):
        self.name, vals = state
        self.update(vals)
        self.active = True


class GetstateOnly:
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def __getstate__(self):
        return [self.a, self.b]


class GetstateReturnsList:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getstate__(self):
        return [self.x, self.y]

    def __setstate__(self, state):
        self.x, self.y = state[0], state[1]


class GetstateRecursesInfintely:
    def __getstate__(self):
        return GetstateRecursesInfintely()


class ListSubclassWithInit(list):
    def __init__(self, attr):
        self.attr = attr
        super().__init__()


NamedTuple = collections.namedtuple('NamedTuple', 'a, b, c')


class ObjWithJsonPickleRepr:
    def __init__(self):
        self.data = {'a': self}

    def __repr__(self):
        return jsonpickle.encode(self)


class OldStyleClass:
    pass


class SetSubclass(set):
    pass


def func(x):
    return x


class ThingWithFunctionRefs:
    def __init__(self):
        self.fn = func


class ThingWithQueue:
    def __init__(self):
        self.child_1 = queue.Queue()
        self.child_2 = queue.Queue()
        self.childref_1 = self.child_1
        self.childref_2 = self.child_2


class ThingWithSlots:
    __slots__ = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class ThingWithInheritedSlots(ThingWithSlots):
    __slots__ = ('c',)

    def __init__(self, a, b, c):
        ThingWithSlots.__init__(self, a, b)
        self.c = c


class ThingWithIterableSlots:
    __slots__ = iter('ab')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class ThingWithStringSlots:
    __slots__ = 'ab'

    def __init__(self, a, b):
        self.ab = a + b


class ThingWithSelfAsDefaultFactory(collections.defaultdict):
    """defaultdict subclass that uses itself as its default factory"""

    def __init__(self):
        self.default_factory = self

    def __call__(self):
        return self.__class__()


class ThingWithClassAsDefaultFactory(collections.defaultdict):
    """defaultdict subclass that uses its class as its default factory"""

    def __init__(self):
        self.default_factory = self.__class__

    def __call__(self):
        return self.__class__()


class ThingWithDefaultFactoryRegistry:
    """Counts calls to ThingWithDefaultFactory.__init__()"""

    count = 0  # Incremented by ThingWithDefaultFactory().


class ThingWithDefaultFactory:
    """Class that ensures that provides a default factory"""

    def __new__(cls, *args, **kwargs):
        """Ensure that jsonpickle uses our constructor during initialization"""
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        self.default_factory = self.__class__
        # Keep track of how many times this constructor has been run.
        ThingWithDefaultFactoryRegistry.count += 1


class IntEnumTest(enum.IntEnum):
    X = 1
    Y = 2


class StringEnumTest(enum.Enum):
    A = 'a'
    B = 'b'


class SubEnum(enum.Enum):
    a = 1
    b = 2


class EnumClass:
    def __init__(self):
        self.enum_a = SubEnum.a
        self.enum_b = SubEnum.b


class MessageTypes(enum.Enum):
    STATUS = ('STATUS',)
    CONTROL = 'CONTROL'


class MessageStatus(enum.Enum):
    OK = ('OK',)
    ERROR = 'ERROR'


class MessageCommands(enum.Enum):
    STATUS_ALL = 'STATUS_ALL'


class Message:
    def __init__(self, message_type, command, status=None, body=None):
        self.message_type = MessageTypes(message_type)
        if command:
            self.command = MessageCommands(command)
        if status:
            self.status = MessageStatus(status)
        if body:
            self.body = body


class ThingWithTimedeltaAttribute:
    def __init__(self, offset):
        self.offset = datetime.timedelta(offset)

    def __getinitargs__(self):
        return (self.offset,)


class FailSafeTestCase(SkippableTest):
    class BadClass:
        def __getstate__(self):
            raise ValueError('Intentional error')

    good = 'good'

    to_pickle = [BadClass(), good]

    def test_no_error(self):
        encoded = jsonpickle.encode(self.to_pickle, fail_safe=lambda e: None)
        decoded = jsonpickle.decode(encoded)
        assert decoded[0] is None
        assert decoded[1] == 'good'

    def test_error_recorded(self):
        exceptions = []

        def recorder(exception):
            exceptions.append(exception)

        jsonpickle.encode(self.to_pickle, fail_safe=recorder)
        assert len(exceptions) == 1
        assert isinstance(exceptions[0], Exception)

    def test_custom_err_msg(self):
        CUSTOM_ERR_MSG = 'custom err msg'
        encoded = jsonpickle.encode(self.to_pickle, fail_safe=lambda e: CUSTOM_ERR_MSG)
        decoded = jsonpickle.decode(encoded)
        assert decoded[0] == CUSTOM_ERR_MSG


class IntKeysObject:
    def __init__(self):
        self.data = {0: 0}

    def __getstate__(self):
        return self.__dict__


class ExceptionWithArguments(Exception):
    def __init__(self, value):
        super().__init__('test')
        self.value = value


class ThingWithExclusion:
    _jsonpickle_exclude = ["foo"]

    def __init__(self, a):
        self.foo = 1
        self.bar = a


class ThingWithExcludeSubclass:
    def __init__(self, foo):
        self.foo = foo
        self.thing = ThingWithExclusion(3)


def test_list_subclass(pickler, unpickler):
    obj = ListSubclass()
    obj.extend([1, 2, 3])
    flattened = pickler.flatten(obj)
    assert tags.OBJECT in flattened
    assert tags.SEQ in flattened
    assert len(flattened[tags.SEQ]) == 3
    for v in obj:
        assert v in flattened[tags.SEQ]
    restored = unpickler.restore(flattened)
    assert type(restored) is ListSubclass
    assert restored == obj


def test_list_subclass_with_init(pickler, unpickler):
    obj = ListSubclassWithInit('foo')
    assert obj.attr == 'foo'
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert type(inflated) is ListSubclassWithInit


def test_list_subclass_with_data(pickler, unpickler):
    obj = ListSubclass()
    obj.extend([1, 2, 3])
    data = SetSubclass([1, 2, 3])
    obj.data = data
    flattened = pickler.flatten(obj)
    restored = unpickler.restore(flattened)
    assert restored == obj
    assert type(restored.data) is SetSubclass
    assert restored.data == data


def test_set_subclass(pickler, unpickler):
    obj = SetSubclass([1, 2, 3])
    flattened = pickler.flatten(obj)
    assert tags.OBJECT in flattened
    assert tags.SEQ in flattened
    assert len(flattened[tags.SEQ]) == 3
    for v in obj:
        assert v in flattened[tags.SEQ]
    restored = unpickler.restore(flattened)
    assert type(restored) is SetSubclass
    assert restored == obj


def test_set_subclass_with_data(pickler, unpickler):
    obj = SetSubclass([1, 2, 3])
    data = ListSubclass()
    data.extend([1, 2, 3])
    obj.data = data
    flattened = pickler.flatten(obj)
    restored = unpickler.restore(flattened)
    assert restored.data.__class__ == ListSubclass
    assert restored.data == data


def test_decimal(pickler, unpickler):
    obj = decimal.Decimal('0.5')
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert isinstance(inflated, decimal.Decimal)


def test_oldstyleclass(pickler, unpickler):
    obj = OldStyleClass()
    obj.value = 1234
    flattened = pickler.flatten(obj)
    assert flattened['value'] == 1234
    inflated = unpickler.restore(flattened)
    assert inflated.value == 1234


def test_dictsubclass(pickler, unpickler):
    obj = DictSubclass()
    obj['key1'] = 1
    expect = {
        tags.OBJECT: 'object_test.DictSubclass',
        'key1': 1,
        '__dict__': {},
    }
    flattened = pickler.flatten(obj)
    assert expect == flattened
    inflated = unpickler.restore(flattened)
    assert type(inflated) is DictSubclass
    assert inflated['key1'] == 1
    assert inflated.name == 'Test'


def test_dictsubclass_notunpickable(pickler, unpickler):
    pickler.unpicklable = False
    obj = DictSubclass()
    obj['key1'] = 1
    flattened = pickler.flatten(obj)
    assert flattened['key1'] == 1
    assert tags.OBJECT not in flattened
    inflated = unpickler.restore(flattened)
    assert inflated['key1'] == 1


def test_getstate_dict_subclass_structure(pickler):
    obj = GetstateDict('test')
    obj['key1'] = 1
    flattened = pickler.flatten(obj)
    assert tags.OBJECT in flattened
    assert 'object_test.GetstateDict' == flattened[tags.OBJECT]
    assert tags.STATE in flattened
    assert tags.TUPLE in flattened[tags.STATE]
    assert ['test' == {'key1': 1}], flattened[tags.STATE][tags.TUPLE]


def test_getstate_dict_subclass_roundtrip_simple(pickler, unpickler):
    obj = GetstateDict('test')
    obj['key1'] = 1
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert inflated['key1'] == 1
    assert inflated.name == 'test'


def test_getstate_dict_subclass_roundtrip_cyclical(pickler, unpickler):
    obj = GetstateDict('test')
    obj['key1'] = 1
    # The "name" field of obj2 points to obj (reference)
    obj2 = GetstateDict(obj)
    # The "obj2" key in obj points to obj2 (cyclical reference)
    obj['obj2'] = obj2
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    # The dict must be preserved
    assert inflated['key1'] == 1
    # __getstate__/__setstate__ must have been run
    assert inflated.name == 'test'
    assert inflated.active is True
    assert inflated['obj2'].active is True
    # The reference must be preserved
    assert inflated is inflated['obj2'].name


def test_getstate_list_simple(pickler, unpickler):
    obj = GetstateReturnsList(1, 2)
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert inflated.x == 1
    assert inflated.y == 2


def test_getstate_list_inside_list(pickler, unpickler):
    obj1 = GetstateReturnsList(1, 2)
    obj2 = GetstateReturnsList(3, 4)
    obj = [obj1, obj2]
    flattened = pickler.flatten(obj)
    inflated = unpickler.restore(flattened)
    assert inflated[0].x == 1
    assert inflated[0].y == 2
    assert inflated[1].x == 3
    assert inflated[1].y == 4


def test_getstate_with_getstate_only(pickler, unpickler):
    obj = GetstateOnly()
    a = obj.a = 'this object implements'
    b = obj.b = '__getstate__ but not __setstate__'
    expect = [a, b]
    flat = pickler.flatten(obj)
    actual = flat[tags.STATE]
    assert expect == actual
    restored = unpickler.restore(flat)
    assert expect == restored


def test_thing_with_queue(pickler, unpickler):
    obj = ThingWithQueue()
    flattened = pickler.flatten(obj)
    restored = unpickler.restore(flattened)
    assert type(restored.child_1) is type(queue.Queue())
    assert type(restored.child_2) is type(queue.Queue())
    # Check references
    assert restored.child_1 is restored.childref_1
    assert restored.child_2 is restored.childref_2


def test_thing_with_func(pickler, unpickler):
    obj = ThingWithFunctionRefs()
    obj.ref = obj
    flattened = pickler.flatten(obj)
    restored = unpickler.restore(flattened)
    assert restored.fn is obj.fn
    expect = 'success'
    actual1 = restored.fn(expect)
    assert expect == actual1
    assert restored is restored.ref


def test_thing_with_compiled_regex(pickler, unpickler):
    rgx = re.compile(r'(.*)(cat)')
    obj = Thing(rgx)
    flattened = pickler.flatten(obj)
    restored = unpickler.restore(flattened)
    match = restored.name.match('fatcat')
    assert 'fat' == match.group(1)
    assert 'cat' == match.group(2)


def test_base_object_roundrip(pickler, unpickler):
    roundtrip = unpickler.restore(pickler.flatten(object()))
    assert type(roundtrip) is object


def test_enum34(pickler, unpickler):
    restore = unpickler.restore
    flatten = pickler.flatten

    def roundtrip(obj):
        return restore(flatten(obj))

    assert roundtrip(IntEnumTest.X) is IntEnumTest.X
    assert roundtrip(IntEnumTest) is IntEnumTest
    assert roundtrip(StringEnumTest.A) is StringEnumTest.A
    assert roundtrip(StringEnumTest) is StringEnumTest


def test_bytes_unicode(pickler, unpickler):
    b1 = b'foo'
    b2 = b'foo\xff'
    u1 = 'foo'
    # unicode strings get encoded/decoded as is
    encoded = pickler.flatten(u1)
    assert encoded == u1
    assert isinstance(encoded, str)
    decoded = unpickler.restore(encoded)
    assert decoded == u1
    assert isinstance(decoded, str)
    # bytestrings are wrapped in py 3
    encoded = pickler.flatten(b1)
    assert encoded != u1
    encoded_ustr = util.b64encode(b'foo')
    assert {tags.B64: encoded_ustr} == encoded
    assert isinstance(encoded[tags.B64], str)
    decoded = unpickler.restore(encoded)
    assert decoded == b1
    assert isinstance(decoded, bytes)
    # bytestrings that we can't decode to UTF-8 will always be wrapped
    encoded = pickler.flatten(b2)
    assert encoded != b2
    encoded_ustr = util.b64encode(b'foo\xff')
    assert {tags.B64: encoded_ustr} == encoded
    assert isinstance(encoded[tags.B64], str)
    decoded = unpickler.restore(encoded)
    assert decoded == b2
    assert isinstance(decoded, bytes)


def test_nested_objects(pickler, unpickler):
    obj = ThingWithTimedeltaAttribute(99)
    flattened = pickler.flatten(obj)
    restored = unpickler.restore(flattened)
    assert restored.offset == datetime.timedelta(99)


def test_threading_lock(pickler, unpickler):
    obj = Thing('lock')
    obj.lock = threading.Lock()
    lock_class = obj.lock.__class__
    # Roundtrip and make sure we get a lock object.
    json = pickler.flatten(obj)
    clone = unpickler.restore(json)
    assert isinstance(clone.lock, lock_class)
    assert not clone.lock.locked()
    # Serializing a locked lock should create a locked clone.
    assert obj.lock.acquire()
    json = pickler.flatten(obj)
    obj.lock.release()
    # Restore the locked lock state.
    clone = unpickler.restore(json)
    assert clone.lock.locked()
    clone.lock.release()


def _test_array_roundtrip(pickler, unpickler, obj):
    """Roundtrip an array and test invariants"""
    json = pickler.flatten(obj)
    clone = unpickler.restore(json)
    assert isinstance(clone, array.array)
    assert obj.typecode == clone.typecode
    assert len(obj) == len(clone)
    for j, k in zip(obj, clone):
        assert j == k
    assert obj == clone


def test_array_handler_numeric(pickler, unpickler):
    """Test numeric array.array typecodes that work in Python2+3"""
    typecodes = ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'f', 'd')
    for typecode in typecodes:
        obj = array.array(typecode, (1, 2, 3))
        _test_array_roundtrip(pickler, unpickler, obj)


def test_exceptions_with_arguments(pickler, unpickler):
    """Ensure that we can roundtrip Exceptions that take arguments"""
    obj = ExceptionWithArguments('example')
    json = pickler.flatten(obj)
    clone = unpickler.restore(json)
    assert obj.value == clone.value
    assert obj.args == clone.args


def test_getstate_does_not_recurse_infinitely():
    """Serialize an object with a __getstate__ that recurses forever"""
    obj = GetstateRecursesInfintely()
    pickler = jsonpickle.pickler.Pickler(max_depth=5)
    actual = pickler.flatten(obj)
    assert isinstance(actual, dict)
    assert tags.OBJECT in actual


def test_defaultdict_roundtrip():
    """Make sure we can handle collections.defaultdict(list)"""
    # setup
    defaultdict = collections.defaultdict
    defdict = defaultdict(list)
    defdict['a'] = 1
    defdict['b'].append(2)
    defdict['c'] = defaultdict(dict)
    # jsonpickle work your magic
    encoded = jsonpickle.encode(defdict)
    newdefdict = jsonpickle.decode(encoded)
    # jsonpickle never fails
    assert newdefdict['a'] == 1
    assert newdefdict['b'] == [2]
    assert type(newdefdict['c']) is defaultdict
    assert defdict.default_factory == list
    assert newdefdict.default_factory == list


def test_default_factory_initialization():
    """Ensure that ThingWithDefaultFactory()'s constructor is called"""
    thing = ThingWithDefaultFactory()
    assert ThingWithDefaultFactoryRegistry.count == 1

    thing.is_awesome = True
    encoded = jsonpickle.encode(thing)
    new_thing = jsonpickle.decode(encoded)

    assert ThingWithDefaultFactoryRegistry.count == 2
    assert new_thing.is_awesome


def test_defaultdict_roundtrip_simple_lambda():
    """Make sure we can handle defaultdict(lambda: defaultdict(int))"""
    # setup a sparse collections.defaultdict with simple lambdas
    defaultdict = collections.defaultdict
    defdict = defaultdict(lambda: defaultdict(int))
    defdict[0] = 'zero'
    defdict[1] = defaultdict(lambda: defaultdict(dict))
    defdict[1][0] = 'zero'
    # roundtrip
    encoded = jsonpickle.encode(defdict, keys=True)
    newdefdict = jsonpickle.decode(encoded, keys=True)
    assert newdefdict[0] == 'zero'
    assert type(newdefdict[1]) is defaultdict
    assert newdefdict[1][0] == 'zero'
    assert newdefdict[1][1] == {}  # inner defaultdict
    assert newdefdict[2][0] == 0  # outer defaultdict
    assert type(newdefdict[3]) is defaultdict
    # outer-most defaultdict
    assert newdefdict[3].default_factory == int


def test_defaultdict_roundtrip_simple_lambda2():
    """Serialize a defaultdict that contains a lambda"""
    defaultdict = collections.defaultdict
    payload = {'a': defaultdict(lambda: 0)}
    defdict = defaultdict(lambda: 0, payload)
    # roundtrip
    encoded = jsonpickle.encode(defdict, keys=True)
    decoded = jsonpickle.decode(encoded, keys=True)
    assert type(decoded) is defaultdict
    assert type(decoded['a']) is defaultdict


def test_defaultdict_and_things_roundtrip_simple_lambda():
    """Serialize a default dict that contains a lambda and objects"""
    thing = Thing('a')
    defaultdict = collections.defaultdict
    defdict = defaultdict(lambda: 0)
    obj = [defdict, thing, thing]
    # roundtrip
    encoded = jsonpickle.encode(obj, keys=True)
    decoded = jsonpickle.decode(encoded, keys=True)
    assert decoded[0].default_factory() == 0
    assert decoded[1] is decoded[2]


def _test_defaultdict_tree(tree, cls):
    tree['A']['B'] = 1
    tree['A']['C'] = 2
    # roundtrip
    encoded = jsonpickle.encode(tree)
    newtree = jsonpickle.decode(encoded)
    # make sure we didn't lose anything
    assert type(newtree) is cls
    assert type(newtree['A']) is cls
    assert newtree['A']['B'] == 1
    assert newtree['A']['C'] == 2
    # ensure that the resulting default_factory is callable and creates
    # a new instance of cls.
    assert type(newtree['A'].default_factory()) == cls
    # we've never seen 'D' before so the reconstructed defaultdict tree
    # should create an instance of cls.
    assert type(newtree['A']['D']) is cls
    # ensure that proxies do not escape into user code
    assert type(newtree.default_factory) is not jsonpickle.unpickler._Proxy
    assert type(newtree['A'].default_factory) is not jsonpickle.unpickler._Proxy
    assert type(newtree['A']['Z'].default_factory) is not jsonpickle.unpickler._Proxy
    return newtree


def test_defaultdict_subclass_with_self_as_default_factory():
    """Serialize a defaultdict subclass with self as its default factory"""
    cls = ThingWithSelfAsDefaultFactory
    tree = cls()
    newtree = _test_defaultdict_tree(tree, cls)
    assert type(newtree['A'].default_factory) is cls
    assert newtree.default_factory is newtree
    assert newtree['A'].default_factory is newtree['A']
    assert newtree['Z'].default_factory is newtree['Z']


def test_defaultdict_subclass_with_class_as_default_factory():
    """Serialize a defaultdict with a class as its default factory"""
    cls = ThingWithClassAsDefaultFactory
    tree = cls()
    newtree = _test_defaultdict_tree(tree, cls)
    assert newtree.default_factory is cls
    assert newtree['A'].default_factory is cls
    assert newtree['Z'].default_factory is cls


def test_posix_stat_result():
    """Serialize a posix.stat() result"""
    try:
        import posix
    except ImportError:
        return
    expect = posix.stat(__file__)
    encoded = jsonpickle.encode(expect)
    actual = jsonpickle.decode(encoded)
    assert expect == actual


def test_repr_using_jsonpickle():
    """Serialize an object that uses jsonpickle in its __repr__ definition"""
    thing = ObjWithJsonPickleRepr()
    thing.child = ObjWithJsonPickleRepr()
    thing.child.parent = thing
    encoded = jsonpickle.encode(thing)
    decoded = jsonpickle.decode(encoded)
    assert id(decoded) == id(decoded.child.parent)


def test_broken_repr_dict_key():
    """Tests that we can pickle dictionaries with keys that have
    broken __repr__ implementations.
    """
    br = BrokenReprThing('test')
    obj = {br: True}
    pickler = jsonpickle.pickler.Pickler()
    flattened = pickler.flatten(obj)
    assert '<BrokenReprThing "test">' in flattened
    assert flattened['<BrokenReprThing "test">']


def test_ordered_dict_python3():
    """Ensure that we preserve dict order on python3"""
    # Python3.6+ preserves dict order.
    obj = {'z': 'Z', 'x': 'X', 'y': 'Y'}
    clone = jsonpickle.decode(jsonpickle.encode(obj))
    expect = ['z', 'x', 'y']
    actual = list(clone.keys())
    assert expect == actual


def test_ordered_dict():
    """Serialize an OrderedDict"""
    d = collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)])
    encoded = jsonpickle.encode(d)
    decoded = jsonpickle.decode(encoded)
    assert d == decoded


def test_ordered_dict_unpicklable():
    """Serialize an OrderedDict with unpicklable=False"""
    d = collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)])
    encoded = jsonpickle.encode(d, unpicklable=False)
    decoded = jsonpickle.decode(encoded)
    assert d == decoded


def test_ordered_dict_reduces():
    """Ensure that OrderedDict is reduce()-able"""
    d = collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)])
    has_reduce, has_reduce_ex = util.has_reduce(d)
    assert util.is_reducible(d)
    assert has_reduce or has_reduce_ex


def test_int_keys_in_object_with_getstate_only():
    """Serialize objects with dict keys that implement __getstate__ only"""
    obj = IntKeysObject()
    encoded = jsonpickle.encode(obj, keys=True)
    decoded = jsonpickle.decode(encoded, keys=True)
    assert obj.data == decoded.data


def test_ordered_dict_int_keys():
    """Serialize dicts with int keys and OrderedDict values"""
    d = {
        1: collections.OrderedDict([(2, -2), (3, -3)]),
        4: collections.OrderedDict([(5, -5), (6, -6)]),
    }
    encoded = jsonpickle.encode(d, keys=True)
    decoded = jsonpickle.decode(encoded, keys=True)
    assert isinstance(decoded[1], collections.OrderedDict)
    assert isinstance(decoded[4], collections.OrderedDict)
    assert -2 == decoded[1][2]
    assert -3 == decoded[1][3]
    assert -5 == decoded[4][5]
    assert -6 == decoded[4][6]
    assert d == decoded


def test_ordered_dict_nested():
    """Serialize nested dicts with OrderedDict values"""
    bottom = collections.OrderedDict([('z', 1), ('a', 2)])
    middle = collections.OrderedDict([('c', bottom)])
    top = collections.OrderedDict([('b', middle)])
    encoded = jsonpickle.encode(top)
    decoded = jsonpickle.decode(encoded)
    assert top == decoded
    # test unpicklable=False
    encoded = jsonpickle.encode(top, unpicklable=False)
    decoded = jsonpickle.decode(encoded)
    assert top == decoded


def test_deque_roundtrip():
    """Make sure we can handle collections.deque"""
    old_deque = collections.deque([0, 1, 2], maxlen=5)
    encoded = jsonpickle.encode(old_deque)
    new_deque = jsonpickle.decode(encoded)
    assert encoded != 'nil'
    assert old_deque[0] == 0
    assert new_deque[0] == 0
    assert old_deque[1] == 1
    assert new_deque[1] == 1
    assert old_deque[2] == 2
    assert new_deque[2] == 2
    assert old_deque.maxlen == 5
    assert new_deque.maxlen == 5


def test_namedtuple_roundtrip():
    """Serialize a NamedTuple"""
    old_nt = NamedTuple(0, 1, 2)
    encoded = jsonpickle.encode(old_nt)
    new_nt = jsonpickle.decode(encoded)
    assert type(old_nt) is type(new_nt)
    assert old_nt is not new_nt
    assert old_nt.a == new_nt.a
    assert old_nt.b == new_nt.b
    assert old_nt.c == new_nt.c
    assert old_nt[0] == new_nt[0]
    assert old_nt[1] == new_nt[1]
    assert old_nt[2] == new_nt[2]


def test_counter_roundtrip():
    counter = collections.Counter({1: 2})
    encoded = jsonpickle.encode(counter)
    decoded = jsonpickle.decode(encoded)
    assert type(decoded) is collections.Counter
    # the integer key becomes a string when keys=False
    assert decoded.get('1') == 2


def test_counter_roundtrip_with_keys():
    counter = collections.Counter({1: 2})
    encoded = jsonpickle.encode(counter, keys=True)
    decoded = jsonpickle.decode(encoded, keys=True)
    assert type(decoded) is collections.Counter
    assert decoded.get(1) == 2


def test_list_with_fd():
    """Serialize a list with an file descriptor"""
    with open(__file__) as fd:
        _test_list_with_fd(fd)
    _test_list_with_fd(fd)  # fd is closed.


def _test_list_with_fd(fd):
    """Serialize a list with an file descriptor"""
    obj = [fd]
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert [None] == newobj


def test_thing_with_fd():
    """Serialize an object with a file descriptor"""
    with open(__file__) as fd:
        _test_thing_with_fd(fd)
    _test_thing_with_fd(fd)  # fd is closed.


def _test_thing_with_fd(fd):
    """Serialize an object with a file descriptor"""
    obj = Thing(fd)
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.name is None


def test_dict_with_fd():
    """Serialize a dict with a file descriptor"""
    with open(__file__) as fd:
        _test_dict_with_fd(fd)
    _test_dict_with_fd(fd)


def _test_dict_with_fd(fd):
    """Serialize a dict with a file descriptor"""
    obj = {'fd': fd}
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj['fd'] is None


def test_thing_with_lamda():
    """Serialize an object with a lambda"""
    obj = Thing(lambda: True)
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert not hasattr(newobj, 'name')


def test_newstyleslots():
    """Serialize an object with new-style slots"""
    obj = ThingWithSlots(True, False)
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.a
    assert not newobj.b


def test_newstyleslots_inherited():
    """Serialize an object with inherited new-style slots"""
    obj = ThingWithInheritedSlots(True, False, None)
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.a
    assert not newobj.b
    assert newobj.c is None


def test_newstyleslots_inherited_deleted_attr():
    """Serialize an object with inherited and deleted new-style slots"""
    obj = ThingWithInheritedSlots(True, False, None)
    del obj.c
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.a
    assert not newobj.b
    assert not hasattr(newobj, 'c')


def test_newstyleslots_with_children():
    """Serialize an object with slots containing objects"""
    obj = ThingWithSlots(Thing('a'), Thing('b'))
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.a.name == 'a'
    assert newobj.b.name == 'b'


def test_newstyleslots_with_children_inherited():
    """Serialize an object with inherited slots containing objects"""
    obj = ThingWithInheritedSlots(Thing('a'), Thing('b'), Thing('c'))
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.a.name == 'a'
    assert newobj.b.name == 'b'
    assert newobj.c.name == 'c'


def test_newstyleslots_iterable():
    """Seriazlie an object with iterable slots"""
    obj = ThingWithIterableSlots('alpha', 'bravo')
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.a == 'alpha'
    assert newobj.b == 'bravo'


def test_newstyleslots_string_slot():
    """Serialize an object with string slots"""
    obj = ThingWithStringSlots('a', 'b')
    jsonstr = jsonpickle.encode(obj)
    newobj = jsonpickle.decode(jsonstr)
    assert newobj.ab == 'ab'


def test_enum34_nested():
    """Serialize enums as nested member variables in an object"""
    ec = EnumClass()
    encoded = jsonpickle.encode(ec)
    decoded = jsonpickle.decode(encoded)
    assert ec.enum_a == decoded.enum_a
    assert ec.enum_b == decoded.enum_b


def test_enum_references():
    """Serialize duplicate enums so that reference IDs are used"""
    a = IntEnumTest.X
    b = IntEnumTest.X
    enums_list = [a, b]
    encoded = jsonpickle.encode(enums_list)
    decoded = jsonpickle.decode(encoded)
    assert enums_list == decoded


def test_enum_unpicklable():
    """Serialize enums when unpicklable=False is specified"""
    obj = Message(MessageTypes.STATUS, MessageCommands.STATUS_ALL)
    encoded = jsonpickle.encode(obj, unpicklable=False)
    decoded = jsonpickle.decode(encoded)
    assert 'message_type' in decoded
    assert 'command' in decoded


def test_enum_int_key_and_value():
    """Serialize Integer enums as dict keys and values"""
    thing = Thing('test')
    value = IntEnumTest.X
    value2 = IntEnumTest.Y
    expect = {
        '__first__': thing,
        'thing': thing,
        value: value,
        value2: value2,
    }
    string = jsonpickle.encode(expect, keys=True)
    actual = jsonpickle.decode(string, keys=True)
    assert 'test' == actual['__first__'].name
    assert value == actual[value]
    assert value2 == actual[value2]

    actual_first = actual['__first__']
    actual_thing = actual['thing']
    assert actual_first is actual_thing


def test_enum_string_key_and_value():
    """Encode enums dict keys and values"""
    thing = Thing('test')
    value = StringEnumTest.A
    value2 = StringEnumTest.B
    expect = {
        value: value,
        '__first__': thing,
        value2: value2,
    }
    string = jsonpickle.encode(expect, keys=True)
    actual = jsonpickle.decode(string, keys=True)
    assert 'test' == actual['__first__'].name
    assert value == actual[value]
    assert value2 == actual[value2]


def test_multiple_string_enums_when_make_refs_is_false():
    """Enums do not create cycles when make_refs=False"""
    # The make_refs=False code path will fallback to repr() when encoding
    # objects that it believes introduce a cycle.  It does this to break
    # out of what would be infinite recursion during traversal.
    # This test ensures that enums do not trigger cycles and are properly
    # encoded under make_refs=False.
    expect = {
        'a': StringEnumTest.A,
        'aa': StringEnumTest.A,
    }
    string = jsonpickle.encode(expect, make_refs=False)
    actual = jsonpickle.decode(string)
    assert expect == actual


# Test classes for ExternalHandlerTestCase
class Mixin:
    def ok(self):
        return True


class UnicodeMixin(str, Mixin):
    def __add__(self, rhs):
        obj = super().__add__(rhs)
        return UnicodeMixin(obj)


class UnicodeMixinHandler(handlers.BaseHandler):
    def flatten(self, obj, data):
        data['value'] = obj
        return data

    def restore(self, obj):
        return UnicodeMixin(obj['value'])


def test_unicode_mixin():
    obj = UnicodeMixin('test')
    assert isinstance(obj, UnicodeMixin)
    assert obj == 'test'

    # Encode into JSON
    handlers.register(UnicodeMixin, UnicodeMixinHandler)
    content = jsonpickle.encode(obj)

    # Resurrect from JSON
    new_obj = jsonpickle.decode(content)
    handlers.unregister(UnicodeMixin)

    new_obj += ' passed'

    assert new_obj == 'test passed'
    assert isinstance(new_obj, UnicodeMixin)
    assert new_obj.ok()


def test_datetime_with_tz_copies_refs():
    """Ensure that we create copies of referenced objects"""
    utc = datetime.timezone.utc
    d0 = datetime.datetime(2020, 5, 5, 5, 5, 5, 5, tzinfo=utc)
    d1 = datetime.datetime(2020, 5, 5, 5, 5, 5, 5, tzinfo=utc)
    obj = [d0, d1]
    encoded = jsonpickle.encode(obj, make_refs=False)
    decoded = jsonpickle.decode(encoded)
    assert len(decoded) == 2
    assert decoded[0] == d0
    assert decoded[1] == d1


def test_with_exclude():
    """Does the _jsonpickle_exclude work"""
    obj = ThingWithExclusion('baz')
    encoded = jsonpickle.encode(obj)
    decoded = jsonpickle.decode(encoded)
    assert decoded.bar == 'baz'
    assert not hasattr(decoded, 'foo')


def test_contained_exclusion():
    """_jsonpickle_exclude should work only on the class it is defined in"""
    obj = ThingWithExcludeSubclass('alpha')
    encoded = jsonpickle.encode(obj)
    decoded = jsonpickle.decode(encoded)
    assert decoded.foo == 'alpha'
    assert not hasattr(decoded.thing, 'foo')
