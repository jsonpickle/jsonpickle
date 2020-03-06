from __future__ import absolute_import, division, unicode_literals
import array
import enum
import collections
import datetime
import decimal
import re
import threading
import unittest

import jsonpickle
from jsonpickle import compat
from jsonpickle import handlers
from jsonpickle import tags
from jsonpickle import util
from jsonpickle.compat import queue, PY2, PY3_ORDERED_DICT

import pytest

from helper import SkippableTest


class Thing(object):
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


class GetstateOnly(object):
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def __getstate__(self):
        return [self.a, self.b]


class GetstateReturnsList(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getstate__(self):
        return [self.x, self.y]

    def __setstate__(self, state):
        self.x, self.y = state[0], state[1]


class GetstateRecursesInfintely(object):
    def __getstate__(self):
        return GetstateRecursesInfintely()


class ListSubclassWithInit(list):
    def __init__(self, attr):
        self.attr = attr
        super(ListSubclassWithInit, self).__init__()


NamedTuple = collections.namedtuple('NamedTuple', 'a, b, c')


class ObjWithJsonPickleRepr(object):
    def __init__(self):
        self.data = {'a': self}

    def __repr__(self):
        return jsonpickle.encode(self)


class OldStyleClass:
    pass


class SetSubclass(set):
    pass


class ThingWithFunctionRefs(object):
    def __init__(self):
        self.fn = func


def func(x):
    return x


class ThingWithQueue(object):
    def __init__(self):
        self.child_1 = queue.Queue()
        self.child_2 = queue.Queue()
        self.childref_1 = self.child_1
        self.childref_2 = self.child_2


class ThingWithSlots(object):

    __slots__ = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class ThingWithInheritedSlots(ThingWithSlots):

    __slots__ = ('c',)

    def __init__(self, a, b, c):
        ThingWithSlots.__init__(self, a, b)
        self.c = c


class ThingWithIterableSlots(object):

    __slots__ = iter('ab')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class ThingWithStringSlots(object):
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


class IntEnumTest(enum.IntEnum):
    X = 1
    Y = 2


class StringEnumTest(enum.Enum):
    A = 'a'
    B = 'b'


class SubEnum(enum.Enum):
    a = 1
    b = 2


class EnumClass(object):
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


class Message(object):
    def __init__(self, message_type, command, status=None, body=None):
        self.message_type = MessageTypes(message_type)
        if command:
            self.command = MessageCommands(command)
        if status:
            self.status = MessageStatus(status)
        if body:
            self.body = body


class ThingWithTimedeltaAttribute(object):
    def __init__(self, offset):
        self.offset = datetime.timedelta(offset)

    def __getinitargs__(self):
        return (self.offset,)


class FailSafeTestCase(SkippableTest):
    class BadClass(object):
        def __getstate__(self):
            raise ValueError('Intentional error')

    good = 'good'

    to_pickle = [BadClass(), good]

    def test_no_error(self):
        encoded = jsonpickle.encode(self.to_pickle, fail_safe=lambda e: None)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(decoded[0], None)
        self.assertEqual(decoded[1], 'good')

    def test_error_recorded(self):
        exceptions = []

        def recorder(exception):
            exceptions.append(exception)

        jsonpickle.encode(self.to_pickle, fail_safe=recorder)
        self.assertEqual(len(exceptions), 1)
        self.assertTrue(isinstance(exceptions[0], Exception))

    def test_custom_err_msg(self):
        CUSTOM_ERR_MSG = 'custom err msg'
        encoded = jsonpickle.encode(self.to_pickle, fail_safe=lambda e: CUSTOM_ERR_MSG)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(decoded[0], CUSTOM_ERR_MSG)


class IntKeysObject(object):
    def __init__(self):
        self.data = {0: 0}

    def __getstate__(self):
        return self.__dict__


class ExceptionWithArguments(Exception):
    def __init__(self, value):
        super(ExceptionWithArguments, self).__init__('test')
        self.value = value


class AdvancedObjectsTestCase(SkippableTest):
    def setUp(self):
        self.pickler = jsonpickle.pickler.Pickler()
        self.unpickler = jsonpickle.unpickler.Unpickler()

    def tearDown(self):
        self.pickler.reset()
        self.unpickler.reset()

    def test_defaultdict_roundtrip(self):
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
        self.assertEqual(newdefdict['a'], 1)
        self.assertEqual(newdefdict['b'], [2])
        self.assertEqual(type(newdefdict['c']), defaultdict)
        self.assertEqual(defdict.default_factory, list)
        self.assertEqual(newdefdict.default_factory, list)

    def test_defaultdict_roundtrip_simple_lambda(self):
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
        self.assertEqual(newdefdict[0], 'zero')
        self.assertEqual(type(newdefdict[1]), defaultdict)
        self.assertEqual(newdefdict[1][0], 'zero')
        self.assertEqual(newdefdict[1][1], {})  # inner defaultdict
        self.assertEqual(newdefdict[2][0], 0)  # outer defaultdict
        self.assertEqual(type(newdefdict[3]), defaultdict)
        # outer-most defaultdict
        self.assertEqual(newdefdict[3].default_factory, int)

    def test_defaultdict_roundtrip_simple_lambda2(self):
        defaultdict = collections.defaultdict
        payload = {'a': defaultdict(lambda: 0)}
        defdict = defaultdict(lambda: 0, payload)
        # roundtrip
        encoded = jsonpickle.encode(defdict, keys=True)
        decoded = jsonpickle.decode(encoded, keys=True)
        self.assertEqual(type(decoded), defaultdict)
        self.assertEqual(type(decoded['a']), defaultdict)

    def test_defaultdict_and_things_roundtrip_simple_lambda(self):
        thing = Thing('a')
        defaultdict = collections.defaultdict
        defdict = defaultdict(lambda: 0)
        obj = [defdict, thing, thing]
        # roundtrip
        encoded = jsonpickle.encode(obj, keys=True)
        decoded = jsonpickle.decode(encoded, keys=True)
        self.assertEqual(decoded[0].default_factory(), 0)
        self.assertIs(decoded[1], decoded[2])

    def test_defaultdict_subclass_with_self_as_default_factory(self):
        cls = ThingWithSelfAsDefaultFactory
        tree = cls()
        newtree = self._test_defaultdict_tree(tree, cls)
        self.assertEqual(type(newtree['A'].default_factory), cls)
        self.assertTrue(newtree.default_factory is newtree)
        self.assertTrue(newtree['A'].default_factory is newtree['A'])
        self.assertTrue(newtree['Z'].default_factory is newtree['Z'])

    def test_defaultdict_subclass_with_class_as_default_factory(self):
        cls = ThingWithClassAsDefaultFactory
        tree = cls()
        newtree = self._test_defaultdict_tree(tree, cls)
        self.assertTrue(newtree.default_factory is cls)
        self.assertTrue(newtree['A'].default_factory is cls)
        self.assertTrue(newtree['Z'].default_factory is cls)

    def _test_defaultdict_tree(self, tree, cls):
        tree['A']['B'] = 1
        tree['A']['C'] = 2
        # roundtrip
        encoded = jsonpickle.encode(tree)
        newtree = jsonpickle.decode(encoded)
        # make sure we didn't lose anything
        self.assertEqual(type(newtree), cls)
        self.assertEqual(type(newtree['A']), cls)
        self.assertEqual(newtree['A']['B'], 1)
        self.assertEqual(newtree['A']['C'], 2)
        # ensure that the resulting default_factory is callable and creates
        # a new instance of cls.
        self.assertEqual(type(newtree['A'].default_factory()), cls)
        # we've never seen 'D' before so the reconstructed defaultdict tree
        # should create an instance of cls.
        self.assertEqual(type(newtree['A']['D']), cls)
        # ensure that proxies do not escape into user code
        self.assertNotEqual(type(newtree.default_factory), jsonpickle.unpickler._Proxy)
        self.assertNotEqual(
            type(newtree['A'].default_factory), jsonpickle.unpickler._Proxy
        )
        self.assertNotEqual(
            type(newtree['A']['Z'].default_factory), jsonpickle.unpickler._Proxy
        )
        return newtree

    def test_deque_roundtrip(self):
        """Make sure we can handle collections.deque"""
        old_deque = collections.deque([0, 1, 2], maxlen=5)
        encoded = jsonpickle.encode(old_deque)
        new_deque = jsonpickle.decode(encoded)
        self.assertNotEqual(encoded, 'nil')
        self.assertEqual(old_deque[0], 0)
        self.assertEqual(new_deque[0], 0)
        self.assertEqual(old_deque[1], 1)
        self.assertEqual(new_deque[1], 1)
        self.assertEqual(old_deque[2], 2)
        self.assertEqual(new_deque[2], 2)
        self.assertEqual(old_deque.maxlen, 5)
        self.assertEqual(new_deque.maxlen, 5)

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
        counter = collections.Counter({1: 2})
        encoded = jsonpickle.encode(counter)
        decoded = jsonpickle.decode(encoded)
        self.assertTrue(type(decoded) is collections.Counter)
        # the integer key becomes a string when keys=False
        self.assertEqual(decoded.get('1'), 2)

    def test_counter_roundtrip_with_keys(self):
        counter = collections.Counter({1: 2})
        encoded = jsonpickle.encode(counter, keys=True)
        decoded = jsonpickle.decode(encoded, keys=True)
        self.assertTrue(type(decoded) is collections.Counter)
        self.assertEqual(decoded.get(1), 2)

    issue281 = pytest.mark.xfail(
        'sys.version_info >= (3, 8)',
        reason='https://github.com/jsonpickle/jsonpickle/issues/281',
    )

    @issue281
    def test_list_with_fd(self):
        fd = open(__file__, 'r')
        fd.close()
        obj = [fd]
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual([None], newobj)

    @issue281
    def test_thing_with_fd(self):
        fd = open(__file__, 'r')
        fd.close()
        obj = Thing(fd)
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(None, newobj.name)

    @issue281
    def test_dict_with_fd(self):
        fd = open(__file__, 'r')
        fd.close()
        obj = {'fd': fd}
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(None, newobj['fd'])

    def test_thing_with_lamda(self):
        obj = Thing(lambda: True)
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertFalse(hasattr(newobj, 'name'))

    def test_newstyleslots(self):
        obj = ThingWithSlots(True, False)
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertTrue(newobj.a)
        self.assertFalse(newobj.b)

    def test_newstyleslots_inherited(self):
        obj = ThingWithInheritedSlots(True, False, None)
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertTrue(newobj.a)
        self.assertFalse(newobj.b)
        self.assertEqual(newobj.c, None)

    def test_newstyleslots_inherited_deleted_attr(self):
        obj = ThingWithInheritedSlots(True, False, None)
        del obj.c
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertTrue(newobj.a)
        self.assertFalse(newobj.b)
        self.assertFalse(hasattr(newobj, 'c'))

    def test_newstyleslots_with_children(self):
        obj = ThingWithSlots(Thing('a'), Thing('b'))
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(newobj.a.name, 'a')
        self.assertEqual(newobj.b.name, 'b')

    def test_newstyleslots_with_children_inherited(self):
        obj = ThingWithInheritedSlots(Thing('a'), Thing('b'), Thing('c'))
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(newobj.a.name, 'a')
        self.assertEqual(newobj.b.name, 'b')
        self.assertEqual(newobj.c.name, 'c')

    def test_newstyleslots_iterable(self):
        obj = ThingWithIterableSlots('alpha', 'bravo')
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(newobj.a, 'alpha')
        self.assertEqual(newobj.b, 'bravo')

    def test_newstyleslots_string_slot(self):
        obj = ThingWithStringSlots('a', 'b')
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(newobj.ab, 'ab')

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

    def test_list_subclass_with_init(self):
        obj = ListSubclassWithInit('foo')
        self.assertEqual(obj.attr, 'foo')
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(type(inflated), ListSubclassWithInit)

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
        self.assertEqual(restored.data.__class__, ListSubclass)
        self.assertEqual(restored.data, data)

    def test_decimal(self):
        obj = decimal.Decimal('0.5')
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertTrue(isinstance(inflated, decimal.Decimal))

    def test_repr_using_jsonpickle(self):
        thing = ObjWithJsonPickleRepr()
        thing.child = ObjWithJsonPickleRepr()
        thing.child.parent = thing

        encoded = jsonpickle.encode(thing)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(id(decoded), id(decoded.child.parent))

    def test_broken_repr_dict_key(self):
        """Tests that we can pickle dictionaries with keys that have
        broken __repr__ implementations.
        """
        br = BrokenReprThing('test')
        obj = {br: True}
        pickler = jsonpickle.pickler.Pickler()
        flattened = pickler.flatten(obj)
        self.assertTrue('<BrokenReprThing "test">' in flattened)
        self.assertTrue(flattened['<BrokenReprThing "test">'])

    def test_ordered_dict_python3(self):
        """Ensure that we preserve dict order on python3"""
        if not PY3_ORDERED_DICT:
            return
        # Python3.6+ preserves dict order.
        obj = {'z': 'Z', 'x': 'X', 'y': 'Y'}
        clone = jsonpickle.decode(jsonpickle.encode(obj))
        expect = ['z', 'x', 'y']
        actual = list(clone.keys())
        self.assertEqual(expect, actual)

    def test_ordered_dict(self):
        d = collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)])
        encoded = jsonpickle.encode(d)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(d, decoded)

    def test_ordered_dict_unpicklable(self):
        d = collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)])
        encoded = jsonpickle.encode(d, unpicklable=False)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(d, decoded)

    def test_ordered_dict_reduces(self):
        d = collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)])
        has_reduce, has_reduce_ex = util.has_reduce(d)
        self.assertTrue(util.is_reducible(d))
        self.assertTrue(has_reduce or has_reduce_ex)

    def test_int_keys_in_object_with_getstate_only(self):
        obj = IntKeysObject()
        encoded = jsonpickle.encode(obj, keys=True)
        decoded = jsonpickle.decode(encoded, keys=True)
        self.assertEqual(obj.data, decoded.data)

    def test_ordered_dict_int_keys(self):
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

    def test_ordered_dict_nested(self):
        bottom = collections.OrderedDict([('z', 1), ('a', 2)])
        middle = collections.OrderedDict([('c', bottom)])
        top = collections.OrderedDict([('b', middle)])

        encoded = jsonpickle.encode(top)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(top, decoded)

        # test unpicklable=False
        encoded = jsonpickle.encode(top, unpicklable=False)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(top, decoded)

    def test_posix_stat_result(self):
        try:
            import posix
        except ImportError:
            return
        expect = posix.stat(__file__)
        encoded = jsonpickle.encode(expect)
        actual = jsonpickle.decode(encoded)
        self.assertEqual(expect, actual)

    def test_oldstyleclass(self):
        obj = OldStyleClass()
        obj.value = 1234

        flattened = self.pickler.flatten(obj)
        self.assertEqual(1234, flattened['value'])

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1234, inflated.value)

    def test_dictsubclass(self):
        obj = DictSubclass()
        obj['key1'] = 1

        expect = {
            tags.OBJECT: 'object_test.DictSubclass',
            'key1': 1,
            '__dict__': {},
        }
        flattened = self.pickler.flatten(obj)
        self.assertEqual(expect, flattened)

        inflated = self.unpickler.restore(flattened)
        self.assertEqual(type(inflated), DictSubclass)
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
        self.assertEqual('object_test.GetstateDict', flattened[tags.OBJECT])
        self.assertTrue(tags.STATE in flattened)
        self.assertTrue(tags.TUPLE in flattened[tags.STATE])
        self.assertEqual(['test', {'key1': 1}], flattened[tags.STATE][tags.TUPLE])

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

    def test_getstate_with_getstate_only(self):
        obj = GetstateOnly()
        a = obj.a = 'this object implements'
        b = obj.b = '__getstate__ but not __setstate__'
        expect = [a, b]
        flat = self.pickler.flatten(obj)
        actual = flat[tags.STATE]
        self.assertEqual(expect, actual)
        restored = self.unpickler.restore(flat)
        self.assertEqual(expect, restored)

    def test_getstate_does_not_recurse_infinitely(self):
        obj = GetstateRecursesInfintely()
        pickler = jsonpickle.pickler.Pickler(max_depth=5)
        pickler.flatten(obj)

    def test_thing_with_queue(self):
        obj = ThingWithQueue()
        flattened = self.pickler.flatten(obj)
        restored = self.unpickler.restore(flattened)
        self.assertEqual(type(restored.child_1), type(queue.Queue()))
        self.assertEqual(type(restored.child_2), type(queue.Queue()))
        # Check references
        self.assertTrue(restored.child_1 is restored.childref_1)
        self.assertTrue(restored.child_2 is restored.childref_2)

    def test_thing_with_func(self):
        obj = ThingWithFunctionRefs()
        obj.ref = obj
        flattened = self.pickler.flatten(obj)
        restored = self.unpickler.restore(flattened)
        self.assertTrue(restored.fn is obj.fn)

        expect = 'success'
        actual1 = restored.fn(expect)
        self.assertEqual(expect, actual1)
        self.assertTrue(restored is restored.ref)

    def test_thing_with_compiled_regex(self):
        rgx = re.compile(r'(.*)(cat)')
        obj = Thing(rgx)

        flattened = self.pickler.flatten(obj)
        restored = self.unpickler.restore(flattened)
        match = restored.name.match('fatcat')
        self.assertEqual('fat', match.group(1))
        self.assertEqual('cat', match.group(2))

    def test_base_object_roundrip(self):
        roundtrip = self.unpickler.restore(self.pickler.flatten(object()))
        self.assertEqual(type(roundtrip), object)

    def test_enum34(self):
        restore = self.unpickler.restore
        flatten = self.pickler.flatten

        def roundtrip(obj):
            return restore(flatten(obj))

        self.assertTrue(roundtrip(IntEnumTest.X) is IntEnumTest.X)
        self.assertTrue(roundtrip(IntEnumTest) is IntEnumTest)

        self.assertTrue(roundtrip(StringEnumTest.A) is StringEnumTest.A)
        self.assertTrue(roundtrip(StringEnumTest) is StringEnumTest)

    def test_enum34_nested(self):
        ec = EnumClass()
        encoded = jsonpickle.encode(ec)
        decoded = jsonpickle.decode(encoded)

        self.assertEqual(ec.enum_a, decoded.enum_a)
        self.assertEqual(ec.enum_b, decoded.enum_b)

    def test_enum_references(self):
        a = IntEnumTest.X
        b = IntEnumTest.X

        enums_list = [a, b]
        encoded = jsonpickle.encode(enums_list)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(enums_list, decoded)

    def test_enum_unpicklable(self):
        obj = Message(MessageTypes.STATUS, MessageCommands.STATUS_ALL)
        encoded = jsonpickle.encode(obj, unpicklable=False)
        decoded = jsonpickle.decode(encoded)
        self.assertTrue('message_type' in decoded)
        self.assertTrue('command' in decoded)

    def test_enum_int_key_and_value(self):
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
        self.assertEqual('test', actual['__first__'].name)
        self.assertEqual(value, actual[value])
        self.assertEqual(value2, actual[value2])

        actual_first = actual['__first__']
        actual_thing = actual['thing']
        self.assertTrue(actual_first is actual_thing)

    def test_enum_string_key_and_value(self):
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
        self.assertEqual('test', actual['__first__'].name)
        self.assertEqual(value, actual[value])
        self.assertEqual(value2, actual[value2])

    def test_bytes_unicode(self):
        b1 = b'foo'
        b2 = b'foo\xff'
        u1 = 'foo'

        # unicode strings get encoded/decoded as is
        encoded = self.pickler.flatten(u1)
        self.assertTrue(encoded == u1)
        self.assertTrue(isinstance(encoded, compat.ustr))
        decoded = self.unpickler.restore(encoded)
        self.assertTrue(decoded == u1)
        self.assertTrue(isinstance(decoded, compat.ustr))

        # bytestrings are wrapped in PY3 but in PY2 we try to decode first
        encoded = self.pickler.flatten(b1)
        if PY2:
            self.assertEqual(encoded, u1)
            self.assertTrue(isinstance(encoded, compat.ustr))
        else:
            self.assertNotEqual(encoded, u1)
            encoded_ustr = util.b64encode(b'foo')
            self.assertEqual({tags.B64: encoded_ustr}, encoded)
            self.assertTrue(isinstance(encoded[tags.B64], compat.ustr))
        decoded = self.unpickler.restore(encoded)
        self.assertTrue(decoded == b1)
        if PY2:
            self.assertTrue(isinstance(decoded, compat.ustr))
        else:
            self.assertTrue(isinstance(decoded, bytes))

        # bytestrings that we can't decode to UTF-8 will always be wrapped
        encoded = self.pickler.flatten(b2)
        self.assertNotEqual(encoded, b2)
        encoded_ustr = util.b64encode(b'foo\xff')
        self.assertEqual({tags.B64: encoded_ustr}, encoded)
        self.assertTrue(isinstance(encoded[tags.B64], compat.ustr))
        decoded = self.unpickler.restore(encoded)
        self.assertEqual(decoded, b2)
        self.assertTrue(isinstance(decoded, bytes))

    def test_backcompat_bytes_quoted_printable(self):
        """Test decoding bytes objects from older jsonpickle versions"""

        b1 = b'foo'
        b2 = b'foo\xff'

        # older versions of jsonpickle used a quoted-printable encoding
        expect = b1
        actual = self.unpickler.restore({tags.BYTES: 'foo'})
        self.assertEqual(expect, actual)

        expect = b2
        actual = self.unpickler.restore({tags.BYTES: 'foo=FF'})
        self.assertEqual(expect, actual)

    def test_nested_objects(self):
        obj = ThingWithTimedeltaAttribute(99)
        flattened = self.pickler.flatten(obj)
        restored = self.unpickler.restore(flattened)
        self.assertEqual(restored.offset, datetime.timedelta(99))

    def test_threading_lock(self):
        obj = Thing('lock')
        obj.lock = threading.Lock()
        lock_class = obj.lock.__class__
        # Roundtrip and make sure we get a lock object.
        json = self.pickler.flatten(obj)
        clone = self.unpickler.restore(json)
        self.assertTrue(isinstance(clone.lock, lock_class))
        self.assertFalse(clone.lock.locked())

        # Serializing a locked lock should create a locked clone.
        self.assertTrue(obj.lock.acquire())
        json = self.pickler.flatten(obj)
        obj.lock.release()
        # Restore the locked lock state.
        clone = self.unpickler.restore(json)
        self.assertTrue(clone.lock.locked())
        clone.lock.release()

    def _test_array_roundtrip(self, obj):
        """Roundtrip an array and test invariants"""
        json = self.pickler.flatten(obj)
        clone = self.unpickler.restore(json)
        self.assertTrue(isinstance(clone, array.array))
        self.assertEqual(obj.typecode, clone.typecode)
        self.assertEqual(len(obj), len(clone))
        for j, k in zip(obj, clone):
            self.assertEqual(j, k)
        self.assertEqual(obj, clone)

    def test_array_handler_numeric(self):
        """Test numeric array.array typecodes that work in Python2+3"""
        typecodes = ('b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'f', 'd')
        for typecode in typecodes:
            obj = array.array(typecode, (1, 2, 3))
            self._test_array_roundtrip(obj)

    def test_array_handler_python2(self):
        """Python2 allows the "c" byte/char typecode"""
        if PY2:
            obj = array.array('c', bytes('abcd'))
            self._test_array_roundtrip(obj)

    def test_exceptions_with_arguments(self):
        """Ensure that we can roundtrip Exceptions that take arguments"""
        obj = ExceptionWithArguments('example')
        json = self.pickler.flatten(obj)
        clone = self.unpickler.restore(json)
        self.assertEqual(obj.value, clone.value)
        self.assertEqual(obj.args, clone.args)


# Test classes for ExternalHandlerTestCase
class Mixin(object):
    def ok(self):
        return True


class UnicodeMixin(str, Mixin):
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
        self.assertTrue(isinstance(obj, UnicodeMixin))
        self.assertEqual(obj, 'test')

        # Encode into JSON
        content = jsonpickle.encode(obj)

        # Resurrect from JSON
        new_obj = jsonpickle.decode(content)
        new_obj += ' passed'

        self.assertEqual(new_obj, 'test passed')
        self.assertTrue(isinstance(new_obj, UnicodeMixin))
        self.assertTrue(new_obj.ok())


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FailSafeTestCase))
    suite.addTest(unittest.makeSuite(AdvancedObjectsTestCase))
    suite.addTest(unittest.makeSuite(ExternalHandlerTestCase))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
