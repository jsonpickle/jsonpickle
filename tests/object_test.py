from __future__ import absolute_import, division, unicode_literals
import collections
import decimal
import re
import unittest
import datetime

import jsonpickle
from jsonpickle import compat
from jsonpickle import handlers
from jsonpickle import tags
from jsonpickle import util
from jsonpickle.compat import queue, PY2, encodebytes

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


try:
    import enum
except ImportError:
    import types

    class enum(types.ModuleType):

        class Enum(object):
            pass

        class IntEnum(object):
            pass


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
    STATUS = 'STATUS',
    CONTROL = 'CONTROL'


class MessageStatus(enum.Enum):
    OK = 'OK',
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
        return self.offset,


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
        self.assertNotEqual(type(newtree.default_factory),
                            jsonpickle.unpickler._Proxy)
        self.assertNotEqual(type(newtree['A'].default_factory),
                            jsonpickle.unpickler._Proxy)
        self.assertNotEqual(type(newtree['A']['Z'].default_factory),
                            jsonpickle.unpickler._Proxy)
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

    def test_list_with_fd(self):
        fd = open(__file__, 'r')
        fd.close()
        obj = [fd]
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual([None], newobj)

    def test_thing_with_fd(self):
        fd = open(__file__, 'r')
        fd.close()
        obj = Thing(fd)
        jsonstr = jsonpickle.encode(obj)
        newobj = jsonpickle.decode(jsonstr)
        self.assertEqual(None, newobj.name)

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
        obj = decimal.Decimal(1)
        flattened = self.pickler.flatten(obj)
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(type(inflated), decimal.Decimal)

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
        self.assertEqual('object_test.GetstateDict',
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
            b64ustr = encodebytes(b'foo').decode('utf-8')
            self.assertEqual({tags.B64: b64ustr}, encoded)
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
        b64ustr = encodebytes(b'foo\xff').decode('utf-8')
        self.assertEqual({tags.B64: b64ustr}, encoded)
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
    suite.addTest(unittest.makeSuite(AdvancedObjectsTestCase))
    suite.addTest(unittest.makeSuite(ExternalHandlerTestCase))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
