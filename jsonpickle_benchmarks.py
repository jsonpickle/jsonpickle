# Run "make benchmark" in the directory that this file is in.
# This will test the version of jsonpickle inside the local directory,
# NOT the pip installed version.
# To test the pip installed version, run this outside the jsonpickle directory:
# py.test --benchmark-only ./jsonpickle_benchmarks.py --benchmark-histogram
# MAKE SURE you have pytest-benchmark pip installed.
import jsonpickle
import itertools

# DEFINTIIONS:
# HOMOGENOUS - Except for sets, each container may only have integers, floats, strings,
# and itself inside it
# HETEROGENOUS - Except for sets, each container must have at least one of every other
# container type being tested
# SIMPLE - Each container must have two each of integers, floats, and strings
# COMPLEX - Each container must have four each of integers, floats, and strings

# SETUP
###################################################################################


class SlotPickleMixin(object):
    def __getstate__(self):
        all_slots = itertools.chain.from_iterable(
            getattr(cls, '__slots__', []) for cls in self.__class__.__mro__
        )
        return dict(
            (slot, getattr(self, slot)) for slot in all_slots if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in dict(state).items():
            setattr(self, slot, value)


class MyClass(object):
    __slots__ = ['idk', 'idk2']

    def __init__(self):
        self.idk = {
            'a': 0,
            'b': [0.5783, 1, 2.5789],
            'c': {0, 1.87559, 2},
            'd': ['0', '1', '2'],
            'e': {'0', '1', '2'},
            'f': {'a': 0, 'b': {'a': 1}},
        }
        self.idk2 = [
            'a',
            'b',
            'c',
            0,
            0.18752,
            2,
            ['a', 'b', 'c', 0.8579, 1, 2.59757],
            (self.idk, self.idk),
        ]


class MyClassGetState(SlotPickleMixin):
    __slots__ = ['idk', 'idk2']

    def __init__(self):
        self.idk = {
            'a': 0,
            'b': [0.5783, 1, 2.5789],
            'c': {0, 1.87559, 2},
            'd': ['0', '1', '2'],
            'e': {'0', '1', '2'},
            'f': {'a': 0, 'b': {'a': 1}},
        }
        self.idk2 = [
            'a',
            'b',
            'c',
            0,
            0.18752,
            2,
            ['a', 'b', 'c', 0.8579, 1, 2.59757],
            (self.idk, self.idk),
        ]

    def __getstate__(self):
        return SlotPickleMixin.__getstate__(self)

    def __setstate__(self, object_state):
        SlotPickleMixin.__setstate__(self, object_state)


class MyClassBasic(object):
    def __init__(self):
        self.idk = {'a': 0}
        self.idk2 = ['a', 0]


HOMOGENOUS_COMPLEX_DICT = {
    'a': 0,
    'b': 4.2,
    'c': {6.9: 'd'},
    -1: {5: 5},
    6: {8.42: -2.5},
}
HOMOGENOUS_COMPLEX_LIST = [
    'a',
    0,
    'b',
    4.2,
    'c',
    [6.9, 'd'],
    -1,
    [5, 5],
    6,
    [8.42, -2.5],
]
HOMOGENOUS_COMPLEX_TUPLE = (
    'a',
    0,
    'b',
    4.2,
    'c',
    (6.9, 'd'),
    -1,
    (5, 5),
    6,
    (8.42, -2.5),
)
# can't put a set inside a set :(
HOMOGENOUS_COMPLEX_SET = {
    'a',
    0,
    'b',
    4.2,
    'c',
    (6.9, 'd'),
    -1,
    (5, 5),
    6,
    (8.42, -2.5),
}

HETEROGENOUS_COMPLEX_DICT = {
    'a': (0, '1'),
    2.53: [3, 1],
    'c': {4.2, 6.9},
    'd': {1: 5.432},
}
HETEROGENOUS_COMPLEX_LIST = [
    'a',
    (0, '1'),
    2.53,
    [3, 1],
    'c',
    {4.2, 6.9},
    'd',
    {1: 5.432},
]
HETEROGENOUS_COMPLEX_TUPLE = (
    'a',
    (0, '1'),
    2.53,
    [3, 1],
    'c',
    {4.2, 6.9},
    'd',
    {1: 5.432},
)
HETEROGENOUS_COMPLEX_SET = {
    'a',
    (0, '1'),
    2.53,
    (3, 1),
    'c',
    (4.2, 6.9),
    'd',
    (1, 5.432),
}

simple_dict_encoded = jsonpickle.encode({'a': 0, 'b': 4.2, 3: 6.9})
simple_list_encoded = jsonpickle.encode(['a', 0, 'b', 4.2, 3, 6.9])
simple_tuple_encoded = jsonpickle.encode(('a', 0, 'b', 4.2, 3, 6.9))
simple_set_encoded = jsonpickle.encode({'a', 0, 'b', 4.2, 3, 6.9})

homogenous_dict_encoded = jsonpickle.encode(HOMOGENOUS_COMPLEX_DICT)
homogenous_list_encoded = jsonpickle.encode(HOMOGENOUS_COMPLEX_LIST)
homogenous_tuple_encoded = jsonpickle.encode(HOMOGENOUS_COMPLEX_TUPLE)
homogenous_set_encoded = jsonpickle.encode(HOMOGENOUS_COMPLEX_SET)

heterogenous_dict_encoded = jsonpickle.encode(HETEROGENOUS_COMPLEX_DICT)
heterogenous_list_encoded = jsonpickle.encode(HETEROGENOUS_COMPLEX_LIST)
heterogenous_tuple_encoded = jsonpickle.encode(HETEROGENOUS_COMPLEX_TUPLE)
heterogenous_set_encoded = jsonpickle.encode(HETEROGENOUS_COMPLEX_SET)

basic_class_encoded = jsonpickle.encode(MyClassBasic())
class_encoded = jsonpickle.encode(MyClass())
state_class_encoded = jsonpickle.encode(MyClassGetState())
###################################################################################

# SIMPLE PRIMITIVE ENCODE/DECODE


def test_simple_dict_encode(benchmark):
    benchmark(jsonpickle.encode, {'a': 0})


def test_simple_dict_decode(benchmark):
    benchmark(jsonpickle.decode, simple_dict_encoded)


def test_simple_list_encode(benchmark):
    benchmark(jsonpickle.encode, ['a', 0])


def test_simple_list_decode(benchmark):
    benchmark(jsonpickle.decode, simple_list_encoded)


def test_simple_tuple_encode(benchmark):
    benchmark(jsonpickle.encode, ('a', 0))


def test_simple_tuple_decode(benchmark):
    benchmark(jsonpickle.decode, simple_tuple_encoded)


def test_simple_set_encode(benchmark):
    benchmark(jsonpickle.encode, {'a', 0})


def test_simple_set_decode(benchmark):
    benchmark(jsonpickle.decode, simple_set_encoded)


# COMPLEX HOMOGENOUS PRIMITIVE ENCODE/DECODE
def test_complex_homogenous_dict_encode(benchmark):
    benchmark(jsonpickle.encode, HOMOGENOUS_COMPLEX_DICT)


def test_complex_homogenous_dict_decode(benchmark):
    benchmark(jsonpickle.decode, homogenous_dict_encoded)


def test_complex_homogenous_list_encode(benchmark):
    benchmark(jsonpickle.encode, HOMOGENOUS_COMPLEX_LIST)


def test_complex_homogenous_list_decode(benchmark):
    benchmark(jsonpickle.decode, homogenous_list_encoded)


def test_complex_homogenous_tuple_encode(benchmark):
    benchmark(jsonpickle.encode, HOMOGENOUS_COMPLEX_TUPLE)


def test_complex_homogenous_tuple_decode(benchmark):
    benchmark(jsonpickle.decode, homogenous_tuple_encoded)


def test_complex_homogenous_set_encode(benchmark):
    benchmark(jsonpickle.encode, HOMOGENOUS_COMPLEX_SET)


def test_complex_homogenous_set_decode(benchmark):
    benchmark(jsonpickle.decode, homogenous_set_encoded)


# COMPLEX HETEROGENOUS PRIMITIVE ENCODE/DECODE
def test_complex_heterogenous_dict_encode(benchmark):
    benchmark(jsonpickle.encode, HETEROGENOUS_COMPLEX_DICT)


def test_complex_heterogenous_dict_decode(benchmark):
    benchmark(jsonpickle.decode, heterogenous_dict_encoded)


def test_complex_heterogenous_list_encode(benchmark):
    benchmark(jsonpickle.encode, HETEROGENOUS_COMPLEX_LIST)


def test_complex_heterogenous_list_decode(benchmark):
    benchmark(jsonpickle.decode, heterogenous_list_encoded)


def test_complex_heterogenous_tuple_encode(benchmark):
    benchmark(jsonpickle.encode, HETEROGENOUS_COMPLEX_TUPLE)


def test_complex_heterogenous_tuple_decode(benchmark):
    benchmark(jsonpickle.decode, heterogenous_tuple_encoded)


def test_complex_heterogenous_set_encode(benchmark):
    benchmark(jsonpickle.encode, HETEROGENOUS_COMPLEX_SET)


def test_complex_heterogenous_set_decode(benchmark):
    benchmark(jsonpickle.decode, heterogenous_set_encoded)


# CUSTOM CLASS ENCODE/DECODE
def test_basic_class_encode(benchmark):
    benchmark(jsonpickle.encode, MyClassBasic())


def test_basic_class_decode(benchmark):
    benchmark(jsonpickle.decode, basic_class_encoded)


def test_advanced_class_encode(benchmark):
    benchmark(jsonpickle.encode, MyClass())


def test_advanced_class_decode(benchmark):
    benchmark(jsonpickle.decode, class_encoded)


def test_state_class_encode(benchmark):
    benchmark(jsonpickle.encode, MyClassGetState())


def test_state_class_decode(benchmark):
    benchmark(jsonpickle.decode, state_class_encoded)
