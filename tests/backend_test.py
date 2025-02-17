import decimal
from hashlib import md5
from warnings import warn

import pytest
from helper import SkippableTest

import jsonpickle
import jsonpickle.ext.yaml


class Thing:
    def __init__(self, name):
        self.name = name
        self.child = None


class A:
    def __init__(self):
        self.id = md5(str(id(self)).encode()).hexdigest()[:5]  # unique enough hash


class BSlots:
    __slots__ = ["a2", "a1", "a3"]

    def __init__(self):
        self.a2 = A()  # set attribs not in alphabetical order
        self.a1 = A()
        self.a3 = self.a1  # create a reference


SAMPLE_DATA = {'things': [Thing('data')]}


class BackendBase(SkippableTest):
    def _is_installed(self, backend):
        if not jsonpickle.util.is_installed(backend):
            return self.skip('%s not available; please install' % backend)

    def set_backend(self, *args):
        backend = args[0]

        self._is_installed(backend)

        jsonpickle.load_backend(*args)
        jsonpickle.set_preferred_backend(backend)

    def set_preferred_backend(self, backend):
        self._is_installed(backend)
        jsonpickle.set_preferred_backend(backend)

    def tearDown(self):
        # always reset to default backend
        jsonpickle.set_preferred_backend('json')

    def assert_roundtrip(self, json_input):
        expect = SAMPLE_DATA
        actual = jsonpickle.decode(json_input)
        assert expect['things'][0].name == actual['things'][0].name
        assert expect['things'][0].child == actual['things'][0].child

        pickled = jsonpickle.encode(SAMPLE_DATA)
        actual = jsonpickle.decode(pickled)
        assert expect['things'][0].name == actual['things'][0].name
        assert expect['things'][0].child == actual['things'][0].child

    def test_None_dict_key(self):
        """Ensure that backends produce the same result for None dict keys"""
        data = {None: None}
        expect = {'null': None}
        pickle = jsonpickle.encode(data)
        actual = jsonpickle.decode(pickle)
        assert expect == actual

    def test_encode_with_indent_and_separators(self):
        obj = {
            'a': 1,
            'b': 2,
        }
        expect = '{\n' '    "a": 1,\n' '    "b": 2\n' '}'
        actual = jsonpickle.encode(obj, indent=4, separators=(',', ': '))
        assert expect == actual


class JsonTestCase(BackendBase):
    def setUp(self):
        self.set_preferred_backend('json')

    def test_backend(self):
        expected_pickled = (
            '{"things": [{'
            '"py/object": "backend_test.Thing", '
            '"name": "data", '
            '"child": null} '
            ']}'
        )
        self.assert_roundtrip(expected_pickled)


class SimpleJsonTestCase(BackendBase):
    def setUp(self):
        self.set_preferred_backend('simplejson')

    def test_backend(self):
        expected_pickled = (
            '{"things": [{'
            '"py/object": "backend_test.Thing", '
            '"name": "data", '
            '"child": null}'
            ']}'
        )
        self.assert_roundtrip(expected_pickled)

    def test_decimal(self):
        # Default behavior: Decimal is preserved
        obj = decimal.Decimal(0.5)
        as_json = jsonpickle.dumps(obj)
        clone = jsonpickle.loads(as_json)
        assert isinstance(clone, decimal.Decimal)
        assert obj == clone

        # Custom behavior: we want to use simplejson's Decimal support.
        jsonpickle.set_encoder_options('simplejson', use_decimal=True, sort_keys=True)

        jsonpickle.set_decoder_options('simplejson', use_decimal=True)

        # use_decimal mode allows Decimal objects to pass-through to simplejson.
        # The end result is we get a simple '0.5' value as our json string.
        as_json = jsonpickle.dumps(obj, unpicklable=True, use_decimal=True)
        assert as_json == '0.5'
        # But when loading we get back a Decimal.
        clone = jsonpickle.loads(as_json)
        assert isinstance(clone, decimal.Decimal)

        # side-effect: floats become decimals too!
        obj = 0.5
        as_json = jsonpickle.dumps(obj)
        clone = jsonpickle.loads(as_json)
        assert isinstance(clone, decimal.Decimal)
        # options are persisted unless we disable them
        jsonpickle.set_encoder_options('simplejson', use_decimal=False)
        jsonpickle.set_decoder_options('simplejson', use_decimal=False)

    def test_sort_keys(self):
        jsonpickle.set_encoder_options('simplejson', sort_keys=True)
        b = BSlots()
        with pytest.raises(TypeError):
            jsonpickle.encode(b, keys=True, warn=True)
        # return encoder options to default
        jsonpickle.set_encoder_options('simplejson', sort_keys=False)


def has_module(module):
    try:
        __import__(module)
    except ImportError:
        warn(module + ' module not available for testing, ' 'consider installing')
        return False
    return True


class UJsonTestCase(BackendBase):
    def setUp(self):
        self.set_preferred_backend('ujson')

    def test_backend(self):
        expected_pickled = (
            '{"things":[{'
            r'"py\/object":"backend_test.Thing",'
            '"name":"data","child":null}'
            ']}'
        )
        self.assert_roundtrip(expected_pickled)


class YamlTestCase(BackendBase):
    def setUp(self):
        jsonpickle.ext.yaml.register()
        self.set_preferred_backend('yaml')

    def test_backend(self):
        expected_pickled = (
            'things:\n'
            '  - py/object: backend_test.Thing\n'
            '    name: data\n'
            '    child: null\n'
        )
        self.assert_roundtrip(expected_pickled)
