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
    __slots__ = ["a2", "a1", "a3"]  # ruff: ignore[RUF023]

    def __init__(self):
        self.a2 = A()  # set attribs not in alphabetical order
        self.a1 = A()
        self.a3 = self.a1  # create a reference


class DecimalSubclass(decimal.Decimal):
    pass


SAMPLE_DATA = {"things": [Thing("data")]}


class BackendBase(SkippableTest):
    def _is_installed(self, backend):
        if not jsonpickle.util._is_installed(backend):
            return self.skip(f"{backend} not available; please install")

    def set_backend(self, *args):
        backend = args[0]

        self._is_installed(backend)

        jsonpickle.load_backend(*args)
        jsonpickle.set_preferred_backend(backend)

    def set_preferred_backend(self, backend):
        self._is_installed(backend)
        jsonpickle.set_preferred_backend(backend)

    def teardown_method(self):
        # always reset to default backend
        jsonpickle.set_preferred_backend("json")

    def assert_roundtrip(self, json_input):
        expect = SAMPLE_DATA
        actual = jsonpickle.decode(json_input)
        assert expect["things"][0].name == actual["things"][0].name
        assert expect["things"][0].child == actual["things"][0].child

        pickled = jsonpickle.encode(SAMPLE_DATA)
        actual = jsonpickle.decode(pickled)
        assert expect["things"][0].name == actual["things"][0].name
        assert expect["things"][0].child == actual["things"][0].child

    def test_None_dict_key(self):
        """Ensure that backends produce the same result for None dict keys"""
        data = {None: None}
        expect = {"null": None}
        pickle = jsonpickle.encode(data)
        actual = jsonpickle.decode(pickle)
        assert expect == actual

    def test_encode_with_indent_and_separators(self):
        obj = {
            "a": 1,
            "b": 2,
        }
        expect = '{\n    "a": 1,\n    "b": 2\n}'
        actual = jsonpickle.encode(obj, indent=4, separators=(",", ": "))
        assert expect == actual


class JsonTestCase(BackendBase):
    def setup_method(self):
        self.set_preferred_backend("json")

    def test_backend(self):
        expected_pickled = (
            '{"things": [{'
            '"py/object": "backend_test.Thing", '
            '"name": "data", '
            '"child": null} '
            "]}"
        )
        self.assert_roundtrip(expected_pickled)


class SimpleJsonTestCase(BackendBase):
    def setup_method(self):
        self.set_preferred_backend("simplejson")

    def test_backend(self):
        expected_pickled = (
            '{"things": [{'
            '"py/object": "backend_test.Thing", '
            '"name": "data", '
            '"child": null}'
            "]}"
        )
        self.assert_roundtrip(expected_pickled)

    def test_decimal(self):
        # Default behavior: Decimal is preserved
        obj = decimal.Decimal(0.5)  # ruff: ignore[RUF032]
        as_json = jsonpickle.dumps(obj)
        clone = jsonpickle.loads(as_json)
        assert isinstance(clone, decimal.Decimal)
        assert obj == clone

        # Custom behavior: we want to use simplejson's Decimal support.
        jsonpickle.set_encoder_options("simplejson", use_decimal=True, sort_keys=True)

        jsonpickle.set_decoder_options("simplejson", use_decimal=True)

        # use_decimal mode allows Decimal objects to pass-through to simplejson.
        # The end result is we get a simple '0.5' value as our json string.
        as_json = jsonpickle.dumps(obj, unpicklable=True, use_decimal=True)
        assert as_json == "0.5"
        # But when loading we get back a Decimal.
        clone = jsonpickle.loads(as_json)
        assert isinstance(clone, decimal.Decimal)

        # side-effect: floats become decimals too!
        obj = 0.5
        as_json = jsonpickle.dumps(obj)
        clone = jsonpickle.loads(as_json)
        assert isinstance(clone, decimal.Decimal)
        # options are persisted unless we disable them
        jsonpickle.set_encoder_options("simplejson", use_decimal=False)
        jsonpickle.set_decoder_options("simplejson", use_decimal=False)

    def test_sort_keys(self):
        jsonpickle.set_encoder_options("simplejson", sort_keys=True)
        b = BSlots()
        with pytest.raises(TypeError):
            jsonpickle.encode(b, keys=True, warn=True)
        # return encoder options to default
        jsonpickle.set_encoder_options("simplejson", sort_keys=False)


def has_module(module):
    try:
        __import__(module)
    except ImportError:
        warn(module + " module not available for testing, consider installing")
        return False
    return True


class UJsonTestCase(BackendBase):
    def setup_method(self):
        self.set_preferred_backend("ujson")

    def test_backend(self):
        expected_pickled = (
            '{"things":[{'
            r'"py\/object":"backend_test.Thing",'
            '"name":"data","child":null}'
            "]}"
        )
        self.assert_roundtrip(expected_pickled)


class YamlTestCase(BackendBase):
    def setup_method(self):
        jsonpickle.ext.yaml.register()
        self.set_preferred_backend("yaml")

    def teardown_method(self):
        jsonpickle.remove_backend("yaml")
        super().teardown_method()

    def test_backend(self):
        expected_pickled = (
            "things:\n"
            "  - py/object: backend_test.Thing\n"
            "    name: data\n"
            "    child: null\n"
        )
        self.assert_roundtrip(expected_pickled)


@pytest.fixture
def simplejson_use_decimal():
    """Serve Decimal via simplejson's use_decimal mode, then restore defaults"""
    if not jsonpickle.util._is_installed("simplejson"):
        pytest.skip("simplejson not available; please install")
    jsonpickle.set_preferred_backend("simplejson")
    jsonpickle.set_encoder_options("simplejson", use_decimal=True, sort_keys=True)
    jsonpickle.set_decoder_options("simplejson", use_decimal=True)
    try:
        yield
    finally:
        jsonpickle.set_encoder_options("simplejson", use_decimal=False, sort_keys=False)
        jsonpickle.set_decoder_options("simplejson", use_decimal=False)
        jsonpickle.set_preferred_backend("json")


@pytest.fixture
def decimal_passthrough():
    """
    Register PassthroughHandler for Decimal and all of its subclasses
    """
    jsonpickle.handlers.register(
        decimal.Decimal, jsonpickle.handlers.PassthroughHandler, base=True
    )
    try:
        yield
    finally:
        jsonpickle.handlers.unregister(decimal.Decimal)


def test_decimal_passthrough_handler(simplejson_use_decimal, decimal_passthrough):
    """
    Ensure that PassthroughHandler reproduces use_decimal=True without needing the
    encode() flag that was removed in jsonpickle v5.
    """
    obj = decimal.Decimal("0.5")

    assert jsonpickle.dumps(obj, unpicklable=True) == "0.5"
    clone = jsonpickle.loads(jsonpickle.dumps(obj))
    assert isinstance(clone, decimal.Decimal)
    assert obj == clone

    assert jsonpickle.dumps(DecimalSubclass("0.5")) == "0.5"

    assert jsonpickle.dumps({"a": obj}) == '{"a": 0.5}'
    assert jsonpickle.dumps({"a": {"b": [obj]}}) == '{"a": {"b": [0.5]}}'
    assert jsonpickle.dumps([obj, decimal.Decimal("2.1")]) == "[0.5, 2.1]"

    shared = decimal.Decimal("2.1")
    values = [
        decimal.Decimal("2.1"),
        [decimal.Decimal("2.1"), decimal.Decimal("0.5")],
        # ensure distinct instances of equal value are not deduplicated by either mode
        [decimal.Decimal("2.1"), decimal.Decimal("2.1")],
        # nor should the same instance be turned into a py/id reference
        [shared, shared],
        {"a": decimal.Decimal("2.1")},
        {"a": {"b": [decimal.Decimal("2.1")]}},
        {decimal.Decimal("2.1"): "a"},
        DecimalSubclass("2.1"),
    ]
    for value in values:
        with_handler = jsonpickle.dumps(value)
        jsonpickle.handlers.unregister(decimal.Decimal)
        try:
            with_flag = jsonpickle.dumps(value, use_decimal=True)
        finally:
            jsonpickle.handlers.register(
                decimal.Decimal, jsonpickle.handlers.PassthroughHandler, base=True
            )
        assert with_handler == with_flag, value


def test_decimal_passthrough_repeated_instance(
    simplejson_use_decimal, decimal_passthrough
):
    """
    Make sure that the same Decimal instance repeats instead of becoming a py/id ref
    when make_refs and/or unpickleable are set to False. This guards against a bug
    encountered when drafting the passthrough handler, as it interacted with make_refs.

    Bug: A handler is dispatched after _mkref() has already logged a reference, so
    without the newly added _unlog_ref(), the second occurrence would encode as a py/id
    entry, which is a reference the backend-level decode cannot resolve, which would give
    us None instead of the original value.
    """
    shared = decimal.Decimal("2.1")

    assert jsonpickle.dumps([shared, shared]) == "[2.1, 2.1]"
    assert jsonpickle.loads(jsonpickle.dumps([shared, shared])) == [shared, shared]

    for kwargs in ({"make_refs": False}, {"unpicklable": False}):
        encoded = jsonpickle.dumps([shared, shared], **kwargs)
        assert encoded == "[2.1, 2.1]"
        assert jsonpickle.loads(encoded) == [shared, shared]


def test_contiguous_decimal_passthrough_ref_ids(
    simplejson_use_decimal, decimal_passthrough
):
    """
    Skipping a passthrough ref shouldn't shift py/id numbering
    """
    shared = decimal.Decimal("2.1")
    first, second = Thing("a"), Thing("b")

    # the decimals consume no reference IDs, so the Things should still number 1 and 2
    encoded = jsonpickle.dumps([shared, first, shared, second, first, second])
    assert encoded.endswith('{"py/id": 1}, {"py/id": 2}]')

    restored = jsonpickle.loads(encoded)
    assert restored[0] == restored[2] == shared
    # references still resolve to the same object, not a copy or None
    assert restored[4] is restored[1]
    assert restored[5] is restored[3]


def test_decimal_passthrough_ref_ids(simplejson_use_decimal, decimal_passthrough):
    """
    Ensure that interleaving passthrough and referenced objects matches use_decimal
    """

    def build():
        shared = decimal.Decimal("2.1")
        thing = Thing("a")
        return [shared, thing, shared, thing, {"t": thing, "d": shared}]

    with_handler = jsonpickle.dumps(build())
    jsonpickle.handlers.unregister(decimal.Decimal)
    try:
        with_flag = jsonpickle.dumps(build(), use_decimal=True)
    finally:
        jsonpickle.handlers.register(
            decimal.Decimal, jsonpickle.handlers.PassthroughHandler, base=True
        )
    assert with_handler == with_flag
