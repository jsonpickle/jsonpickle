"""Test serializing pymongo bson structures"""

import datetime
import json
import pickle

import pytest

try:
    import bson
    import bson.int64
    import bson.tz_util
except ImportError:
    pytest.skip("bson is not available", allow_module_level=True)

import jsonpickle


class Object:
    def __init__(self, offset):
        self.offset = datetime.timedelta(offset)

    def __getinitargs__(self):
        return (self.offset,)


def test_FixedOffsetSerializable():
    fo = bson.tz_util.FixedOffset(-60 * 5, "EST")
    serialized = jsonpickle.dumps(fo)
    restored = jsonpickle.loads(serialized)
    assert vars(restored) == vars(fo)


def test_timedelta():
    td = datetime.timedelta(-1, 68400)
    serialized = jsonpickle.dumps(td)
    restored = jsonpickle.loads(serialized)
    assert restored == td


def test_stdlib_pickle():
    fo = bson.tz_util.FixedOffset(-60 * 5, "EST")
    serialized = pickle.dumps(fo)
    restored = pickle.loads(serialized)
    assert vars(restored) == vars(fo)


def test_nested_objects():
    o = Object(99)
    serialized = jsonpickle.dumps(o)
    restored = jsonpickle.loads(serialized)
    assert restored.offset == datetime.timedelta(99)


def test_datetime_with_fixed_offset():
    fo = bson.tz_util.FixedOffset(-60 * 5, "EST")
    dt = datetime.datetime.now().replace(tzinfo=fo)
    serialized = jsonpickle.dumps(dt)
    restored = jsonpickle.loads(serialized)
    assert restored == dt


def test_datetime_with_fixed_offset_incremental():
    """Test creating an Unpickler and incrementally encoding"""
    obj = datetime.datetime(2019, 1, 29, 18, 9, 8, 826000, tzinfo=bson.tz_util.utc)
    doc = jsonpickle.dumps(obj)
    # Restore the json using a custom unpickler context.
    unpickler = jsonpickle.unpickler.Unpickler()
    jsonpickle.loads(doc, context=unpickler)
    # Incrementally restore using the same context
    clone = json.loads(doc, object_hook=lambda x: unpickler.restore(x, reset=False))
    assert obj.tzinfo.__reduce__() == clone.tzinfo.__reduce__()


def test_int64_roundtrip():
    value = bson.int64.Int64(9223372036854775807)
    string = jsonpickle.encode(value, handle_readonly=True)
    actual = jsonpickle.decode(string)
    assert value == actual
