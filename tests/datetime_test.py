#
# Copyright (C) 2013 Jason R. Coombs <jaraco@jaraco.com>
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import datetime
import time

import jsonpickle
from jsonpickle import tags


class ObjWithDate:
    def __init__(self):
        ts = datetime.datetime.now()  # ruff: ignore[DTZ005]
        self.data = {"a": "a", "ts": ts}
        self.data_ref = {"b": "b", "ts": ts}


# UTC implementation from Python 2.7 docs
class UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return datetime.timedelta()

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return datetime.timedelta()


utc = UTC()


# Payloads produced by jsonpickle before the fold flag was encoded.
LEGACY_DATETIME = (
    '{"py/object": "datetime.datetime", "__reduce__": '
    '[{"py/type": "datetime.datetime"}, ["B+cLBQEeAAAAAA=="]]}'
)
LEGACY_DATE = (
    '{"py/object": "datetime.date", "__reduce__": '
    '[{"py/type": "datetime.date"}, ["B+cLBQ=="]]}'
)
LEGACY_TIME = (
    '{"py/object": "datetime.time", "__reduce__": '
    '[{"py/type": "datetime.time"}, ["AR4PAeJA"]]}'
)


class MyDatetime(datetime.datetime):
    """datetime subclass; no handler is registered for it"""


class MyTime(datetime.time):
    """time subclass; no handler is registered for it"""


class TimestampedVariable:
    def __init__(self, value=None):
        self._value = value
        self._dt_read = datetime.datetime.now(tz=datetime.timezone.utc)
        self._dt_write = self._dt_read

    def get(self, default_value=None):
        if self._dt_read is None and self._dt_write is None:
            value = default_value
            self._value = value
            self._dt_write = datetime.datetime.now(tz=datetime.timezone.utc)
        else:
            value = self._value
        self._dt_read = datetime.datetime.now(tz=datetime.timezone.utc)
        return value

    def set(self, new_value):
        self._dt_write = datetime.datetime.now(tz=datetime.timezone.utc)
        self._value = new_value

    def erasable(self, td=datetime.timedelta(seconds=1)):
        dt_now = datetime.datetime.now(tz=datetime.timezone.utc)
        td_read = dt_now - self._dt_read
        td_write = dt_now - self._dt_write
        return td_read > td and td_write > td

    def __repr__(self):
        dt_now = datetime.datetime.now(tz=datetime.timezone.utc)
        td_read = dt_now - self._dt_read
        td_write = dt_now - self._dt_write
        s = f"<TimestampedVariable>\n value: {self._value}\n dt_read: {self._dt_read} ({td_read} ago)\n dt_write: {self._dt_write} ({td_write} ago)\n"
        return s


class PersistantVariables:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data.setdefault(key, TimestampedVariable(None))

    def __setitem__(self, key, value):
        return self._data.setdefault(key, TimestampedVariable(value))

    def __repr__(self):
        return str(self._data)


def test_object_with_inner_datetime_refs():
    pvars = PersistantVariables()
    pvars["z"] = 1
    pvars["z2"] = 2
    pickled = jsonpickle.encode(pvars)
    obj = jsonpickle.decode(pickled)
    # ensure the references are valid
    assert obj["z"]._dt_read is obj["z"]._dt_write
    assert obj["z2"]._dt_read is obj["z2"]._dt_write
    # ensure the values are valid
    assert obj["z"].get() == 1
    assert obj["z2"].get() == 2
    # ensure get() updates _dt_read
    assert obj["z"]._dt_read is not obj["z"]._dt_write
    assert obj["z2"]._dt_read is not obj["z2"]._dt_write


def _roundtrip(obj):
    """Roundtrip encode and decode an object and assert equality"""
    pickled = jsonpickle.encode(obj)
    unpickled = jsonpickle.decode(pickled)
    assert obj == unpickled


def test_datetime():
    """Roundtrip datetime objects"""
    _roundtrip(datetime.datetime.now())  # ruff: ignore[DTZ005]


def test_date():
    """Roundtrip date objects"""
    _roundtrip(datetime.datetime.today())  # ruff: ignore[DTZ002]


def test_time():
    """Roundtrip time objects"""
    _roundtrip(datetime.datetime.now().time())  # ruff: ignore[DTZ005]


def test_timedelta():
    """Roundtrip timedelta objects"""
    _roundtrip(datetime.timedelta(days=3))


def test_utc():
    """Roundtrip datetime objectcs with UTC timezone info"""
    _roundtrip(datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=utc))


def test_unpickleable():
    """Date objects are human-readable strings when unpicklable is False"""
    obj = datetime.datetime.now()  # ruff: ignore[DTZ005]
    pickler = jsonpickle.pickler.Pickler(unpicklable=False)
    flattened = pickler.flatten(obj)
    assert obj.isoformat() == flattened


def test_object_with_datetime():
    test_obj = ObjWithDate()
    json = jsonpickle.encode(test_obj)
    test_obj_decoded = jsonpickle.decode(json)
    assert test_obj_decoded.data["ts"] == test_obj_decoded.data_ref["ts"]


def test_datetime_with_zoneinfo():
    """Roundtrip datetime objects with ZoneInfo tzinfo"""
    from zoneinfo import ZoneInfo

    now = datetime.datetime.now()  # ruff: ignore[DTZ005]
    SaoPaulo = ZoneInfo("America/Sao_Paulo")
    NewYork = ZoneInfo("America/New_York")
    now_sp = now.replace(tzinfo=SaoPaulo)
    now_us = now.replace(tzinfo=NewYork)
    _roundtrip(now_sp)
    _roundtrip(now_us)


def test_datetime_fold():
    """Roundtrip preserves the PEP 495 fold flag"""
    for obj in (
        datetime.datetime(2023, 11, 5, 1, 30, fold=1),  # ruff: ignore[DTZ001]
        datetime.datetime(2023, 11, 5, 1, 30, tzinfo=utc, fold=1),
        datetime.time(1, 30, fold=1),
        datetime.time(1, 30, tzinfo=utc, fold=1),
    ):
        assert jsonpickle.decode(jsonpickle.encode(obj)).fold == 1
        assert jsonpickle.decode(jsonpickle.encode(obj.replace(fold=0))).fold == 0


def test_datetime_fold_is_encoded():
    """fold=0 and fold=1 do not encode to the same payload"""
    obj = datetime.datetime(2023, 11, 5, 1, 30)  # ruff: ignore[DTZ001]
    assert jsonpickle.encode(obj) != jsonpickle.encode(obj.replace(fold=1))
    t = datetime.time(1, 30)
    assert jsonpickle.encode(t) != jsonpickle.encode(t.replace(fold=1))


def test_datetime_fold_ambiguous_time():
    """During a DST fall-back the fold flag selects the actual instant"""
    from zoneinfo import ZoneInfo

    ambiguous = [
        (
            ZoneInfo("America/New_York"),
            datetime.datetime(2023, 11, 5, 1, 30),  # ruff: ignore[DTZ001]
        ),
        (
            ZoneInfo("Europe/Paris"),
            datetime.datetime(2023, 10, 29, 2, 30),  # ruff: ignore[DTZ001]
        ),
    ]
    for tz, naive in ambiguous:
        for fold in (0, 1):
            obj = naive.replace(tzinfo=tz, fold=fold)
            unpickled = jsonpickle.decode(jsonpickle.encode(obj))
            # __eq__ ignores fold for same-zone operands, so compare the instants
            assert unpickled.utcoffset() == obj.utcoffset()
            assert unpickled.tzname() == obj.tzname()
            assert unpickled.astimezone(datetime.timezone.utc) == obj.astimezone(
                datetime.timezone.utc
            )


def test_datetime_subclass_fold():
    """Subclasses go through py/reduce instead of the handler, but keep fold"""
    for obj in (
        MyDatetime(2023, 11, 5, 1, 30, fold=1),
        MyTime(1, 30, fold=1),
    ):
        unpickled = jsonpickle.decode(jsonpickle.encode(obj))
        assert type(unpickled) is type(obj)
        assert unpickled.fold == 1


def test_datetime_legacy_payloads():
    """Payloads written before fold was encoded still decode"""
    legacy = [
        (LEGACY_DATETIME, datetime.datetime(2023, 11, 5, 1, 30)),  # ruff: ignore[DTZ001]
        (LEGACY_DATE, datetime.date(2023, 11, 5)),
        (LEGACY_TIME, datetime.time(1, 30, 15, 123456)),
    ]
    for payload, expect in legacy:
        actual = jsonpickle.decode(payload)
        assert actual == expect
        assert type(actual) is type(expect)
        assert getattr(actual, "fold", 0) == 0


def test_struct_time():
    expect = time.struct_time([1, 2, 3, 4, 5, 6, 7, 8, 9])
    json = jsonpickle.encode(expect)
    actual = jsonpickle.decode(json)
    assert type(actual) is time.struct_time
    assert expect == actual


def test_struct_time_chars():
    pickler = jsonpickle.pickler.Pickler()
    unpickler = jsonpickle.unpickler.Unpickler()
    expect = time.struct_time("123456789")
    flattened = pickler.flatten(expect)
    actual = unpickler.restore(flattened)
    assert expect == actual


def test_datetime_structure():
    pickler = jsonpickle.pickler.Pickler()
    unpickler = jsonpickle.unpickler.Unpickler()
    obj = datetime.datetime.now()  # ruff: ignore[DTZ005]
    flattened = pickler.flatten(obj)
    assert tags.OBJECT in flattened
    assert "__reduce__" in flattened
    inflated = unpickler.restore(flattened)
    assert obj == inflated


def test_datetime_inside_int_keys_defaults():
    t = datetime.time(hour=10)
    s = jsonpickle.encode({1: t, 2: t})
    d = jsonpickle.decode(s)
    assert d["1"] == d["2"]
    assert d["1"] is d["2"]
    assert isinstance(d["1"], datetime.time)


def test_datetime_inside_int_keys_with_keys_enabled():
    t = datetime.time(hour=10)
    s = jsonpickle.encode({1: t, 2: t}, keys=True)
    d = jsonpickle.decode(s, keys=True)
    assert d[1] == d[2]
    assert d[1] is d[2]
    assert isinstance(d[1], datetime.time)


def test_datetime_repr_not_unpicklable():
    obj = datetime.datetime.now()  # ruff: ignore[DTZ005]
    pickler = jsonpickle.pickler.Pickler(unpicklable=False)
    flattened = pickler.flatten(obj)
    assert tags.REPR not in flattened
    assert tags.MODULE not in flattened
    assert tags.OBJECT not in flattened
    assert obj.isoformat() == flattened


def test_datetime_dict_keys_defaults():
    """Test that we handle datetime objects as keys."""
    datetime_dict = {datetime.datetime(2008, 12, 31): True}  # ruff: ignore[DTZ001]
    pickled = jsonpickle.encode(datetime_dict)
    expect = {"datetime.datetime(2008, 12, 31, 0, 0)": True}
    actual = jsonpickle.decode(pickled)
    assert expect == actual


def test_datetime_dict_keys_with_keys_enabled():
    """Test that we handle datetime objects as keys."""
    datetime_dict = {datetime.datetime(2008, 12, 31): True}  # ruff: ignore[DTZ001]
    pickled = jsonpickle.encode(datetime_dict, keys=True)
    expect = datetime_dict
    actual = jsonpickle.decode(pickled, keys=True)
    assert expect == actual
