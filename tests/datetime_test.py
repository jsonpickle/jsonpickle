#
# Copyright (C) 2013 Jason R. Coombs <jaraco@jaraco.com>
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import datetime
import sys
import time
import unittest

import jsonpickle
from jsonpickle import tags


class ObjWithDate:
    def __init__(self):
        ts = datetime.datetime.now()
        self.data = dict(a='a', ts=ts)
        self.data_ref = dict(b='b', ts=ts)


# UTC implementation from Python 2.7 docs
class UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return datetime.timedelta()

    def tzname(self, dt):
        return 'UTC'

    def dst(self, dt):
        return datetime.timedelta()


utc = UTC()


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
        s = '<TimestampedVariable>\n'
        s += ' value: ' + str(self._value) + '\n'
        s += ' dt_read: ' + str(self._dt_read) + ' (%s ago)' % td_read + '\n'
        s += ' dt_write: ' + str(self._dt_write) + ' (%s ago)' % td_write + '\n'
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
    pvars['z'] = 1
    pvars['z2'] = 2
    pickled = jsonpickle.encode(pvars)
    obj = jsonpickle.decode(pickled)
    # ensure the references are valid
    assert obj['z']._dt_read is obj['z']._dt_write
    assert obj['z2']._dt_read is obj['z2']._dt_write
    # ensure the values are valid
    assert obj['z'].get() == 1
    assert obj['z2'].get() == 2
    # ensure get() updates _dt_read
    assert obj['z']._dt_read is not obj['z']._dt_write
    assert obj['z2']._dt_read is not obj['z2']._dt_write


def _roundtrip(obj):
    """Roundtrip encode and decode an object and assert equality"""
    pickled = jsonpickle.encode(obj)
    unpickled = jsonpickle.decode(pickled)
    assert obj == unpickled


def test_datetime():
    """Roundtrip datetime objects"""
    _roundtrip(datetime.datetime.now())


def test_date():
    """Roundtrip date objects"""
    _roundtrip(datetime.datetime.today())


def test_time():
    """Roundtrip time objects"""
    _roundtrip(datetime.datetime.now().time())


def test_timedelta():
    """Roundtrip timedelta objects"""
    _roundtrip(datetime.timedelta(days=3))


def test_utc():
    """Roundtrip datetime objectcs with UTC timezone info"""
    _roundtrip(datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=utc))


def test_unpickleable():
    """Date objects are human-readable strings when unpicklable is False"""
    obj = datetime.datetime.now()
    pickler = jsonpickle.pickler.Pickler(unpicklable=False)
    flattened = pickler.flatten(obj)
    assert obj.isoformat() == flattened


def test_object_with_datetime():
    test_obj = ObjWithDate()
    json = jsonpickle.encode(test_obj)
    test_obj_decoded = jsonpickle.decode(json)
    assert test_obj_decoded.data['ts'] == test_obj_decoded.data_ref['ts']


@unittest.skipIf(sys.version_info < (3, 9), 'only tested for python >= 3.9')
def test_datetime_with_zoneinfo():
    """Roundtrip datetime objects with ZoneInfo tzinfo"""
    from zoneinfo import ZoneInfo

    now = datetime.datetime.now()
    SaoPaulo = ZoneInfo('America/Sao_Paulo')
    NewYork = ZoneInfo('America/New_York')
    now_sp = now.replace(tzinfo=SaoPaulo)
    now_us = now.replace(tzinfo=NewYork)
    _roundtrip(now_sp)
    _roundtrip(now_us)


def test_struct_time():
    expect = time.struct_time([1, 2, 3, 4, 5, 6, 7, 8, 9])
    json = jsonpickle.encode(expect)
    actual = jsonpickle.decode(json)
    assert type(actual) is time.struct_time
    assert expect == actual


def test_struct_time_chars():
    pickler = jsonpickle.pickler.Pickler()
    unpickler = jsonpickle.unpickler.Unpickler()
    expect = time.struct_time('123456789')
    flattened = pickler.flatten(expect)
    actual = unpickler.restore(flattened)
    assert expect == actual


def test_datetime_structure():
    pickler = jsonpickle.pickler.Pickler()
    unpickler = jsonpickle.unpickler.Unpickler()
    obj = datetime.datetime.now()
    flattened = pickler.flatten(obj)
    assert tags.OBJECT in flattened
    assert '__reduce__' in flattened
    inflated = unpickler.restore(flattened)
    assert obj == inflated


def test_datetime_inside_int_keys_defaults():
    t = datetime.time(hour=10)
    s = jsonpickle.encode({1: t, 2: t})
    d = jsonpickle.decode(s)
    assert d['1'] == d['2']
    assert d['1'] is d['2']
    assert isinstance(d['1'], datetime.time)


def test_datetime_inside_int_keys_with_keys_enabled():
    t = datetime.time(hour=10)
    s = jsonpickle.encode({1: t, 2: t}, keys=True)
    d = jsonpickle.decode(s, keys=True)
    assert d[1] == d[2]
    assert d[1] is d[2]
    assert isinstance(d[1], datetime.time)


def test_datetime_repr_not_unpicklable():
    obj = datetime.datetime.now()
    pickler = jsonpickle.pickler.Pickler(unpicklable=False)
    flattened = pickler.flatten(obj)
    assert tags.REPR not in flattened
    assert tags.MODULE not in flattened
    assert tags.OBJECT not in flattened
    assert obj.isoformat() == flattened


def test_datetime_dict_keys_defaults():
    """Test that we handle datetime objects as keys."""
    datetime_dict = {datetime.datetime(2008, 12, 31): True}
    pickled = jsonpickle.encode(datetime_dict)
    expect = {'datetime.datetime(2008, 12, 31, 0, 0)': True}
    actual = jsonpickle.decode(pickled)
    assert expect == actual


def test_datetime_dict_keys_with_keys_enabled():
    """Test that we handle datetime objects as keys."""
    datetime_dict = {datetime.datetime(2008, 12, 31): True}
    pickled = jsonpickle.encode(datetime_dict, keys=True)
    expect = datetime_dict
    actual = jsonpickle.decode(pickled, keys=True)
    assert expect == actual
