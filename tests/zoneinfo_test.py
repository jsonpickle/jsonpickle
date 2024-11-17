import sys

import jsonpickle


def _roundtrip(obj):
    """Verify object equality after encoding and decoding to/from jsonpickle"""
    pickled = jsonpickle.encode(obj)
    unpickled = jsonpickle.decode(pickled)
    assert obj == unpickled


def test_zoneinfo():
    """zoneinfo objects can roundtrip"""
    if sys.version_info < (3, 9):
        return
    from zoneinfo import ZoneInfo

    _roundtrip(ZoneInfo('Australia/Brisbane'))
