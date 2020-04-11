__author__ = "Mihir Shrestha"

import pytest
import jsonpickle


def test_initialTest():
    x = "hello"
    y = jsonpickle.decode(jsonpickle.encode(x))
    assert x == y
