import jsonpickle
from enum import Enum


# Simple Enum Class
class Type(Enum):
    ONE = 1
    TWO = 2


class TestEnumObject:
    def __init__(self, num, objectType):
        self.num = num
        self.type = objectType

    def __repr__(self):
        return "TestEnumObject(num={}, objectType={})".format(self.num, self.type)

    def __str__(self):
        return "TestEnumObject(num={}, objectType={})".format(self.num, self.type)


# Dummy class to test @property issue
class Dummy(object):
    def __init__(self, name="Dummy"):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def changeNameToDummy(self):
        self.name = "Dummy"


def test_initialTest():
    x = "hello"
    y = jsonpickle.decode(jsonpickle.encode(x))
    assert x == y


def test_propertyIssue():
    # Getting a decoded Dummy class
    decodedDummy = jsonpickle.decode(jsonpickle.encode(Dummy))

    # Creating a new dummy object with the parameter "Thanos"
    dummy = decodedDummy("Thanos")
    assert dummy.name == "Thanos"

    dummy.name = "Rocket"
    assert dummy.name == "Rocket"

    dummy.changeNameToDummy()
    assert dummy.name == "Dummy"


def test_dictionaryIdentity():
    x = {}
    y = [x, x]
    z = jsonpickle.decode(jsonpickle.encode(y))

    # Testing if it works with just the unencoded list
    y[0]["name"] = "David"
    assert y[0] == y[1]

    # Testing it it works with the encoded list
    z[0]["name"] = "David"
    assert z[0] == z[1]


def test_enumIssue():
    a = TestEnumObject(1, Type.ONE)
    b = TestEnumObject(2, Type.ONE)
    c = TestEnumObject(3, Type.TWO)
    d = TestEnumObject(1, Type.ONE)

    # Different enum values, so they both should not equal each other
    differentEnums = [a, c]
    differentEnumsDecoded = jsonpickle.decode(jsonpickle.encode(differentEnums))
    assert str(differentEnumsDecoded[0]) != str(differentEnumsDecoded[1])

    # Same enum values, but different numbers, so they should still not equal each other
    sameEnumsDifValues = [a, b]
    sameEnumsDifValuesDecoded = jsonpickle.decode(jsonpickle.encode(sameEnumsDifValues))
    assert str(sameEnumsDifValuesDecoded[0]) != str(sameEnumsDifValuesDecoded[1])

    # Exact same enum and num values, so both should equal other other.
    exactlySameEnums = [a, d]
    exactlySameEnumsDecoded = jsonpickle.decode(jsonpickle.encode(exactlySameEnums))
    assert str(exactlySameEnumsDecoded[0]) == str(exactlySameEnumsDecoded[1])
