import jsonpickle
from enum import Enum


# Simple Enum Class
class Type(Enum):
    ONE = 1
    TWO = 2


# Class to test Enum issue with
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

    def __str__(self):
        return "Dummy(name='{}')".format(str(self._name))


# Initial test to see if pytest works
def test_initialTest():
    x = "hello"
    y = jsonpickle.decode(jsonpickle.encode(x))
    assert x == y


# Test to see if the @property issue has been fixed
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


# Test to see if dictionary identity is preserved or not
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


# Test to see if enum issue has been resolved or not
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


# Test to see if the parameter to ignore null fields works in JSON representation
def test_removeNullFields():
    newDummy = Dummy("Mihir")
    newDummy.age = 21
    newDummy.ethnicity = "Asian"
    newDummy.children = None
    newDummy.phone = None
    newDummy.email = None

    # This returns all the attributes
    allAttributes = vars(newDummy)

    # Getting all the null attributes
    nullAttributes = []
    for attr in allAttributes:
        if getattr(newDummy, attr) is None:
            nullAttributes.append(attr)

    # Encoding dummy object by default that also includes null values
    withNullEncoded = jsonpickle.encode(newDummy)

    # Checking to see if the JSON representation contains all attribute values
    for attr in allAttributes:
        assert attr in withNullEncoded

    # Encoding dummy object that will does not include null values
    withoutNullEncoded = jsonpickle.encode(newDummy, nullValues=False)

    # Checking to see if the JSON representation contains only non-null attribute values
    for attr in allAttributes:
        if attr in nullAttributes:
            assert attr not in withoutNullEncoded
        else:
            assert attr in withoutNullEncoded


