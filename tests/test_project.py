import jsonpickle


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
