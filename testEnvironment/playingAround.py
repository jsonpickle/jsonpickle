__author__ = "Mihir Shrestha"

import jsonpickle


class Celsius:
    def __init__(self, temperature=0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self._temperature * 1.8) + 32

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = value

    def __str__(self):
        return str(self._temperature)


class CelsiusTest:
    def __init__(self, temperature=0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self._temperature * 1.8) + 32

    def getTemperature(self):
        return self._temperature

    @property
    def test(self):
        return 5

    def setTemperature(self, temperature):
        if temperature < -273:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = temperature

    def __str__(self):
        return str(self._temperature)


# Creating two objects from the two different classes; Celsius has @property, CelsiusTest does not.
c = Celsius(25)
cTest = CelsiusTest(25)

# Encoding the two classes
encodedCelsius = jsonpickle.encode(Celsius)
encodedCelsiusTest = jsonpickle.encode(CelsiusTest)

# Decoding the two encoded classes
decodedCelsius = jsonpickle.decode(encodedCelsius)
decodedCelsiusTest = jsonpickle.decode(encodedCelsiusTest)

# Creating two new objects with the decoded classes
c1 = decodedCelsius(25)
cTest1 = decodedCelsiusTest(25)

# Checking to see if getting the temperature works
print(c1.temperature)
print(cTest1.getTemperature())

# Altering the temperature to see if function works
c1.temperature = 30
cTest1.setTemperature(30)

print(c1.temperature)
print(cTest1.getTemperature())



