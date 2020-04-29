import inspect
import jsonpickle


def calculateArea(length, width):
    return length * width


print(jsonpickle.encode(calculateArea))
