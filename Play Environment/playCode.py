# Demonstration Code
import jsonpickle


class Student:
    def __init__(self):
        self.name = None
        self.age = 50
        self.hairColor = "Black"
        self.ethnicity = None
        self.eyeColor = "Black"
        self.enrolled = True
        self.x = [1,2,3,4,5,6,7]


mihir = Student()

withNull = jsonpickle.encode(mihir)
print(withNull)

withoutNull = jsonpickle.encode(mihir, nullValues=False)
print(withoutNull)
