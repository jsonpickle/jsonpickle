# Demonstration Code
import jsonpickle


# Function that shows null values not being encoded
def classExample():
    # Null Values Decoding
    class Student:
        def __init__(self):
            self.name = None
            self.age = 50
            self.hairColor = "Black"
            self.ethnicity = None
            self.eyeColor = "Black"
            self.enrolled = True
            self.x = [1, 2, 3, 4, 5, 6, 7]

        def __str__(self):
            try:
                return f"{self.name} is {self.age} years old"
            except Exception as e:
                print(e)
                return ""

    mihir = Student()

    withNull = jsonpickle.encode(mihir)
    withoutNull = jsonpickle.encode(mihir, nullValues=False)
    print("JSON Data with null info   :", withNull)
    print("JSON Data without null info:", withoutNull)

    withNullDecoded = jsonpickle.decode(withNull)
    withoutNullDecoded = jsonpickle.decode(withoutNull)
    print("Decoded object with null info   :", withNullDecoded)
    print("Decoded object without null info:", withoutNullDecoded)


# Function that shows null values not being encoded in a dict
def dictExample():
    sampleDict = {"name": "Mihir", "age": 21, "hairColor": "black", "eyeColor": None, "ethnicity": None, "isStudent": True}
    encodedDictWithNull = jsonpickle.encode(sampleDict, nullValues=True)
    encodedDictWithoutNull = jsonpickle.encode(sampleDict, nullValues=False)
    print(f"This is the encoded dict with null values   : {encodedDictWithNull}")
    print(f"This is the encoded dict without null values: {encodedDictWithoutNull}")


dictExample()
