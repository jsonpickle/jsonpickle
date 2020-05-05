# Demonstration Code
import jsonpickle


# Function that shows null values not being encoded
def classExample():
    """
    For some classes, it can be much more concise to ignore/drop any field that is null. Could there be an optional parameter that enables this?
    """

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

    print("\nStart of class example:")
    withNull = jsonpickle.encode(mihir)
    withoutNull = jsonpickle.encode(mihir, nullValues=False)
    print("JSON Data with null info   :", withNull)
    print("JSON Data without null info:", withoutNull)

    withNullDecoded = jsonpickle.decode(withNull)
    withoutNullDecoded = jsonpickle.decode(withoutNull)
    print("Decoded object with null info   :", withNullDecoded)
    print("Decoded object without null info:", withoutNullDecoded)
    print()


# Function that shows null values not being encoded in a dict
def dictExample():
    print("Start of dict example:")
    sampleDict = {"name": "Mihir", "age": 21, "hairColor": "black", "eyeColor": None, "ethnicity": None,
                  "isStudent": True}
    encodedDictWithNull = jsonpickle.encode(sampleDict, nullValues=True)
    encodedDictWithoutNull = jsonpickle.encode(sampleDict, nullValues=False)
    print(f"This is the encoded dict with null values   : {encodedDictWithNull}")
    print(f"This is the encoded dict without null values: {encodedDictWithoutNull}")
    print()


# Function that shows functions can be encoded in JSON format
def functionExample():
    print("Start of function encoding example")

    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n - 1) + fibonacci(n - 2)

    filename = "functionDump.json"
    with open(filename, "w") as f:
        functionEncoded = jsonpickle.encode(fibonacci, encodeFunctionItself=True)
        f.write(functionEncoded)

    print("Encoded function:", functionEncoded)
    print(f"Function has been dumped to {filename}")
    print()


classExample()
dictExample()
functionExample()

with open("functionDump.json", 'r') as f:
    jsonCode = f.read()
    fibonacci = jsonpickle.decode(jsonCode, encodeFunctionItself=True)
    print(fibonacci)
    exec(fibonacci)

for i in range(10):
    print(fibonacci(i))
