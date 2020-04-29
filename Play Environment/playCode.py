import jsonpickle


with open("test.json", "r") as f:
    readJson = f.read()
    calculateArea = jsonpickle.decode(readJson, encodeFunctionItself=True)
    exec(calculateArea)

print(calculateArea(5, 10))

