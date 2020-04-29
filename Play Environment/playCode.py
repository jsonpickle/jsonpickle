import jsonpickle


with open("test.json", "r") as f:
    readJson = f.read()
    decoded = jsonpickle.decode(readJson, encodeFunctionItself=True)
    exec(decoded)

print(calculateArea(5, 10))

