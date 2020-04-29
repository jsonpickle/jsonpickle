import jsonpickle


with open("test.json", "r") as f:
    readJson = f.read()
    getPerimeter = jsonpickle.decode(readJson, encodeFunctionItself=True)
    exec(getPerimeter)

print(getPerimeter(10, 10))
