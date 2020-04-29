import jsonpickle
import dill as pickle


def calculateArea(length, width):
    return length * width


print(jsonpickle.encode(calculateArea))


with open("save.pkl", "wb") as f:
    pickle.dump(calculateArea, f)
