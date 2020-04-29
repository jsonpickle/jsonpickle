import jsonpickle
import dill as pickle


with open("jsonWithDillStuff.json", "r") as f:
    pickle.dump(calculateArea, f)
