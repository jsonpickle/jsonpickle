__author__ = "Mihir Shrestha"
import dill as pickle

with open("save.pkl", 'rb') as f:
    calculateArea = pickle.load(f)

print(calculateArea(5, 3))
