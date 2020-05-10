# issue number 275
# encoding the lambda function
# Marie Clancy
# file number 1

import jsonpickle

#random lambda expression
lambdaExpression1 = lambda x : x + 10
print(lambdaExpression1(8))

# filename
filename = "functionLambda.json"
# load filename in write mode
with open(filename, "w") as m:
    # encoding function... decode what we just wrote
    functionEncoded = jsonpickle.encode(lambdaExpression1, encodeFunctionItself=True)
    m.write(functionEncoded)

