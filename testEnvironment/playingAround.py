import jsonpickle


class T:
    def __init__(self, name):
        self.name = name
        self.age = None
        self.ethnicity = None
        self.hairColor = None
        self.height = None
        self.weight = None
        self.race = "Human"


t = T("Mihir")
encT = jsonpickle.encode(t, nullValues=False)
print(encT)
decT = jsonpickle.decode(encT)


