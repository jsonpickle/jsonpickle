import jsonpickle


class Node:
    def __init__(self, name):
        self._name = name
        self._children = []
        self._parent = None

    def add_child(self, child, index=-1):
        if index == -1:
            index = len(self._children)
        self._children.insert(index, child)
        child._parent = self


class Document(Node):
    def __init__(self, name):
        Node.__init__(self, name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        ret_str = 'Document "%s"\n' % self._name
        for c in self._children:
            ret_str += repr(c)
        return ret_str


class Question(Node):
    def __init__(self, name):
        Node.__init__(self, name)

    def __str__(self):
        return f'Question "{self._name}", parent: "{self._parent._name}"\n'

    def __repr__(self):
        return self.__str__()


class Section(Node):
    def __init__(self, name):
        Node.__init__(self, name)

    def __str__(self):
        ret_str = f'Section "{self._name}", parent: "{self._parent._name}"\n'
        for c in self._children:
            ret_str += repr(c)
        return ret_str

    def __repr__(self):
        return self.__str__()


def test_cyclical():
    """Test that we can pickle cyclical data structure

    This test is ensures that we can reference objects which
    first appear within a list (in other words, not a top-level
    object or attribute).  Later children will reference that
    object through its "_parent" field.

    This makes sure that we handle this case correctly.

    """
    document = Document('My Document')
    section1 = Section('Section 1')
    section2 = Section('Section 2')
    question1 = Question('Question 1')
    question2 = Question('Question 2')
    question3 = Question('Question 3')
    question4 = Question('Question 4')
    document.add_child(section1)
    document.add_child(section2)
    section1.add_child(question1)
    section1.add_child(question2)
    section2.add_child(question3)
    section2.add_child(question4)
    pickled = jsonpickle.encode(document)
    unpickled = jsonpickle.decode(pickled)
    assert str(document) == str(unpickled)
