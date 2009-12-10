
import unittest

import sphinxtogithub

class TestFileObject(object):

    before = """
    <title>Breathe's documentation &mdash; BreatheExample v0.0.1 documentation</title>
    <link rel="stylesheet" href="_static/default.css" type="text/css" />
    <link rel="stylesheet" href="_static/pygments.css" type="text/css" />
    """

    after = """
    <title>Breathe's documentation &mdash; BreatheExample v0.0.1 documentation</title>
    <link rel="stylesheet" href="static/default.css" type="text/css" />
    <link rel="stylesheet" href="static/pygments.css" type="text/css" />
    """

    def read(self):

        return self.before

    def write(self, text):

        self.written = text

class TestOpener(object):

    def __init__(self):

        self.file_object = TestFileObject()

    def __call__(self, name, readmode="r"):
        
        self.name = name

        return self.file_object



class TestFileHandler(unittest.TestCase):

    def testProcess(self):

        filepath = "filepath"
        
        opener = TestOpener()
        file_handler = sphinxtogithub.FileHandler(filepath, [], opener)

        file_handler.process()

        self.assertEqual(opener.file_object.written, TestFileObject.before)
        self.assertEqual(opener.name, filepath)

    def testProcessWithReplacers(self):

        filepath = "filepath"
        
        replacers = []
        replacers.append(sphinxtogithub.Replacer("_static/default.css", "static/default.css"))
        replacers.append(sphinxtogithub.Replacer("_static/pygments.css", "static/pygments.css"))

        opener = TestOpener()
        file_handler = sphinxtogithub.FileHandler(filepath, replacers, opener)

        file_handler.process()

        self.assertEqual(opener.file_object.written, TestFileObject.after)



def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestFileHandler("testProcess"))
    suite.addTest(TestFileHandler("testProcessWithReplacers"))

    return suite

