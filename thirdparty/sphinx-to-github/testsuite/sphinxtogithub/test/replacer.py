
import unittest

import sphinxtogithub

class TestReplacer(unittest.TestCase):

    before = """
    <title>Breathe's documentation &mdash; BreatheExample v0.0.1 documentation</title>
    <link rel="stylesheet" href="_static/default.css" type="text/css" />
    <link rel="stylesheet" href="_static/pygments.css" type="text/css" />
    """

    after = """
    <title>Breathe's documentation &mdash; BreatheExample v0.0.1 documentation</title>
    <link rel="stylesheet" href="static/default.css" type="text/css" />
    <link rel="stylesheet" href="_static/pygments.css" type="text/css" />
    """

    def testReplace(self):

        replacer = sphinxtogithub.Replacer("_static/default.css", "static/default.css")
        self.assertEqual(replacer.process(self.before), self.after)


def testSuite():

    suite = unittest.TestSuite()

    suite.addTest(TestReplacer("testReplace"))

    return suite


