import sys
import unittest
from warnings import warn

import jsonpickle
from jsonpickle._samples import Thing
from jsonpickle.compat import unicode
from jsonpickle.compat import PY2
from jsonpickle.compat import PY3
from jsonpickle.compat import PY32

SAMPLE_DATA = {'things': [Thing('data')]}


class BackendTestCase(unittest.TestCase):

    def _is_installed(self, backend):
        if not jsonpickle.util.is_installed(backend):
            if hasattr(self, 'skipTest'):
                doit = self.skipTest
            else:
                doit = self.fail
            doit('%s not available; please install' % backend)

    def set_backend(self, *args):
        backend = args[0]

        self._is_installed(backend)

        jsonpickle.load_backend(*args)
        jsonpickle.set_preferred_backend(backend)

    def set_preferred_backend(self, backend):
        self._is_installed(backend)
        jsonpickle.set_preferred_backend(backend)

    def tearDown(self):
        # always reset to default backend
        jsonpickle.set_preferred_backend('json')

    def assertEncodeDecode(self, json_input):
        expect = SAMPLE_DATA
        actual = jsonpickle.decode(json_input)
        self.assertEqual(expect['things'][0].name, actual['things'][0].name)
        self.assertEqual(expect['things'][0].child, actual['things'][0].child)

        pickled = jsonpickle.encode(SAMPLE_DATA)
        actual = jsonpickle.decode(pickled)
        self.assertEqual(expect['things'][0].name, actual['things'][0].name)
        self.assertEqual(expect['things'][0].child, actual['things'][0].child)


class JsonTestCase(BackendTestCase):
    def setUp(self):
        self.set_preferred_backend('json')

    def test_backend(self):
        expected_pickled = (
                '{"things": [{'
                    '"py/object": "jsonpickle._samples.Thing",'
                    ' "name": "data",'
                    ' "child": null}'
                ']}')
        self.assertEncodeDecode(expected_pickled)


class SimpleJsonTestCase(BackendTestCase):
    def setUp(self):
        if PY32:
            return
        self.set_preferred_backend('simplejson')

    def test_backend(self):
        if PY32:
            self.skipTest('no simplejson for python3.2')
            return
        expected_pickled = (
                '{"things": [{'
                    '"py/object": "jsonpickle._samples.Thing",'
                    ' "name": "data",'
                    ' "child": null}'
                ']}')
        self.assertEncodeDecode(expected_pickled)


def has_module(module):
    try:
        __import__(module)
    except ImportError:
        warn(module + ' module not available for testing, '
             'consider installing')
        return False
    return True


class DemjsonTestCase(BackendTestCase):
    def setUp(self):
        if PY2:
            self.set_preferred_backend('demjson')

    def test_backend(self):
        if PY3:
            self.skipTest('no demjson for python3')
            return
        expected_pickled = unicode(
                '{"things":[{'
                    '"child":null,'
                    '"name":"data",'
                    '"py/object":"jsonpickle._samples.Thing"}'
                ']}')
        self.assertEncodeDecode(expected_pickled)


class JsonlibTestCase(BackendTestCase):
    def setUp(self):
        if PY2:
            self.set_preferred_backend('jsonlib')

    def test_backend(self):
        if PY3:
            self.skipTest('no jsonlib for python3')
            return
        expected_pickled = (
                '{"things":[{'
                    '"py\/object":"jsonpickle._samples.Thing",'
                    '"name":"data","child":null}'
                ']}')
        self.assertEncodeDecode(expected_pickled)


class YajlTestCase(BackendTestCase):
    def setUp(self):
        if PY2:
            self.set_preferred_backend('yajl')

    def test_backend(self):
        if PY3:
            self.skipTest('no yajl for python3')
            return
        expected_pickled = (
                '{"things":[{'
                    '"py/object":"jsonpickle._samples.Thing",'
                    '"name":"data","child":null}'
                ']}')
        self.assertEncodeDecode(expected_pickled)


class UJsonTestCase(BackendTestCase):

    def setUp(self):
        self.set_preferred_backend('ujson')

    def test_backend(self):
        expected_pickled = (
                '{"things":[{'
                    '"py\/object":"jsonpickle._samples.Thing",'
                    '"name":"data","child":null}'
                ']}')
        self.assertEncodeDecode(expected_pickled)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(JsonTestCase))
    suite.addTest(unittest.makeSuite(UJsonTestCase))
    if not PY32:
        suite.addTest(unittest.makeSuite(SimpleJsonTestCase))
    if PY2:
        if has_module('demjson'):
            suite.addTest(unittest.makeSuite(DemjsonTestCase))
        if has_module('yajl'):
            suite.addTest(unittest.makeSuite(YajlTestCase))
        if has_module('jsonlib'):
            suite.addTest(unittest.makeSuite(JsonlibTestCase))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
