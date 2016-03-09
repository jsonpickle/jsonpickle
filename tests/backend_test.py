# -*- coding: utf-8 -*-

import unittest
from warnings import warn

import jsonpickle
from jsonpickle.compat import unicode
from jsonpickle.compat import PY2
from jsonpickle.compat import PY3
from jsonpickle.compat import PY32

from helper import SkippableTest


class Thing(object):

    def __init__(self, name):
        self.name = name
        self.child = None


SAMPLE_DATA = {'things': [Thing('data')]}


class BackendBase(SkippableTest):

    def _is_installed(self, backend):
        if not jsonpickle.util.is_installed(backend):
            return self.skip('%s not available; please install' % backend)

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

    def test_None_dict_key(self):
        """Ensure that backends produce the same result for None dict keys"""
        data = {None: None}
        expect = {'null': None}
        pickle = jsonpickle.encode(data)
        actual = jsonpickle.decode(pickle)
        self.assertEqual(expect, actual)


class JsonTestCase(BackendBase):
    def setUp(self):
        self.set_preferred_backend('json')

    def test_backend(self):
        expected_pickled = (
            '{"things": [{'
            '"py/object": "backend_test.Thing", '
            '"name": "data", '
            '"child": null} '
            ']}')
        self.assertEncodeDecode(expected_pickled)


class SimpleJsonTestCase(BackendBase):
    def setUp(self):
        if PY32:
            return
        self.set_preferred_backend('simplejson')

    def test_backend(self):
        if PY32:
            return self.skip('no simplejson for python3.2')
        expected_pickled = (
            '{"things": [{'
            '"py/object": "backend_test.Thing", '
            '"name": "data", '
            '"child": null}'
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


class DemjsonTestCase(BackendBase):

    def setUp(self):
        self.set_preferred_backend('demjson')

    def test_backend(self):
        expected_pickled = unicode(
            '{"things":[{'
            '"child":null,'
            '"name":"data",'
            '"py/object":"backend_test.Thing"}'
            ']}')
        self.assertEncodeDecode(expected_pickled)

    def test_int_dict_keys_with_numeric_keys(self):
        jsonpickle.set_encoder_options('demjson', strict=False)
        int_dict = {1000: [1, 2]}
        pickle = jsonpickle.encode(int_dict, numeric_keys=True)
        actual = jsonpickle.decode(pickle)
        self.assertEqual(actual[1000], [1, 2])


class JsonlibTestCase(BackendBase):
    def setUp(self):
        if PY2:
            self.set_preferred_backend('jsonlib')

    def test_backend(self):
        if PY3:
            return self.skip('no jsonlib for python3')
        expected_pickled = (
            '{"things":[{'
            '"py\/object":"backend_test.Thing",'
            '"name":"data","child":null}'
            ']}')
        self.assertEncodeDecode(expected_pickled)


class YajlTestCase(BackendBase):
    def setUp(self):
        self.set_preferred_backend('yajl')

    def test_backend(self):
        expected_pickled = (
            '{"things":[{'
            '"py/object":"backend_test.Thing",'
            '"name":"data","child":null}'
            ']}')
        self.assertEncodeDecode(expected_pickled)


class UJsonTestCase(BackendBase):

    def setUp(self):
        self.set_preferred_backend('ujson')

    def test_backend(self):
        expected_pickled = (
            '{"things":[{'
            '"py\/object":"backend_test.Thing",'
            '"name":"data","child":null}'
            ']}')
        self.assertEncodeDecode(expected_pickled)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(JsonTestCase))
    suite.addTest(unittest.makeSuite(UJsonTestCase))
    if not PY32:
        suite.addTest(unittest.makeSuite(SimpleJsonTestCase))
    if has_module('demjson'):
        suite.addTest(unittest.makeSuite(DemjsonTestCase))
    if has_module('yajl'):
        suite.addTest(unittest.makeSuite(YajlTestCase))
    if PY2:
        if has_module('jsonlib'):
            suite.addTest(unittest.makeSuite(JsonlibTestCase))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
