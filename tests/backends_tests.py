from samples import Thing

import jsonpickle
import unittest

SAMPLE_DATA = {'things': [Thing('data')]}

class BackendTestCase(unittest.TestCase):

    def _is_installed(self, backend):
        if not jsonpickle.util.is_installed(backend):
            self.fail('%s module not available, please install' % backend)

    def set_preferred_backend(self, backend):
        self._is_installed(backend)
        jsonpickle.set_preferred_backend(backend)

    def tearDown(self):
        # always reset to default backend
        jsonpickle.set_preferred_backend('json')

    def assertEncodeDecode(self, expected_pickled):
        pickled = jsonpickle.encode(SAMPLE_DATA)

        self.assertEqual(expected_pickled, pickled)
        unpickled = jsonpickle.decode(pickled)
        self.assertEqual(SAMPLE_DATA['things'][0].name, unpickled['things'][0].name)


class JsonTestCase(BackendTestCase):
    def setUp(self):
        self.set_preferred_backend('json')

    def test(self):
        expected_pickled = '''
        {"things": [{"py/object": "samples.Thing", "name": "data", "child": null}]}
        '''.strip()
        self.assertEncodeDecode(expected_pickled)


class SimpleJsonTestCase(BackendTestCase):
    def setUp(self):
        self.set_preferred_backend('simplejson')

    def test(self):
        expected_pickled = '''
        {"things": [{"py/object": "samples.Thing", "name": "data", "child": null}]}
        '''.strip()
        self.assertEncodeDecode(expected_pickled)


class DemjsonTestCase(BackendTestCase):
    def setUp(self):
        self.set_preferred_backend('demjson')

    def test(self):
        expected_pickled = u'''
        {"things":[{"child":null,"name":"data","py/object":"samples.Thing"}]}
        '''.strip()
        self.assertEncodeDecode(expected_pickled)


class JsonlibTestCase(BackendTestCase):
    def setUp(self):
        self.set_preferred_backend('jsonlib')

    def test(self):
        expected_pickled = '''
        {"things":[{"py\/object":"samples.Thing","name":"data","child":null}]}
        '''.strip()
        self.assertEncodeDecode(expected_pickled)


class YajlTestCase(BackendTestCase):
    def setUp(self):
        self.set_preferred_backend('yajl')

    def test(self):
        expected_pickled = '''
        {"things":[{"py/object":"samples.Thing","name":"data","child":null}]}
        '''.strip()
        self.assertEncodeDecode(expected_pickled)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(JsonTestCase))
    suite.addTest(unittest.makeSuite(SimpleJsonTestCase))
    suite.addTest(unittest.makeSuite(DemjsonTestCase))
    suite.addTest(unittest.makeSuite(YajlTestCase))
    suite.addTest(unittest.makeSuite(JsonlibTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
