import sys

if sys.version_info >= (3, 9):
    import unittest
    from zoneinfo import ZoneInfo

    import jsonpickle

    class ZoneInfoSimpleTestCase(unittest.TestCase):
        def _roundtrip(self, obj):
            """
            pickle and then unpickle object, then assert the new object is the
            same as the original.
            """
            pickled = jsonpickle.encode(obj)
            unpickled = jsonpickle.decode(pickled)
            self.assertEqual(obj, unpickled)

        def test_zoneinfo(self):
            """
            jsonpickle should pickle a zoneinfo object
            """
            self._roundtrip(ZoneInfo("Australia/Brisbane"))

    def suite():
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(ZoneInfoSimpleTestCase))
        return suite

    if __name__ == '__main__':
        unittest.main(defaultTest='suite')
