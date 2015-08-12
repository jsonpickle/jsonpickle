# -*- coding: utf-8 -*-
"""Test miscellaneous objects from the standard library"""

import uuid
import unittest

import jsonpickle


class UUIDTestCase(unittest.TestCase):

    def test_random_uuid(self):
        u = uuid.uuid4()
        encoded = jsonpickle.encode(u)
        decoded = jsonpickle.decode(encoded)

        expect = u.hex
        actual = decoded.hex
        self.assertEqual(expect, actual)

    def test_known_uuid(self):
        hex = '28b56adbd18f44e2a5556bba2f23e6f6'
        exemplar = uuid.UUID(hex)
        encoded = jsonpickle.encode(exemplar)
        decoded = jsonpickle.decode(encoded)

        expect = hex
        actual = decoded.hex
        self.assertEqual(expect, actual)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(UUIDTestCase))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
