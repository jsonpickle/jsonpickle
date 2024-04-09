"""Test serializing ecdsa keys"""

import unittest

import pytest
from helper import SkippableTest

import jsonpickle


@pytest.fixture(scope='module', autouse=True)
def gmpy_extension():
    """Initialize the gmpy extension for this test module"""
    try:
        jsonpickle.ext.gmpy.register_handlers()
        yield  # control to the test function.
        jsonpickle.ext.gmpy.unregister_handlers()
    except AttributeError:
        pytest.skip(
            "gmpy was not detected, please try installing it for more complete tests!"
        )


class EcdsaTestCase(SkippableTest):
    def setUp(self):
        try:
            from ecdsa import NIST384p
            from ecdsa.keys import SigningKey

            self.NIST384p = NIST384p
            self.SigningKey = SigningKey
            self.should_skip = False
        except ImportError:
            self.should_skip = True

    def test_roundtrip(self):
        if self.should_skip:
            return self.skip('ecdsa module is not installed')

        message = 'test'.encode('utf-8')
        key_pair = self.SigningKey.generate(curve=self.NIST384p)
        sig = key_pair.sign(message)

        serialized = jsonpickle.dumps(key_pair.get_verifying_key())
        restored = jsonpickle.loads(serialized)
        self.assertTrue(restored.verify(sig, message))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(EcdsaTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
