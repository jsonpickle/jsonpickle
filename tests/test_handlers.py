# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Jason R. Coombs <jaraco@jaraco.com>
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import unittest

import jsonpickle


class CustomObject(object):
    "A class to be serialized by a custom handler"
    def __eq__(self, other):
        return True

class NullHandler(jsonpickle.handlers.BaseHandler):
    _handles = CustomObject,

    def flatten(self, obj, data):
        return data

    def restore(self, obj):
        return CustomObject()

class HandlerTests(unittest.TestCase):
    def roundtrip(self, ob):
        encoded = jsonpickle.encode(ob)
        decoded = jsonpickle.decode(encoded)
        self.assertEqual(decoded, ob)
        return decoded

    def test_references(self):
        """
        Ensure objects handled by a custom handler are properly dereferenced.
        """
        ob = CustomObject()
        # create a dictionary which contains several references to ob
        subject = dict(a=ob, b=ob, c=ob)
        # ensure the subject can be roundtripped
        new_subject = self.roundtrip(subject)
        assert new_subject['a'] is new_subject['b']
        assert new_subject['b'] is new_subject['c']

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(HandlerTests, 'test_references'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
