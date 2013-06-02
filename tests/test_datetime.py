# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Jason R. Coombs <jaraco@jaraco.com>
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import unittest
import datetime

import jsonpickle

from jsonpickle._samples import ObjWithDate


# UTC implementation from Python 2.7 docs
class UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        return datetime.timedelta()

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return datetime.timedelta()

utc = UTC()


class DateTimeTests(unittest.TestCase):
    def _roundtrip(self, obj):
        """
        pickle and then unpickle object, then assert the new object is the
        same as the original.
        """
        pickled = jsonpickle.encode(obj)
        unpickled = jsonpickle.decode(pickled)
        self.assertEquals(obj, unpickled)

    def test_datetime(self):
        """
        jsonpickle should pickle a datetime object
        """
        self._roundtrip(datetime.datetime.now())

    def test_date(self):
        """
        jsonpickle should pickle a date object
        """
        self._roundtrip(datetime.datetime.today())

    def test_time(self):
        """
        jsonpickle should pickle a time object
        """
        self._roundtrip(datetime.datetime.now().time())

    def test_timedelta(self):
        """
        jsonpickle should pickle a timedelta object
        """
        self._roundtrip(datetime.timedelta(days=3))

    def test_utc(self):
        """
        jsonpickle should be able to encode and decode a datetime with a
        simple, pickleable UTC tzinfo.
        """
        self._roundtrip(datetime.datetime.utcnow().replace(tzinfo=utc))

    def test_unpickleable(self):
        """
        If 'unpickleable' is set on the Pickler, the date objects should be
        simple, human-readable strings.
        """
        obj = datetime.datetime.now()
        pickler = jsonpickle.Pickler(unpicklable=False)
        flattened = pickler.flatten(obj)
        self.assertEqual(str(obj), flattened)

    def test_object_with_datetime(self):
        test_obj = ObjWithDate()
        json = jsonpickle.encode(test_obj)
        test_obj_decoded = jsonpickle.decode(json)
        self.assertEqual(test_obj_decoded.data['ts'],
                         test_obj_decoded.data_ref['ts'])



def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DateTimeTests, 'test_utc'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
