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
    def test_utc(self):
        """
        jsonpickle should be able to encode and decode a datetime with a
        simple, pickleable UTC tzinfo.
        """
        now = datetime.datetime.utcnow().replace(tzinfo=utc)
        pickled = jsonpickle.encode(now)
        unpickled = jsonpickle.decode(pickled)
        self.assertEquals(now, unpickled)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DateTimeTests, 'test_utc'))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
