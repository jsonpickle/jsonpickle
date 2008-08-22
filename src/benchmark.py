#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import timeit

number = 1000

cjson = """\
import feedparser
import jsonpickle
import jsonpickle.tests.thirdparty_tests as test
doc = feedparser.parse(test.RSS_DOC)

jsonpickle._use_cjson()

pickled = jsonpickle.encode(doc)
unpickled = jsonpickle.decode(pickled)
if doc['feed']['title'] != unpickled['feed']['title']:
    print 'Not a match'
"""

print 'Using cjson'
cjson_test = timeit.Timer(stmt=cjson)
print "%.9f sec/pass " % (cjson_test.timeit(number=number) / number)


simplejson = """\
import feedparser
import jsonpickle
import jsonpickle.tests.thirdparty_tests as test
doc = feedparser.parse(test.RSS_DOC)

jsonpickle._use_simplejson()

pickled = jsonpickle.encode(doc)
unpickled = jsonpickle.decode(pickled)
if doc['feed']['title'] != unpickled['feed']['title']:
    print 'Not a match'
"""

print 'Using simplejson'
simplejson_test = timeit.Timer(stmt=simplejson)
print "%.9f sec/pass " % (simplejson_test.timeit(number=number) / number)

