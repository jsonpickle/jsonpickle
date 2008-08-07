# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import doctest
import unittest
import tempfile
import os.path

import jsonpickle
from jsonpickle.tests.classes import Thing, DictSubclass

class PicklingTestCase(unittest.TestCase):
    def setUp(self):
        self.pickler = jsonpickle.pickler.Pickler()
        self.unpickler = jsonpickle.unpickler.Unpickler()
        
    def test_string(self):
        self.assertEqual('a string', self.pickler.flatten('a string'))
        self.assertEqual('a string', self.unpickler.restore('a string'))
    
    def test_unicode(self):
        self.assertEqual(u'a string', self.pickler.flatten(u'a string'))
        self.assertEqual(u'a string', self.unpickler.restore(u'a string'))
    
    def test_int(self):
        self.assertEqual(3, self.pickler.flatten(3))
        self.assertEqual(3, self.unpickler.restore(3))
    
    def test_float(self):
        self.assertEqual(3.5, self.pickler.flatten(3.5))
        self.assertEqual(3.5, self.unpickler.restore(3.5))
    
    def test_boolean(self):    
        self.assertTrue(self.pickler.flatten(True))
        self.assertFalse(self.pickler.flatten(False))
        self.assertTrue(self.unpickler.restore(True))
        self.assertFalse(self.unpickler.restore(False))
    
    def test_none(self):
        self.assertTrue(self.pickler.flatten(None) is None)
        self.assertTrue(self.unpickler.restore(None) is None)
    
    def test_list(self):
        # multiple types of values
        listA = [1, 35.0, 'value']
        self.assertEqual(listA, self.pickler.flatten(listA))
        self.assertEqual(listA, self.unpickler.restore(listA))
        # nested list
        listB = [40, 40, listA, 6]
        self.assertEqual(listB, self.pickler.flatten(listB))
        self.assertEqual(listB, self.unpickler.restore(listB))
        # 2D list
        listC = [[1, 2], [3, 4]]
        self.assertEqual(listC, self.pickler.flatten(listC))
        self.assertEqual(listC, self.unpickler.restore(listC))
        # empty list
        listD = []
        self.assertEqual(listD, self.pickler.flatten(listD))
        self.assertEqual(listD, self.unpickler.restore(listD))
        
    def test_set(self):
        setA = set(['orange', 'apple', 'grape'])
        self.assertEqual(setA, self.pickler.flatten(setA))
        self.assertEqual(list(setA), self.unpickler.restore(setA))
        
    def test_dict(self):
        dictA = {'key1': 1.0, 'key2': 20, 'key3': 'thirty'}
        self.assertEqual(dictA, self.pickler.flatten(dictA))
        self.assertEqual(dictA, self.unpickler.restore(dictA))
        dictB = {}
        self.assertEqual(dictB, self.pickler.flatten(dictB))
        self.assertEqual(dictB, self.unpickler.restore(dictB))
    
    def test_tuple(self):
        # currently all collections are converted to lists
        tupleA = (4, 16, 32)
        self.assertEqual(tupleA, self.pickler.flatten(tupleA))
        self.assertEqual(list(tupleA), self.unpickler.restore(tupleA))
        tupleB = (4,)
        self.assertEqual(tupleB, self.pickler.flatten(tupleB))
        self.assertEqual(list(tupleB), self.unpickler.restore(tupleB))
        
    def test_class(self):
        inst = Thing('test name')
        inst.child = Thing('child name') 
        
        flattened = self.pickler.flatten(inst)
        self.assertEqual('test name', flattened['name'])
        child = flattened['child']
        self.assertEqual('child name', child['name'])
        
        inflated = self.unpickler.restore(flattened)
        self.assertEqual('test name', inflated.name)
        self.assertTrue(type(inflated) is Thing)
        self.assertEqual('child name', inflated.child.name)
        self.assertTrue(type(inflated.child) is Thing)
    
    def test_classlist(self):
        array = [Thing('one'), Thing('two'), 'a string']
        
        flattened = self.pickler.flatten(array)
        self.assertEqual('one', flattened[0]['name'])
        self.assertEqual('two', flattened[1]['name'])
        self.assertEqual('a string', flattened[2])        
        
        inflated = self.unpickler.restore(flattened)
        self.assertEqual('one', inflated[0].name)
        self.assertTrue(type(inflated[0]) is Thing)
        self.assertEqual('two', inflated[1].name)
        self.assertTrue(type(inflated[1]) is Thing)
        self.assertEqual('a string', inflated[2])
        
    def test_classdict(self):
        dict = {'k1':Thing('one'), 'k2':Thing('two'), 'k3':3}
        
        flattened = self.pickler.flatten(dict)
        self.assertEqual('one', flattened['k1']['name'])
        self.assertEqual('two', flattened['k2']['name'])
        self.assertEqual(3, flattened['k3'])        
        
        inflated = self.unpickler.restore(flattened)
        self.assertEqual('one', inflated['k1'].name)
        self.assertTrue(type(inflated['k1']) is Thing)
        self.assertEqual('two', inflated['k2'].name)
        self.assertTrue(type(inflated['k2']) is Thing)
        self.assertEqual(3, inflated['k3'])
        
        #TODO show that non string keys fail
    
    def test_oldstyleclass(self):
        from pickle import _EmptyClass
        
        obj = _EmptyClass()
        obj.value = 1234
        
        flattened = self.pickler.flatten(obj)
        self.assertEqual(1234, flattened['value'])
        
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1234, inflated.value)
        
    def test_struct_time(self):
        from time import struct_time
        t = struct_time('123456789')
        
        flattened = self.pickler.flatten(t)
        self.assertEqual(['1', '2', '3', '4', '5', '6', '7', '8', '9'], flattened)
         
    def test_dictsubclass(self):
        obj = DictSubclass()
        obj['key1'] = 1
        
        flattened = self.pickler.flatten(obj)
        self.assertEqual({'key1': 1}, flattened['classdictitems__'])
        self.assertEqual(flattened['classname__'], 'DictSubclass')
        
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1, inflated['key1'])

    def test_dictsubclass_notunpickable(self):
        self.pickler.unpicklable = False
        
        obj = DictSubclass()
        obj['key1'] = 1
                
        flattened = self.pickler.flatten(obj)
        self.assertEqual(1, flattened['key1'])
        self.assertFalse(flattened.has_key('classdictitems__'))
        
        inflated = self.unpickler.restore(flattened)
        self.assertEqual(1, inflated['key1'])
    
    def test_collectionsubclass(self):
        pass
    
    def test_userobjects(self):
        pass


class JSONPickleTestCase(unittest.TestCase):
    def setUp(self):
        self.obj = Thing('A name')
        self.expected_json = '{"classname__": "Thing", "child": null, "name": "A name", "classmodule__": "jsonpickle.tests.classes"}' 
        
    def test_dumps(self):
        pickled = jsonpickle.dumps(self.obj)
        self.assertEqual(self.expected_json, pickled)
    
    def test_dumps_notunpicklable(self):
        pickled = jsonpickle.dumps(self.obj, unpicklable=False)
        self.assertEqual('{"name": "A name", "child": null}', pickled)
      
    def test_dump(self):
        # TODO test writing to file
        pass
    
    def test_loads(self):
        unpickled = jsonpickle.loads(self.expected_json)
        self.assertEqual(self.obj.name, unpickled.name)
        self.assertEqual(type(self.obj), type(unpickled))
    
    def test_load(self):
        # TODO test reading to file
        pass

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(PicklingTestCase))
    suite.addTest(unittest.makeSuite(JSONPickleTestCase))
    suite.addTest(doctest.DocTestSuite(jsonpickle.pickler))
    suite.addTest(doctest.DocTestSuite(jsonpickle.unpickler))
    suite.addTest(doctest.DocTestSuite(jsonpickle))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
