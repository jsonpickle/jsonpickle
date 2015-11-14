# -*- coding: utf-8 -*-

import unittest
import datetime
import jsonpickle

from helper import SkippableTest

try:
    import numpy as np
    from numpy.compat import asbytes
    from numpy.testing import assert_equal
except ImportError:
    np = None


class NumpyTestCase(SkippableTest):

    def setUp(self):
        if np is None:
            self.should_skip = True
            return
        self.should_skip = False
        import jsonpickle.ext.numpy
        jsonpickle.ext.numpy.register_handlers()

    def tearDown(self):
        if self.should_skip:
            return
        import jsonpickle.ext.numpy
        jsonpickle.ext.numpy.unregister_handlers()

    def roundtrip(self, obj):
        return jsonpickle.decode(jsonpickle.encode(obj))

    def test_dtype_roundtrip(self):
        if self.should_skip:
            return self.skip('numpy is not importable')
        dtypes = [
            np.int,
            np.float,
            np.complex,
            np.int32,
            np.str,
            np.object,
            np.unicode,
            np.dtype([('f0', 'i4'), ('f1', 'i1')]),
            np.dtype('1i4', align=True),
            np.dtype('M8[7D]'),
            np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                               ('rtile', '>f4', (64, 36))], (3,)),
                      ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                  ('bright', '>f4', (8, 36))])]),
            np.dtype({'names': ['f0', 'f1', 'f2'],
                      'formats': ['<u4', '<u2', '<u2'],
                      'offsets':[0, 0, 2]}, align=True)
        ]
        for dtype in dtypes:
            self.assertEqual(self.roundtrip(dtype), dtype)

    def test_generic_roundtrip(self):
        if self.should_skip:
            return self.skip('numpy is not importable')
        values = [
            np.int_(1),
            np.int32(-2),
            np.float_(2.5),
            np.nan,
            -np.inf,
            np.inf,
            np.datetime64('2014-01-01'),
            np.str_('foo'),
            np.unicode_('bar'),
            np.object_({'a': 'b'}),
            np.complex_(1 - 2j)
        ]
        for value in values:
            decoded = self.roundtrip(value)
            assert_equal(decoded, value)
            self.assertTrue(isinstance(decoded, type(value)))

    def test_ndarray_roundtrip(self):
        if self.should_skip:
            return self.skip('numpy is not importable')
        arrays = [
            np.random.random((10, 20)),
            np.array([[True, False, True]]),
            np.array(['foo', 'bar']),
            np.array([b'baz']),
            np.array(['2010', 'NaT', '2030']).astype('M'),
            np.rec.array(asbytes('abcdefg') * 100, formats='i2,a3,i4',
                         shape=3, byteorder='big'),
            np.rec.array([(1, 11, 'a'), (2, 22, 'b'),
                          (3, 33, 'c'), (4, 44, 'd'),
                          (5, 55, 'ex'), (6, 66, 'f'),
                          (7, 77, 'g')],
                         formats='u1,f4,a1'),
            np.array(['1960-03-12', datetime.date(1960, 3, 12)],
                     dtype='M8[D]'),
            np.array([0, 1, -1, np.inf, -np.inf, np.nan], dtype='f2'),
            np.rec.array([('NGC1001', 11), ('NGC1002', 1.), ('NGC1003', 1.)],
                         dtype=[('target', 'S20'), ('V_mag', '>f4')])
        ]
        for array in arrays:
            decoded = self.roundtrip(array)
            assert_equal(decoded, array)
            self.assertEqual(decoded.dtype, array.dtype)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NumpyTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
