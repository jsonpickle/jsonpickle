# -*- coding: utf-8 -*-

import unittest
import datetime
import warnings

import jsonpickle

from helper import SkippableTest

try:
    import numpy as np
    import numpy.testing as npt
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

    def test_shape(self):
        """test that shapes containing zeros, which cannot be represented as nested lists, are deserialized correctly"""
        a = np.eye(3)[3:]
        _a = self.roundtrip(a)
        npt.assert_array_equal(a, _a)

    def test_accuracy(self):
        """test if the string representation maintains accuracy"""
        rand = np.random.randn(3, 3)
        _rand = self.roundtrip(rand)
        npt.assert_array_equal(rand, _rand)

    def test_b64(self):
        """test that binary encoding works"""
        a = np.random.rand(10, 10)    # array of substantial size is stored as b64
        _a = self.roundtrip(a)
        print(jsonpickle.encode(a))
        npt.assert_array_equal(a, _a)

    def test_views(self):
        """Test that views are maintained under serialization"""
        rng = np.arange(20)  # a range of an array
        view = rng[10:]  # a view referencing a portion of an array
        data = [rng, view]

        _data = self.roundtrip(data)

        _data[0][15] = -1
        assert _data[1][5] == -1

    def test_strides(self):
        """test that cases with non-standard strides and offsets work correctly"""
        arr = np.eye(3)
        view = arr[1:, 1:]
        assert view.base is arr
        data = [arr, view]

        _data = self.roundtrip(data)

        # test that the deserialized arrays indeed view the same memory
        _arr, _view = _data
        _arr[1, 2] = -1
        assert _view[0, 1] == -1
        assert _view.base is _arr

    def test_weird_arrays(self):
        """test that we disallow serialization of references to arrays that do not effectively own their memory"""
        a = np.arange(9)
        b = a[5:]
        a.strides = 1

        # this is kinda fishy; a has overlapping memory, _a does not
        with warnings.catch_warnings(record=True) as w:
            _a = self.roundtrip(a)
            # assert len(w) == 1
            npt.assert_array_equal(a, _a)

        # this also requires a deepcopy to work
        with warnings.catch_warnings(record=True) as w:
            _a, _b = self.roundtrip([a, b])
            assert len(w) == 1
            npt.assert_array_equal(a, _a)
            npt.assert_array_equal(b, _b)

    def test_transpose(self):
        """test handling of non-c-contiguous memory layout"""
        # simple case; view a c-contiguous array
        a = np.arange(9).reshape(3, 3)
        b = a[1:, 1:]
        assert b.base is a.base
        _a, _b = self.roundtrip([a, b])
        assert _b.base is _a.base
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # a and b both view the same contiguous array
        a = np.arange(9).reshape(3, 3).T
        b = a[1:, 1:]
        assert b.base is a.base
        _a, _b = self.roundtrip([a, b])
        assert _b.base is _a.base
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # view an f-contiguous array
        a = a.copy()
        a.strides = a.strides[::-1]
        b = a[1:, 1:]
        assert b.base is a
        _a, _b = self.roundtrip([a, b])
        assert _b.base is _a
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # now a.data.contiguous is False; we have to make a deepcopy to make this work
        a = np.arange(8).reshape(2, 2, 2).copy()
        a.strides = a.strides[0], a.strides[2], a.strides[1]
        b = a[1:, 1:]
        assert b.base is a
        with warnings.catch_warnings(record=True) as w:
            _a, _b = self.roundtrip([a, b])
            assert len(w) == 1
            npt.assert_array_equal(a, _a)
            npt.assert_array_equal(b, _b)

    def test_buffer(self):
        """test behavior with memoryviews which are not ndarrays"""
        buffer = b'abcdefgh'
        a = np.frombuffer(buffer, dtype=np.byte)
        with warnings.catch_warnings(record=True) as w:
            _a = self.roundtrip(a)
            npt.assert_array_equal(a, _a)
            assert len(w) == 1

    def test_as_strided(self):
        """test object with array interface which isnt an ndarray, like the result of as_strided"""
        a = np.arange(10)
        b = np.lib.stride_tricks.as_strided(a, shape=(5,), strides=(a.dtype.itemsize * 2,))
        data = [a, b]

        with warnings.catch_warnings(record=True) as w:
            # as_strided returns a DummyArray object, which we can not currently serialize correctly
            # FIXME: would be neat to add support for all objects implementing the __array_interface__
            _data = self.roundtrip(data)
            assert len(w) == 1

        # as we were warned, deserialized result is no longer a view
        with self.assertRaises(Exception):
            _data[0][0] = -1
            assert (_data[1][0] == -1)

    def test_immutable(self):
        """test that immutability flag is copied correctly"""

    # def test_newstrided(self):
    #     a = np.arange(10)
    #     q = np.ndarray(buffer=a.data, strides=a.strides*2, shape=(5,6), dtype=a.dtype)
    #     assert q.base is a
    #     print(q)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NumpyTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
