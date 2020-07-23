from __future__ import absolute_import, division, unicode_literals
import unittest
import datetime
import warnings

import jsonpickle
from jsonpickle import handlers
from jsonpickle.compat import PY2

from helper import SkippableTest

try:
    import numpy as np
    import numpy.testing as npt
    from numpy.compat import asbytes
    from numpy.testing import assert_equal

    from jsonpickle.ext import numpy as numpy_ext
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
            np.dtype('f4,i4,f2,i1'),
            np.dtype(('f4', 'i4'), ('f2', 'i1')),
            np.dtype('1i4', align=True),
            np.dtype('M8[7D]'),
            np.dtype(
                {
                    'names': ['f0', 'f1', 'f2'],
                    'formats': ['<u4', '<u2', '<u2'],
                    'offsets': [0, 0, 2],
                },
                align=True,
            ),
        ]

        if not PY2:
            dtypes.extend(
                [
                    np.dtype([('f0', 'i4'), ('f2', 'i1')]),
                    np.dtype(
                        [
                            (
                                'top',
                                [
                                    ('tiles', ('>f4', (64, 64)), (1,)),
                                    ('rtile', '>f4', (64, 36)),
                                ],
                                (3,),
                            ),
                            (
                                'bottom',
                                [
                                    ('bleft', ('>f4', (8, 64)), (1,)),
                                    ('bright', '>f4', (8, 36)),
                                ],
                            ),
                        ]
                    ),
                ]
            )

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
            np.complex_(1 - 2j),
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
            np.array(['baz'.encode('utf-8')]),
            np.array(['2010', 'NaT', '2030']).astype('M'),
            np.rec.array(asbytes('abcdefg') * 100, formats='i2,a3,i4', shape=3),
            np.rec.array(
                [
                    (1, 11, 'a'),
                    (2, 22, 'b'),
                    (3, 33, 'c'),
                    (4, 44, 'd'),
                    (5, 55, 'ex'),
                    (6, 66, 'f'),
                    (7, 77, 'g'),
                ],
                formats='u1,f4,a1',
            ),
            np.array(['1960-03-12', datetime.date(1960, 3, 12)], dtype='M8[D]'),
            np.array([0, 1, -1, np.inf, -np.inf, np.nan], dtype='f2'),
        ]

        if not PY2:
            arrays.extend(
                [
                    np.rec.array(
                        [('NGC1001', 11), ('NGC1002', 1.0), ('NGC1003', 1.0)],
                        dtype=[('target', 'S20'), ('V_mag', 'f4')],
                    )
                ]
            )
        for array in arrays:
            decoded = self.roundtrip(array)
            assert_equal(decoded, array)
            self.assertEqual(decoded.dtype, array.dtype)

    def test_shapes_containing_zeroes(self):
        """Test shapes which cannot be represented as nested lists"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.eye(3)[3:]
        _a = self.roundtrip(a)
        npt.assert_array_equal(a, _a)

    def test_accuracy(self):
        """Test the accuracy of the string representation"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        rand = np.random.randn(3, 3)
        _rand = self.roundtrip(rand)
        npt.assert_array_equal(rand, _rand)

    def test_b64(self):
        """Test the binary encoding"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.random.rand(10, 10)  # array of substantial size is stored as b64
        _a = self.roundtrip(a)
        npt.assert_array_equal(a, _a)

    def test_views(self):
        """Test views under serialization"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        rng = np.arange(20)  # a range of an array
        view = rng[10:]  # a view referencing a portion of an array
        data = [rng, view]

        _data = self.roundtrip(data)

        _data[0][15] = -1
        self.assertEqual(_data[1][5], -1)

    def test_strides(self):
        """Test non-standard strides and offsets"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        arr = np.eye(3)
        view = arr[1:, 1:]
        self.assertTrue(view.base is arr)
        data = [arr, view]

        _data = self.roundtrip(data)

        # test that the deserialized arrays indeed view the same memory
        _arr, _view = _data
        _arr[1, 2] = -1
        self.assertEqual(_view[0, 1], -1)
        self.assertTrue(_view.base is _arr)

    def test_weird_arrays(self):
        """Test references to arrays that do not effectively own their memory"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.arange(9)
        b = a[5:]
        a.strides = 1

        # this is kinda fishy; a has overlapping memory, _a does not
        if PY2:
            warn_count = 0
        else:
            warn_count = 1
        with warnings.catch_warnings(record=True) as w:
            _a = self.roundtrip(a)
            self.assertEqual(len(w), warn_count)
            npt.assert_array_equal(a, _a)

        # this also requires a deepcopy to work
        with warnings.catch_warnings(record=True) as w:
            _a, _b = self.roundtrip([a, b])
            self.assertEqual(len(w), warn_count)
            npt.assert_array_equal(a, _a)
            npt.assert_array_equal(b, _b)

    def test_transpose(self):
        """test handling of non-c-contiguous memory layout"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        # simple case; view a c-contiguous array
        a = np.arange(9).reshape(3, 3)
        b = a[1:, 1:]
        self.assertTrue(b.base is a.base)
        _a, _b = self.roundtrip([a, b])
        self.assertTrue(_b.base is _a.base)
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # a and b both view the same contiguous array
        a = np.arange(9).reshape(3, 3).T
        b = a[1:, 1:]
        self.assertTrue(b.base is a.base)
        _a, _b = self.roundtrip([a, b])
        self.assertTrue(_b.base is _a.base)
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # view an f-contiguous array
        a = a.copy()
        a.strides = a.strides[::-1]
        b = a[1:, 1:]
        self.assertTrue(b.base is a)
        _a, _b = self.roundtrip([a, b])
        self.assertTrue(_b.base is _a)
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # now a.data.contiguous is False; we have to make a deepcopy to make
        # this work note that this is a pretty contrived example though!
        a = np.arange(8).reshape(2, 2, 2).copy()
        a.strides = a.strides[0], a.strides[2], a.strides[1]
        b = a[1:, 1:]
        self.assertTrue(b.base is a)

        if PY2:
            warn_count = 0
        else:
            warn_count = 1

        with warnings.catch_warnings(record=True) as w:
            _a, _b = self.roundtrip([a, b])
            self.assertEqual(len(w), warn_count)
            npt.assert_array_equal(a, _a)
            npt.assert_array_equal(b, _b)

    def test_fortran_base(self):
        """Test a base array in fortran order"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.asfortranarray(np.arange(100).reshape((10, 10)))
        _a = self.roundtrip(a)
        npt.assert_array_equal(a, _a)

    def test_buffer(self):
        """test behavior with memoryviews which are not ndarrays"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        bstring = 'abcdefgh'.encode('utf-8')
        a = np.frombuffer(bstring, dtype=np.byte)
        if PY2:
            warn_count = 0
        else:
            warn_count = 1
        with warnings.catch_warnings(record=True) as w:
            _a = self.roundtrip(a)
            npt.assert_array_equal(a, _a)
            self.assertEqual(len(w), warn_count)

    def test_as_strided(self):
        """Test the result of as_strided()

        as_strided() returns an object that implements the array interface but
        is not an ndarray.

        """
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.arange(10)
        b = np.lib.stride_tricks.as_strided(
            a, shape=(5,), strides=(a.dtype.itemsize * 2,)
        )
        data = [a, b]

        with warnings.catch_warnings(record=True) as w:
            # as_strided returns a DummyArray object, which we can not
            # currently serialize correctly FIXME: would be neat to add
            # support for all objects implementing the __array_interface__
            _data = self.roundtrip(data)
            self.assertEqual(len(w), 1)

        # as we were warned, deserialized result is no longer a view
        _data[0][0] = -1
        self.assertEqual(_data[1][0], 0)

    def test_immutable(self):
        """test that immutability flag is copied correctly"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.arange(10)
        a.flags.writeable = False
        _a = self.roundtrip(a)
        try:
            _a[0] = 0
            self.assertTrue(False, 'item assignment must raise')
        except ValueError:
            self.assertTrue(True)

    def test_byteorder(self):
        """Test the byteorder for text and binary encodings"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        # small arr is stored as text
        a = np.arange(10).newbyteorder()
        b = a[:].newbyteorder()
        _a, _b = self.roundtrip([a, b])
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

        # bigger arr is stored as binary
        a = np.arange(100).newbyteorder()
        b = a[:].newbyteorder()
        _a, _b = self.roundtrip([a, b])
        npt.assert_array_equal(a, _a)
        npt.assert_array_equal(b, _b)

    def test_zero_dimensional_array(self):
        if self.should_skip:
            return self.skip('numpy is not importable')
        self.roundtrip(np.array(0.0))

    def test_nested_data_list_of_dict_with_list_keys(self):
        """Ensure we can handle numpy arrays within a nested structure"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        obj = [{'key': [np.array(0)]}]
        self.roundtrip(obj)

        obj = [{'key': [np.array([1.0])]}]
        self.roundtrip(obj)

    def test_nested_data(self):
        dict_1 = jsonpickle.decode(
            '{"0": {"py/object": "numpy.ndarray", "base": {"py/object": "numpy.ndarray", "values": "eJwBQAG//gAAAAAAAPA/VQqW3jPepD++wi4Xbj2xP+Eqr2DOdpU/mWetMS0IcD/m16Qof+6EP2u0UAKlSm0/IdTAjhapQb8tN4GncxpfPzkmJYS9XVw/+msZK9RTnLxuuyglnIZFP1+1FwKKYEE/7u8BXJxhYz9U37jb6qWCv0bczkAkLy+/JezKR4iwdz+I7Yif/EFYP55JeY33qj0/49HWOShSXz+TPzOdt12vPPgWuoKqYrA/7v6zLZ9Ko7++32mPR9mmv5HUyfMj5JE/Lc0iuzhxij+/PZPl/8h1P3lgGpQVeGI/WYbxdAsabL9j6H/jqXlhPz3xQW1qgeM/1czqT+R3tr/ov+7Bs6q2P/AKik9eqJY/wpyS7g4kfb/RBtjkPeRkP2xnXCPi3oA/PlCKOijbgj+vvoM4JUxyv394C9d7bTu/IKCbnQ==", "shape": [4, 10], "dtype": "float64", "byteorder": "<"}, "strides": [80], "shape": [4], "dtype": "float64", "values": [1.0, -9.828060650793317e-17, 2.1764592196729556e-16, 0.6095478185593844]}, "1": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 8, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.040757771416811174, 0.0006569158961776519, 0.06400552455496811, -0.08776690436461661]}, "2": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 16, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.06734359804137122, 0.0005303071849859612, -0.03767869408511425, 0.08854220852549555]}, "3": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 24, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.020961022044313608, 0.002365880383932143, -0.044626461273707715, 0.022126649479569716]}, "4": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 32, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.003914047755495442, -0.009105524855030168, 0.017471849207556792, -0.007114466026144574]}, "5": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 40, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.010220521381238922, -0.00023791616004384587, 0.012911265574394004, 0.0025502404084825976]}, "6": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 48, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.0035756323655023368, 0.005783588738888877, 0.005318641278065638, 0.008237616256838463]}, "7": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 56, "strides": [80], "shape": [4], "dtype": "float64", "values": [-0.0005389557022740279, 0.0014805762313148969, 0.002254526277123911, 0.0092070715775564]}, "8": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 64, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.0018983964382814453, 0.00045269531026295186, -0.0034303878560855536, -0.004467149156183999]}, "9": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 72, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.0017313338035641736, 0.0019116776305568787, 0.0021332090509505338, -0.00041851304471929995]}}'
        )
        dict_2 = jsonpickle.decode(
            '{"0": {"py/object": "numpy.ndarray", "base": {"py/object": "numpy.ndarray", "values": "eJwBQAG//gAAAAAAAPA/hs9S2ciZab9k2+isy/SzP4Casbt62Fe/lQCfUucnkj/iGcKfgfxlvwGtIV8RQTI/RgwE4z/xUr/OPGXjAbhBP7/elRvFZlC/NNRw4dW3W7w03D7Ft+A3P36vmVffTmi/L49PrPopdj9ThqpLh556v0cIl/WgznA/aqKsqpNEdr+6KhsYA5l4P3zlEUrZMIC/uDKaQ9rsXz+EV7YIEvV2vGYEScR0Ib8/11vTXiARi7/tLbTSznBqv3KO0JTwEXy/6jHOARzZ3r4v+1caXKN4v6+R4qpdLmA/N344TtQjYL9MvGCU/EGAv6d38Ze3gdc/jFiKFSRxez/1UGZRs6u0P9V9LqV9CYA/JuDyhIrtoD+Twrp4OD5Ev+E/5pV7CUw/KaCJOrEAgb/Fjp+kUvNmv6BW858GgmM/V2CfZw==", "shape": [4, 10], "dtype": "float64", "byteorder": "<"}, "strides": [80], "shape": [4], "dtype": "float64", "values": [1.0, -6.010406432404402e-18, -1.991228920248446e-17, 0.3672923072643876]}, "1": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 8, "strides": [80], "shape": [4], "dtype": "float64", "values": [-0.0031250880079713465, 0.0003643463762687566, 0.12160424987907134, 0.006699696496658924]}, "2": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 16, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.07795403453279232, -0.0029672968055844398, -0.013216259855125685, 0.08074494111344051]}, "3": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 24, "strides": [80], "shape": [4], "dtype": "float64", "values": [-0.0014554213110159753, 0.005411128226999114, -0.003227619124691597, 0.007830602267687429]}, "4": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 32, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.01773034517077628, -0.006498840807269329, -0.006853046198589051, 0.03306229470166526]}, "5": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 40, "strides": [80], "shape": [4], "dtype": "float64", "values": [-0.002683880969985384, 0.0041033065294318665, -7.354756260802008e-06, -0.0006177688350899761]}, "6": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 48, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.00027853654967797964, -0.005436493704828219, -0.006015167023632625, 0.0008556226201097383]}, "7": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 56, "strides": [80], "shape": [4], "dtype": "float64", "values": [-0.0011561511892643913, 0.006005298697071277, 0.001975233978628497, -0.008302101706434044]}, "8": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 64, "strides": [80], "shape": [4], "dtype": "float64", "values": [0.0005407342166935615, -0.007905671666082574, -0.0019702097504390686, -0.0028015722391375474]}, "9": {"py/object": "numpy.ndarray", "base": {"py/id": 1}, "offset": 72, "strides": [80], "shape": [4], "dtype": "float64", "values": [-0.0010010647659707687, 0.001948559902675102, -0.007938359525807702, 0.002381337107730655]}}'
        )

        # demonstrate that objects themselves can be encoded and decoded
        self.roundtrip(dict_1)
        self.roundtrip(dict_2)

        # tuple however fails (note lists also fail)
        obj = (dict_1, dict_2)
        self.roundtrip(obj)

    def test_size_threshold_None(self):
        if self.should_skip:
            return self.skip('numpy is not importable')
        handler = numpy_ext.NumpyNDArrayHandlerView(size_threshold=None)
        handlers.registry.unregister(np.ndarray)
        handlers.registry.register(np.ndarray, handler, base=True)
        self.roundtrip(np.array([0, 1]))

    def test_ndarray_dtype_object(self):
        if self.should_skip:
            return self.skip('numpy is not importable')
        a = np.array(['F' + str(i) for i in range(30)], dtype=np.object)
        buf = jsonpickle.encode(a)
        # This is critical for reproducing the numpy segfault issue when
        # restoring ndarray of dtype object
        del a
        _a = jsonpickle.decode(buf)
        a = np.array(['F' + str(i) for i in range(30)], dtype=np.object)
        npt.assert_array_equal(a, _a)

    def test_np_random(self):
        """Ensure random.random() arrays can be serialized"""
        if self.should_skip:
            return self.skip('numpy is not importable')
        obj = np.random.random(100)
        encoded = jsonpickle.encode(obj)
        clone = jsonpickle.decode(encoded)
        self.assertEqual(100, len(clone))
        for idx, (expect, actual) in enumerate(zip(obj, clone)):
            self.assertEqual(
                expect,
                actual,
                'Item at index %s differs.  %s != %s' % (idx, expect, actual),
            )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NumpyTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
