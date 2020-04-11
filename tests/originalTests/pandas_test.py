from __future__ import absolute_import, division, unicode_literals
import unittest

import jsonpickle

from helper import SkippableTest

try:
    import pandas as pd
    import numpy as np
    from pandas.testing import assert_series_equal
    from pandas.testing import assert_frame_equal
    from pandas.testing import assert_index_equal
except ImportError:
    np = None


class PandasTestCase(SkippableTest):
    def setUp(self):
        if np is None:
            self.should_skip = True
            return
        self.should_skip = False
        import jsonpickle.ext.pandas

        jsonpickle.ext.pandas.register_handlers()

    def tearDown(self):
        if self.should_skip:
            return
        import jsonpickle.ext.pandas

        jsonpickle.ext.pandas.unregister_handlers()

    def roundtrip(self, obj):
        return jsonpickle.decode(jsonpickle.encode(obj))

    def test_series_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')
        ser = pd.Series(
            {
                'an_int': np.int_(1),
                'a_float': np.float_(2.5),
                'a_nan': np.nan,
                'a_minus_inf': -np.inf,
                'an_inf': np.inf,
                'a_str': np.str_('foo'),
                'a_unicode': np.unicode_('bar'),
                'date': np.datetime64('2014-01-01'),
                'complex': np.complex_(1 - 2j),
                # TODO: the following dtypes are not currently supported.
                # 'object': np.object_({'a': 'b'}),
            }
        )
        decoded_ser = self.roundtrip(ser)
        assert_series_equal(decoded_ser, ser)

    def test_dataframe_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')
        df = pd.DataFrame(
            {
                'an_int': np.int_([1, 2, 3]),
                'a_float': np.float_([2.5, 3.5, 4.5]),
                'a_nan': np.array([np.nan] * 3),
                'a_minus_inf': np.array([-np.inf] * 3),
                'an_inf': np.array([np.inf] * 3),
                'a_str': np.str_('foo'),
                'a_unicode': np.unicode_('bar'),
                'date': np.array([np.datetime64('2014-01-01')] * 3),
                'complex': np.complex_([1 - 2j, 2 - 1.2j, 3 - 1.3j]),
                # TODO: the following dtypes are not currently supported.
                # 'object': np.object_([{'a': 'b'}]*3),
            }
        )
        decoded_df = self.roundtrip(df)
        assert_frame_equal(decoded_df, df)

    def test_multindex_dataframe_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        df = pd.DataFrame(
            {
                'idx_lvl0': ['a', 'b', 'c'],
                'idx_lvl1': np.int_([1, 1, 2]),
                'an_int': np.int_([1, 2, 3]),
                'a_float': np.float_([2.5, 3.5, 4.5]),
                'a_nan': np.array([np.nan] * 3),
                'a_minus_inf': np.array([-np.inf] * 3),
                'an_inf': np.array([np.inf] * 3),
                'a_str': np.str_('foo'),
                'a_unicode': np.unicode_('bar'),
            }
        )
        df = df.set_index(['idx_lvl0', 'idx_lvl1'])

        decoded_df = self.roundtrip(df)
        assert_frame_equal(decoded_df, df)

    def test_dataframe_with_interval_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        df = pd.DataFrame(
            {'a': [1, 2], 'b': [3, 4]}, index=pd.IntervalIndex.from_breaks([1, 2, 4])
        )

        decoded_df = self.roundtrip(df)
        assert_frame_equal(decoded_df, df)

    def test_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.Index(range(5, 10))
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_datetime_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.date_range(start='2019-01-01', end='2019-02-01', freq='D')
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_ragged_datetime_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-05'])
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_timedelta_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.timedelta_range(start='1 day', periods=4, closed='right')
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_period_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_int64_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.Int64Index([-1, 0, 3, 4])
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_uint64_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.UInt64Index([0, 3, 4])
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_float64_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.Float64Index([0.1, 3.7, 4.2])
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_interval_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.IntervalIndex.from_breaks(range(5))
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_datetime_interval_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.IntervalIndex.from_breaks(pd.date_range('2019-01-01', '2019-01-10'))
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_multi_index_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        idx = pd.MultiIndex.from_product(((1, 2, 3), ('a', 'b')))
        decoded_idx = self.roundtrip(idx)
        assert_index_equal(decoded_idx, idx)

    def test_timestamp_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        obj = pd.Timestamp('2019-01-01')
        decoded_obj = self.roundtrip(obj)
        assert decoded_obj == obj

    def test_period_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        obj = pd.Timestamp('2019-01-01')
        decoded_obj = self.roundtrip(obj)
        assert decoded_obj == obj

    def test_interval_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')

        obj = pd.Interval(2, 4, closed=str('left'))
        decoded_obj = self.roundtrip(obj)
        assert decoded_obj == obj

    def test_b64(self):
        """Test the binary encoding"""
        if self.should_skip:
            return self.skip('pandas is not importable')
        # array of substantial size is stored as b64
        a = np.random.rand(20, 10)
        index = ['Row' + str(i) for i in range(1, a.shape[0] + 1)]
        columns = ['Col' + str(i) for i in range(1, a.shape[1] + 1)]
        df = pd.DataFrame(a, index=index, columns=columns)
        decoded_df = self.roundtrip(df)
        assert_frame_equal(decoded_df, df)

    def test_series_list_index(self):
        """Test pandas using series with a list index"""
        expect = pd.Series(0, index=[1, 2, 3])
        actual = self.roundtrip(expect)

        self.assertEqual(expect.values[0], actual.values[0])
        self.assertEqual(0, actual.values[0])

        self.assertEqual(expect.index[0], actual.index[0])
        self.assertEqual(expect.index[1], actual.index[1])
        self.assertEqual(expect.index[2], actual.index[2])

    def test_series_multi_index(self):
        """Test pandas using series with a multi-index"""
        expect = pd.Series(0, index=[[1], [2], [3]])
        actual = self.roundtrip(expect)

        self.assertEqual(expect.values[0], actual.values[0])
        self.assertEqual(0, actual.values[0])

        self.assertEqual(expect.index[0], actual.index[0])
        self.assertEqual(expect.index[0][0], actual.index[0][0])
        self.assertEqual(expect.index[0][1], actual.index[0][1])
        self.assertEqual(expect.index[0][2], actual.index[0][2])

    def test_series_multi_index_strings(self):
        """Test multi-index with strings"""
        lets = ['A', 'B', 'C']
        nums = ['1', '2', '3']
        midx = pd.MultiIndex.from_product([lets, nums])
        expect = pd.Series(0, index=midx)
        actual = self.roundtrip(expect)

        self.assertEqual(expect.values[0], actual.values[0])
        self.assertEqual(0, actual.values[0])

        self.assertEqual(expect.index[0], actual.index[0])
        self.assertEqual(expect.index[1], actual.index[1])
        self.assertEqual(expect.index[2], actual.index[2])
        self.assertEqual(expect.index[3], actual.index[3])
        self.assertEqual(expect.index[4], actual.index[4])
        self.assertEqual(expect.index[5], actual.index[5])
        self.assertEqual(expect.index[6], actual.index[6])
        self.assertEqual(expect.index[7], actual.index[7])
        self.assertEqual(expect.index[8], actual.index[8])

        self.assertEqual(('A', '1'), actual.index[0])
        self.assertEqual(('A', '2'), actual.index[1])
        self.assertEqual(('A', '3'), actual.index[2])
        self.assertEqual(('B', '1'), actual.index[3])
        self.assertEqual(('B', '2'), actual.index[4])
        self.assertEqual(('B', '3'), actual.index[5])
        self.assertEqual(('C', '1'), actual.index[6])
        self.assertEqual(('C', '2'), actual.index[7])
        self.assertEqual(('C', '3'), actual.index[8])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(PandasTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
