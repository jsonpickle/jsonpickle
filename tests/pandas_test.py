from __future__ import absolute_import, division, unicode_literals
import unittest

import jsonpickle

from helper import SkippableTest

try:
    import pandas as pd
    import numpy as np
    from pandas.testing import assert_series_equal, assert_frame_equal

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
        ser = pd.Series({
                'an_int': np.int_(1),
                'a_float': np.float_(2.5),
                'a_nan': np.nan,
                'a_minus_inf': -np.inf,
                'an_inf': np.inf,
                'a_str': np.str_('foo'),
                'a_unicode': np.unicode_('bar'),
                # TODO: the following dtypes are not currently supported.
                # 'object': np.object_({'a': 'b'}),
                # 'date': np.datetime64('2014-01-01'),
                # 'complex': np.complex_(1 - 2j)
            })
        decoded_ser = self.roundtrip(ser)
        assert_series_equal(decoded_ser, ser)

    def test_dataframe_roundtrip(self):
        if self.should_skip:
            return self.skip('pandas is not importable')
        df = pd.DataFrame({
                'an_int': np.int_([1, 2, 3]),
                'a_float': np.float_([2.5, 3.5, 4.5]),
                'a_nan': np.array([np.nan]*3),
                'a_minus_inf': np.array([-np.inf]*3),
                'an_inf': np.array([np.inf]*3),
                'a_str': np.str_('foo'),
                'a_unicode': np.unicode_('bar'),
                # TODO: the following dtypes are not currently supported.
                # 'object': np.object_([{'a': 'b'}]*3),
                # 'date': np.array([np.datetime64('2014-01-01')]*3),
                # 'complex': np.complex_([1 - 2j, 2-1.2j, 3-1.3j])
            })
        decoded_df = self.roundtrip(df)
        assert_frame_equal(decoded_df, df)

    def test_b64(self):
        """Test the binary encoding"""
        if self.should_skip:
            return self.skip('pandas is not importable')
        a = np.random.rand(20, 10)  # array of substantial size is stored as b64
        index = ['Row'+str(i) for i in range(1, a.shape[0]+1)]
        columns = ['Col'+str(i) for i in range(1, a.shape[1]+1)]
        df = pd.DataFrame(a, index=index, columns=columns)
        decoded_df = self.roundtrip(df)
        assert_frame_equal(decoded_df, df)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(PandasTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
