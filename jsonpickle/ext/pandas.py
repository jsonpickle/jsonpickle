from __future__ import absolute_import

import pandas as pd
from io import StringIO
import zlib

from ..handlers import BaseHandler, register, unregister
from ..util import b64decode, b64encode
from ..backend import json

__all__ = ['register_handlers', 'unregister_handlers']


class PandasProcessor(object):

    def __init__(self, size_threshold=500, compression=zlib):
        """
        :param size_threshold: nonnegative int or None
            valid values for 'size_threshold' are all nonnegative
            integers and None.  If size_threshold is None,
            dataframes are always stored as csv strings
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compresion is None, no compression is applied
        """
        self.size_threshold = size_threshold
        self.compression = compression

    def flatten_pandas(self, buf, data, meta=None):
        if self.size_threshold is not None and len(buf) > self.size_threshold:
            if self.compression:
                buf = self.compression.compress(buf.encode())
                data['comp'] = True
            data['values'] = b64encode(buf)
            data['txt'] = False
        else:
            data['values'] = buf
            data['txt'] = True

        data['meta'] = meta
        return data

    def restore_pandas(self, data):
        if data.get('txt', True):
            # It's just text...
            buf = data['values']
        else:
            buf = b64decode(data['values'])
            if data.get('comp', False):
                buf = self.compression.decompress(buf).decode()
        meta = data.get('meta', {})
        return (buf, meta)


def make_read_csv_params(meta):
    meta_dtypes = meta.get('dtypes', {})

    parse_dates = []
    converters = {}
    dtype = {}
    for k, v in meta_dtypes.items():
        if v.startswith('datetime'):
            parse_dates.append(k)
        elif v.startswith('complex'):
            converters[k] = complex
        else:
            dtype[k] = v

    return dict(dtype=dtype, parse_dates=parse_dates, converters=converters)


class PandasDfHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        dtype = obj.dtypes.to_dict()

        # Handles named multi-indexes
        index_col = list(obj.index.names) if list(obj.index.names) != [None] else 0

        meta = {'dtypes': {k: str(dtype[k]) for k in dtype},
                'index_col': index_col}

        data = self.pp.flatten_pandas(obj.to_csv(), data, meta)
        return data

    def restore(self, data):
        csv, meta = self.pp.restore_pandas(data)
        params = make_read_csv_params(meta)
        df = pd.read_csv(StringIO(csv),
                         index_col=meta.get('index_col', None),
                         **params)
        return df


class PandasSeriesHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        dtypes = {k: str(pd.np.dtype(type(obj[k]))) for k in obj.keys()}
        meta = {'dtypes': dtypes, 'name': obj.name}
        # Save series as two rows rather than two cols to make preserving the
        # type easier.
        data = self.pp.flatten_pandas(obj.to_frame().T.to_csv(), data, meta)
        return data

    def restore(self, data):
        csv, meta = self.pp.restore_pandas(data)
        params = make_read_csv_params(meta)
        df = pd.read_csv(StringIO(csv), **params)
        ser = pd.Series(data=df.iloc[:, 1:].values[0],
                        index=df.columns[1:].values,
                        name=meta.get('name', None))
        return ser


class PandasIndexHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        meta = {'dtype': str(obj.dtype), 'name': obj.name}
        buf = json.dumps(obj.tolist())
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data):
        buf, meta = self.pp.restore_pandas(data)
        dtype = meta.get('dtype', None)
        name = meta.get('name', None)
        idx = pd.Index(json.loads(buf), dtype=dtype, name=name)
        return idx


def register_handlers():
    register(pd.DataFrame, PandasDfHandler, base=True)
    register(pd.Series, PandasSeriesHandler, base=True)
    register(pd.Index, PandasIndexHandler, base=True)


def unregister_handlers():
    unregister(pd.DataFrame)
    unregister(pd.Series)
    unregister(pd.Index)
