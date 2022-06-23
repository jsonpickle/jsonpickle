from __future__ import absolute_import

import zlib
from io import StringIO

import pandas as pd

from .. import decode, encode
from ..handlers import BaseHandler, register, unregister
from ..util import b64decode, b64encode
from .numpy import register_handlers as register_numpy_handlers
from .numpy import unregister_handlers as unregister_numpy_handlers

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
            if compression is None, no compression is applied
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


def make_read_csv_params(meta, context):
    meta_dtypes = context.restore(meta.get('dtypes', {}), reset=False)
    # The header is used to select the rows of the csv from which
    # the columns names are retrieved
    header = meta.get('header', [0])
    parse_dates = []
    converters = {}
    timedeltas = []
    dtype = {}
    for k, v in meta_dtypes.items():
        if v.startswith('datetime'):
            parse_dates.append(k)
        elif v.startswith('complex'):
            converters[k] = complex
        elif v.startswith('timedelta'):
            timedeltas.append(k)
            dtype[k] = 'object'
        else:
            dtype[k] = v

    return (
        dict(
            dtype=dtype, header=header, parse_dates=parse_dates, converters=converters
        ),
        timedeltas,
    )


class PandasDfHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        dtype = obj.dtypes.to_dict()

        meta = {
            'dtypes': self.context.flatten(
                {k: str(dtype[k]) for k in dtype}, reset=False
            ),
            'index': encode(obj.index),
            'column_level_names': obj.columns.names,
            'header': list(range(len(obj.columns.names))),
        }

        data = self.pp.flatten_pandas(
            obj.reset_index(drop=True).to_csv(index=False), data, meta
        )
        return data

    def restore(self, data):
        csv, meta = self.pp.restore_pandas(data)
        params, timedeltas = make_read_csv_params(meta, self.context)
        # None makes it compatible with objects serialized before
        # column_levels_names has been introduced.
        column_level_names = meta.get('column_level_names', None)
        df = (
            pd.read_csv(StringIO(csv), **params)
            if data['values'].strip()
            else pd.DataFrame()
        )
        for col in timedeltas:
            df[col] = pd.to_timedelta(df[col])

        df.set_index(decode(meta['index']), inplace=True)
        # restore the column level(s) name(s)
        if column_level_names:
            df.columns.names = column_level_names
        return df


class PandasSeriesHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        """Flatten the index and values for reconstruction"""
        data['name'] = obj.name
        # This relies on the numpy handlers for the inner guts.
        data['index'] = self.context.flatten(obj.index, reset=False)
        data['values'] = self.context.flatten(obj.values, reset=False)
        return data

    def restore(self, data):
        """Restore the flattened data"""
        name = data['name']
        index = self.context.restore(data['index'], reset=False)
        values = self.context.restore(data['values'], reset=False)
        return pd.Series(values, index=index, name=name)


class PandasIndexHandler(BaseHandler):

    pp = PandasProcessor()
    index_constructor = pd.Index

    def name_bundler(self, obj):
        return {'name': obj.name}

    def flatten(self, obj, data):
        name_bundle = self.name_bundler(obj)
        meta = dict(dtype=str(obj.dtype), **name_bundle)
        buf = encode(obj.tolist())
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data):
        buf, meta = self.pp.restore_pandas(data)
        dtype = meta.get('dtype', None)
        name_bundle = {k: v for k, v in meta.items() if k in {'name', 'names'}}
        if 'names' in name_bundle:
            # pandas won't accept the names arg in a future version
            names = name_bundle.pop('names')
            # name arg must be hashable so cast to tuple first
            name_bundle['name'] = tuple(names)
        idx = self.index_constructor(decode(buf), dtype=dtype, **name_bundle)
        return idx


class PandasPeriodIndexHandler(PandasIndexHandler):
    index_constructor = pd.PeriodIndex


class PandasMultiIndexHandler(PandasIndexHandler):
    def name_bundler(self, obj):
        return {'names': obj.names}


class PandasTimestampHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        meta = {'isoformat': obj.isoformat()}
        buf = ''
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data):
        _, meta = self.pp.restore_pandas(data)
        isoformat = meta['isoformat']
        obj = pd.Timestamp(isoformat)
        return obj


class PandasPeriodHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        meta = {
            'start_time': encode(obj.start_time),
            'freqstr': obj.freqstr,
        }
        buf = ''
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data):
        _, meta = self.pp.restore_pandas(data)
        start_time = decode(meta['start_time'])
        freqstr = meta['freqstr']
        obj = pd.Period(start_time, freqstr)
        return obj


class PandasIntervalHandler(BaseHandler):
    pp = PandasProcessor()

    def flatten(self, obj, data):
        meta = {
            'left': encode(obj.left),
            'right': encode(obj.right),
            'closed': obj.closed,
        }
        buf = ''
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data):
        _, meta = self.pp.restore_pandas(data)
        left = decode(meta['left'])
        right = decode(meta['right'])
        closed = str(meta['closed'])
        obj = pd.Interval(left, right, closed=closed)
        return obj


def register_handlers():
    register_numpy_handlers()
    register(pd.DataFrame, PandasDfHandler, base=True)
    register(pd.Series, PandasSeriesHandler, base=True)
    register(pd.Index, PandasIndexHandler, base=True)
    register(pd.PeriodIndex, PandasPeriodIndexHandler, base=True)
    register(pd.MultiIndex, PandasMultiIndexHandler, base=True)
    register(pd.Timestamp, PandasTimestampHandler, base=True)
    register(pd.Period, PandasPeriodHandler, base=True)
    register(pd.Interval, PandasIntervalHandler, base=True)


def unregister_handlers():
    unregister_numpy_handlers()
    unregister(pd.DataFrame)
    unregister(pd.Series)
    unregister(pd.Index)
    unregister(pd.PeriodIndex)
    unregister(pd.MultiIndex)
    unregister(pd.Timestamp)
    unregister(pd.Period)
    unregister(pd.Interval)
