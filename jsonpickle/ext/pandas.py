import warnings
import zlib
from io import StringIO
from types import ModuleType
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd

from .. import decode, encode
from ..handlers import BaseHandler, ContextType, register, unregister
from ..tags_pd import REVERSE_TYPE_MAP, TYPE_MAP
from ..util import b64decode, b64encode
from .numpy import register_handlers as register_numpy_handlers
from .numpy import unregister_handlers as unregister_numpy_handlers

__all__ = ['register_handlers', 'unregister_handlers']


# unused, TODO deprecate then remove
# it's suggested that this return str instead of Any, but i'm not sure bc of obj.item()
def pd_encode(obj: Any, **kwargs: Dict[str, Any]) -> Any:
    if isinstance(obj, np.generic):
        # convert pandas/numpy scalar to native Python type
        return obj.item()
    return encode(obj, **kwargs)  # type: ignore[arg-type]


# unused, TODO deprecate then remove
def pd_decode(s: str, **kwargs: Dict[str, Any]) -> Any:
    return decode(s, **kwargs)  # type: ignore[arg-type]


def rle_encode(types_list: List[str]) -> List[List[object]]:
    """
    Encodes a list of type codes using Run-Length Encoding (RLE). This allows for object columns in dataframes to contain items of different types without massively bloating the encoded representation.
    """
    if not types_list:
        return []

    encoded = []
    current_type = types_list[0]
    count = 1

    for typ in types_list[1:]:
        if typ == current_type:
            count += 1
        else:
            encoded.append([current_type, count])
            current_type = typ
            count = 1
    encoded.append([current_type, count])

    return encoded


def rle_decode(encoded_list: List[Tuple[str, int]]) -> List[str]:
    """
    Decodes a Run-Length Encoded (RLE) list back into the original list of type codes.
    """
    decoded = []
    for typ, count in encoded_list:
        decoded.extend([typ] * count)
    return decoded


class PandasProcessor:
    def __init__(
        self, size_threshold: int = 500, compression: ModuleType = zlib
    ) -> None:
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

    def flatten_pandas(
        self, buf: str, data: Dict[str, Any], meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self.size_threshold is not None and len(buf) > self.size_threshold:
            if self.compression:
                buf = self.compression.compress(buf.encode())
                data['comp'] = True
            # we have a mypy error here, not sure what the fix is so i'm silencing it temporarily
            data['values'] = b64encode(buf)  # type: ignore[arg-type]
            data['txt'] = False
        else:
            data['values'] = buf
            data['txt'] = True

        data['meta'] = meta
        return data

    def restore_pandas(self, data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if data.get('txt', True):
            # It's just text...
            buf = data['values']
        else:
            buf = b64decode(data['values'])
            if data.get('comp', False):
                buf = self.compression.decompress(buf).decode()
        meta = data.get('meta', {})
        return (buf, meta)


def make_read_csv_params(
    # we can remove the valid-type ignore once we convert ContextType to a Type Alias when 3.9 is dropped
    meta: Dict[str, Any],
    context: ContextType,  # type: ignore[valid-type]
) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    meta_dtypes = context.restore(meta.get('dtypes', {}), reset=False)  # type: ignore[attr-defined]
    # The header is used to select the rows of the csv from which
    # the columns names are retrieved
    header = meta.get('header', [0])
    parse_dates = []
    converters = {}
    timedeltas = []
    # this is only for pandas v2+ due to a backwards-incompatible change
    parse_datetime_v2 = {}
    dtype = {}
    for k, v in meta_dtypes.items():
        if v.startswith('datetime'):
            parse_dates.append(k)
            parse_datetime_v2[k] = v
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
        parse_datetime_v2,
    )


class PandasDfHandler(BaseHandler):
    pp: PandasProcessor = PandasProcessor()

    def flatten(self, obj: pd.DataFrame, data: Dict[str, Any]) -> Dict[str, Any]:
        pp = PandasProcessor()
        # handle multiindex columns
        if isinstance(obj.columns, pd.MultiIndex):
            columns: Union[List[Tuple[Any, ...]], List[str]] = [
                tuple(col) for col in obj.columns
            ]
            column_names: Union[
                List[List[object]], List[Hashable], List[str], Hashable
            ] = obj.columns.names
            is_multicolumns = True
        else:
            columns = obj.columns.tolist()
            column_names = obj.columns.name
            is_multicolumns = False

        # handle multiindex index
        if isinstance(obj.index, pd.MultiIndex):
            index_values = [tuple(idx) for idx in obj.index.values]
            index_names: Union[List[Hashable], List[str], Hashable] = obj.index.names
            is_multiindex = True
        else:
            index_values = obj.index.tolist()
            index_names = obj.index.name
            is_multiindex = False

        data_columns = {}
        type_codes = []
        for col in obj.columns:
            col_data = obj[col]
            dtype_name = col_data.dtype.name

            if dtype_name == "object":
                # check if items are complex types
                if col_data.apply(
                    lambda x: isinstance(x, (list, dict, set, tuple, np.ndarray))
                ).any():
                    # if items are complex, erialize each item individually
                    serialized_values = col_data.apply(lambda x: encode(x)).tolist()
                    data_columns[col] = serialized_values
                    type_codes.append("py/jp")
                else:
                    # treat it as regular object dtype
                    data_columns[col] = col_data.tolist()
                    type_codes.append(TYPE_MAP.get(dtype_name, "object"))
            else:
                # for other dtypes, store their values directly
                data_columns[col] = col_data.tolist()
                type_codes.append(TYPE_MAP.get(dtype_name, "object"))

        # store index data
        index_encoded = encode(index_values, keys=True)

        rle_types = rle_encode(type_codes)
        # prepare metadata
        meta = {
            "dtypes_rle": rle_types,
            "index": index_encoded,
            "index_names": index_names,
            "columns": encode(columns, keys=True),
            "column_names": column_names,
            "is_multiindex": is_multiindex,
            "is_multicolumns": is_multicolumns,
        }

        # serialize data_columns with keys=True to allow for non-object keys
        data_encoded = encode(data_columns, keys=True)

        # use PandasProcessor to flatten
        data = pp.flatten_pandas(data_encoded, data, meta)
        return data

    def restore(self, obj: Dict[str, Any]) -> pd.DataFrame:
        data_encoded, meta = self.pp.restore_pandas(obj)
        try:
            data_columns = decode(data_encoded, keys=True)
        except Exception:
            # this may be a specific type of jsondecode error for pre-v3.4 encoding schemes, but also might not be
            warnings.warn(
                (
                    "jsonpickle versions at and above v3.4 have a different encoding scheme for pandas dataframes."
                    # stack level 6 is where the user called jsonpickle from
                    " If you're not decoding an object encoded in pre-v3.4 jsonpickle, please file a bug report on our GitHub!"
                ),
                stacklevel=6,
            )
            return self.restore_v3_3(obj)

        # get type codes, un-RLE-ed
        try:
            rle_types = meta["dtypes_rle"]
        except KeyError:
            # was definitely encoded with pre-v3.4 scheme, but warn anyway
            warnings.warn(
                (
                    "jsonpickle versions at and above v3.4 have a different encoding scheme for pandas dataframes."
                    " Please update your jsonpickle and re-encode these objects!"
                ),
                stacklevel=6,
            )
            return self.restore_v3_3(obj)
        type_codes = rle_decode(rle_types)

        # handle multicolumns
        columns_decoded = decode(meta["columns"], keys=True)
        if meta.get("is_multicolumns", False):
            columns = pd.MultiIndex.from_tuples(
                columns_decoded, names=meta.get("column_names")  # type: ignore[arg-type]
            )
        else:
            columns = columns_decoded

        # progressively reconstruct dataframe as a dict
        df_data = {}
        dtypes = {}
        for col, type_code in zip(columns, type_codes):
            col_data = data_columns[col]
            if type_code == "py/jp":
                # deserialize each item in the column
                col_values = [decode(item) for item in col_data]
                df_data[col] = col_values
            else:
                df_data[col] = col_data
                # used later to get correct dtypes
                dtype_str = REVERSE_TYPE_MAP.get(type_code, "object")
                dtypes[col] = dtype_str

        # turn dict into df
        df = pd.DataFrame(df_data)
        df.columns = columns

        # apply dtypes
        for col in df.columns:
            dtype_str = dtypes.get(col, "object")
            try:
                dtype = np.dtype(dtype_str)
                df[col] = df[col].astype(dtype)
            except Exception:
                msg = (
                    f"jsonpickle was unable to properly deserialize "
                    f"the column {col} into its inferred dtype. "
                    f"Please file a bugreport on the jsonpickle GitHub! "
                )
                warnings.warn(msg)

        # decode and set the index
        index_values = decode(meta["index"], keys=True)
        if meta.get("is_multiindex", False):
            index = pd.MultiIndex.from_tuples(
                index_values, names=meta.get("index_names")  # type: ignore[arg-type]
            )
        else:
            index = pd.Index(index_values, name=meta.get("index_names"))
        df.index = index

        # restore column names for easy readability
        if "column_names" in meta:
            if meta.get("is_multicolumns", False):
                names: Any = meta.get("column_names")
                df.columns.names = names
            else:
                df.columns.name = meta.get("column_names")

        return df

    def restore_v3_3(self, data: Dict[str, Any]) -> pd.DataFrame:
        csv, meta = self.pp.restore_pandas(data)
        params, timedeltas, parse_datetime_v2 = make_read_csv_params(meta, self.context)
        # None makes it compatible with objects serialized before
        # column_levels_names has been introduced.
        column_level_names = meta.get("column_level_names", None)
        df = (
            pd.read_csv(StringIO(csv), **params)
            if data["values"].strip()
            else pd.DataFrame()
        )
        for col in timedeltas:
            df[col] = pd.to_timedelta(df[col])
        df = df.astype(parse_datetime_v2)

        df.set_index(decode(meta["index"]), inplace=True)
        # restore the column level(s) name(s)
        if column_level_names:
            df.columns.names = column_level_names
        return df


class PandasSeriesHandler(BaseHandler):
    pp: PandasProcessor = PandasProcessor()

    def flatten(self, obj: pd.Series[Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten the index and values for reconstruction"""
        data['name'] = obj.name
        # This relies on the numpy handlers for the inner guts.
        data['index'] = self.context.flatten(obj.index, reset=False)
        data['values'] = self.context.flatten(obj.values, reset=False)
        return data

    def restore(self, data: Dict[str, Any]) -> pd.Series[Any]:
        """Restore the flattened data"""
        name = data['name']
        index = self.context.restore(data['index'], reset=False)
        values = self.context.restore(data['values'], reset=False)
        return pd.Series(values, index=index, name=name)  # type: ignore[no-any-return]


class PandasIndexHandler(BaseHandler):
    pp: PandasProcessor = PandasProcessor()
    index_constructor: Type[pd.Index[Any]] = pd.Index

    def name_bundler(self, obj: pd.Index[Any]) -> Dict[str, Optional[Hashable]]:
        return {'name': obj.name}

    def flatten(self, obj: pd.Index[Any], data: Dict[str, Any]) -> Dict[str, Any]:
        name_bundle = self.name_bundler(obj)
        meta = dict(dtype=str(obj.dtype), **name_bundle)
        buf = encode(obj.tolist())
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data: Dict[str, Any]) -> pd.Index[Any]:
        buf, meta = self.pp.restore_pandas(data)
        dtype = meta.get('dtype', None)
        name_bundle = {
            'name': (tuple if v is not None else lambda x: x)(v)  # type: ignore[misc]
            for k, v in meta.items()
            if k in {'name', 'names'}
        }
        idx = self.index_constructor(decode(buf), dtype=dtype, **name_bundle)  # type: ignore[arg-type]
        return idx


class PandasPeriodIndexHandler(PandasIndexHandler):
    index_constructor: Type[pd.PeriodIndex] = pd.PeriodIndex


class PandasMultiIndexHandler(PandasIndexHandler):
    # sequence is technically the type pd.core.indexes.frozen.FrozenList
    def name_bundler(self, obj: pd.MultiIndex) -> Dict[str, Sequence[Optional[Hashable]]]:  # type: ignore[override]
        return {'names': obj.names}


class PandasTimestampHandler(BaseHandler):
    pp: PandasProcessor = PandasProcessor()

    def flatten(self, obj: pd.Timestamp, data: Dict[str, Any]) -> Dict[str, Any]:
        meta = {'isoformat': obj.isoformat()}
        buf = ''
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data: Dict[str, Any]) -> pd.Timestamp:
        _, meta = self.pp.restore_pandas(data)
        isoformat = meta['isoformat']
        obj = pd.Timestamp(isoformat)
        return obj


class PandasPeriodHandler(BaseHandler):
    pp: PandasProcessor = PandasProcessor()

    def flatten(self, obj: pd.Period, data: Dict[str, Any]) -> Dict[str, Any]:
        meta = {
            'start_time': encode(obj.start_time),
            'freqstr': obj.freqstr,
        }
        buf = ''
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data: Dict[str, Any]) -> pd.Period:
        _, meta = self.pp.restore_pandas(data)
        start_time = decode(meta['start_time'])
        freqstr = meta['freqstr']
        obj = pd.Period(start_time, freqstr)
        return obj


class PandasIntervalHandler(BaseHandler):
    pp: PandasProcessor = PandasProcessor()

    def flatten(self, obj: pd.Interval[Any], data: Dict[str, Any]) -> Dict[str, Any]:
        meta = {
            'left': encode(obj.left),
            'right': encode(obj.right),
            'closed': obj.closed,
        }
        buf = ''
        data = self.pp.flatten_pandas(buf, data, meta)
        return data

    def restore(self, data: Dict[str, Any]) -> pd.Interval[Any]:
        _, meta = self.pp.restore_pandas(data)
        left = decode(meta['left'])
        right = decode(meta['right'])
        closed: Literal['both', 'neither', 'left', 'right'] = str(meta['closed'])  # type: ignore[assignment]
        obj = pd.Interval(left, right, closed=closed)
        return obj


def register_handlers() -> None:
    register_numpy_handlers()
    register(pd.DataFrame, PandasDfHandler, base=True)
    register(pd.Series, PandasSeriesHandler, base=True)
    register(pd.Index, PandasIndexHandler, base=True)
    register(pd.PeriodIndex, PandasPeriodIndexHandler, base=True)
    register(pd.MultiIndex, PandasMultiIndexHandler, base=True)
    register(pd.Timestamp, PandasTimestampHandler, base=True)
    register(pd.Period, PandasPeriodHandler, base=True)
    register(pd.Interval, PandasIntervalHandler, base=True)


def unregister_handlers() -> None:
    unregister_numpy_handlers()
    unregister(pd.DataFrame)
    unregister(pd.Series)
    unregister(pd.Index)
    unregister(pd.PeriodIndex)
    unregister(pd.MultiIndex)
    unregister(pd.Timestamp)
    unregister(pd.Period)
    unregister(pd.Interval)
