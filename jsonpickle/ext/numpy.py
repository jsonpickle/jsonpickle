from __future__ import annotations

import ast
import json
import sys
import warnings
import zlib
from types import ModuleType
from typing import Any, Dict, Tuple

import numpy as np

# do the annotations import so python doesn't complain about numpy type hints missing if numpy isn't installed


_np_version: Tuple[int, ...] = tuple(int(x) for x in np.__version__.split('.')[:3])
# numpy.typing was introduced in 1.20.0
if _np_version >= (1, 20, 0):
    from numpy.typing import ArrayLike, DTypeLike, NDArray
else:
    NDArray = Any  # type: ignore[assignment,misc]
    ArrayLike = Any  # type: ignore[assignment,misc]
    DTypeLike = Any  # type: ignore[assignment,misc]


from ..handlers import BaseHandler, register, unregister
from ..util import b64decode, b64encode

__all__ = ['register_handlers', 'unregister_handlers']

native_byteorder: str = '<' if sys.byteorder == 'little' else '>'


# considered typing arr as ArrayLike but then we get a mypy error with no attribute dtype
def get_byteorder(arr: ArrayLike) -> str:
    """translate equals sign to native order"""
    byteorder = arr.dtype.byteorder  # type: ignore[union-attr]
    return native_byteorder if byteorder == '=' else byteorder


class NumpyBaseHandler(BaseHandler):
    def flatten_dtype(self, dtype: DTypeLike, data: Dict[str, Any]) -> None:
        if hasattr(dtype, 'tostring'):
            data['dtype'] = dtype.tostring()  # type: ignore[union-attr]
        else:
            dtype = str(dtype)
            prefix = '(numpy.record, '
            if dtype.startswith(prefix):
                dtype = dtype[len(prefix) : -1]
            data['dtype'] = dtype

    def restore_dtype(self, data: Dict[str, Any]) -> np.dtype:
        dtype = data['dtype']
        if dtype.startswith(('{', '[')):
            dtype = ast.literal_eval(dtype)
        return np.dtype(dtype)


class NumpyDTypeHandler(NumpyBaseHandler):
    def flatten(self, obj: DTypeLike, data: Dict[str, Any]) -> Dict[str, Any]:
        self.flatten_dtype(obj, data)
        return data

    def restore(self, data: Dict[str, Any]) -> DTypeLike:
        return self.restore_dtype(data)


class NumpyGenericHandler(NumpyBaseHandler):
    def flatten(self, obj: NDArray, data: Dict[str, Any]) -> Dict[str, Any]:
        self.flatten_dtype(obj.dtype.newbyteorder('N'), data)
        data['value'] = self.context.flatten(obj.tolist(), reset=False)
        return data

    def restore(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = self.context.restore(data['value'], reset=False)
        return self.restore_dtype(data).type(value)


class NumpyDatetimeHandler(NumpyGenericHandler):
    """Extend NumpyGenericHandler to handle nanosecond-resolution datetime64"""

    def restore(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = self.context.restore(data['value'], reset=False)
        dtype = data['dtype']
        if dtype.endswith('[ns]'):
            return self.restore_dtype(data).type(value, 'ns')
        return self.restore_dtype(data).type(value)


class UnpickleableNumpyGenericHandler(NumpyGenericHandler):
    """
    From issue #381, this is used for simplifying the output of numpy arrays
    when unpicklable=False (the default is True).
    """

    # TODO: narrow return value down from Any
    def flatten(self, obj: NDArray, data: Dict[str, Any]) -> Any:
        if not self.context.unpicklable:
            return self.context.flatten(obj.tolist(), reset=False)
        else:
            return super(NumpyGenericHandler, self).flatten(obj, data)

    def restore(self, data: Dict[str, Any]):
        raise NotImplementedError


class NumpyNDArrayHandler(NumpyBaseHandler):
    """Stores arrays as text representation, without regard for views"""

    def flatten_flags(self, obj: NDArray, data: Dict[str, Any]) -> None:
        if obj.flags.writeable is False:
            data['writeable'] = False

    def restore_flags(self, data: Dict[str, Any], arr: NDArray) -> None:
        if not data.get('writeable', True):
            arr.flags.writeable = False

    def flatten(self, obj: NDArray, data: Dict[str, Any]) -> Dict[str, Any]:
        self.flatten_dtype(obj.dtype.newbyteorder('N'), data)
        self.flatten_flags(obj, data)
        data['values'] = self.context.flatten(obj.tolist(), reset=False)
        if 0 in obj.shape:
            # add shape information explicitly as it cannot be
            # inferred from an empty list
            data['shape'] = obj.shape
        return data

    def restore(self, data: Dict[str, Any]) -> NDArray:
        values = self.context.restore(data['values'], reset=False)
        arr = np.array(
            values, dtype=self.restore_dtype(data), order=data.get('order', 'C')
        )
        shape = data.get('shape', None)
        if shape is not None:
            arr = arr.reshape(shape)

        self.restore_flags(data, arr)
        return arr


class NumpyNDArrayHandlerBinary(NumpyNDArrayHandler):
    """stores arrays with size greater than 'size_threshold' as
    (optionally) compressed base64

    Notes
    -----
    This would be easier to implement using np.save/np.load, but
    that would be less language-agnostic
    """

    def __init__(
        self, size_threshold: int = 16, compression: ModuleType = zlib
    ) -> None:
        """
        :param size_threshold: nonnegative int or None
            valid values for 'size_threshold' are all nonnegative
            integers and None
            if size_threshold is None, values are always stored as nested lists
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compression is None, no compression is applied
        """
        self.size_threshold = size_threshold
        self.compression = compression

    def flatten_byteorder(self, obj: NDArray, data: Dict[str, Any]) -> None:
        byteorder = obj.dtype.byteorder
        if byteorder != '|':
            data['byteorder'] = get_byteorder(obj)

    def restore_byteorder(self, data: Dict[str, Any], arr: NDArray) -> None:
        byteorder = data.get('byteorder', None)
        if byteorder:
            arr.dtype = arr.dtype.newbyteorder(byteorder)  # type: ignore[misc]

    def flatten(self, obj: NDArray, data: Dict[str, Any]) -> Dict[str, Any]:
        """encode numpy to json"""
        if self.size_threshold is None or self.size_threshold >= obj.size:
            # encode as text
            data = super().flatten(obj, data)
        else:
            # encode as binary
            if obj.dtype == object:
                # There's a bug deep in the bowels of numpy that causes a
                # segfault when round-tripping an ndarray of dtype object.
                # E.g., the following will result in a segfault:
                #     import numpy as np
                #     arr = np.array([str(i) for i in range(3)],
                #                    dtype=object)
                #     dtype = arr.dtype
                #     shape = arr.shape
                #     buf = arr.tobytes()
                #     del arr
                #     arr = np.ndarray(buffer=buf, dtype=dtype,
                #                      shape=shape).copy()
                # So, save as a binary-encoded list in this case
                buf = json.dumps(obj.tolist()).encode()
            elif hasattr(obj, 'tobytes'):
                # numpy docstring is lacking as of 1.11.2,
                # but this is the option we need
                buf = obj.tobytes(order='A')
            else:
                # numpy < 1.9 compatibility
                buf = obj.tostring(order='a')  # type: ignore[attr-defined]
            if self.compression:
                buf = self.compression.compress(buf)
            data['values'] = b64encode(buf)
            data['shape'] = obj.shape
            self.flatten_dtype(obj.dtype.newbyteorder('N'), data)
            self.flatten_byteorder(obj, data)
            self.flatten_flags(obj, data)

            if not obj.flags.c_contiguous:
                data['order'] = 'F'

        return data

    def restore(self, data: Dict[str, Any]) -> NDArray:
        """decode numpy from json"""
        values = data['values']
        if isinstance(values, list):
            # decode text representation
            arr = super().restore(data)
        elif isinstance(values, (int, float)):
            # single-value array
            arr = np.array([values], dtype=self.restore_dtype(data))
        else:
            # decode binary representation
            dtype = self.restore_dtype(data)
            buf = b64decode(values)
            if self.compression:
                buf = self.compression.decompress(buf)
            # See note above about segfault bug for numpy dtype object. Those
            # are saved as a list to work around that.
            if dtype == object:
                values = json.loads(buf.decode())
                arr = np.array(values, dtype=dtype, order=data.get('order', 'C'))
                shape = data.get('shape', None)
                if shape is not None:
                    arr = arr.reshape(shape)
            else:
                arr = np.ndarray(
                    buffer=buf,
                    dtype=dtype,
                    shape=data.get('shape'),  # type: ignore[arg-type]
                    order=data.get('order', 'C'),
                ).copy()  # make a copy, to force the result to own the data
                self.restore_byteorder(data, arr)
            self.restore_flags(data, arr)

        return arr


class NumpyNDArrayHandlerView(NumpyNDArrayHandlerBinary):
    """Pickles references inside ndarrays, or array-views

    Notes
    -----
    The current implementation has some restrictions.

    'base' arrays, or arrays which are viewed by other arrays,
    must be f-or-c-contiguous.
    This is not such a large restriction in practice, because all
    numpy array creation is c-contiguous by default.
    Relaxing this restriction would be nice though; especially if
    it can be done without bloating the design too much.

    Furthermore, ndarrays which are views of array-like objects
    implementing __array_interface__,
    but which are not themselves nd-arrays, are deepcopied with
    a warning (by default),
    as we cannot guarantee whatever custom logic such classes
    implement is correctly reproduced.
    """

    def __init__(
        self,
        mode: str = 'warn',
        size_threshold: int = 16,
        compression: ModuleType = zlib,
    ) -> None:
        """
        :param mode: {'warn', 'raise', 'ignore'}
            How to react when encountering array-like objects whose
            references we cannot safely serialize
        :param size_threshold: nonnegative int or None
            valid values for 'size_threshold' are all nonnegative
            integers and None
            if size_threshold is None, values are always stored as nested lists
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compression is None, no compression is applied
        """
        super().__init__(size_threshold, compression)
        self.mode = mode

    def flatten(self, obj: NDArray, data: Dict[str, Any]) -> Dict[str, Any]:
        """encode numpy to json"""
        base = obj.base
        if base is None and obj.flags.forc:
            # store by value
            data = super().flatten(obj, data)
            # ensure that views on arrays stored as text
            # are interpreted correctly
            if not obj.flags.c_contiguous:
                data['order'] = 'F'
        elif isinstance(base, np.ndarray) and base.flags.forc:
            # store by reference
            data['base'] = self.context.flatten(base, reset=False)

            offset = obj.ctypes.data - base.ctypes.data
            if offset:
                data['offset'] = offset

            if not obj.flags.c_contiguous:
                data['strides'] = obj.strides

            data['shape'] = obj.shape
            self.flatten_dtype(obj.dtype.newbyteorder('N'), data)
            self.flatten_flags(obj, data)

            if get_byteorder(obj) != '|':
                byteorder = 'S' if get_byteorder(obj) != get_byteorder(base) else None
                if byteorder:
                    data['byteorder'] = byteorder

            if self.size_threshold is None or self.size_threshold >= obj.size:
                # not used in restore since base is present, but
                # include values for human-readability
                super(NumpyNDArrayHandlerBinary, self).flatten(obj, data)
        else:
            # store a deepcopy or fail
            if self.mode == 'warn':
                msg = (
                    "ndarray is defined by reference to an object "
                    "we do not know how to serialize. "
                    "A deep copy is serialized instead, breaking "
                    "memory aliasing."
                )
                warnings.warn(msg)
            elif self.mode == 'raise':
                msg = (
                    "ndarray is defined by reference to an object we do "
                    "not know how to serialize."
                )
                raise ValueError(msg)
            data = super().flatten(obj.copy(), data)

        return data

    def restore(self, data: Dict[str, Any]) -> NDArray:
        """decode numpy from json"""
        base = data.get('base', None)
        if base is None:
            # decode array with owndata=True
            arr = super().restore(data)
        else:
            # decode array view, which references the data of another array
            base = self.context.restore(base, reset=False)
            if not isinstance(base, np.ndarray):
                # the object is probably a nested list
                base = np.array(base)
            assert (
                base.flags.forc
            ), "Current implementation assumes base is C or F contiguous"

            arr = np.ndarray(
                buffer=base.data,
                dtype=self.restore_dtype(data).newbyteorder(data.get('byteorder', '|')),
                shape=data.get('shape'),  # type: ignore[arg-type]
                offset=data.get('offset', 0),
                strides=data.get('strides', None),
            )

            self.restore_flags(data, arr)

        return arr


def register_handlers(
    ndarray_mode: str = 'warn',
    ndarray_size_threshold: int = 16,
    ndarray_compression: ModuleType = zlib,
) -> None:
    """Register handlers for numpy types

    :param ndarray_abc_xyz: Forward constructor arguments to NumpyNDArrayHandlerView.
        Options with an 'ndarray_' prefix correspond to the same-named
        NumpyNDArrayHandlerView constructor options, sans the 'ndarray_' prefix.
    """
    ndarray_handler = NumpyNDArrayHandlerView(
        mode=ndarray_mode,
        size_threshold=ndarray_size_threshold,
        compression=ndarray_compression,
    )
    register(np.ndarray, ndarray_handler, base=True)  # type: ignore[arg-type]
    register(np.dtype, NumpyDTypeHandler, base=True)
    register(np.generic, NumpyGenericHandler, base=True)
    # Numpy 1.20 has custom dtypes that must be registered separately.
    register(np.dtype(np.void).__class__, NumpyDTypeHandler, base=True)
    register(np.dtype(np.float32).__class__, NumpyDTypeHandler, base=True)
    register(np.dtype(np.int32).__class__, NumpyDTypeHandler, base=True)
    register(np.dtype(np.datetime64).__class__, NumpyDTypeHandler, base=True)
    register(np.datetime64, NumpyDatetimeHandler, base=True)


def unregister_handlers() -> None:
    """Remove numpy handlers from the handler registry"""
    unregister(np.dtype)
    unregister(np.generic)
    unregister(np.ndarray)
    # Numpy 1.20 dtypes
    unregister(np.dtype(np.void).__class__)
    unregister(np.dtype(np.float32).__class__)
    unregister(np.dtype(np.int32).__class__)
    unregister(np.dtype(np.datetime64).__class__)
