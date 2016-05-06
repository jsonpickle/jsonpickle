# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
import zlib
import warnings

import numpy as np

import ast
import jsonpickle
from jsonpickle.compat import unicode

__all__ = ['register_handlers', 'unregister_handlers']

native_byteorder = '<' if sys.byteorder == 'little' else '>'


class NumpyBaseHandler(jsonpickle.handlers.BaseHandler):

    def restore_dtype(self, data):
        dtype = data['dtype']
        if dtype.startswith(('{', '[')):
            return ast.literal_eval(dtype)
        return np.dtype(dtype)

    def flatten_dtype(self, dtype, data):
        if hasattr(dtype, 'tostring'):
            data['dtype'] = dtype.tostring()
        else:
            dtype = unicode(dtype)
            prefix = '(numpy.record, '
            if dtype.startswith(prefix):
                dtype = dtype[len(prefix):-1]
            data['dtype'] = dtype


class NumpyDTypeHandler(NumpyBaseHandler):

    def flatten(self, obj, data):
        self.flatten_dtype(obj, data)
        return data

    def restore(self, data):
        return self.restore_dtype(data)


class NumpyGenericHandler(NumpyBaseHandler):

    def flatten(self, obj, data):
        self.flatten_dtype(obj.dtype, data)
        data['value'] = self.context.flatten(obj.tolist(), reset=False)
        return data

    def restore(self, data):
        value = self.context.restore(data['value'], reset=False)
        return self.restore_dtype(data).type(value)


class NumpyNDArrayHandler(NumpyBaseHandler):

    def flatten_flags(self, obj, data):
        if obj.flags.writeable is False:
            data['writeable'] = False

    def restore_flags(self, data, arr):
        if not data.get('writeable', True):
            arr.flags.writeable = False

    def flatten(self, obj, data):
        self.flatten_dtype(obj.dtype, data)
        self.flatten_flags(obj, data)
        data['values'] = self.context.flatten(obj.tolist(), reset=False)
        if 0 in obj.shape:
            # add shape information explicitly as it cannot be inferred from an empty list
            data['shape'] = obj.shape
        if obj.flags.f_contiguous:  # needed by views; move logic there? what about byteorder of text viewed by other?
            data['order'] = 'F'
        return data

    def restore(self, data):
        values = self.context.restore(data['values'], reset=False)
        arr = np.array(
            values,
            dtype=self.restore_dtype(data),
            order=data.get('order', 'C')
        )
        shape = data.get('shape', None)
        if shape is not None:
            arr = arr.reshape(shape)

        self.restore_flags(data, arr)
        return arr


class NumpyNDArrayHandlerBinary(NumpyNDArrayHandler):
    """stores arrays with size greater than 'size_treshold' as (optionally) compressed base64

    Notes
    -----
    This would be easier to implement using np.save/np.load, but that would be less language-agnostic
    """

    def __init__(self, size_treshold=16, compression=zlib):
        """
        :param size_treshold: nonnegative int or None
            valid values for 'size_treshold' are all nonnegative integers and None
            if size_treshold is None, values are always stored as nested lists
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compresion is None, no compression is applied
        """
        self.size_treshold = size_treshold
        self.compression = compression

    def flatten_byteorder(self, obj, data):
        data['byteorder'] = native_byteorder if obj.dtype.byteorder == '=' else obj.dtype.byteorder

    def restore_byteorder(self, data, arr):
        arr.dtype = arr.dtype.newbyteorder(data['byteorder'])

    def flatten(self, obj, data):
        """encode numpy to json"""
        if self.size_treshold > obj.size or self.size_treshold is None:
            # store as json
            data = super(NumpyNDArrayHandlerBinary, self).flatten(obj, data)
        else:
            # store as binary
            self.flatten_dtype(obj.dtype, data)
            buffer = obj.tobytes(order=None)    # store as C or Fortran order
            if self.compression:
                buffer = self.compression.compress(buffer)
            values = jsonpickle.util.b64encode(buffer)
            data['values'] = self.context.flatten(values, reset=False)
            data['shape'] = obj.shape
            if obj.flags.f_contiguous:
                data['order'] = 'F'
            self.flatten_byteorder(obj, data)
            self.flatten_flags(obj, data)
        return data

    def restore(self, data):
        """decode numpy from json"""
        values = data['values']
        if isinstance(values, list):
            # decode text representation
            arr = super(NumpyNDArrayHandlerBinary, self).restore(data)
        else:
            # decode binary representation
            values = self.context.restore(values, reset=False)
            buffer = jsonpickle.util.b64decode(values)
            if self.compression:
                buffer = self.compression.decompress(buffer)
            arr = np.ndarray(
                buffer=buffer,
                dtype=self.restore_dtype(data),
                shape=data.get('shape'),
                order=data.get('order', 'C')
            ).copy() # make a copy, to force the result to own the data
            self.restore_byteorder(data, arr)
            self.restore_flags(data, arr)

        return arr


class NumpyNDArrayHandlerView(NumpyNDArrayHandlerBinary):
    """Pickles references inside ndarrays, or array-views

    Notes
    -----
    The current implementation has some restrictions.

    'base' arrays, or arrays which are viewed by other arrays, must be f-or-c-contiguous.
    This is not such a large restriction in practice, because all numpy array creation is c-contiguous by default.
    Relaxing this restriction would be nice though; especially if it can be done without bloating the design too much.

    Furthermore, ndarrays which are views of array-like objects implementing __array_interface__,
    but which are not themselves nd-arrays, are deepcopied with a warning (by default),
    as we cannot guarantee whatever custom logic such classes implement is correctly reproduced.
    """
    def __init__(self, mode='warn', size_treshold=16, compression=zlib):
        """
        :param mode: {'warn', 'raise', 'ignore'}
            How to react when encountering array-like objects whos references we cannot safely serialize
        :param size_treshold: nonnegative int or None
            valid values for 'size_treshold' are all nonnegative integers and None
            if size_treshold is None, values are always stored as nested lists
        :param compression: a compression module or None
            valid values for 'compression' are {zlib, bz2, None}
            if compresion is None, no compression is applied
        """
        super(NumpyNDArrayHandlerView, self).__init__(size_treshold, compression)
        self.mode = mode

    def flatten(self, obj, data):
        """encode numpy to json"""
        base = obj.base
        if base is None and obj.flags.forc:
            # store by value
            data = super(NumpyNDArrayHandlerView, self).flatten(obj, data)
        elif isinstance(base, np.ndarray) and base.data.contiguous:
            # store by reference
            self.flatten_dtype(obj.dtype, data)
            data['base'] = self.context.flatten(base, reset=False)
            # for a view, always store shape
            data['shape'] = obj.shape

            offset = obj.ctypes.data - base.ctypes.data
            if offset:
                data['offset'] = offset

            if not obj.data.c_contiguous:
                data['strides'] = obj.strides

            self.flatten_byteorder(obj, data)
            self.flatten_flags(obj, data)
        else:
            # store a deepcopy or fail
            if self.mode == 'warn':
                msg = "ndarray is defined by reference to an object we do not know how to serialize. " \
                      "A deep copy is serialized instead, breaking memory aliasing."
                warnings.warn(msg)
            elif self.mode == 'raise':
                msg = "ndarray is defined by reference to an object we do not know how to serialize."
                raise ValueError(msg)
            data = super(NumpyNDArrayHandlerView, self).flatten(obj.copy(), data)

        return data

    def restore(self, data):
        """decode numpy from json"""
        base = data.get('base', None)
        if base is None:
            # decode array with owndata=True
            arr = super(NumpyNDArrayHandlerView, self).restore(data)
        else:
            # decode array view, which references the data of another array
            base = self.context.restore(base, reset=False)
            assert base.data.contiguous, \
                "Current implementation assumes base is C or F contiguous"

            arr = np.ndarray(
                buffer=base.data,
                dtype=self.restore_dtype(data),
                shape=data.get('shape'),
                offset=data.get('offset', 0),
                strides=data.get('strides', None)
            )
            self.restore_byteorder(data, arr)
            self.restore_flags(data, arr)

        return arr


def register_handlers():
    jsonpickle.handlers.register(np.dtype, NumpyDTypeHandler, base=True)
    jsonpickle.handlers.register(np.generic, NumpyGenericHandler, base=True)
    jsonpickle.handlers.register(np.ndarray, NumpyNDArrayHandlerView(), base=True)


def unregister_handlers():
    jsonpickle.handlers.unregister(np.dtype)
    jsonpickle.handlers.unregister(np.generic)
    jsonpickle.handlers.unregister(np.ndarray)
