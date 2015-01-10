# -*- coding: utf-8 -*-

from __future__ import absolute_import

import ast
import jsonpickle
import numpy as np

__all__ = ['register_handlers', 'unregister_handlers']


class NumpyBaseHandler(jsonpickle.handlers.BaseHandler):
    def restore_dtype(self, data):
        dtype = data['dtype']
        return np.dtype(dtype if not dtype.startswith(('{', '[')) else ast.literal_eval(dtype))


class NumpyDTypeHandler(NumpyBaseHandler):
    def flatten(self, obj, data):
        data['dtype'] = str(obj)
        return data

    def restore(self, data):
        return self.restore_dtype(data)


class NumpyGenericHandler(NumpyBaseHandler):
    def flatten(self, obj, data):
        data['dtype'] = str(obj.dtype)
        data['value'] = self.context.flatten(obj.tolist())
        return data

    def restore(self, data):
        return self.restore_dtype(data).type(self.context.restore(data['value']))


class NumpyNDArrayHandler(NumpyBaseHandler):
    def flatten(self, obj, data):
        data['dtype'] = str(obj.dtype)
        data['values'] = self.context.flatten(obj.tolist())
        return data

    def restore(self, data):
        return np.array(self.context.restore(data['values']),
                        dtype=self.restore_dtype(data))


def register_handlers():
    jsonpickle.handlers.register(np.dtype, NumpyDTypeHandler, base=True)
    jsonpickle.handlers.register(np.generic, NumpyGenericHandler, base=True)
    jsonpickle.handlers.register(np.ndarray, NumpyNDArrayHandler, base=True)


def unregister_handlers():
    jsonpickle.handlers.unregister(np.dtype)
    jsonpickle.handlers.unregister(np.generic)
    jsonpickle.handlers.unregister(np.ndarray)
