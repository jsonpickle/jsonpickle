import sys
import datetime
import collections

import jsonpickle


class DatetimeHandler(jsonpickle.handlers.BaseHandler):
    """
    Datetime objects use __reduce__, and they generate binary strings encoding
    the payload. This handler encodes that payload to reconstruct the
    object.
    """
    _handles = datetime.datetime, datetime.date, datetime.time

    def flatten(self, obj, data):
        pickler = self._base
        if not pickler.unpicklable:
            return unicode(obj)
        cls, args = obj.__reduce__()
        flatten = pickler.flatten
        args = [args[0].encode('base64')] + [flatten(i, reset=False) for i in args[1:]]
        data['__reduce__'] = (flatten(cls, reset=False), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        value = args[0].decode('base64')
        unpickler = self._base
        restore = unpickler.restore
        cls = restore(cls, reset=False)
        params = (value,) + tuple([restore(i, reset=False) for i in args[1:]])
        return cls.__new__(cls, *params)


class SimpleReduceHandler(jsonpickle.handlers.BaseHandler):
    """
    Follow the __reduce__ protocol to pickle an object. As long as the factory
    and its arguments are pickleable, this should pickle any object that
    implements the reduce protocol.
    """
    _handles = [datetime.timedelta]
    if sys.version_info >= (2, 7):
        _handles.append(collections.OrderedDict)

    def flatten(self, obj, data):
        pickler = self._base
        if not pickler.unpicklable:
            return unicode(obj)
        flatten = pickler.flatten
        data['__reduce__'] = [flatten(i, reset=False) for i in obj.__reduce__()]
        return data

    def restore(self, obj):
        unpickler = self._base
        restore = unpickler.restore
        factory, args = [restore(i, reset=False) for i in obj['__reduce__']]
        return factory(*args)
