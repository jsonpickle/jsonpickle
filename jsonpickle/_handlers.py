import datetime

import jsonpickle

class DatetimeHandler(jsonpickle.handlers.BaseHandler):
    """
    Datetime objects use __reduce__, and they generate binary strings encoding
    the payload. This handler encodes that payload to reconstruct the
    object.
    """
    def flatten(self, obj, data):
        cls, args = obj.__reduce__()
        pickler = jsonpickle.Pickler()
        args = [args[0].encode('base64')] + map(pickler.flatten, args[1:])
        data['__reduce__'] = (pickler.flatten(cls), args)
        return data

    def restore(self, obj):
        cls, args = obj['__reduce__']
        value = args[0].decode('base64')
        unpickler = jsonpickle.Unpickler()
        cls = unpickler.restore(cls)
        params = map(unpickler.restore, args[1:])
        params = (value,) + tuple(params)
        return cls.__new__(cls, *params)

class SimpleReduceHandler(jsonpickle.handlers.BaseHandler):
    """
    Follow the __reduce__ protocol to pickle an object. As long as the factory
    and its arguments are pickleable, this should pickle any object that
    implements the reduce protocol.
    """
    def flatten(self, obj, data):
        pickler = jsonpickle.Pickler()
        data['__reduce__'] = map(pickler.flatten, obj.__reduce__())
        return data

    def restore(self, obj):
        unpickler = jsonpickle.Unpickler()
        cls, args = map(unpickler.restore, obj['__reduce__'])
        return cls.__new__(cls, *args)

jsonpickle.handlers.registry.register(datetime.datetime, DatetimeHandler)
jsonpickle.handlers.registry.register(datetime.date, DatetimeHandler)
jsonpickle.handlers.registry.register(datetime.time, DatetimeHandler)
jsonpickle.handlers.registry.register(datetime.timedelta, SimpleReduceHandler)
