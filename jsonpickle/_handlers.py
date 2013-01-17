import jsonpickle
import datetime

class DatetimeHandler(jsonpickle.handlers.BaseHandler):
	def flatten(self, obj, data):
		cls, args = obj.__reduce__()
		pickler = jsonpickle.Pickler()
		args = [args[0].encode('base64')] + map(pickler.flatten, args[1:])
		data['__reduce__'] = args
		return data

	def restore(self, obj):
		args = obj['__reduce__']
		value = args[0].decode('base64')
		unpickler = jsonpickle.Unpickler()
		params = map(unpickler.restore, args[1:])
		params = (value,) + tuple(params)
		return datetime.datetime.__new__(datetime.datetime, *params)

jsonpickle.util.NEEDS_REPR = ()
jsonpickle.handlers.registry.register(datetime.datetime, DatetimeHandler)
