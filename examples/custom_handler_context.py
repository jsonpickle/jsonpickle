import jsonpickle
import jsonpickle.handlers


class A:
    def __init__(self, identity: str):
        self.identity = identity


@jsonpickle.handlers.register(A)
class AHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj: A, data: dict, handler_context: dict) -> dict:
        data["payload"] = f"{handler_context['foo']}:{obj.identity}"
        return data

    def restore(self, data: dict, handler_context: dict) -> object:
        payload = data["payload"]
        return A(f"{payload}-{handler_context['foo']}")


a0 = A("first")
a1 = A("second")

encoded0 = jsonpickle.encode(a0, context={"foo": "bar"})
encoded1 = jsonpickle.encode(a1, context={"foo": "baz"})

print(encoded0)
print(encoded1)

decoded = jsonpickle.decode(encoded0, context={"foo": "qux"})
assert decoded.identity == "bar:first-qux"
print("Context has been propagated through the handler!")
