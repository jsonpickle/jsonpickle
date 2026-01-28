import jsonpickle
import jsonpickle.handlers


class RideTicket:
    def __init__(self, route: str, base_price: float, currency: str | None = None):
        self.route = route
        self.base_price = base_price
        self.currency = currency

    def __repr__(self) -> str:
        return (
            f"RideTicket(route={self.route!r}, base_price={self.base_price}, "
            f"currency={self.currency})"
        )


@jsonpickle.handlers.register(RideTicket)
class RideTicketHandler(jsonpickle.handlers.BaseHandler):
    # if either flatten or restore have an argument called handler_context
    # then it will be passed through encode/decode into the handler
    # handler_context doesn't need to be a dict, but it's recommended that it be
    def flatten(self, obj: RideTicket, data: dict, handler_context: dict) -> dict:
        currency = handler_context["currency"]
        data["route"] = obj.route
        data["base_price"] = obj.base_price
        data["price_label"] = f"{currency} {obj.base_price}"
        return data

    def restore(self, data: dict, handler_context: dict) -> object:
        return RideTicket(
            route=data["route"],
            base_price=data["base_price"],
            currency=handler_context["currency"],
        )


def main():
    local_commute = RideTicket("city bus", 1.50)
    encoded_local = jsonpickle.encode(
        local_commute, handler_context={"currency": "USD"}
    )

    print("Encoded:")
    print(encoded_local)

    decoded = jsonpickle.decode(encoded_local, handler_context={"currency": "EUR"})
    print("Decoded with different currency context:")
    print(decoded)
    assert decoded.currency == "EUR"
    print("Context changed how the handler interpreted the ticket!")


if __name__ == "__main__":
    main()
