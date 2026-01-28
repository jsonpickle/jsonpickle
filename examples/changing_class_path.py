"""
This example simulates the case where the path to a class changes after an object has been encoded.
"""

import sys
import types

import jsonpickle


class Thing:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Thing(value={self.value!r})"


def make_other_file_module():
    # this simulates the path changing after the object has been encoded
    module = types.ModuleType("other_file")

    class Thing2:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return f"other_file.Thing2(value={self.value!r})"

    Thing2.__module__ = "other_file"
    module.Thing2 = Thing2
    sys.modules["other_file"] = module
    return module


def main():
    original = Thing("hello")
    encoded = jsonpickle.encode(original)
    print("Encoded:")
    print(encoded)

    other_file = make_other_file_module()

    decoded = jsonpickle.decode(
        encoded,
        # this argument is key for remapping the serialized class to the new class!
        classes={"__main__.Thing": other_file.Thing2},
    )
    print("Decoded:")
    print(decoded)

    assert isinstance(decoded, other_file.Thing2), (
        "decode did not use the remapped class!"
    )
    assert decoded.value == "hello", "Attribute value did not round-trip properly!"

    print("Remap succeeded!")


if __name__ == "__main__":
    main()
