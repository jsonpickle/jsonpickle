from utilities import ensure_no_files_overwritten

import jsonpickle

ensure_no_files_overwritten(
    expected_contents='{"py/object": "__main__.Example", "data": {"BAR": 1, "foo": 0}}'
)


class Example:
    def __init__(self):
        self.data = {"foo": 0, "BAR": 1}

    def get_foo(self):
        return self.data["foo"]

    def __eq__(self, other):
        # ensure that jsonpickle preserves important stuff like data across encode/decoding
        return self.data == other.data and self.get_foo() == other.get_foo()


# instantiate the class
ex = Example()
# encode the class. this returns a string, like json.dumps
encoded_instance = jsonpickle.encode(ex)
assert (
    encoded_instance
    == '{"py/object": "__main__.Example", "data": {"BAR": 1, "foo": 0}}'
)
print(
    f"jsonpickle successfully encoded the instance of the Example class! It looks like: {encoded_instance}"
)


with open("example.json", "w+") as f:
    f.write(encoded_instance)
    print(
        "jsonpickle successfully wrote the instance of the Example class to example.json!"
    )

with open("example.json", "r+") as f:
    written_instance = f.read()
    # decode the file into a copy of the original object
    decoded_instance = jsonpickle.decode(written_instance)
    print("jsonpickle successfully decoded the instance of the Example class!")

# this should be true, if it isn't then you should file a bug report
assert decoded_instance == ex
print(
    "jsonpickle successfully decoded the instance of the Example class, and it matches the original instance!"
)
