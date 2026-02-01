import pytest


class SkippableTest:
    def skip(self, msg: str):
        pytest.skip(msg)
