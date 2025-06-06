import sys
from pathlib import Path

collect_ignore = ['contrib', 'examples', 'build']


def pytest_ignore_collect(collection_path, config):
    p = Path(collection_path)
    if any(directory in p.parts for directory in collect_ignore):
        return True
    # atheris isn't available on python 3.13+, so we disable fuzzing when that's the case
    if sys.version_info >= (3, 13):
        if "fuzzing" in p.parts:
            return True
    return False


def pytest_addoption(parser):
    parser.addoption(
        '--repeat', action='store', help='Number of times to repeat each test'
    )


def pytest_generate_tests(metafunc):
    if metafunc.config.option.repeat is not None:
        count = int(metafunc.config.option.repeat)

        # We're going to duplicate these tests by parametrizing them,
        # which requires that each test has a fixture to accept the parameter.
        # We can add a new fixture like so:
        metafunc.fixturenames.append('tmp_ct')

        # Now we parametrize. This is what happens when we do e.g.,
        # @pytest.mark.parametrize('tmp_ct', range(count))
        # def test_foo(): pass
        metafunc.parametrize('tmp_ct', range(count))
