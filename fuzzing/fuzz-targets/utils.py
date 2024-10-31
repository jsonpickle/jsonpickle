import traceback  # pragma: no cover
from typing import Dict, List, Tuple, Union  # pragma: no cover

import atheris  # pragma: no cover


@atheris.instrument_func
def is_expected_error(
    exception: Exception, expected_errors: Dict[str, List[Tuple[str, Union[int]]]]
):  # pragma: no cover
    """Checks if a given exception matches any of the expected errors.

    This function inspects the traceback of the provided exception to determine if
    it originated from a file and line number listed in the expected errors, and if
    the exception message contains a specified substring.

    Args:
        exception (Exception): The exception to be checked.
        expected_errors (Dict[str, List[Tuple[str, Union[int]]]]): A dictionary where
            keys are filenames and values are lists of tuples. Each tuple contains a
            substring of an expected exception message and a line number (or -1 to
            match any line number).

    Returns:
        bool: True if the exception matches any of the expected errors, False otherwise.
    """
    tb = traceback.extract_tb(exception.__traceback__)
    if not tb:
        return False

    last_frame = tb[-1]
    exception_origin_file = last_frame.filename

    if exception_origin_file in expected_errors:
        origin_line = int(last_frame.lineno)
        exception_message = str(exception).lower()
        for message, line in expected_errors[exception_origin_file]:
            if message.lower() in exception_message and (
                line == -1 or line == origin_line
            ):
                return True
    return False
