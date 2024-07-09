import os
import sys


def ensure_no_files_overwritten(expected_contents):
    # ensure we don't overwrite the user's files, if they run it on their machine
    if os.path.exists("example.json"):
        with open("example.json", "r") as f:
            contents = f.read()
        if contents != expected_contents:
            # exit with a non-zero error code
            sys.exit(
                "I don't want to overwrite the example.json file in my current directory! Please delete example.txt and try again!"
            )
