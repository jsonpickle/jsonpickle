import os
import sys


def ensure_no_files_overwritten(expected_contents):
    # ensure we don't overwrite the user's files, if they run it on their machine
    if os.path.exists(os.path.join(os.getcwd(), 'example.txt')):
        with open("example.txt", "r") as f:
            contents = f.read()
        if contents != expected_contents:
            print(
                "I don't want to overwrite the example.txt file in my current directory! Please delete example.txt and try again!"
            )
            # exit with a non-zero error code
            sys.exit(1)
