"""
Downgrade to pip 19.0 before installing requirements, working
around limitations introduced in 19.1 (ref
https://github.com/pypa/pip/issues/6434)
"""

import sys
import subprocess
import shlex


def main():
	subprocess.check_call(shlex.split(
		'python -m pip install pip<19.1'
	))
	subprocess.check_call(shlex.split(
		'python -m pip install') + sys.argv[1:])


__name__ == '__main__' and main()
