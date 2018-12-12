"""
In order to support installation of pep517 from source,
pip from master must be installed.
"""

import subprocess
import sys


def main():
	cmd = [
		sys.executable,
		'-m', 'pip', 'install',
		'git+https://github.com/pypa/pip',
	]
	subprocess.run(cmd)
	cmd[-1:] = sys.argv[1:]
	subprocess.run(cmd)


__name__ == '__main__' and main()
