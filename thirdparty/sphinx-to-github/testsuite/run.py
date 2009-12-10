#!/usr/bin/env python

import unittest
import sys

sys.path.append(".")

from sphinxtogithub.test import directoryhandler, filehandler, replacer, renamer


if __name__ == "__main__":

    suites = [
            filehandler.testSuite(),
            directoryhandler.testSuite(),
            replacer.testSuite(),
            renamer.testSuite(),
            ]

    suite = unittest.TestSuite(suites)
    
    runner = unittest.TextTestRunner()

    runner.run(suite)

