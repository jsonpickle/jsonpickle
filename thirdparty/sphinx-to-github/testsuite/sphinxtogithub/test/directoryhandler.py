
import unittest
import os

import sphinxtogithub


class TestRenamer(object):

    def __call__(self, from_, to):

        self.from_ = from_
        self.to = to

class TestDirectoryHandler(unittest.TestCase):

    def setUp(self):

        self.directory = "_static"
        self.new_directory = "static"
        self.root = os.path.join("build", "html")
        renamer = TestRenamer()
        self.dir_handler = sphinxtogithub.DirectoryHandler(self.directory, self.root, renamer)

    def tearDown(self):
        
        self.dir_handler = None
    

    def testPath(self):

        self.assertEqual(self.dir_handler.path(), os.path.join(self.root, self.directory))

    def testRelativePath(self):

        dir_name = "css"
        dir_path = os.path.join(self.root, self.directory, dir_name)
        filename = "cssfile.css"

        self.assertEqual(
                self.dir_handler.relative_path(dir_path, filename),
                os.path.join(self.directory, dir_name, filename)
                )

    def testNewRelativePath(self):

        dir_name = "css"
        dir_path = os.path.join(self.root, self.directory, dir_name)
        filename = "cssfile.css"

        self.assertEqual(
                self.dir_handler.new_relative_path(dir_path, filename),
                os.path.join(self.new_directory, dir_name, filename)
                )

    def testProcess(self):

        self.dir_handler.process()

        self.assertEqual(
                self.dir_handler.renamer.to,
                os.path.join(self.root, self.new_directory)
                )

        self.assertEqual(
                self.dir_handler.renamer.from_,
                os.path.join(self.root, self.directory)
                )


def testSuite():
    suite = unittest.TestSuite()

    suite.addTest(TestDirectoryHandler("testPath"))
    suite.addTest(TestDirectoryHandler("testRelativePath"))
    suite.addTest(TestDirectoryHandler("testNewRelativePath"))
    suite.addTest(TestDirectoryHandler("testProcess"))

    return suite

