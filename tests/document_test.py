import unittest
import jsonpickle

from jsonpickle._samples import Document, Section, Question

class DocumentTestCase(unittest.TestCase):
    def test_cyclical(self):
        """Test that we can pickle cyclical data structure

        This test is ensures that we can reference objects which
        first appear within a list (in other words, not a top-level
        object or attribute).  Later children will reference that
        object through its "_parent" field.

        This makes sure that we handle this case correctly.

        """
        document = Document('My Document')
        section1 = Section('Section 1')
        section2 = Section('Section 2')
        question1 = Question('Question 1')
        question2 = Question('Question 2')
        question3 = Question('Question 3')
        question4 = Question('Question 4')

        document.add_child(section1)
        document.add_child(section2)
        section1.add_child(question1)
        section1.add_child(question2)
        section2.add_child(question3)
        section2.add_child(question4)

        pickled = jsonpickle.encode(document)
        unpickled = jsonpickle.decode(pickled)

        self.assertEqual(str(document), str(unpickled))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DocumentTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
