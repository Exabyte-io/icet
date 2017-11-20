from __future__ import print_function
import os
import unittest
from glob import glob
import test_structure_container as TestStructureContainer

class ScriptTestCase(unittest.TestCase):
    def __init__(self, methodname='testfile', filename=None):
        unittest.TestCase.__init__(self, methodname)
        self.filename = filename

    def testfile(self):
        try:
            with open(self.filename) as fd:
                exec(compile(fd.read(), self.filename, 'exec'), {})
        except KeyboardInterrupt:
            raise RuntimeError('Keyboard interrupt')
        except ImportError as ex:
            module = ex.args[0].split()[-1].replace("'", '').split('.')[0]
            if module in ['scipy', 'matplotlib']:
                raise unittest.SkipTest('no {} module'.format(module))
            else:
                raise

    def id(self):
        return self.filename

    def __str__(self):
        return self.filename.split('test/')[-1]

    def __repr__(self):
        return "ScriptTestCase(filename='%s')" % self.filename


def run_test(verbosity=2, files=None):
    '''
    Run test
    '''

    test_dir = os.path.dirname(os.path.realpath(__file__))
    testSuite = unittest.TestSuite()

    if files:
        files = [os.path.join(test_dir, f) for f in files]
    else:
        files = glob(test_dir + '/*')
    sdirtests = []  # tests from subdirectories: only one level assumed
    tests = []
    for f in files:
        if os.path.isdir(f):
            # add test subdirectories (like calculators)
            sdirtests.extend(glob(f + '/*.py'))
        else:
            # add py files in testdir
            if f.endswith('.py'):
                tests.append(f)
    tests.sort()
    sdirtests.sort()
    tests.extend(sdirtests)  # run test subdirectories at the end
    for test in tests:
        if test.endswith('__.py'):
            continue
        elif test.endswith('_container.py'):
            testSuite.addTest(TestStructureContainer.suite())
        else:
            testSuite.addTest(ScriptTestCase(filename=os.path.abspath(test)))

    print('')
    ttr = unittest.TextTestRunner(verbosity=verbosity)
    results = ttr.run(testSuite)

    return results


if __name__ == '__main__':
    results = run_test()
    assert len(results.failures) == 0, 'At least one test failed'
    assert len(results.errors) == 0, 'At least one test failed'
