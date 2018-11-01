import os
import sys
import unittest
from icet.io.logging import set_log_config


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

    @property
    def id(self):
        return self.filename.split('tests')[-1]

    def __str__(self):
        return self.id

    def __repr__(self):
        return "ScriptTestCase(filename='%s')" % self.filename


def find_script_tests(suite, script_dir):

    tests = []
    for root, dirs, files in os.walk(script_dir):
        for f in files:
            if f.endswith('.py'):
                tests.append(os.path.join(root, f))

    tests.sort()
    for test in tests:
        if test.endswith('__.py'):
            continue
        test = ScriptTestCase(filename=test)
        suite.addTest(test)


if __name__ == "__main__":

    from generate_structures_for_testing import create_database

    create_database()

    set_log_config(level='ERROR')

    # Find testing dirs
    main_dir = os.path.dirname(__file__)
    unittest_dir = os.path.join(main_dir, 'unittest')
    integration_dir = os.path.join(main_dir, 'integration')

    # Collect tests
    suite = unittest.TestLoader().discover(unittest_dir, pattern="*.py")
    find_script_tests(suite, integration_dir)

    # Run tests
    ttr = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    results = ttr.run(suite)

    assert len(results.failures) == 0, "At least one test failed"
    assert len(results.errors) == 0, "At least one test failed"
