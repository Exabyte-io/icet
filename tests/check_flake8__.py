import subprocess
import sys
import os


"""
This script tests code for pyflakes and pep8 compliance using
flake8. It is usually run as part of the test suite.

The following parameters can be specified to control the behavior of this
script.

Parameters
----------
test_directories : list of str
    names of directories to be tested; paths have to be provided relative to
    the icet root directory
"""

# General settings
path = os.path.abspath(__file__)
rootdir = os.path.abspath(os.path.dirname(path) + '/..')
test_directories = ['icet', 'mchammer',
                    'tests', 'benchmark',
                    'tutorial', 'doc', 'examples']

# Prepend directory names with <rootdir>
directories = []
for dir in test_directories:
    fullpath = '{}/{}'.format(rootdir, dir)
    fullpath = os.path.abspath(fullpath)
    directories.append(fullpath)

# Run flake8
print('Testing the following directories:')
for dir in directories:
    print('  {}'.format(dir))
try:
    output = subprocess.check_output(
        'python3 -m flake8'.split() + directories)
except subprocess.CalledProcessError as ex:
    output = ex.output.decode()

# Compile and clean up error and/or warnings
lines = []
for line in output.splitlines():
    lines.append('  ' + line.replace(rootdir, '.'))

# Raise error if there are error and/or warnings
if lines:
    print('\nErrors and warnings:')
    print('\n'.join(lines))
    sys.exit(1)
# OK otherwise
print('OK')
