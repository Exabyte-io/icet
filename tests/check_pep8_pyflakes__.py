import subprocess
import sys


'''
This script tests code for pyflakes and pep8 compliance. It is usually run as
part of the test suite.

The following parameters can be specified to control the behavior of this
script.

Parameters
----------
rootdir : str
    absolute path to hiphive root directory or path relative to the directory
    where this script is executed

test_directories : list of str
    names of directories to be tested; paths have to be provided relative to
    the HiPhive root directory

warnings_to_ignore_pyflakes : list of str
    list of pyflakes warnings that should be ignored

warnings_to_ignore_pep8 : list of str
    list of pep8 warnings that should be ignored
'''

# General settings
rootdir = '../'
test_directories = ['icetdev', 'tests', 'doc', 'examples']
warnings_to_ignore_pyflakes = []
warnings_to_ignore_pep8 = []

# Prepend directory names with <rootdir>
directories = []
for dir in test_directories:
    directories.append('{}/{}'.format(rootdir, dir))

# Run pyflakes3
print('Running pyflakes...')
try:
    output = subprocess.check_output(['pyflakes3'] + directories)
except subprocess.CalledProcessError as ex:
    output = ex.output.decode()

lines = []
for line in output.splitlines():
    # Ignore these:
    for txt in warnings_to_ignore_pyflakes:
        if txt in line:
            break
    else:
        lines.append(line.replace(rootdir, ''))
if lines:
    print('Errors.')
    print('\n'.join(lines))
    sys.exit(1)
print('OK')

# Run pep8
print('\nRunning pep8...')
try:
    output = subprocess.check_output(['pep8'] + directories)
except subprocess.CalledProcessError as ex:
    output = ex.output.decode()

lines = []
for line in output.splitlines():
    # Ignore these:
    for txt in warnings_to_ignore_pep8:
        if txt in line:
            break
    else:
        lines.append(line.replace(rootdir, ''))
if lines:
    print('Errors.')
    print('\n'.join(lines))
    sys.exit(1)
print('OK')
