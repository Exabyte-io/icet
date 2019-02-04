#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from glob import glob

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


icet_cpp_module = Extension(
    '_icet',
    glob('src/*.cpp'),
    include_dirs=[
        'src/3rdparty/pybind11/include/',
        'src/3rdparty/boost_1_68_0/',
        'src/3rdparty/eigen3/'
    ],
    language='c++')


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """
    Returns a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """
    Returns the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append("-DVERSION_INFO='{}'"
                        .format(self.distribution.get_version()))
            opts.append(cpp_flag(self.compiler))
            opts.append('-O3')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append("/DVERSION_INFO=\\'{}\\'"
                        .format(self.distribution.get_version()))
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = opts
        build_ext.build_extensions(self)


if sys.version_info < (3, 5, 0, 'final', 0):
    raise SystemExit('Python 3.5 or later is required!')


with open('README.rst', encoding='utf-8') as fd:
    long_description = fd.read()

with open('icet/__init__.py', encoding='utf-8') as fd:
    try:
        lines = ''
        for item in fd.readlines():
            item = item
            lines += item + '\n'
    except Exception as exc:
        raise Exception('Caught exception {}'.format(exc))


author_list = ['Mattias Ångqvist',
               'William A. Muñoz',
               'J. Magnus Rahm',
               'Erik Fransson',
               'Céline Durniak',
               'Piotr Rozyczko',
               'Thomas Holm Rod',
               'Paul Erhart']
authors = ', '.join(author_list[:-1]) + ', and ' + author_list[-1]

version = re.search("__version__ = '(.*)'", lines).group(1)
maintainer = re.search("__maintainer__ = '(.*)'", lines).group(1)
email = re.search("__email__ = '(.*)'", lines).group(1)
description = re.search("__description__ = '(.*)'", lines).group(1)
url = re.search("__url__ = '(.*)'", lines).group(1)
license = re.search("__license__ = '(.*)'", lines).group(1)

classifiers = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: {}'.format(license),
    'Topic :: Scientific/Engineering :: Physics']


if __name__ == '__main__':

    setup(
        name='icet',
        version=version,
        author=authors,
        author_email=email,
        description=description,
        long_description=long_description,
        ext_modules=[icet_cpp_module],
        install_requires=['pybind11>=2.2',
                          'ase',
                          'numpy',
                          'scipy',
                          'sklearn',
                          'pandas>=0.23',
                          'spglib>1.12.0'],
        packages=find_packages(),
        cmdclass={'build_ext': BuildExt},
        zip_safe=False,
        classifiers=classifiers,
        license=license,
        url=url,
    )
