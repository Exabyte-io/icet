#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import re
from multiprocessing import Process
import glob


def find_eigen(hint=None):
    """
    Find the location of the Eigen 3 include directory. This will return
    ``None`` on failure.
    """
    # List the standard locations including a user supplied hint.
    search_dirs = [] if hint is None else hint
    search_dirs += [
        "/usr/local/include/eigen3",
        "/usr/local/homebrew/include/eigen3",
        "/opt/local/var/macports/software/eigen3",
        "/opt/local/include/eigen3",
        "/usr/include/eigen3",
        "/usr/include/local",
        "/usr/include",
        "src/3rdparty/eigen3/"
    ]

    # Loop over search paths and check for the existence of the Eigen/Dense
    # header.
    for d in search_dirs:
        path = os.path.join(d, "Eigen", "Dense")
        if os.path.exists(path):
            # Determine the version.
            vf = os.path.join(d, "Eigen", "src", "Core", "util", "Macros.h")
            if not os.path.exists(vf):
                continue
            src = open(vf, "r").read()
            v1 = re.findall("#define EIGEN_WORLD_VERSION (.+)", src)
            v2 = re.findall("#define EIGEN_MAJOR_VERSION (.+)", src)
            v3 = re.findall("#define EIGEN_MINOR_VERSION (.+)", src)
            if not len(v1) or not len(v2) or not len(v3):
                continue
            v = "{0}.{1}.{2}".format(v1[0], v2[0], v3[0])
            print("Found Eigen version {0} in: {1}".format(v, d))
            return d
    return None


eigen_include = find_eigen()
if eigen_include is None:
    raise RuntimeError("Required library Eigen not found. "
                       "Check the documentation for solutions.")


ext_modules = [
    Extension(
        '_icet',
        ['src/ClusterCounts.cpp',
         'src/Cluster.cpp',
         'src/ClusterExpansionCalculator.cpp',
         'src/ClusterSpace.cpp',
         'src/Geometry.cpp',
         'src/LocalOrbitListGenerator.cpp',
         'src/ManyBodyNeighborList.cpp',
         'src/NeighborList.cpp',
         'src/Orbit.cpp',
         'src/OrbitList.cpp',
         'src/PermutationMatrix.cpp',
         'src/PyBinding.cpp',
         'src/Structure.cpp',
         'src/Symmetry.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            'src/3rdparty/pybind11/include/',
            'src/3rdparty/boost_1_68_0/',
            eigen_include,
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
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
    """Return the -std=c++[11/14] compiler flag.

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
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            opts.append('-O3')
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


if sys.version_info < (3, 5, 0, 'final', 0):
    raise SystemExit('Python 3.5 or later is required!')


with open('README.md', encoding="utf-8") as fd:
    long_description = fd.read()

with open('icet/__init__.py', encoding='utf-8') as fd:
    try:
        lines = ''
        for item in fd.readlines():
            item = item
            lines += item + "\n"
    except Exception as e:
        raise Exception("Caught exception {}".format(e))


version = re.search("__version__ = '(.*)'", lines).group(1)
maintainer = re.search("__maintainer__ = '(.*)'", lines).group(1)
email = re.search("__email__ = '(.*)'", lines).group(1)
description = re.search("__description__ = '(.*)'", lines).group(1)
authors = 'Mattias Ångqvist, William A. Muñoz, J. Magnus Rahm, Erik Fransson, Céline Durniak, Piotr Rozyczko, Thomas Holm Rod and Paul Erhart'
# authors = 'Mattias Ångqvist, William Armando Muñoz, Thomas Holm Rod and Paul Erhart'

homepage = 'https://gitlab.com/materials-modeling/icet'
license = 'Mozilla Public License Version 2.0'

classifiers = [
    'Development Status :: 4 - Beta',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: UNIX',
    'Operating System :: LINUX',
    'Topic :: Scientific/Engineering :: Physics']


def setup_cpp():
    setup(
        name='icet_cpp',
        version=version,
        author=authors,
        author_email=email,
        description=description,
        long_description=long_description,
        ext_modules=ext_modules,
        install_requires=['pybind11>=2.2',
                          'ase',
                          'numpy',
                          'scipy',
                          'sklearn',
                          'pandas>=0.23',
                          'spglib>1.11.0.19'],

        packages=find_packages(),
        cmdclass={'build_ext': BuildExt},
        zip_safe=False,
        classifiers=classifiers,
        license=license,
        url=homepage,
    )


def setup_python_icet():
    setup(
        name='icet',
        version=version,
        author=authors,
        author_email=email,
        description=description,
        long_description=long_description,
        install_requires=['ase',
                          'numpy',
                          'scipy',
                          'sklearn',
                          'pandas>=0.23',
                          'spglib>1.11.0.19'],
        packages=find_packages(),
        classifiers=classifiers,
        license=license,
        url=homepage,
    )


def setup_python_mchammer():
    setup(
        name='mchammer',
        version=version,
        author=authors,
        author_email=email,
        description=description,
        long_description=long_description,
        install_requires=['ase',
                          'numpy',
                          'scipy',
                          'sklearn',
                          'pandas>=0.23',
                          'spglib>1.11.0.19'],
        packages=find_packages(),
        classifiers=classifiers,
        license=license,
        url=homepage,
    )


if __name__ == '__main__':

    # Install python icet
    install_icet_process = Process(target=setup_python_icet)
    install_icet_process.start()
    install_icet_process.join()

    # setup_python mchammer
    install_mchammer_process = Process(target=setup_python_mchammer)
    install_mchammer_process.start()
    install_mchammer_process.join()

    # Install cpp
    install_cpp_process = Process(target=setup_cpp)
    install_cpp_process.start()
    install_cpp_process.join()
