.. index:: Installation

Installation
************

General users
=============

Installation via pip
--------------------

Stable versions of :program:`icet` are provided via `PyPi
<https://pypi.org/>`_. This implies that :program:`icet` can be installed via::

    pip3 install icet --user

or alternatively::

    python3 -m pip install icet --user

As the package is provided as a `source distribution
<https://packaging.python.org/glossary/#term-source-distribution-or-sdist>`_
the C++ core library is compiled as part of the installation. This requires a
C++11 compliant compiler to be installed on your system, e.g., `GCC 4.8.1 and
above <https://gcc.gnu.org/projects/cxx-status.html#cxx11>`_ or
`Clang 3.3 and above <https://clang.llvm.org/cxx_status.html>`_.

Installation via ``setup.py``
-----------------------------

If installation via pip fails or if you want to use the most recent
(development) version you can clone the repository and install using the
``setup.py`` script as follows::

    git clone git@gitlab.com:materials-modeling/icet.git
    cd icet
    python3 setup.py install --user

Testing
-------

It is always a good idea to test that your installation works as advertised. To
this end, you should run the :program:`icet` test suite, which can be
accomplished as follows::

    curl -O https://icet.materialsmodeling.org/tests.zip
    unzip tests.zip
    python3 tests/main.py


Developers
==========

Compiling the core library
--------------------------

During development you might have to recompile the C++ core library. This can
achieved as follows::

    mkdir build
    cd build
    cmake ..
    make -j4
    cd ..

Note that this approach requires cmake to be installed on your system.

In this case :program:`icet` must be added manually to the ``PYTHONPATH``
environment variable. To this end, when using the Bash shell or similar (bash,
ksh) the following command should be added to the ``.bashrc`` file (or
equivalent)::

    export PYTHONPATH=${PYTHONPATH}:<ICET_PATH>/
    export PYTHONPATH=${PYTHONPATH}:<ICET_PATH>/build/src/

Here, ``ICET_PATH`` must be replaced with the path to the :program:`icet` root
directory. If you are using a C shell (csh, tcsh) the equivalent lines read::

    setenv PYTHONPATH ${PYTHONPATH}:<ICET_PATH>/
    setenv PYTHONPATH ${PYTHONPATH}:<ICET_PATH>/build/src/


Dependencies
------------

:program:`icet` is based on Python3 and invokes functionality from other Python
libraries, including
`ase <https://wiki.fysik.dtu.dk/ase>`_,
`pandas <https://pandas.pydata.org/>`_,
`numpy <http://www.numpy.org/>`_,
`scipy <https://www.scipy.org/>`_,
`scitkit-learn <http://scikit-learn.org/>`_, and
`spglib <https://atztogo.github.io/spglib/>`_.
The :program:`icet` C++ core library depends on
`Eigen <eigen.tuxfamily.org/>`_,
`boost <https://www.boost.org/>`_, and
`pybind11 <https://pybind11.readthedocs.io/>`_,
which are included in the distribution as third-party libraries.
