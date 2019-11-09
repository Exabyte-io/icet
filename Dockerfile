# Base image
FROM ubuntu:18.10

# Base packages
RUN \
  apt-get update -qy && \
  apt-get upgrade -qy && \
  apt-get install -qy \
    python3-pip \
    doxygen \
    git \
    graphviz \
    zip

# Set up Python3 packages via pip
RUN pip3 install --upgrade \
  pip \
  setuptools

# Install entrypoints "manually" to ensure a sufficiently recent version is
# available for flake8 since the version from python3-dev/pip is too old.
RUN pip3 install --upgrade --user \
  entrypoints

RUN pip3 install --upgrade \
  coverage \
  flake8

# Packages needed for icet
RUN pip3 install --upgrade \
  ase \
  mip \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  spglib

# Packages for building documentation
RUN pip3 install --upgrade \
  breathe \
  cloud_sptheme \
  sphinx \
  sphinx-rtd-theme \
  sphinx_autodoc_typehints \
  sphinx_sitemap \
  sphinxcontrib-bibtex
