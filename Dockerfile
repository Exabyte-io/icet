# Base image
FROM python:3.6

# Additional packages
RUN \
  apt-get update -qy && \
  apt-get upgrade -qy && \
  apt-get install -qy \
    doxygen \
    graphviz \
    zip

# Packages for testing
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
