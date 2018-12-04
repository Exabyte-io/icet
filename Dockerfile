# Base image
FROM ubuntu:18.10

# Bring package repos up to speed
RUN apt update -qy
RUN apt upgrade -qy

# packages for icet
RUN apt install -qy \
        python3-pip
RUN pip3 install --upgrade \
         pip \
	 setuptools
RUN pip3 install --upgrade \
         ase \
    	 numpy \
         pandas \
         scikit-learn \
	 scipy \
         spglib

# packages for compilation of documentation
RUN apt install -qy \
        doxygen \
        graphviz \
	zip

RUN pip3 install --upgrade \
    	 breathe \
         coverage \
         cloud_sptheme \
         flake8 \
	 sphinx \
         sphinx-rtd-theme \
         sphinx_autodoc_typehints \
         sphinx_sitemap \
         sphinxcontrib-bibtex
