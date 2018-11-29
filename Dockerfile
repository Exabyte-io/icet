# Base image
FROM ubuntu:18.10

# Install required packages
RUN apt update -qy
RUN apt upgrade -qy
RUN apt install -qy \
        python3-pip
RUN apt install -qy \
        doxygen \
        graphviz

# Set up some Python3 packages via pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

RUN pip3 install --upgrade \
	 h5py \
    	 numpy \
	 scipy

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

RUN pip3 install --upgrade \
         ase \
         pandas \
         scikit-learn \
         spglib
