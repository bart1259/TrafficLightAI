#!bin/bash
# This script downloads and compiles a version of python with a disabeled GIL

# Make directory to store .nogil-python
mkdir .nogil-python
git clone git@github.com:colesbury/nogil.git
mv nogil .nogil-python

cd .nogil-python/nogil

# Compile
./configure
make

# Set enviornment variables
echo "export NO_GIL_PYTHON_DIRECTORY=${PWD}" >> ~/.bashrc 
export NO_GIL_PYTHON_DIRECTORY=${PWD}
echo "alias nogil-python=${PWD}/python" >> ~/.bashrc 
alias nogil-python="${PWD}/python"

# Install pip
nogil-python -m ensurepip

cd ../../