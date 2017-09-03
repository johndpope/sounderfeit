
Sounderfeit
===========

A proof-of-concept overlap-add synthesizer based on a conditional
autoencoder.

Software theoretically cross-platform but only tested on Ubuntu 17.04.

## Requires (Linux)

* numpy, scipy, matplotlib: apt-get install python3-numpy
* boost_python: apt-get install boost_python

Generate data

* STK: https://ccrma.stanford.edu/software/stk/
* h5py: apt-get install python3-h5py || pip install h5py

Training

* TensorFlow: http://tensorflow.org/

Running

* Python 3: brew install python3
* PyQt5: apt-get install python-pyqt5

## Requires (Mac OS X)

* Python 3: brew install python3
* numpy, scipy: pip3 install numpy scipy
* boost-python: brew install boost-python --with-python3

Generate data

* STK: https://ccrma.stanford.edu/software/stk/
* h5py: pip3 install h5py

Training

* TensorFlow: http://tensorflow.org/

Running

* PyQt5: brew install pyqt5

## Build

    python3 setup.py build

## Install

    python3 setup.py install

## Run

    python3 main.py
