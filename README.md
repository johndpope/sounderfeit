
Sounderfeit
===========

A proof-of-concept overlap-add synthesizer based on a conditional
autoencoder.

Software theoretically cross-platform but only tested on Ubuntu 17.04.

(A Mac OS X port is in progress on the "osx" branch, it works but is
not yet merged so feel free to try it.)

# Demo

A demo of the user interface and description of the idea and results
is available at:

https://youtu.be/y1wKMhJdeUw

[![Sounderfeit Demo Video](http://img.youtube.com/vi/y1wKMhJdeUw/0.jpg)](http://www.youtube.com/watch?v=y1wKMhJdeUw "Sounderfeit Demo Video")

## Requires

* numpy, scipy, matplotlib: apt-get install python3-numpy

Generate data

* STK: https://ccrma.stanford.edu/software/stk/
* h5py: apt-get install python3-h5py || pip install h5py

Training

* TensorFlow: http://tensorflow.org/

Running

* Python 3
* PyQt5: apt-get install python3-pyqt5

## Build

    python3 setup.py build

## Install

    python3 setup.py install

## Run

    python3 main.py
