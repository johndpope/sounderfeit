
import os
from setuptools import setup, Extension

if 'STK_DIR' in os.environ:
    stk_dir = os.environ['STK_DIR']
else:
    stk_dir = os.environ["HOME"]+"/.local"

# Get long description
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sounderfeit",
    version = "0.0.1",
    author = "Stephen Sinclair",
    author_email = "radarsat1@gmail.com",
    description = ("A sound synthesizer based on a conditional autoencoder."),
    license = "BSD",
    keywords = "audio",
    url = "http://gitlab.com/sinclairs/sounderfeit",
    packages=[],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Audio",
        "License :: OSI Approved :: BSD License",
    ],
    ext_modules=[
        Extension("sounderfeit/soundersynth", ["sounderfeit/soundersynth.cpp"],
                  libraries = ["boost_python3","stk"],
                  include_dirs = [stk_dir + "/include",
                    '/Users/sinclairs/.local/lib/python3.6/site-packages/numpy/core/include'],
                  library_dirs = [stk_dir + "/lib"],
        )
    ],
    test_suite="tests",
)

