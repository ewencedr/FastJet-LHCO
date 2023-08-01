<div align="center">

# Fastjet Example on LHCO Data


[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![FastJet](https://img.shields.io/badge/-FastJet_3.4.1.2-orange)](https://github.com/scikit-hep/fastjet)

</div>

## Description

This repository shows how cluster jets in particle physics with Python. The jet clustering will be done using the [FastJet](http://fastjet.fr/) package. The FastJet package is a C++ library for jet finding. It is interfaced to Python using the [FastJet python](https://pypi.org/project/fastjet/) package. 

The [LHC-Olympics dataset](https://lhco2020.github.io/homepage/) is used for this example. It consists of dijet events that are clustered to get the constituents and jet features of the two jets.

The FastJet library offers two main interfaces on how to perform clustering on HEP data.
- The Awkward interface
- The Classic interface (ToDo)

This repository contains two notebooks, one for each interface.

The Awkward interface is the new interface made to handle multi-event data, whereas the classic interface is the same as the C++ library, designed to handle the data in a particle-at-a-time fashion. The Awkward interface is the recommended interface to use, since it is faster and more memory efficient. However, the classic interface is still maintained and used.

The Awkward interface has its name from the usage of the [Awkward Array](https://awkward-array.org/) library. The Awkward Array library is a library for manipulating arrays of complex data structures as they appear in particle physics. The library is designed to efficiently manipulate large collections of heterogeneous data, in particular particle physics events represented as nested, variable-sized records. In simple words, Awkward Arrays are like numpy arrays, but with the ability to handle nested data structures. They additionally can be easily combined with libraries such as [Vector](https://github.com/scikit-hep/vector) to directly include vector operations on the data. Vector also includes for example Lorentz vectors and is therefore very useful for particle physics.

For more details on how to use the FastJet python package, see the [FastJet python package documentation](https://fastjet.readthedocs.io/en/latest/).