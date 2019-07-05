# Constrained quantum learning

### Using machine learning to train a Gaussian quantum circuit with PNRs to produce cubic phase resource states with high fidelity and probability.


This repository contains the source code used to produce the results presented in *"Near-deterministic production of universal quantum photonic gates enhanced by machine learning"* [arXiv:1809.04680](https://arxiv.org/abs/1809.04680).

## Contents

The following two scripts perform a constrained variational quantum circuit optimization, using both a global search (basin hopping) and a local search (BFGS optimization) to maximize the fidelity (and probability of generating) the cubic phase resource state in the last mode.

* `two_mode.py`: a Python script to generate the results of the two-mode gadget architecture presented in the paper. Here, a two mode squeezed displaced state is incident on a beamsplitter, with the first mode measured by a photon-number resolving detector.

* `three_mode.py`: a Python script to generate the results of the three-mode gadget architecture presented in the paper. Here, a three mode squeezed displaced state is incident on an interferometer consisting of three beamsplitters, with the first and second modes measured by photon-number resolving detectors.


## Requirements

To construct and optimize the constrained variational quantum circuits, these scripts use the Fock backend of [Strawberry Fields](https://github.com/XanaduAI/strawberryfields). In addition, SciPy is required for use of the global Basin Hopping optimization method, as well as the local BFGS optimization method.

**Due to subsequent interface upgrades, these scripts will work only with Strawberry Fields version <= 0.10.0.**

## Authors

Krishna Kumar Sabapathy, Haoyu Qi, Josh Izaac, and Christian Weedbrook.

If you are doing any research using this source code and Strawberry Fields, please cite the following two papers:

> Krishna Kumar Sabapathy, Haoyu Qi, Josh Izaac, and Christian Weedbrook.  Near-deterministic production of universal quantum photonic gates enhanced by machine learning. arXiv, 2018. [arXiv:1809.04680](https://arxiv.org/abs/1809.04680)

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. arXiv, 2018. [Quantum, 3, 129](https://quantum-journal.org/papers/q-2019-03-11-129/) (2019).

## License

This source code is free and open source, released under the Apache License, Version 2.0.
