# nqs-cpp

Neural Quantum State solver for molecular electronic structure, with Restricted Boltzmann Machine (RBM) ansatz and Stochastic Reconfiguration optimization.

## Dependencies

- C++17 compiler with OpenMP support
- [Eigen](https://eigen.tuxfamily.org/) (>= 3.3)
- [Spectra](https://spectralib.org/) (>= 1.0)
- [nlohmann/json](https://github.com/nlohmann/json) (>= 3.0)
- [pybind11](https://pybind11.readthedocs.io/) (with Python + scipy + numpy)
- CMake (>= 3.16)

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

Example input files are in `examples/`. 

Data files are located in `data/`.
