# C++ Benchmarks (Kokkos & Halide)

These benchmarks must be compiled and run on a **GPU compute node**.

## Kokkos

Uses in-tree build from `../../kokkos/`.

```bash
# On compute node with nvcc
cd benchmark/cpp/kokkos

# CUDA backend
cmake -B build-cuda \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -DCMAKE_CXX_COMPILER=$(pwd)/../../../kokkos/bin/nvcc_wrapper
cmake --build build-cuda -j8

./build-cuda/jacobi_2d_kokkos 4096 10 20   # N=4096, steps=10, repeat=20
./build-cuda/reduction_kokkos 10000000 20   # N=10M, repeat=20

# OpenMP backend (CPU comparison)
cmake -B build-omp \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_SERIAL=ON
cmake --build build-omp -j8

OMP_NUM_THREADS=16 ./build-omp/jacobi_2d_kokkos 4096 10 20
OMP_NUM_THREADS=16 ./build-omp/reduction_kokkos 10000000 20
```

## Halide

Halide must be built first from `../../halide/`.

```bash
# Step 1: Build Halide (one-time)
cd ../../halide
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PWD/install \
    -DWITH_TUTORIALS=OFF -DWITH_TESTS=OFF \
    -DWITH_PYTHON_BINDINGS=OFF \
    -DTARGET_CUDA=ON
cmake --build build -j8
cmake --install build

# Step 2: Build benchmarks
cd benchmark/cpp/halide
cmake -B build -DCMAKE_PREFIX_PATH=$(pwd)/../../../halide/install
cmake --build build

# Run: GPU mode
HL_TARGET=host-cuda ./build/jacobi_2d_halide 4096 10 20 0  # no shared mem
HL_TARGET=host-cuda ./build/jacobi_2d_halide 4096 10 20 1  # with shared mem

# Run: CPU mode
./build/jacobi_2d_halide 4096 10 20 0
```
