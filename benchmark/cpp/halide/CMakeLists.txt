cmake_minimum_required(VERSION 3.16)
project(halide_benchmarks CXX)

# Halide must be built/installed first. Set CMAKE_PREFIX_PATH to its install dir.
# Example:
#   cd /path/to/halide && cmake -B build -DCMAKE_INSTALL_PREFIX=$PWD/install \
#       -DWITH_TUTORIALS=OFF -DWITH_TESTS=OFF -DWITH_PYTHON_BINDINGS=OFF
#   cmake --build build -j$(nproc) && cmake --install build
#
# Then build this benchmark:
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/halide/install
#   cmake --build build

find_package(Halide REQUIRED)

add_executable(jacobi_2d_halide jacobi_2d.cpp)
target_link_libraries(jacobi_2d_halide PRIVATE Halide::Halide)
