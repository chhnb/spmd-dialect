cmake_minimum_required(VERSION 3.16)
project(kokkos_benchmarks CXX)

# --- Build Kokkos from source (in-tree) ---
# Point KOKKOS_SRC to our cloned kokkos directory
set(KOKKOS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../kokkos"
    CACHE PATH "Path to Kokkos source")

# Enable desired backends
option(Kokkos_ENABLE_CUDA "Enable CUDA backend" ON)
option(Kokkos_ENABLE_OPENMP "Enable OpenMP backend" ON)
option(Kokkos_ENABLE_SERIAL "Enable Serial backend" ON)

add_subdirectory(${KOKKOS_SRC} ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

# --- Benchmarks ---
add_executable(jacobi_2d_kokkos jacobi_2d.cpp)
target_link_libraries(jacobi_2d_kokkos Kokkos::kokkos)

add_executable(reduction_kokkos reduction.cpp)
target_link_libraries(reduction_kokkos Kokkos::kokkos)

add_executable(hydro_swe_kokkos hydro_swe.cpp)
target_link_libraries(hydro_swe_kokkos Kokkos::kokkos)

add_executable(hydro_swe_fp32_kokkos hydro_swe_fp32.cpp)
target_link_libraries(hydro_swe_fp32_kokkos Kokkos::kokkos)
add_executable(notransfer_test kokkos_notransfer.cpp)
target_link_libraries(notransfer_test Kokkos::kokkos)

add_executable(hydro_swe_swap hydro_swe_swap.cpp)
target_link_libraries(hydro_swe_swap Kokkos::kokkos)

add_executable(hydro_refactored_kokkos hydro_refactored.cpp)
target_link_libraries(hydro_refactored_kokkos Kokkos::kokkos)

add_executable(characterization_all characterization_all.cpp)
target_link_libraries(characterization_all Kokkos::kokkos)
