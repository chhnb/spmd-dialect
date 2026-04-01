/**
 * Global reduction (sum) — Kokkos implementation.
 *
 * Uses Kokkos::parallel_reduce which automatically does
 * hierarchical reduction (warp shuffle + shared memory tree on GPU).
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

using View1D = Kokkos::View<double*>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    int repeat = (argc > 2) ? atoi(argv[2]) : 20;
    int warmup = 5;

    printf("Kokkos Reduction: N=%d, warmup=%d, repeat=%d\n", N, warmup, repeat);

    View1D data("data", N);

    // Fill with values
    Kokkos::parallel_for("fill", N, KOKKOS_LAMBDA(const int i) {
      data(i) = 1.0 / (1.0 + i);
    });
    Kokkos::fence();

    // Warmup
    for (int w = 0; w < warmup; ++w) {
      double result = 0.0;
      Kokkos::parallel_reduce("reduce", N,
        KOKKOS_LAMBDA(const int i, double& lsum) {
          lsum += data(i);
        }, result);
      Kokkos::fence();
    }

    // Timed runs
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      Kokkos::fence();
      auto t0 = std::chrono::high_resolution_clock::now();

      double result = 0.0;
      Kokkos::parallel_reduce("reduce", N,
        KOKKOS_LAMBDA(const int i, double& lsum) {
          lsum += data(i);
        }, result);
      Kokkos::fence();

      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    printf("  min=%.3fms  median=%.3fms  avg=%.3fms  max=%.3fms\n",
           times.front(), times[times.size() / 2], sum / times.size(), times.back());

    printf("CSV: kokkos_reduction,%d,%.3f,%.3f,%.3f,%.3f\n",
           N, times.front(), times[times.size() / 2],
           sum / times.size(), times.back());
  }
  Kokkos::finalize();
  return 0;
}
