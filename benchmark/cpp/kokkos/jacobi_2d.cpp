/**
 * Jacobi 2D 5-point stencil — Kokkos implementation.
 *
 * Build (after Kokkos is installed):
 *   cmake -B build -DCMAKE_PREFIX_PATH=<kokkos_install> -DKokkos_ENABLE_CUDA=ON
 *   cmake --build build
 *
 * Run:
 *   ./build/jacobi_2d_kokkos 4096 10 20
 *                             ^N   ^steps ^repeat
 */

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

using View2D = Kokkos::View<double**>;

struct JacobiStep {
  View2D u;
  View2D u_new;
  int N;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1) {
      u_new(i, j) = 0.25 * (u(i - 1, j) + u(i + 1, j) +
                             u(i, j - 1) + u(i, j + 1));
    }
  }
};

struct CopyBack {
  View2D src;
  View2D dst;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    dst(i, j) = src(i, j);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int steps = (argc > 2) ? atoi(argv[2]) : 10;
    int repeat = (argc > 3) ? atoi(argv[3]) : 20;
    int warmup = 5;

    printf("Kokkos Jacobi 2D: N=%d, steps=%d, warmup=%d, repeat=%d\n",
           N, steps, warmup, repeat);

    View2D u("u", N, N);
    View2D u_new("u_new", N, N);

    // Init boundary: top row = 1.0
    Kokkos::parallel_for("init_bc", N, KOKKOS_LAMBDA(const int j) {
      u(0, j) = 1.0;
      u_new(0, j) = 1.0;
    });
    Kokkos::fence();

    using MDRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    MDRange range({0, 0}, {N, N});

    // Warmup
    for (int w = 0; w < warmup; ++w) {
      for (int s = 0; s < steps; ++s) {
        Kokkos::parallel_for("jacobi", range, JacobiStep{u, u_new, N});
        Kokkos::parallel_for("copy", range, CopyBack{u_new, u});
      }
      Kokkos::fence();
    }

    // Timed runs
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
      Kokkos::fence();
      auto t0 = std::chrono::high_resolution_clock::now();

      for (int s = 0; s < steps; ++s) {
        Kokkos::parallel_for("jacobi", range, JacobiStep{u, u_new, N});
        Kokkos::parallel_for("copy", range, CopyBack{u_new, u});
      }
      Kokkos::fence();

      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    printf("  min=%.3fms  median=%.3fms  avg=%.3fms  max=%.3fms\n",
           times.front(), times[times.size() / 2], sum / times.size(), times.back());

    // CSV output
    printf("CSV: kokkos_jacobi_2d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
           N, steps, times.front(), times[times.size() / 2],
           sum / times.size(), times.back());
  }
  Kokkos::finalize();
  return 0;
}
