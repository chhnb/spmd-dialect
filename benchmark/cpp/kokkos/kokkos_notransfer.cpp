// Measure: how much time does the transfer kernel take?
// Run compute-only (swap pointers) vs compute+transfer
#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using View1D = Kokkos::View<double*>;
using View2Di = Kokkos::View<int**>;
using View2D = Kokkos::View<double**>;

// Minimal compute kernel (just copy to measure overhead)
struct DummyCompute {
    int CEL;
    View1D src, dst;
    KOKKOS_INLINE_FUNCTION void operator()(int i) const {
        int pos = i + 1;
        if (pos > CEL) return;
        dst(pos) = src(pos) * 1.001;  // trivial compute
    }
};

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        int N = argc > 1 ? atoi(argv[1]) : 8192;
        int CEL = N * N;
        int steps = 10;
        int stride = CEL + 1;

        View1D A("A", stride), B("B", stride);
        Kokkos::parallel_for("init", CEL, KOKKOS_LAMBDA(int i) { A(i+1) = 1.0; });
        Kokkos::fence();

        // Method 1: compute + transfer (copy back)
        for (int w = 0; w < 3; w++) {
            for (int s = 0; s < steps; s++) {
                Kokkos::parallel_for("compute", CEL, DummyCompute{CEL, A, B});
                Kokkos::parallel_for("transfer", CEL, KOKKOS_LAMBDA(int i) {
                    int pos = i + 1; if (pos > CEL) return;
                    A(pos) = B(pos);
                });
            }
            Kokkos::fence();
        }

        std::vector<double> t1, t2;
        for (int r = 0; r < 10; r++) {
            Kokkos::fence();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                Kokkos::parallel_for("compute", CEL, DummyCompute{CEL, A, B});
                Kokkos::parallel_for("transfer", CEL, KOKKOS_LAMBDA(int i) {
                    int pos = i + 1; if (pos > CEL) return;
                    A(pos) = B(pos);
                });
            }
            Kokkos::fence();
            t1.push_back(std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
        }

        // Method 2: compute only (swap pointers)
        View1D X("X", stride), Y("Y", stride);
        Kokkos::parallel_for("init", CEL, KOKKOS_LAMBDA(int i) { X(i+1) = 1.0; });
        Kokkos::fence();

        for (int w = 0; w < 3; w++) {
            for (int s = 0; s < steps; s++) {
                Kokkos::parallel_for("compute", CEL, DummyCompute{CEL, X, Y});
                auto tmp = X; X = Y; Y = tmp;  // pointer swap
            }
            Kokkos::fence();
        }

        for (int r = 0; r < 10; r++) {
            Kokkos::fence();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < steps; s++) {
                Kokkos::parallel_for("compute", CEL, DummyCompute{CEL, X, Y});
                auto tmp = X; X = Y; Y = tmp;
            }
            Kokkos::fence();
            t2.push_back(std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count());
        }

        std::sort(t1.begin(), t1.end());
        std::sort(t2.begin(), t2.end());
        printf("N=%d CEL=%d\n", N, CEL);
        printf("  compute+transfer: %.3f ms\n", t1[5]);
        printf("  compute+swap:     %.3f ms\n", t2[5]);
        printf("  transfer overhead: %.1f%%\n", (t1[5]-t2[5])/t1[5]*100);
    }
    Kokkos::finalize();
}
