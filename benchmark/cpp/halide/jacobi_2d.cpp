/**
 * Jacobi 2D 5-point stencil — Halide implementation.
 *
 * Demonstrates two modes:
 * 1. No shared memory: simple gpu_tile schedule
 * 2. With shared memory: store_in(MemoryType::GPUShared) on the input
 *
 * Build: requires Halide installed (cmake -DCMAKE_PREFIX_PATH=<halide_install>)
 */

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace Halide;

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int steps_per_call = (argc > 2) ? atoi(argv[2]) : 10;
    int repeat = (argc > 3) ? atoi(argv[3]) : 20;
    int warmup = 5;
    bool use_shared = (argc > 4) ? (atoi(argv[4]) != 0) : false;

    printf("Halide Jacobi 2D: N=%d, steps=%d, shared=%d, warmup=%d, repeat=%d\n",
           N, steps_per_call, use_shared, warmup, repeat);

    // --- Define the algorithm ---
    // For a single step: u_new(x,y) = 0.25*(u(x-1,y)+u(x+1,y)+u(x,y-1)+u(x,y+1))
    // We chain `steps_per_call` stages.

    ImageParam input(Float(32), 2, "input");

    std::vector<Func> stages;
    stages.push_back(Func("stage_0"));
    stages[0](x, y) = BoundaryConditions::constant_exterior(input, 0.0f)(x, y);

    Var x("x"), y("y"), xi("xi"), yi("yi");

    for (int s = 0; s < steps_per_call; ++s) {
        Func f("stage_" + std::to_string(s + 1));
        Func& prev = stages.back();
        f(x, y) = 0.25f * (prev(x - 1, y) + prev(x + 1, y) +
                            prev(x, y - 1) + prev(x, y + 1));
        stages.push_back(f);
    }

    Func& output = stages.back();

    // --- Schedule ---
    Target target = get_jit_target_from_environment();
    if (target.has_gpu_feature()) {
        // GPU schedule
        output.gpu_tile(x, y, xi, yi, 32, 8);

        for (int s = 1; s < (int)stages.size() - 1; ++s) {
            if (use_shared) {
                // Compute intermediate at block level → shared memory
                stages[s].compute_at(output, x)
                    .gpu_threads(x, y);
            } else {
                // Inline (no shared memory)
                stages[s].compute_inline();
            }
        }
    } else {
        // CPU schedule
        output.parallel(y).vectorize(x, 8);
        for (int s = 1; s < (int)stages.size() - 1; ++s) {
            stages[s].compute_at(output, y);
        }
    }

    output.compile_jit(target);

    // --- Allocate buffers ---
    Buffer<float> buf_in(N, N);
    Buffer<float> buf_out(N, N);

    // Init: top row = 1
    buf_in.fill(0.0f);
    for (int j = 0; j < N; ++j) buf_in(j, 0) = 1.0f;
    input.set(buf_in);

    // --- Warmup ---
    for (int w = 0; w < warmup; ++w) {
        output.realize(buf_out);
        buf_in.copy_from(buf_out);
        input.set(buf_in);
    }

    // --- Timed runs ---
    std::vector<double> times;
    for (int r = 0; r < repeat; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        output.realize(buf_out);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);

        buf_in.copy_from(buf_out);
        input.set(buf_in);
    }

    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    const char* mode = use_shared ? "shared" : "inline";
    printf("  [%s] min=%.3fms  median=%.3fms  avg=%.3fms  max=%.3fms\n",
           mode, times.front(), times[times.size() / 2], sum / times.size(), times.back());

    printf("CSV: halide_jacobi_2d_%s,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
           mode, N, steps_per_call, times.front(), times[times.size() / 2],
           sum / times.size(), times.back());

    return 0;
}
