# Representative NCU Summary

These metrics summarize three representative kernels for the 3060 cost-model decomposition.

| Case | Kernel | Block | Grid | SM Throughput % | Achieved Occupancy % | DRAM Read (B) | DRAM Write (B) | FP32/FP64 Inst |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| heat2d_128_graph | `k_heat2d` | `(256, 1, 1)` | `(64, 1, 1)` | 7.81 | 33.07 | 69,120 | 1,408 | 95,764 |
| osher_64_persistent | `persistent_kernel` | `(256, 1, 1)` | `(16, 1, 1)` | 43.59 | 16.67 | 1,084,416 | 38,784 | 12,943,003 |
| osher_128_graph | `shallow_water_step` | `(256, 1, 1)` | `(64, 1, 1)` | 60.54 | 31.36 | 4,174,592 | 1,369,472 | 10,026,496 |

## Notes

- `heat2d_128_graph` uses the `k_heat2d` kernel from `pk_matrix_benchmark.cu` under the `graph` strategy.
- `osher_64_persistent` uses the `persistent_kernel` path from `hydro_cuda_osher.cu`.
- `osher_128_graph` uses the `shallow_water_step` kernel under the `graph` path.
- FP instruction count is currently a proxy metric (`fp32_inst` for Heat2D, `fp64_inst` for Osher).
