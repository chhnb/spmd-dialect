#pragma once
//===- TargetDescriptor.h - Hardware target parameters --------------------===//
//
// Describes the hardware target that drives scheduling and memory promotion
// decisions in PlanSPMDSchedule and PromoteGroupMemory.
//
// See design-v1.md §8 for the full spec.

namespace mlir {
namespace spmd {

struct TargetDescriptor {
  enum BackendKind { CPU, CUDA, ROCM, SPIRV };

  BackendKind backend;
  int simdWidth;          ///< CPU SIMD / GPU vector width (elements)
  int subgroupWidth;      ///< GPU warp/wavefront size; 1 for CPU
  int maxGroupSize;       ///< max threads per block / work-items per workgroup
  int maxGroupMemBytes;   ///< shared/workgroup memory limit (bytes)
  int cacheLineBytes;
  int l1Bytes;
  int registerBudget;     ///< per-lane register budget (words)
  bool supportsGroupBarrier;

  /// Default descriptor for a generic CPU target.
  static TargetDescriptor cpuDefault() {
    TargetDescriptor td;
    td.backend             = CPU;
    td.simdWidth           = 8;   // AVX-256 float8
    td.subgroupWidth       = 1;
    td.maxGroupSize        = 256;
    td.maxGroupMemBytes    = 48 * 1024; // 48 KB
    td.cacheLineBytes      = 64;
    td.l1Bytes             = 32 * 1024; // 32 KB L1D
    td.registerBudget      = 16;
    td.supportsGroupBarrier = true;
    return td;
  }
};

} // namespace spmd
} // namespace mlir
