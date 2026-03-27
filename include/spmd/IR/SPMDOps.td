// SPMDOps.td — Op definitions for the spmd dialect

#ifndef SPMD_OPS
#define SPMD_OPS

include "spmd/IR/SPMDDialect.td"
include "spmd/IR/SPMDAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

class SPMD_Op<string mnemonic, list<Trait> traits = []>
    : Op<SPMD_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// spmd.yield
//===----------------------------------------------------------------------===//

def SPMD_YieldOp : SPMD_Op<"yield", [
    Pure,
    Terminator,
    ReturnLike
  ]> {
  let summary = "Region terminator for spmd ops";
  let description = [{
    Terminates the region of spmd.forall, spmd.if, and spmd.reduce.

    In spmd.forall: must be empty (no operands).
    In spmd.if:     operands must match the op's result types.
    In spmd.reduce: exactly one operand matching the result type.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// spmd.forall
//===----------------------------------------------------------------------===//

def SPMD_ForallOp : SPMD_Op<"forall", [
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Logical parallel iteration domain (SPMD)";
  let description = [{
    Represents a rectangular iteration domain over which all instances
    execute logically in parallel. The order of execution is unspecified.

    Cross-iteration conflicting accesses not modeled through spmd.reduce
    (or future spmd.atomic_*) are undefined behavior.

    Syntax (full form):
      spmd.forall (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
          [attributes {...}] { ... spmd.yield }

    Shorthand (lb=0, step=1):
      spmd.forall (%i, %j) in (%ub0, %ub1) [attributes {...}] { ... spmd.yield }

    Optional schedule attributes:
      spmd.mapping       : #spmd.level<...>
      spmd.tile_sizes    : DenseI64ArrayAttr (length = rank, all positive)
      spmd.order         : DenseI64ArrayAttr (permutation of [0..rank-1])
      spmd.memory_policy : #spmd.memory_policy<...>

    Verifier checks:
      - lbs/ubs/steps have equal length (rank >= 1)
      - block args count = rank, all index type
      - spmd.order, if present, is a valid permutation
      - spmd.tile_sizes, if present, has length = rank and all-positive values
      - spmd.mapping, if present, is a valid LevelAttr
      - steps must be > 0 at runtime (static constants verified statically)
  }];

  let arguments = (ins
    Variadic<Index>:$lowerBounds,
    Variadic<Index>:$upperBounds,
    Variadic<Index>:$steps
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Returns the number of iteration dimensions.
    unsigned getRank() { return getLowerBounds().size(); }

    // Returns the induction variable for dimension i.
    BlockArgument getInductionVar(unsigned i) {
      return getBody().front().getArgument(i);
    }

    // Returns all induction variables.
    Block::BlockArgListType getInductionVars() {
      return getBody().front().getArguments();
    }
  }];
}

//===----------------------------------------------------------------------===//
// spmd.if
//===----------------------------------------------------------------------===//

def SPMD_IfOp : SPMD_Op<"if", [
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Per-instance conditional branch";
  let description = [{
    Per-SPMD-instance conditional. Different lanes may take different
    branches; this is legal. Backends may lower to branch or predication.

    Without results:
      spmd.if %cond { ... spmd.yield }

    With results:
      %r = spmd.if %cond -> (f32) {
        spmd.yield %x : f32
      } else {
        spmd.yield %y : f32
      }

    Verifier:
      - cond must be i1
      - if op has results, both then and else regions must be present,
        and yield types must match result types
      - if op has no results, else region may be absent
  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        AnyRegion:$elseRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// spmd.reduce
//===----------------------------------------------------------------------===//

def SPMD_ReduceOp : SPMD_Op<"reduce"> {
  let summary = "Structured parallel reduction over a 1D iteration domain";
  let description = [{
    Reduces over the range [lb, ub) with given step, using the combiner
    specified by spmd.kind. The initial value is init. The body computes
    the contribution for each iteration and yields exactly one value.

    MVP: single-dimension only. Multi-dimensional reductions should be
    expressed as nested spmd.reduce.

    Syntax:
      %r = spmd.reduce (%k) = (%lb) to (%ub) step (%step)
               init(%init)
               attributes {spmd.kind = #spmd.reduction_kind<add>} {
        %contrib = ...
        spmd.yield %contrib : f32
      }

    Verifier:
      - spmd.kind must be present and valid
      - init type == result type
      - body yields exactly one value matching result type
      - MVP: body restricted to pure arithmetic + non-volatile loads
      - Floating-point add/mul: reassociation only valid under fast-math

    Execution order is unspecified; combiner must be associative and
    commutative for correct parallel execution.
  }];

  let arguments = (ins
    Index:$lowerBound,
    Index:$upperBound,
    Index:$step,
    AnyType:$init
  );
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// spmd.barrier
//===----------------------------------------------------------------------===//

def SPMD_BarrierOp : SPMD_Op<"barrier"> {
  let summary = "Group-scope synchronization barrier";
  let description = [{
    Synchronizes all active lanes in the current group and establishes
    a group-memory visibility fence.

    Syntax:
      spmd.barrier {spmd.scope = #spmd.scope<group>}

    MVP: only group scope is supported.

    Verifier:
      - spmd.scope must be present (MVP: only group)
      - Must be nested inside a spmd.forall with spmd.mapping = group

    Usage: not generated by frontends. Inserted by PromoteGroupMemory
    pass in S2 between cooperative copy and compute phases.
  }];

  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // SPMD_OPS
