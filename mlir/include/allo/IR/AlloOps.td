/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_OPS
#define ALLO_OPS

include "allo/IR/Allo.td"
include "allo/IR/AlloTypes.td"
include "allo/IR/AlloAttrs.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// Basic Operations
//===----------------------------------------------------------------------===//
// Modeled on upstream FuncOps.td.
def KernelOp : Allo_Op<"kernel", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{}];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       DenseI32ArrayAttr:$mapping);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
  CArg<"ArrayRef<int32_t>", "{}">:$mapping,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "allo"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}


def ReturnOp : Allo_Op<"return", [Pure, HasParent<"KernelOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Allo kernel return operation";
  let description = [{}];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, {});
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def InvokeOp : Allo_Op<"invoke",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "invoke operation";
  let description = [{}];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    UnitAttr:$async
  );

  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "KernelOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "FlatSymbolRefAttr":$callee,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, callee, results, operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, callee, results, operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, callee, results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` (`async` $async^)?
      attr-dict `:` functional-type($operands, results)
  }];
}


//===----------------------------------------------------------------------===//
// SPMW Operations
//===----------------------------------------------------------------------===//

def GetWorkerIdOp : Allo_Op<"get_wid", [Pure]> {
  let summary = "get the worker id";
  let description = [{
    Returns the worker ID of the current worker in an SPMW execution context.
    The worker ID identifies and differentiates the workers running in
    parallel.
  }];
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$worker_id);
  let assemblyFormat = [{
    $axis attr-dict `:` type($worker_id)
  }];
}

def GetNumWorkersOp : Allo_Op<"get_nw", [Pure]> {
  let summary = "get the number of workers";
  let description = [{
    Returns the total number of workers in an SPMW execution context. It
    coordinates parallel execution and distributes work appropriately among
    the workers.
  }];
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$num_workers);
  let assemblyFormat = [{
    $axis attr-dict `:` type($num_workers)
  }];
}

def StreamGetOp : Allo_Op<"stream.get",
    [MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "get a value from a stream";
  let description = [{
    Retrieves a value from a stream and destroys the value in the stream. The
    operation has both read and write memory effects.

    The optional `pred` is an i1 predicate: the token is consumed (the FIFO is
    popped) only on a firing where it holds. A null (statically-true) predicate
    is the canonical unconditional get. The predicate must be a function of data
    or induction variables, never of channel status, which would leave KPN.
  }];
  let arguments = (ins Stream:$stream, Variadic<Index>:$indices, Optional<I1>:$pred);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $stream `[` $indices `]` (`if` $pred^)? attr-dict `:` qualified(type($stream)) `,` type($value)
  }];
  let builders = [
    OpBuilder<(ins "Value":$stream, "ValueRange":$indices), [{
      auto streamType = cast<allo::StreamType>(stream.getType());
      build($_builder, $_state, streamType.getBaseType(), stream, indices,
            /*pred=*/::mlir::Value());
    }]>
  ];
  let hasVerifier = 1;
}

def StreamPutOp : Allo_Op<"stream.put",
    [MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "put a value into a stream";
  let description = [{
    Puts a value into a stream; the value is consumed by the next `stream.get`
    operation. The operation has write memory effects.

    The optional `pred` is an i1 predicate: the token is produced only on a
    firing where it holds. A null (statically-true) predicate is the canonical
    unconditional put. The predicate must be a function of data or induction
    variables, never of channel status, which would leave KPN.
  }];
  let arguments = (ins Stream:$stream, Variadic<Index>:$indices, AnyType:$value,
                       Optional<I1>:$pred);
  let results = (outs);
  let assemblyFormat = [{
    $value `,` $stream `[` $indices `]` (`if` $pred^)? attr-dict `:` type($value) `,` qualified(type($stream))
  }];
  let builders = [
    OpBuilder<(ins "Value":$stream, "ValueRange":$indices, "Value":$value), [{
      build($_builder, $_state, stream, indices, value, /*pred=*/::mlir::Value());
    }]>
  ];
  let hasVerifier = 1;
}

def StreamCreateOp : Allo_Op<"stream.create", [MemoryEffects<[MemAlloc]>]> {
  let summary = "create a stream";
  let description = [{
    This operation creates a new local stream.

    The optional `init` attribute preloads the channel with initial tokens (for
    feedback cycles): an ordered list of typed scalar attributes of the stream's
    base type. Its k elements become the earliest tokens in the channel history
    (`[init] ++ [produced]`), breaking a cycle's start dependence.
  }];
  let arguments = (ins OptionalAttr<ArrayAttr>:$init);
  let results = (outs Stream:$stream);
  let assemblyFormat = [{
    (`init` $init^)? attr-dict `:` qualified(type($stream))
  }];
  let builders = [
    OpBuilder<(ins "::mlir::Type":$stream), [{
      build($_builder, $_state, stream, /*init=*/::mlir::ArrayAttr());
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Bit Operations
//===----------------------------------------------------------------------===//

def BitGetSliceOp : Allo_Op<"bit.get_slice", [Pure]> {
  let summary = "get a slice of a integer";
  let arguments = (ins SignlessIntegerLike:$src, Index:$lo, Index:$hi);
  let results = (outs SignlessIntegerLike:$result);
  let assemblyFormat = [{
    $src `[` $lo `:` $hi `]` attr-dict `:` type($result) `from` type($src)
  }];
}

def BitSetSliceOp : Allo_Op<"bit.set_slice", [
  Pure, AllTypesMatch<["src", "result"]>
  ]> {
  let summary = "set a slice of a integer";
  let arguments = (ins SignlessIntegerLike:$src, Index:$lo, Index:$hi, SignlessIntegerLike:$value);
  let results = (outs SignlessIntegerLike:$result);
  let assemblyFormat = [{
    $src `[` $lo `:` $hi `]` `,` $value attr-dict `:` type($value) `into` type($src)
  }];
}

//===----------------------------------------------------------------------===//
// Analysis / scheduling hints
//
// Unchecked assumptions fed to the scheduler's analyses (llvm.assume style).
// They carry no hardware and are stripped before codegen; a false assumption is
// undefined behavior. They are intentionally NOT `Pure` so DCE keeps them until
// the analysis that consumes them runs.
//===----------------------------------------------------------------------===//

def AssumeSSAOp : Allo_Op<"assume.ssa"> {
  let summary = "Assume an SSA predicate holds (an llvm.assume-style hint)";
  let description = [{
    Asserts that `$condition` is true at this program point, seeding the value
    analyses the scheduler relies on (integer ranges / affine constraints on
    dynamic values such as parametric loop bounds and indices). Purely an
    assumption: unchecked (undefined behavior if violated) and stripped before
    codegen.

    Example: assert a parametric trip count is in a known range.
    ```mlir
    %c = arith.cmpi ule, %n, %c128 : index
    allo.assume.ssa %c : i1
    ```
  }];
  let arguments = (ins I1:$condition);
  let assemblyFormat = [{ $condition attr-dict `:` type($condition) }];
}

def AssumeNoDepOp : Allo_Op<"assume.nodep"> {
  let summary = "Assume (no) memory dependence on an array in a loop";
  let description = [{
    The `#pragma HLS dependence` analogue. Asserts the presence or absence of a
    memory dependence on array `$variable`, scoped to the loop identified by its
    induction variable `$iv`. `dependent = false` (the default) prunes a
    dependence the conservative analysis would otherwise add, recovering II when
    the programmer knows two accesses never collide (e.g. a permutation
    scatter); `dependent = true` asserts one exists.

    Note: `dependent = true` is a no-op, accepted for Vitis-signature fidelity
    but not consumed. The analysis is already conservative, testing affine
    accesses exactly and serializing non-affine ones, so it never misses a real
    dependence for an assertion to re-add. A user-known distance > 1 is better
    expressed on `distance` than here.

    Mirrors Vitis's fields: `dep_type` is inter- (across iterations) vs. intra-
    (within one iteration); the optional `direction` (raw/war/waw) and
    `distance` narrow the claim, and omitting them applies it to the whole array
    and all directions. Unchecked (undefined behavior if false); stripped before
    codegen.

    Example: a histogram index set is a permutation, so no inter-iteration RAW.
    ```mlir
    affine.for %i = 0 to 128 {
      allo.assume.nodep %hist[%i]
          {dep_type = #allo<dep_type inter>, direction = #allo<dep_dir raw>}
          : memref<64xi32>
      ...
    }
    ```
  }];
  let arguments = (ins
    AnyMemRef:$variable,
    Index:$iv,
    AssumeDepTypeAttr:$dep_type,
    DefaultValuedAttr<BoolAttr, "false">:$dependent,
    OptionalAttr<AssumeDepDirAttr>:$direction,
    OptionalAttr<I64Attr>:$distance);
  let assemblyFormat = [{
    $variable `[` $iv `]` attr-dict `:` type($variable)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Data & Control Path (dcp) Operations
//===----------------------------------------------------------------------===//

def ScheduledOpInterface : OpInterface<"ScheduledOpInterface"> {
  let description = [{
    An op that participates in the dcp schedule. The default implementations
    read and write the generic `start` (cycle) and `unit` (resource binding)
    attributes, so an op that lacks them (e.g. a region op) reports `nullopt`.
  }];
  let methods = [
    InterfaceMethod<"Get the start cycle of this operation",
      "std::optional<int64_t>", "getStartCycle", (ins), /*methodBody=*/"", [{
        if (auto a = $_op->template getAttrOfType<::mlir::IntegerAttr>("start"))
          return a.getInt();
        return std::nullopt;
      }]>,
    InterfaceMethod<"Set the start cycle of this operation",
      "void", "setStartCycle", (ins "int64_t":$start), "", [{
        $_op->setAttr("start", ::mlir::IntegerAttr::get(
            ::mlir::IntegerType::get($_op->getContext(), 64), start));
      }]>,
    InterfaceMethod<"Get the resource binding of this operation",
      "std::optional<::mlir::FlatSymbolRefAttr>", "getBinding", (ins), "", [{
        if (auto u = $_op->template getAttrOfType<::mlir::FlatSymbolRefAttr>("unit"))
          return u;
        return std::nullopt;
      }]>,
    InterfaceMethod<"Set the resource binding of this operation",
      "void", "setBinding", (ins "::mlir::FlatSymbolRefAttr":$binding), "", [{
        $_op->setAttr("unit", binding);
      }]>
  ];
}

def DCPathRegionOpInterface : OpInterface<"DCPathRegionOpInterface"> {
  let description = [{
    A dcp SCHEDULING REGION: `dcp.pipeline`, `dcp.sequential` or `dcp.select`.
    What the three share is a timing report, and this is how the latency model
    and the emitter read it.

    They are read through an interface rather than by attribute name because
    the name is the failure mode. These are typed ODS arguments with generated
    accessors, but every consumer works over `Operation *` (it handles all three
    uniformly), so before this each one reached them as `"latency"`, `"drain"`,
    `"trip"` string literals: keys no verifier checks, no rename follows, and a
    typo turns into a silently absent number rather than a compile error. That
    is the argument `dcp.module` was landed on, one level down.

    `drain` and `trip` are not properties of all three; a guard has neither and
    a `dcp.sequential` has no trip. They still belong here, because a consumer
    asking "what is this region's trip" wants "none" rather than a cast it has
    to write, and the default below is that answer.
  }];
  let cppNamespace = "::mlir::allo::dcp";
  let methods = [
    InterfaceMethod<"This region's composed span, when one composes exactly",
      "::std::optional<uint64_t>", "getLatency">,
    InterfaceMethod<"Publish this region's composed span",
      "void", "setLatency", (ins "::std::optional<uint64_t>":$span)>,
    InterfaceMethod<"Whether the span above is an upper bound, not exact",
      "bool", "getLatencyBound">,
    InterfaceMethod<"Which controller family paces this region",
      "::std::optional<::mlir::allo::DeterminacyEnum>", "getDeterminacy">,
    InterfaceMethod<"Publish this region's pacing class",
      "void", "setDeterminacy",
      (ins "::std::optional<::mlir::allo::DeterminacyEnum>":$pacing)>,
    InterfaceMethod<"The TERMINAL cycle: last issue pulse to deepest commit",
      "::std::optional<uint64_t>", "getDrain", (ins), /*methodBody=*/"",
      [{ return ::std::nullopt; }]>,
    InterfaceMethod<"Iterations of this region's body, when statically known",
      "::std::optional<uint64_t>", "getTrip", (ins), /*methodBody=*/"",
      [{ return ::std::nullopt; }]>,
  ];
}

class DCPath_Op<string mnemonic, list<Trait> traits = []> : Allo_Op<mnemonic, traits> {
  let cppNamespace = "::mlir::allo::dcp";
}

def DCPathOperatorOp : DCPath_Op<"dcp.operator", [Symbol]> {
  let summary = "An IP operator characterization referenced by dcp.compute";
  let description = [{
    Declares the timing of one instanced (multi-cycle) IP operator: a
    fixed-signature float/cast core the device provides. Injected from the
    device model (Python) before scheduling, and referenced by a `dcp.compute`
    on the IP path (its `op_type`) so schedule legality and register depths are
    self-contained and verifiable.

    `sym_name` is BOTH the referenced symbol and the RTL module name the emitter
    instantiates. `kind` is the abstract operator kind
    (`add`/`mul`/`cmp`/`int_float_cast`/...); `signature` is the concrete
    realized type (e.g. `(f32, f32) -> f32`), matched by exact element types.
    `stall` is the port/back-pressure contract the emitter instantiates it
    with. Combinational ops do NOT get a `dcp.operator`; they take the
    `dcp.compute` `comb_kind` path, with their chaining delay in `dcp.device`.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    TypeAttrOf<FunctionType>:$signature,
    I64Attr:$latency,
    F32Attr:$in_delay,
    F32Attr:$out_delay,
    DefaultValuedAttr<BoolAttr, "true">:$pipelined,
    DefaultValuedAttr<StallContractAttr, "::mlir::allo::StallContractEnum::Ce">:$stall);
  let hasCustomAssemblyFormat = 1;
}

def DCPathDeviceOp : DCPath_Op<"dcp.device"> {
  let summary = "The device technology tables (combinational + storage timing)";
  let description = [{
    The device model injected from Python before scheduling: the native
    (combinational) operator chaining delays and the storage timing. The
    scheduler and reifier read it directly.

    `comb` maps an abstract operator kind (`add`/`sub`/`cmp`/...) to its
    combinational chaining delay (ns). `memory` maps a storage implementation
    (`register`/`lutram`/`bram`/`uram`) to its `{read_latency, write_latency,
    read_delay_ns, write_delay_ns}`; `fifo` is the stream get/put timing;
    `default_memory` is the implementation unbound on-chip arrays use.
  }];
  let arguments = (ins
    DictionaryAttr:$comb,
    DictionaryAttr:$memory,
    OptionalAttr<DictionaryAttr>:$fifo,
    OptionalAttr<StrAttr>:$default_memory);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def DCPathUnitOp : DCPath_Op<"dcp.unit",
    [Symbol, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "An allocated functional-unit instance of an operator type";
  let description = [{
    A single physical instance of `operator`. The allocation is the set of unit
    declarations; binding is `dcpath.compute` ops referencing a unit, and two
    ops on the same unit at different times imply a (derived) mux.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$op_type);
  let assemblyFormat = "$sym_name `of` $op_type attr-dict";
}

def DCPathModuleOp : DCPath_Op<"dcp.module", [
    AffineScope, AutomaticAllocationScope, FunctionOpInterface,
    IsolatedFromAbove, OpAsmOpInterface,
    SingleBlockImplicitTerminator<"DCPathOutputOp">]> {
  let summary = "One scheduled kernel: the container an hw.module is emitted from";
  let description = [{
    A kernel once its schedule is reified: the body holds `dcp` regions and
    nothing else that controls time, and `dcp.instance` is the only way it
    reaches another one. It replaces the `func.func` the frontend and the
    scheduler work on, at the one point the schedule stops being a promise about
    that IR and becomes the structure of the hardware.

    The three timing fields are the kernel's CONTRACT with its callers, and they
    are op arguments rather than discardable attributes so the verifier below
    reaches them. `latency` is the whole-kernel span, absent when the length is
    data-dependent; `latency_bound` marks it a worst case (an assume-bounded
    trip) rather than an exact count; `determinacy` is the class a caller
    composes against. A caller reading a `latency` that is not the cycle the
    kernel actually finishes on places its consumers at the wrong offset, and
    since nothing handshakes, that is a read before the write.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<I64Attr>:$latency,
    UnitAttr:$latency_bound,
    DeterminacyAttr:$determinacy);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let builders = [OpBuilder<(ins
    "::mlir::StringRef":$name, "::mlir::FunctionType":$type,
    "::mlir::allo::DeterminacyEnum":$determinacy)>];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    ::mlir::Region *getCallableRegion() { return &getBody(); }
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "allo"; }
  }];
}

def DCPathOutputOp : DCPath_Op<"dcp.output",
    [Pure, ReturnLike, Terminator, HasParent<"DCPathModuleOp">]> {
  let summary = "Terminator of a dcp.module: the kernel's scalar results";
  let description = [{
    Its operands are the kernel's results, in signature order, each produced by
    the region that computes it. An array result has no meaning at a hardware
    port, so every operand here is a scalar the emitter drives on an output
    port.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def DCPathPipelineOp : DCPath_Op<"dcp.pipeline",
    [RecursiveMemoryEffects, AllTypesMatch<["inits", "results"]>,
     AttrSizedOperandSegments, ScheduledOpInterface, DCPathRegionOpInterface]> {
  let summary = "An II-scheduled loop carrying iter-args (counted or while)";
  let description = [{
    A cyclic scheduling region at initiation interval `ii`. The single block's
    first argument is the iteration counter (the region's control value); the
    remaining arguments are the iter-args (recurrences), initialized by `inits`
    and updated by the terminator. The terminator is `dcp.uncondition` for a
    counted loop (`trip` iterations) or `dcp.condition` for a while loop, whose
    leading `i1` is the per-iteration continue condition and which carries no
    `trip`. `isWhileLoop()` distinguishes the two; `getCarriedValues()` returns
    the loop-carried next-values for either. Recurrence, accumulator init, and
    the loop result are all loop-carried SSA. Registers, muxes, counter and
    flushing control are derived at lowering.

    The counter runs the source loop's induction: `lb`, `lb+step`, ... The lower
    bound and stride are each either compile-time (the `lb`/`step` attributes,
    default 0/1) or a runtime `index` value (the `lbBound`/`stepBound` operands,
    for a data-dependent `for i in range(m, n)`). The induction register holds
    the real IV, so the body's address maps, identity-relative to the IV, index
    correctly for a `lb != 0` loop, and termination is a bound compare
    (`iv+step >= ub`). The upper bound `ub` is symmetrically the compile-time
    `lb + trip*step` when `trip` is known, else the runtime `dynamicBound`
    operand; `trip` is the derived iteration count kept for latency and
    modulo-unroll math. A while loop carries none of these: `dcp.condition`
    gates termination, and its counter free-runs from 0.

    `ii` is optional. It is absent only for a residual *sequential* wrapper
    whose body length is data-dependent (an enclosed loop with a dynamic trip)
    and therefore has no static initiation interval. Its hardware is a
    done-based sequential controller: start iteration i+1 when iteration i
    finishes.

    `drain` is the region's TERMINAL cycle: how long after its last issue pulse
    the deepest output commits. It is what a span is composed from
    (`leafSpan`), because a region's `done` rises exactly
    `drain + 1` cycles after that pulse. `length` is the schedule DEPTH, the
    cycle by which everything has issued, and is a report. The two differ, and
    composing off the wrong one is what makes a consumer sample early.
  }];
  let arguments = (ins
    Optional<Index>:$lbBound,
    Optional<Index>:$dynamicBound,
    Optional<Index>:$stepBound,
    Variadic<AnyType>:$inits,
    OptionalAttr<I64Attr>:$trip,
    OptionalAttr<I64Attr>:$lb,
    OptionalAttr<I64Attr>:$step,
    OptionalAttr<I64Attr>:$ii,
    OptionalAttr<I64Attr>:$length,
    OptionalAttr<I64Attr>:$drain,
    OptionalAttr<I64Attr>:$latency,
    UnitAttr:$latency_bound,
    OptionalAttr<DeterminacyAttr>:$determinacy);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// True iff this is a while loop (its terminator is a `dcp.condition`).
    bool isWhileLoop();
    /// The `dcp.condition` terminator, or null for a counted loop.
    DCPathConditionOp getConditionOp();
    /// The `dcp.uncondition` terminator, or null for a while loop.
    DCPathUnconditionOp getUnconditionOp();
    /// The `i1` continue-condition value, or null for a counted loop.
    ::mlir::Value getConditionValue();
    /// The loop-carried next-values, independent of terminator kind.
    ::mlir::OperandRange getCarriedValues();
  }];
}

def DCPathSequentialOp : DCPath_Op<"dcp.sequential",
    [SingleBlockImplicitTerminator<"DCPathUnconditionOp">, RecursiveMemoryEffects,
     ScheduledOpInterface, DCPathRegionOpInterface]> {
  let summary = "A straight-line (acyclic) scheduling region";
  let description = [{
    A non-loop region whose ops carry `start` relative to the region. Produces
    `results` consumed by later regions; regions compose in program order. The
    region is not isolated: its ops may reference values from earlier regions
    directly.

    `drain` is the terminal cycle its `done` is composed off, `length` the
    schedule depth kept as a report; see `dcp.pipeline`.
  }];
  let arguments = (ins
    OptionalAttr<I64Attr>:$length,
    OptionalAttr<I64Attr>:$drain,
    OptionalAttr<I64Attr>:$latency,
    UnitAttr:$latency_bound,
    OptionalAttr<DeterminacyAttr>:$determinacy);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def DCPathSelectOp : DCPath_Op<"dcp.select",
    [RecursiveMemoryEffects, ScheduledOpInterface,
     DCPathRegionOpInterface]> {
  let summary = "A control-path guard: selects the active data path";
  let description = [{
    The DCP `if`. `$condition` (an `i1`, the explicit control input, like a
    while pipeline's `dcp.condition`) selects which data path is active: `then`
    or `else`. Selecting an *active path* means one branch executes and the
    other is gated off. The branch bodies hold whole `dcp` sub-schedules (a
    guarded loop, store or stream that cannot be speculated) and are
    enable-gated by the predicate, NOT computed-both-and-muxed. When `$results`
    are yielded, both branches supply one value each and the derived hardware
    includes a result-mux (`cond ? then : else`) on those values, but the bodies
    are still gated. `else` is optional, and present iff there are results (a
    mux needs both sources). Enable, registers and result-mux are derived at
    lowering. Latency is optional: absent for a data-dependent guard (unknown),
    present when an affine active-count is known.
  }];
  let arguments = (ins
    I1:$condition,
    OptionalAttr<I64Attr>:$latency,
    UnitAttr:$latency_bound,
    OptionalAttr<DeterminacyAttr>:$determinacy);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                        MaxSizedRegion<1>:$else_region);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def DCPathComputeOp : DCPath_Op<"dcp.compute",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     ScheduledOpInterface]> {
  let summary = "A scheduled, bound compute operation";
  let description = [{
    Computes `result` from `inputs`, issued at cycle `start` within its region,
    optionally bound to functional unit `unit`. Exactly one of two realization
    paths is set: `comb_kind` (a `CombOpKind`) is a combinational op emitted
    inline as a CIRCT `comb` primitive (latency 0, chained); `op_type`
    references a `dcp.operator` and lowers to that IP module instance (its
    latency). The verifier enforces the exclusivity.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<CombOpKindAttr>:$comb_kind,
    OptionalAttr<FlatSymbolRefAttr>:$op_type,
    I64Attr:$start,
    OptionalAttr<FlatSymbolRefAttr>:$unit);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    (`comb` $comb_kind^)? ($op_type^)? `(` $inputs `)` `at` $start
    (`on` $unit^)? attr-dict `:` functional-type($inputs, $result)
  }];
  let hasVerifier = 1;
}

def DCPathInstanceOp : DCPath_Op<"dcp.instance",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, ScheduledOpInterface]> {
  let summary = "A scheduled sub-kernel call (a leaf-datapath CallUnit node)";
  let description = [{
    Invokes `callee` at cycle `start` within its region, consuming `inputs` in
    callee-argument order: the memref inputs are arrays the child masters,
    driving their addr/data/we, the stream inputs are channel ends it handshakes
    on, and the scalar inputs are values it reads. It produces scalar `results`
    at `start + latency`. An `allo.async` carrier marks an `await` SPAWN: it
    starts with its container and is ordered thereafter only by FIFO
    back-pressure, which is a start policy rather than a different kind of node.

    The op references the callee's timing contract (`latency`, `determinacy`)
    exactly as `dcp.compute` references its `dcp.operator`: a determinate invoke
    is a fixed-latency datapath node (a pipelined multiplier that also masters
    ports); an indeterminate one completes on the child's real `done`. `latency`
    is absent for an indeterminate callee (a `while` or data-dependent child).
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    FlatSymbolRefAttr:$callee,
    I64Attr:$start,
    OptionalAttr<I64Attr>:$latency,
    DeterminacyAttr:$determinacy);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $inputs `)` `at` $start $determinacy attr-dict
    `:` functional-type($inputs, $results)
  }];
  let hasVerifier = 1;
}

def DCPathLoadOp : DCPath_Op<"dcp.load",
    [MemoryEffects<[MemRead]>, ScheduledOpInterface]> {
  let summary = "A scheduled, port-bound memory read";
  let description = [{
    Reads `memref` at the address `map(indices)`; the induction variable among
    the indices makes the address timing fall out of the same register-depth
    rule as data. Bound to a `(bank, port)` of the memref. `latency` is the read
    latency of the memref's storage, resolved from the device memory model at
    reify time.
  }];
  let arguments = (ins
    AnyMemRef:$memref,
    Variadic<Index>:$indices,
    AffineMapAttr:$map,
    I64Attr:$start,
    I64Attr:$latency,
    OptionalAttr<I64Attr>:$bank,
    OptionalAttr<I64Attr>:$port);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $memref `[` $indices `]` `map` $map `at` $start `lat` $latency
    (`bank` $bank^)? (`port` $port^)? attr-dict
    `:` type($memref) `->` type($result)
  }];
}

def DCPathStoreOp : DCPath_Op<"dcp.store",
    [MemoryEffects<[MemWrite]>, ScheduledOpInterface]> {
  let summary = "A scheduled, port-bound memory write";
  let arguments = (ins
    AnyType:$value,
    AnyMemRef:$memref,
    Variadic<Index>:$indices,
    AffineMapAttr:$map,
    I64Attr:$start,
    I64Attr:$latency,
    OptionalAttr<I64Attr>:$bank,
    OptionalAttr<I64Attr>:$port);
  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` `map` $map `at` $start `lat` $latency
    (`bank` $bank^)? (`port` $port^)? attr-dict
    `:` type($value) `,` type($memref)
  }];
}

def DCPathUnconditionOp : DCPath_Op<"dcp.uncondition",
    [Pure, ReturnLike, Terminator,
     ParentOneOf<["DCPathPipelineOp", "DCPathSequentialOp", "DCPathSelectOp"]>]> {
  let summary = "Unconditional terminator of a counted pipeline or branch";
  let description = [{
    The terminator of a counted `dcp.pipeline` (termination by `trip` and
    counter), a `dcp.sequential`, or a `dcp.select` branch. Its operands are the
    loop-carried next-values, the region outputs, or the per-branch yielded
    values. Paired with `dcp.condition`, the while-pipeline terminator.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* no operands */ }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def DCPathConditionOp : DCPath_Op<"dcp.condition",
    [Pure, ReturnLike, Terminator, ParentOneOf<["DCPathPipelineOp"]>]> {
  let summary = "Conditional terminator of a while dcp.pipeline";
  let description = [{
    The terminator of a while `dcp.pipeline`: `$condition` (an `i1`) is the
    per-iteration continue condition the derived flushing controller reads, and
    `$carried` are the loop-carried next-values (the `dcp.uncondition` operands
    of a counted loop).
  }];
  let arguments = (ins I1:$condition, Variadic<AnyType>:$carried);
  let assemblyFormat = [{
    $condition (`,` $carried^)? attr-dict
    `:` type($condition) (`,` type($carried)^)?
  }];
}

#endif // ALLO_OPS
