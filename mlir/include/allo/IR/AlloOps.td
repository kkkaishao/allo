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
// Copied and modified from FuncOps.td
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
    This operation returns the worker ID of the current worker in an SPMW execution context.
    The worker ID is typically used to identify and differentiate between multiple workers executing in parallel.
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
    This operation returns the total number of workers in an SPMW execution context.
    This information is crucial for coordinating parallel execution and ensuring that work is distributed appropriately among the workers.
  }];
  let arguments = (ins I32Attr:$axis);
  let results = (outs Index:$num_workers);
  let assemblyFormat = [{
    $axis attr-dict `:` type($num_workers)
  }];
}

def StreamGetOp : Allo_Op<"stream.get", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "get a value from a stream";
  let description = [{
    This operation retrieves a value from a stream, and destroys the value in the stream.
    The operation has both read and write memory effects
  }];
  let arguments = (ins Stream:$stream, Variadic<Index>:$indices);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $stream `[` $indices `]` attr-dict `:` qualified(type($stream)) `,` type($value)
  }];
  let builders = [
    OpBuilder<(ins "Value":$stream, "ArrayRef<Value>":$indices), [{
      auto streamType = cast<allo::StreamType>(stream.getType());
      build($_builder, $_state, streamType.getBaseType(), stream, indices);
    }]>
  ];
  let hasVerifier = 1;
}

def StreamPutOp : Allo_Op<"stream.put", [MemoryEffects<[MemWrite]>]> {
  let summary = "put a value into a stream";
  let description = [{
    This operation puts a value into a stream, and the value in the stream is consumed by the next `stream.get` operation.
    The operation has write memory effects.
  }];
  let arguments = (ins Stream:$stream, Variadic<Index>:$indices, AnyType:$value);
  let results = (outs);
  let assemblyFormat = [{
    $value `,` $stream `[` $indices `]` attr-dict `:` type($value) `,` qualified(type($stream))
  }];
  let hasVerifier = 1;
}

def StreamCreateOp : Allo_Op<"stream.create", [MemoryEffects<[MemAlloc]>]> {
  let summary = "create a stream";
  let description = [{
    This operation creates a new local stream.
  }];
  let arguments = (ins);
  let results = (outs Stream:$stream);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($stream))
  }];
}

def GlobalStreamGetOp : Allo_Op<"stream.get_global", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "get a local handle to a global stream";
  let description = [{
    This operation retrieves a local SSA handle to a global stream symbol.
  }];
  let arguments = (ins FlatSymbolRefAttr:$stream);
  let results = (outs Stream:$handle);
  let assemblyFormat = [{
    $stream attr-dict `:` qualified(type($handle))
  }];
}

def GlobalStreamCreateOp : Allo_Op<"stream.global", [
  Symbol]> {
  let summary = "create a global stream";
  let description = [{
    This operation creates a new global stream.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<Stream>:$streamType);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name attr-dict `:` $streamType
  }];
}

#endif // ALLO_OPS
