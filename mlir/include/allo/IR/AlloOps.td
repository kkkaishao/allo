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
    This operation creates a new stream.
  }];
  let arguments = (ins);
  let results = (outs Stream:$stream);
  let assemblyFormat = [{
    attr-dict `:` qualified(type($stream))
  }];
}

def GlobalStreamGetOp : Allo_Op<"stream.get_global", [
  MemoryEffects<[MemWrite, MemRead]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "get a value from a global stream";
  let description = [{
    This operation retrieves a value from a global stream
  }];
  let arguments = (ins FlatSymbolRefAttr:$stream, Variadic<Index>:$indices);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $stream `[` $indices `]` attr-dict `:` type($value)
  }];
}

def GlobalStreamPutOp : Allo_Op<"stream.put_global", [
  MemoryEffects<[MemWrite]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "put a value into a global stream";
  let description = [{
    This operation puts a value into a global stream
  }];
  let arguments = (ins FlatSymbolRefAttr:$stream, Variadic<Index>:$indices, AnyType:$value);
  let results = (outs);
  let assemblyFormat = [{
    $value `,` $stream `[` $indices `]` attr-dict `:` type($value)
  }];
}

def GlobalStreamCreateOp : Allo_Op<"stream.create_global", [
  MemoryEffects<[MemAlloc]>, Symbol]> {
  let summary = "create a global stream";
  let description = [{
    This operation creates a new global stream.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs Stream:$stream);
  let assemblyFormat = [{
    $sym_name attr-dict `:` qualified(type($stream))
  }];
}


#endif // ALLO_OPS
