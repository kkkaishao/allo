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

//===----------------------------------------------------------------------===//
// ISA Operations
//===----------------------------------------------------------------------===//
def IntegerSetArrayAttr : TypedArrayAttrBase<IntegerSetAttr,
                                      "IntegerSet array attribute">;

def InstructionYieldOp : Allo_Op<"instr.yield", [Pure, ReturnLike, Terminator]> {
  let summary = "yield results from an instruction";
  let description = [{
    This operation yields the results of an instruction in the semantics region of `instr.define` operation.
  }];
  let arguments = (ins Variadic<AnyType>:$results);
  let results = (outs);
  let assemblyFormat = [{
    $results attr-dict `:` type($results)
  }];
}

def InstructionDefineOp : Allo_Op<"instr.define", [Symbol, RecursiveMemoryEffects]> {
  let summary = "define an instruction";
  let description = [{
    This operation defines a tensor instruction in an ISA.

    The definition contains following components:
    1. The instruction name, which is used to identify the instruction
    2. The instruction operand types, typically are tensor/memref types
    3. The instruction result types, typically are tensor/memref types
    4. The instruction indexing scheme, which specifies how to index into the operands and results of the instruction
    5. The instruction semantics, which is a region that defines how the instruction computes its results from its operands.

    The semantics region must be end with a `yield` operation that yields the results of the instruction.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeArrayAttr:$source_types,
    TypeArrayAttr:$destination_types,
    StrArrayAttr:$index_symbols,
    AffineMapArrayAttr:$index_maps,
    StrArrayAttr:$domain_symbols,
    IntegerSetArrayAttr:$index_sets);
  let regions = (region SizedRegion<1>:$region);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name `{`
      `signature` `src` $source_types `dst` $destination_types
      `index_syms` $index_symbols
      `index_maps` $index_maps
      `domain_syms` $domain_symbols
      `index_sets` $index_sets
      `semantics` $region
    `}` attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    Block *getSemantics() {
       return &getRegion().front();
    }
  }];
}

def InstructionEmitOp : Allo_Op<"instr.emit", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  AttrSizedOperandSegments]> {
  let summary = "emit an instruction";
  let description = [{
    This operation emits an instruction defined by `instr.define` operation.

    It takes the following components as input:
    1. The instruction name
    2. The instruction operands (SSA values), which must match the operand types defined in the instruction definition
    3. The indexing symbols and domain symbols (Attributes), which are compile-tile known and used to instantiate the instruction scheme.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$instruction,
    Variadic<AnyType>:$sources,
    Variadic<AnyType>:$destinations, // DPS style
    DictionaryAttr:$symbol_values);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $instruction `params` $symbol_values
    `src``(` $sources `:` type($sources) `)`
    `dst` `(` $destinations `:` type($destinations) `)` attr-dict
  }];
}

#endif // ALLO_OPS
