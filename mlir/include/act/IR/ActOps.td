#ifndef ACT_OPS_TD
#define ACT_OPS_TD

include "act/IR/Act.td"
include "act/IR/ActTypes.td"
include "act/IR/ActAttrs.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

///===----------------------------------------------------------------------===//
/// Primitive types related operations
///===----------------------------------------------------------------------===//

def Act_DeclareBufferOp : Act_Op<"buffer", [Symbol]> {
  let summary = "Declare a named buffer in a hardware memory space";
  let description = [{
    Declares a buffer with a symbol name, a size (number of elements), and a
    buffer element type that describes the access granularity. For example,
    `act.buffer @vecreg size(8) : !act.vector<64xbf16>` declares a buffer with
    8 slots, each holding a 64-element bf16 vector. HBM buffers must have size 1.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$size,
    TypeAttrOf<BufferTypeInterface>:$bufferType);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name `size` `(` $size `)` attr-dict `:` $bufferType
  }];
  let hasVerifier = 1;
}

def Act_DeclareStateOp : Act_Op<"state", [Symbol]> {
  let summary = "Declare a named hardware state register with enumerated values";
  let description = [{
    Declares a state register that can hold one of a fixed set of enumerated
    values. For example, `act.state @mode enums ["os", "ws"] : !act.state<i32>`
    declares a state that can be either "os" or "ws". An optional default value
    can be specified. Enum values are assigned integer indices by declaration order.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrArrayAttr:$enums,
    OptionalAttr<StrAttr>:$defaultState,
    TypeAttrOf<Act_State>:$stateType);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name `enums` $enums (`default` $defaultState^)? attr-dict `:` $stateType
  }];
  let hasVerifier = 1;
}

def Act_WriteStateOp : Act_Op<"state.write", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Write an enumerated value to a state register";
  let description = [{
    Writes a statically-known enum value to a state register. The value must be
    one of the enums declared in the corresponding `act.state` declaration.
    For dynamic control flow, use `scf.if` around separate `act.state.write` ops.
  }];
  let arguments = (ins FlatSymbolRefAttr:$state, StrAttr:$value);
  let results = (outs);
  let assemblyFormat = [{
    $value `to` $state attr-dict
  }];
}

def Act_ReadStateOp : Act_Op<"state.read", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Read the current value of a state register";
  let description = [{
    Reads the current integer value of a state register. The result type must
    match the element type of the state declaration. Can be combined with
    `arith.cmpi` to implement dynamic execution path selection based on state.
  }];
  let arguments = (ins FlatSymbolRefAttr:$state);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $state attr-dict `:` type($value)
  }];
}

def Act_DeclareDescOp : Act_Op<"desc", [Symbol]> {
  let summary = "Declare a named descriptor array with typed fields";
  let description = [{
    Declares a descriptor array with a symbol name, a number of slots, and a
    descriptor type specifying the named fields. Each slot holds one descriptor.
    For example, `act.desc @tma size(8) : !act.desc<["offset":i8, "stride":i8]>`
    declares 8 descriptor slots, each with "offset" and "stride" fields.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$size, TypeAttrOf<Act_Descriptor>:$descType);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name `size` `(` $size `)` attr-dict `:` $descType
  }];
}

def Act_WriteDescFieldOp : Act_Op<"desc.write", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Write a value to a named field of a descriptor";
  let description = [{
    Writes a value to a specific field of a descriptor. The value type must match
    the field type declared in the descriptor type.
  }];
  let arguments = (ins FlatSymbolRefAttr:$desc, StrAttr:$field, AnyType:$value);
  let results = (outs);
  let assemblyFormat = [{
    $value `to` $field `in` $desc attr-dict `:` type($value)
  }];
}

def Act_ReadDescFieldOp : Act_Op<"desc.read", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Read a value from a named field of a descriptor";
  let description = [{
    Reads a value from a specific field of a descriptor. The result type must
    match the field type declared in the descriptor type.
  }];
  let arguments = (ins FlatSymbolRefAttr:$desc, StrAttr:$field);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $field `from` $desc attr-dict `:` type($value)
  }];
}

///===----------------------------------------------------------------------===//
/// Buffer access operations
///===----------------------------------------------------------------------===//

def BufferAccessOpInterface : OpInterface<"BufferAccessOpInterface"> {
  let description = [{
    Interface for buffer access pattern operations (e.g. strided, tiled).
    Each access pattern can verify its compatibility with a buffer and
    materialize itself into concrete memory access IR (e.g. tensor.extract_slice
    or memref.subview).
  }];
  let cppNamespace = "::mlir::act";
  let methods = [
    InterfaceMethod<"Verify that this access pattern is compatible with the given buffer type and size",
    "mlir::LogicalResult", "verifyCompatibility", (ins "BufferTypeInterface":$bufferType, "unsigned":$size)>,

    InterfaceMethod<"Materialize a read: extract a slice from the buffer according to this access pattern",
    "llvm::FailureOr<mlir::Value>", "materialize",
    (ins "OpBuilder&":$builder, "Location":$loc, "Value":$buffer)>,

    InterfaceMethod<"Materialize a write: insert a value into the buffer according to this access pattern",
    "llvm::FailureOr<mlir::Value>", "materialize",
    (ins "OpBuilder&":$builder, "Location":$loc, "Value":$value, "Value":$buffer)>

    // TODO: add method to generate symbolic expression
  ];
}

def BufferRelayoutOpInterface : OpInterface<"BufferRelayoutOpInterface"> {
  let cppNamespace = "::mlir::act";
  let methods = [
    InterfaceMethod<"get the source access pattern operand",
    "mlir::Value", "getSource", (ins)>
  ];
}

class Act_BufferAccessOpBase<string mnemonic, list<Trait> traits = []> :
  Act_Op<mnemonic, !listconcat(traits, [
  DeclareOpInterfaceMethods<BufferAccessOpInterface>, Pure,
  HasParent<"act::DefineOp">])> {
  let results = (outs Act_Pattern); // token representing the access pattern
}

def Act_StridedOp : Act_BufferAccessOpBase<"strided", [AttrSizedOperandSegments]> {
  let summary = "Strided buffer access pattern";
  let description = [{
    Defines a strided access pattern over a buffer. Parameters (basis, counts,
    strides) can be a mixture of static and dynamic values. For each dimension i,
    the accessed indices are: basis[i] + j * strides[i] for j in 0..counts[i]-1.
    Contiguous access is the special case where strides are all 1.
    On-chip buffers must be accessed with 1-D patterns.
  }];
  let arguments = (ins
    Variadic<Index>:$basis,
    Variadic<Index>:$counts,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$staticBasis,
    DenseI64ArrayAttr:$staticCounts,
    DenseI64ArrayAttr:$staticStrides);
  let assemblyFormat = [{
    `basis```custom<DynamicIndexList>(
      $basis,
      $staticBasis,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    `counts```custom<DynamicIndexList>(
      $counts,
      $staticCounts,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    `strides```custom<DynamicIndexList>(
      $strides,
      $staticStrides,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    attr-dict
  }];
  let hasVerifier = 1;
}

def Act_TiledOp : Act_BufferAccessOpBase<"tiled", [AttrSizedOperandSegments]> {
  let summary = "Tiled buffer access pattern";
  let description = [{
    Defines a tiled access pattern over a buffer. Extends strided access with a
    tile_size per dimension: each stride step accesses a contiguous tile of
    tile_size elements rather than a single element. Tile sizes must not exceed
    strides (no overlapping tiles). On-chip buffers must be accessed with 1-D
    patterns, and the tile size must evenly divide the buffer element size.
  }];
  let arguments = (ins
    Variadic<Index>:$basis,
    Variadic<Index>:$counts,
    Variadic<Index>:$strides,
    Variadic<Index>:$tileSizes,
    DenseI64ArrayAttr:$staticBasis,
    DenseI64ArrayAttr:$staticCounts,
    DenseI64ArrayAttr:$staticStrides,
    DenseI64ArrayAttr:$staticTileSizes);
  let assemblyFormat = [{
    `basis```custom<DynamicIndexList>(
      $basis,
      $staticBasis,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    `counts```custom<DynamicIndexList>(
      $counts,
      $staticCounts,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    `strides```custom<DynamicIndexList>(
      $strides,
      $staticStrides,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    `tile_sizes```custom<DynamicIndexList>(
      $tileSizes,
      $staticTileSizes,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    attr-dict
  }];
  let hasVerifier = 1;
}

def Act_ExpandShapeOp : Act_BufferAccessOpBase<"expand_shape",
    [DeclareOpInterfaceMethods<BufferRelayoutOpInterface>]> {
  let summary = "Shape expansion access pattern (inverse of collapse_shape)";
  let description = [{}];
  let arguments = (ins
    Act_Pattern:$source,
    IndexListArrayAttr:$reassociation,
    Variadic<Index>:$outputShape,
    DenseI64ArrayAttr:$staticOutputShape);
  let assemblyFormat = [{
    $source $reassociation `output_shape`
    custom<DynamicIndexList>($outputShape, $staticOutputShape)
    attr-dict
  }];
  let extraClassDeclaration = [{
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(::llvm::cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return ::llvm::cast<IntegerAttr>(indexAttr).getInt();
            })));
      return reassociationIndices;
    }
  }];
}

def Act_CollapseShapeOp : Act_BufferAccessOpBase<"collapse_shape",
    [DeclareOpInterfaceMethods<BufferRelayoutOpInterface>]> {
  let summary = "Shape collapse access pattern (inverse of expand_shape)";
  let description = [{}];
  let arguments = (ins Act_Pattern:$source, IndexListArrayAttr:$reassociation);
  let assemblyFormat = [{
    $source $reassociation
    attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    SmallVector<ReassociationIndices, 4> getReassociationIndices() {
      SmallVector<ReassociationIndices, 4> reassociationIndices;
      for (auto attr : getReassociation())
        reassociationIndices.push_back(llvm::to_vector<2>(
            llvm::map_range(::llvm::cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return ::llvm::cast<IntegerAttr>(indexAttr).getInt();
            })));
      return reassociationIndices;
    }
  }];
}

def Act_TransposeOp : Act_BufferAccessOpBase<"transpose",
    [DeclareOpInterfaceMethods<BufferRelayoutOpInterface>]> {
  let summary = "Transpose access pattern (permutes dimensions)";
  let description = [{
    Permutes the dimensions of the accessed data. For example, a 2D access
    pattern with shape [M, N] transposed with permutation [1, 0] produces
    shape [N, M]. During materialization, this generates a `linalg.transpose`
    on the read path and an inverse transpose on the write path.
  }];
  let arguments = (ins Act_Pattern:$source, DenseI64ArrayAttr:$permutation);
  let assemblyFormat = [{
    $source `permutation` `=` $permutation attr-dict
  }];
  let extraClassDeclaration = [{
    SmallVector<int64_t> getInversePermutation() {
      auto perm = getPermutation();
      SmallVector<int64_t> inv(perm.size());
      for (unsigned i = 0; i < perm.size(); ++i)
        inv[perm[i]] = i;
      return inv;
    }
  }];
}

///===----------------------------------------------------------------------===//
/// Instruction operations
///===----------------------------------------------------------------------===//

def Act_YieldOp : Act_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "Yield values from an addr or compute region";
  let description = [{
    Terminates an addr region (yielding buffer access patterns) or a compute
    region (yielding destination buffer values). The number and types of yielded
    values must match the expectations of the enclosing `act.define` op.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);
  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

def Act_DefineOp : Act_Op<"define", [Symbol,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  IsolatedFromAbove]> {
  let summary = "Define a hardware instruction with access patterns and compute semantics";
  let description = [{
    Defines a named instruction template with source/destination buffer references,
    an `addr` region for buffer access patterns, and a `compute` region for
    computation semantics. The addr region's block arguments are address parameters
    (index type) that control dynamic access patterns. The compute region's first
    N block arguments correspond to source+destination buffers (ranked tensors);
    additional arguments are compute parameters (int/index) for dynamic computation.
    During lowering, `act.emit` inlines the compute region at the call site with
    access patterns materialized into concrete buffer access operations.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefArrayAttr:$sources,
    FlatSymbolRefArrayAttr:$destinations);
  let results = (outs);

  let regions = (region SizedRegion<1>:$access, SizedRegion<1>:$semantics);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    Block &getAccessBlock() { return getRegion(0).front(); }
    Block &getSemanticsBlock() { return getRegion(1).front(); }
    ArrayRef<BlockArgument> getExtraComputeArgs() {
      unsigned nBuffers = getSources().size() + getDestinations().size();
      return ArrayRef<BlockArgument>(getSemanticsBlock().getArguments()).drop_front(nBuffers);
    }
  }];
}

def Act_EmitOp : Act_Op<"emit", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  AttrSizedOperandSegments]> {
  let summary = "Emit (invoke) a defined instruction at the current program point";
  let description = [{
    Invokes an instruction defined by `act.define`. The `addr` parameters are
    mapped to the addr region's block arguments to materialize buffer access
    patterns. The `compute` parameters are mapped to the extra (non-buffer)
    block arguments of the compute region. Both parameter lists accept a mixture
    of static integer attributes and dynamic index-typed SSA values.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$instruction,
    Variadic<Index>:$addrParams,
    Variadic<Index>:$computeParams,
    DenseI64ArrayAttr:$staticAddrParams,
    DenseI64ArrayAttr:$staticComputeParams);
  let results = (outs);
  let assemblyFormat = [{
    $instruction
    `addr```custom<DynamicIndexList>(
      $addrParams,
      $staticAddrParams,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    `compute```custom<DynamicIndexList>(
      $computeParams,
      $staticComputeParams,
      "{}",
      "mlir::AsmParser::Delimiter::Paren")
    attr-dict
  }];
}

def Act_SequenceOp : Act_Op<"sequence", [AutomaticAllocationScope, IsolatedFromAbove, Symbol, NoTerminator]> {
  let summary = "Define a flat sequence of instructions";
  let description = [{
    Defines a named sequence of instructions. The sequence body can contain
    `act.emit` operations to invoke defined instructions. The body is a flat
    instruction stream for now; control flow is rejected until explicit `cf`
    support is added.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
  let extraClassDeclaration = [{
    Block *addEntryBlock() {
      return &getBody().emplaceBlock();
    }
  }];
}

def Act_LaunchOp : Act_Op<"launch", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Launch a sequence of instructions asynchronously";
  let description = [{
    Launches a sequence defined by `act.sequence` to execute. The
    launched sequence can contain any instructions, but must not contain another
    `act.launch`.
  }];
  let arguments = (ins FlatSymbolRefAttr:$sequence, UnitAttr:$async);
  let results = (outs);
  let assemblyFormat = [{
    $sequence (`async` $async^)? attr-dict
  }];
}

#endif // ACT_OPS_TD
