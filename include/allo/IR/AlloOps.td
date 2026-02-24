#ifndef ALLO_OPS
#define ALLO_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "allo/IR/Allo.td"
include "allo/IR/AlloAttrs.td"
include "allo/IR/AlloTypes.td"
include "allo/IR/AlloInterfaces.td"

//===----------------------------------------------------------------------===//
// SPMD Operations
//===----------------------------------------------------------------------===//
def GetProgramIdOp : Allo_Op<"program_id", [Pure]> {
  let arguments = (ins I32Attr:$axi);
  let results = (outs Index:$id);
  let assemblyFormat = "$axi attr-dict `:` type($id)";
  let hasVerifier = 1;
}

def GetNumProgramsOp : Allo_Op<"num_programs", [Pure]> {
  let arguments = (ins I32Attr:$axi);
  let results = (outs Index:$num);
  let assemblyFormat = "$axi attr-dict `:` type($num)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Kernel Operations
//===----------------------------------------------------------------------===//

// The following ops, including `call`, `func`, and `return` are copied and modified from
// https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Func/IR/FuncOps.td
def KernelOp : Allo_Op<"kernel", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface,
  HasParent<"ModuleOp">]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       VirtMapAttr:$virtual_mapping);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
    CArg<"ArrayRef<int64_t>", "{}">:$virtual_mapping)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

    //===------------------------------------------------------------------===//
    // Custom Methods
    //===------------------------------------------------------------------===//

    ArrayRef<int64_t> getVirtualMappingVec() {
      return getVirtualMapping().getRanges();
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def ReturnOp : Allo_Op<"return", [Pure, HasParent<"allo::KernelOp">, 
  /*MemRefsNormalizable, */ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `allo.return` operation represents a return operation within a kernel.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    allo.kernel @foo() : (i32, f8) {
      ...
      allo.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, mlir::ValueRange());
  }]>];

  let assemblyFormat = "attr-dict ($srcs^ `:` type($srcs))?";
  let hasVerifier = 1;
}

def CallOp : Allo_Op<"call", [CallOpInterface, /*MemRefsNormalizable, */                    
  DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `allo.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = allo.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs);
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
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType() {
      return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
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

    // Required by CallOpInterface.
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}


//===----------------------------------------------------------------------===//
// Channel Operations
//===----------------------------------------------------------------------===//
def ChanCreateOp : Allo_Op<"chan.create", [Symbol, HasParent<"ModuleOp">]> {
  let summary = "Channel creation operation";
  let description = [{
    The `chan.create` operation creates a channel with specified element type
    and capacity.

    The result value is a placeholder, which should not be directly used.
    Instead, the channel should be referred by its symbol name in other operations.

    e.g.
    allo.chan.create @buffer_chan : !allo.chan<tensor<16x16xf32>,2>
    allo.chan.create @stream_chan : !allo.chan<f32,16>

    The created channel cannot directly be used for read/write operations.
    Its data must be acquired as a buffer using `allo.chan.acquire_buffer` operation.
    or converted to a stream using `allo.chan.to_stream` operation.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
    I64ArrayAttr:$shape,
    TypeAttrOf<ChannelType>:$chan_type);
  let results = (outs);
  let assemblyFormat = [{
    $sym_name $shape attr-dict `:` $chan_type
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$name, "ChannelType":$chanType, CArg<"ArrayRef<int64_t>", "{}">:$shape), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), name), chanType, shape);
    }]>,
    OpBuilder<(ins "StringAttr":$name, "ChannelType":$chanType, CArg<"ArrayRef<int64_t>", "{}">:$shape), [{
      $_state.addAttribute(SymbolTable::getSymbolAttrName(), name);
      auto shapeAttr = $_builder.getI64ArrayAttr(shape);
      $_state.addAttribute(getShapeAttrName($_state.name), shapeAttr);
      auto chanAttr = TypeAttr::get(chanType);
      $_state.addAttribute(getChanTypeAttrName($_state.name), chanAttr);
    }]>
  ];
}

def ChanGetOp : Allo_Op<"chan.get", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<ChannelAccessOpInterface>,
  MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "get an element from a channel";
  let description = [{

  }];
  let arguments = (ins
    FlatSymbolRefAttr:$channel,
    Variadic<Index>:$indices,
    UnitAttr:$blocking);
  let results = (outs AnyType:$value);
  let assemblyFormat = [{
    $channel `[` $indices `]` (`blocking` $blocking^)? attr-dict `:` type($value)
  }];
}

def ChanPutOp : Allo_Op<"chan.put", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<ChannelAccessOpInterface>,
  MemoryEffects<[MemWrite]>]> {
  let summary = "put an element to a channel";
  let description = [{

  }];
  let arguments = (ins
    FlatSymbolRefAttr:$channel,
    Variadic<Index>:$indices,
    AnyType:$value,
    UnitAttr:$blocking);
  let results = (outs);
  let assemblyFormat = [{
    $value `,` $channel `[` $indices `]` (`blocking` $blocking^)? attr-dict `:` type($value)
  }];
}

def ChanAcquireOp : Allo_Op<"chan.acquire", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<ChannelAccessOpInterface>,
  ]> {
  let summary = "Channel to buffer conversion operation";
  let description = [{
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$channel,
    Variadic<Index>:$indices,
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$size,
    UnitAttr:$blocking);
  let results = (outs Variadic<AnyType>:$buffers);
  let assemblyFormat = [{
    $channel `[` $indices `]` `,` $size (`blocking` $blocking^)? attr-dict `:` type($buffers)
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$chan, "ValueRange":$indices, "Type":$dataType, 
              CArg<"int", "1">:$size), [{
      build($_builder, $_state, $_builder.getStringAttr(chan), indices, dataType, size);
    }]>,
    OpBuilder<(ins "StringAttr":$chan, "ValueRange":$indices, "Type":$dataType, 
              CArg<"int", "1">:$size), [{
      build($_builder, $_state, FlatSymbolRefAttr::get(chan), indices, dataType, size);
    }]>,
    OpBuilder<(ins "FlatSymbolRefAttr":$chan, "ValueRange":$indices, "Type":$dataType,                
              CArg<"int", "1">:$size), [{
      $_state.addOperands(indices);
      $_state.addAttribute(getChannelAttrName($_state.name), chan);
      $_state.addAttribute(getSizeAttrName($_state.name), $_builder.getI32IntegerAttr(size));
      llvm::SmallVector<Type, 4> types(size, dataType);
      $_state.addTypes(types);
    }]>,
  ];
  let skipDefaultBuilders = 1;
}

def ChanReleaseOp : Allo_Op<"chan.release", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<ChannelAccessOpInterface>,
  AttrSizedOperandSegments]> {
  let summary = "Release buffers back to the channel";
  let description = [{
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$channel,
    Variadic<Index>:$indices,
    Variadic<AnyType>:$buffers,
    UnitAttr:$blocking);
  let results = (outs);
  let assemblyFormat = [{
    $channel `[` $indices `]` `,` $buffers (`blocking` $blocking^)? attr-dict `:` type($buffers)
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$chan, "ValueRange":$indices, "ValueRange":$buffers), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), chan), indices, buffers);
    }]>,
    OpBuilder<(ins "StringAttr":$chan, "ValueRange":$indices, "ValueRange":$buffers), [{
      build($_builder, $_state, FlatSymbolRefAttr::get(chan), indices, buffers);
    }]>,
    OpBuilder<(ins "FlatSymbolRefAttr":$chan, "ValueRange":$indices, "ValueRange":$buffers), [{
      $_state.addAttribute(getChannelAttrName($_state.name), chan);
      $_state.addOperands(indices);
      $_state.addOperands(buffers);
      SmallVector<int32_t, 4> segSizes = 
        {static_cast<int32_t>(indices.size()), static_cast<int32_t>(buffers.size())};
      $_state.addAttribute("operandSegmentSizes", $_builder.getDenseI32ArrayAttr(segSizes));
    }]>,
  ];
  let skipDefaultBuilders = 1;
}

def BitExtractOp : Allo_Op<"bit.extract", [Pure]> {
  let summary = "Extract bits from an integer value";
  let description = [{
    The `bit.extract` operation extracts a contiguous sequence of bits from an
    integer value, starting at a specified bit position and with a specified
    width.

    Example:
    ```
    %0 = allo.bit.extract %input, lo 8, width 8 : i8 from i32
    ```
  }];

  let arguments = (ins AnySignlessInteger:$input, Index:$start, I64Attr:$width);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
    $input `,` `lo` $start `width` $width attr-dict `:` type($result) `from` type($input)
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$start, "int64_t":$width), [{
      auto resTy = $_builder.getIntegerType(width);
      build($_builder, $_state, resTy, input, start, width);
    }]>
  ];
}

def BitInsertOp : Allo_Op<"bit.insert", [Pure]> {
  let summary = "Insert bits into an integer value";
  let description = [{
    The `bit.insert` operation inserts a contiguous sequence of bits from a source
    integer value into a destination integer value, starting at a specified bit
    position and with a specified width.

    Example:
    ```
    %0 = allo.bit.insert %dest, %src, 8, 16 : i32
    ```
  }];

  let arguments = (ins AnySignlessInteger:$dest, AnySignlessInteger:$src,
                   Index:$start, I64Attr:$width);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
    $dest `,` $src `,` `lo` $start `width` $width attr-dict `:` type($src) `to` type($dest) `,` type($result)
  }];
  let hasVerifier = 1; 
  let builders = [
    OpBuilder<(ins "Value":$dest, "Value":$src, "Value":$start, "int64_t":$width), [{
      auto resTy = dest.getType();
      build($_builder, $_state, resTy, dest, src, start, width);
    }]>
  ];
}

def BitConcatOp : Allo_Op<"bit.concat", [Pure]> {
  let summary = "Concatenate bits from multiple integer values";
  let description = [{
    The `bit.concat` operation concatenates the bits from multiple integer values
    into a single integer value. The order of concatenation is determined by the
    order of the input operands.

    Example:
    ```
    %0 = allo.bit.concat %input1, %input2, %input3 : i32
    ```
  }];

  let arguments = (ins Variadic<AnySignlessInteger>:$inputs);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$inputs), [{
      // Compute the total width of the concatenated result.
      int64_t totalWidth = 0;
      for (Value input : inputs) {
        auto intTy = cast<IntegerType>(input.getType());
        totalWidth += intTy.getWidth();
      }
      auto resTy = $_builder.getIntegerType(totalWidth);
      build($_builder, $_state, resTy, inputs);
    }]>
  ];
}

#endif // ALLO_OPS