#ifndef ALLO_TYPES
#define ALLO_TYPES

include "allo/IR/Allo.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def StreamType : Allo_Type<"Stream"> {
  let summary = "stream type";
  let description = [{
    Stream type models a sequence of data elements (e.g. FIFO buffer).
    Its data values are destroyed after being read, and cannot be accessed randomly.

    Data types should be scalar types (e.g., int, float).
  }];
  let mnemonic = "stream";
  let parameters = (ins AnyType:$dataType, "std::size_t":$depth);
  let assemblyFormat = "`<` $dataType `,``` $depth `>`";
  let builders = [
    TypeBuilder<(ins "Type":$type), [{
      return $_get($_ctxt, type, /*capacity=*/2);
    }]>
  ];
}

def ChannelType : Allo_Type<"Channel"> {
  let summary = "channel type";
  let description = [{
    Channel type models a communication channel between different
    computational entities. It encapsulates the data type being transmitted
    and the capacity of the channel.

    Data types can be either scalar types (e.g., int, float) or composite types
    (e.g. memref, tensor).
  }];
  let mnemonic = "chan";
  let parameters = (ins AnyType:$dataType, "std::size_t":$capacity);
  let assemblyFormat = "`<` $dataType `,``` $capacity `>`";
  let extraClassDeclaration = [{
    Type getPrimitiveType() const {
      if (auto shaped = llvm::dyn_cast<ShapedType>(getDataType())) {
        return shaped.getElementType();
      }
      return getDataType();
    }
  }];
  let builders = [
    TypeBuilder<(ins "MemRefType":$type, CArg<"std::size_t", "2">:$capacity), [{
      return $_get($_ctxt, type, capacity);
    }]>,
    TypeBuilder<(ins "RankedTensorType":$type, CArg<"std::size_t", "2">:$capacity), [{
      return $_get($_ctxt, type, capacity);
    }]>,
  ];
}

#endif // ALLO_TYPES