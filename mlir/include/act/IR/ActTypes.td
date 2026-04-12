#ifndef ACT_TYPES_TD
#define ACT_TYPES_TD

include "act/IR/Act.td"

def BufferTypeInterface : TypeInterface<"BufferTypeInterface"> {
  let description = [{
    Common interface for all buffer element types (scalar, vector, tile, HBM).
    Each buffer element type describes the access granularity of a hardware
    scratchpad or memory space.
  }];
  let cppNamespace = "::mlir::act";
  let methods = [
    InterfaceMethod<"Get the scalar element type (e.g. f32, bf16)",
    "mlir::Type", "getElementType", (ins)>,
    InterfaceMethod<"Get the rank of a single buffer element (0 for scalar, 1 for vector, etc.)",
    "unsigned", "getRank", (ins)>,
    InterfaceMethod<"Get the shape of a single buffer element",
    "llvm::ArrayRef<int64_t>", "getShape", (ins)>,
  ];
}

class Act_BufferTypeBase<string mnemonic> : Act_Type<mnemonic, [BufferTypeInterface]>;

def Act_HBMBuffer : Act_BufferTypeBase<"HBMBuffer"> {
  let summary = "Off-chip HBM memory element type";
  let description = [{
    Represents the element type for off-chip high-bandwidth memory (HBM) buffers.
    Unlike on-chip buffer types, HBM elements can be multi-dimensional and are
    accessed at byte granularity. Syntax: `!act.hbm<8192xbf16>`.
  }];
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape);
  let mnemonic = "hbm";
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
  }];
  let genVerifyDecl = 1;
}

def Act_ScalarBuffer : Act_BufferTypeBase<"ScalarBuffer"> {
  let summary = "On-chip scalar buffer element type";
  let description = [{
    Represents the element type for on-chip buffers accessed at scalar granularity.
    Each buffer slot holds a single scalar value. Syntax: `!act.scalar<f32>`.
  }];
  let parameters = (ins
    "Type":$elementType);
  let mnemonic = "scalar";
  let assemblyFormat = "`<` $elementType `>`";
  let extraClassDeclaration = [{
    unsigned getRank() const { return 0; }
    ArrayRef<int64_t> getShape() const { return {}; }
  }];
  let genVerifyDecl = 1;
}

def Act_VectorBuffer : Act_BufferTypeBase<"VectorBuffer"> {
  let summary = "On-chip vector buffer element type";
  let description = [{
    Represents the element type for on-chip buffers accessed at vector granularity.
    Each buffer slot holds a 1-D vector of elements. Syntax: `!act.vector<64xbf16>`.
  }];
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape);
  let mnemonic = "vector";
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getRank() const { return 1; }
  }];
  let genVerifyDecl = 1;
}

def Act_TileBuffer : Act_BufferTypeBase<"TileBuffer"> {
  let summary = "On-chip tile buffer element type";
  let description = [{
    Represents the element type for on-chip buffers accessed at tile (2-D+)
    granularity. Each buffer slot holds a multi-dimensional tile of elements.
    Syntax: `!act.tile<4x4xf32>`.
  }];
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape);
  let mnemonic = "tile";
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
  }];
  let genVerifyDecl = 1;
}

def Act_Descriptor : Act_Type<"Descriptor"> {
  let summary = "Hardware descriptor with named typed fields";
  let description = [{
    Represents a hardware descriptor type with named fields, such as a TMA
    descriptor with "offset" and "stride" fields. Each field has a name and
    a primitive integer type. Syntax: `!act.desc<["offset":i8, "stride":i8]>`.
  }];
  let parameters = (ins
    ArrayRefParameter<"StringAttr">:$fields,
    ArrayRefParameter<"Type">:$fieldTypes);
  let mnemonic = "desc";
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Act_State : Act_Type<"State"> {
  let summary = "Hardware state with enumerated values";
  let description = [{
    Represents the type of a hardware state register that holds one of a fixed
    set of enumerated values. The element type is the hardware primitive integer
    type used to store the state. Syntax: `!act.state<i32>`.
  }];
  let parameters = (ins
    "Type":$elementType);
  let mnemonic = "state";
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

def Act_Pattern : Act_Type<"Pattern"> {
  let summary = "A token type representing a access pattern";
  let description = [{
    Represents a token type that encodes a specific access pattern.
    Syntax: `!act.pattern`
  }];
  let mnemonic = "pattern";
}

#endif // ACT_TYPES_TD
