#ifndef ALLO_INTERFACES
#define ALLO_INTERFACES

include "mlir/IR/OpBase.td"

def ChannelAccessOpInterface : OpInterface<"ChannelAccessOpInterface"> {
  let description = [{}];
  let methods = [
    InterfaceMethod<
      "Return the channel symbol reference.",
      "FlatSymbolRefAttr",
      "getChannelAttr"
    >,
    InterfaceMethod<
      "Return the indices operands.",
      "ValueRange",
      "getIndices"
    >,
    InterfaceMethod<
      "Return the mutable indices operands.",
      "MutableOperandRange",
      "getIndicesMutable"
    >,
    InterfaceMethod<
      "Return the blocking attribute.",
      "bool",
      "getBlocking"
    >
  ];
  let cppNamespace = "::mlir::allo";
}

#endif // ALLO_INTERFACES