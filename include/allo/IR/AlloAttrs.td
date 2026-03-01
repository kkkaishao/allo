#ifndef ALLO_ATTRS
#define ALLO_ATTRS

include "mlir/IR/EnumAttr.td"
include "allo/IR/Allo.td"

def PartitionKindEnum : I32EnumAttr<"PartitionKindEnum",
  "HLS array partitioning kind",
  [
    I32EnumAttrCase<"Complete", 0>,
    I32EnumAttrCase<"Block",    1>,
    I32EnumAttrCase<"Cyclic",   2>
  ]> {
  let cppNamespace = "::mlir::allo";
}

def PartitionAttr : Allo_Attr<"Partition"> {
  let mnemonic = "part";
  let parameters = (ins
    ArrayRefParameter<"allo::PartitionKindEnum">:$kinds,
    ArrayRefParameter<"int64_t">:$factors,
    ArrayRefParameter<"int64_t">:$dims);
  let assemblyFormat = [{
    `<` `{` `kinds` `=` `[` $kinds `]` `,` `factors` `=` `[` $factors `]` `,` `dims` `=` `[` $dims `]` `}` `>`
  }];
  let cppNamespace = "::mlir::allo";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    PartitionInfoVec getPartitionInfoVec() const {
      PartitionInfoVec result;
      for (auto [dim, kind, factor] : llvm::zip(getDims(), getKinds(), getFactors())) {
        result.push_back({dim, kind, factor});
      }
      return result;
    }
    PartitionInfoMap getPartitionInfoMap() const {
      PartitionInfoMap result;
      for (auto [dim, kind, factor] : llvm::zip(getDims(), getKinds(), getFactors())) {
        result[dim] = {kind, factor};
      }
      return result;
    }
  }];
}

def BRAMStorage : I32EnumAttrCase<"BRAM", 0>;
def URAMStorage : I32EnumAttrCase<"URAM", 1>;
def LUTRAMStorage : I32EnumAttrCase<"LUTRAM", 2>;
def DDRStorage : I32EnumAttrCase<"DDR", 3>;
def HBMStorage : I32EnumAttrCase<"HBM", 4>;

def StorageKindEnum : I32EnumAttr<"StorageKindEnum",
        "Storage kind enum for array storage types",
        [BRAMStorage, URAMStorage, LUTRAMStorage, DDRStorage, HBMStorage]> {
    let cppNamespace = "mlir::allo";
}

#endif // ALLO_ATTRS