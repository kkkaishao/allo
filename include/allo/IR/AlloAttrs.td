#ifndef ALLO_ATTRS
#define ALLO_ATTRS

include "mlir/IR/EnumAttr.td"
include "allo/IR/Allo.td"

def PartitionKindEnum : I32EnumAttr<"PartitionKindEnum",
  "HLS array partitioning kind",
  [
    I32EnumAttrCase<"Complete", 0, "complete">,
    I32EnumAttrCase<"Block",    1, "block">,
    I32EnumAttrCase<"Cyclic",   2, "cyclic">
  ]> {
  let cppNamespace = "::mlir::allo";
}

def PartitionAttr : Allo_Attr<"Partition"> {
  let mnemonic = "partition";
  let parameters = (ins
    "allo::PartitionKindEnum":$kind,
    "int64_t":$factor,
    "int64_t":$dim);
  let assemblyFormat = [{
    `<` `{` `kind` `=` $kind `,` `factor` `=` $factor `,` `dim` `=` $dim `}` `>`
  }];
  let cppNamespace = "mlir::allo";
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