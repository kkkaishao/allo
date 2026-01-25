#ifndef ALLO_ATTRS
#define ALLO_ATTRS

include "mlir/IR/EnumAttr.td"
include "allo/IR/Allo.td"

def CompletePartition: I32EnumAttrCase<"Complete", 0>;
def BlockPartition: I32EnumAttrCase<"Block", 1>;
def CyclicPartition: I32EnumAttrCase<"Cyclic", 2>;

def PartitionKindEnum : I32EnumAttr<"PartitionKindEnum",
        "Partition kind enum for array partitioning",
        [CompletePartition, BlockPartition, CyclicPartition]> {
    let cppNamespace = "mlir::allo";
}

def PartitionAttr : Allo_Attr<"Partition"> {
  let mnemonic = "partition";
  let parameters = (ins 
    ArrayRefParameter<"allo::PartitionKindEnum">:$kinds, 
    ArrayRefParameter<"unsigned">:$factors
  );
  let assemblyFormat = [{
    `<` `{` `kinds` `=` $kinds `,` `factors` `=` $factors `}` `>`
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

def DataflowRoleAttr : I32EnumAttr<"DataflowRoleEnum",
  "Dataflow role annotations",
  [
    I32EnumAttrCase<"Consumer", 0>,
    I32EnumAttrCase<"Producer", 1>
  ]> {
  let cppNamespace = "mlir::allo";
}


#endif // ALLO_ATTRS