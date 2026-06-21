/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_ATTRS
#define ALLO_ATTRS

include "allo/IR/Allo.td"
include "mlir/IR/EnumAttr.td"

// https://mlir.llvm.org/docs/OpDefinitions/#enum-attributes
def CompletePartition: I32EnumAttrCase<"CompletePartition", 0, "Complete">;
def BlockPartition: I32EnumAttrCase<"BlockPartition", 1, "Block">;
def CyclicPartition: I32EnumAttrCase<"CyclicPartition", 2, "Cyclic">;

def PartitionKindEnum: I32EnumAttr<"PartitionKindEnum",
        "An example partition enum",
        [CompletePartition,BlockPartition,CyclicPartition]> {
    let cppNamespace = "mlir::allo";
    let stringToSymbolFnName = "ConvertToPartitionEnum";
    let symbolToStringFnName = "ConvertToPartitionString";
}

def PartitionAxis : Allo_Attr<"PartitionAxis"> {
  let mnemonic = "part_axis";
  let parameters = (ins
    "allo::PartitionKindEnum":$kind,
    "int64_t":$factor,
    "int64_t":$dim);
  let assemblyFormat = [{
    `(` $dim `,` `` $kind `,` `` $factor`)`
  }];
  let cppNamespace = "mlir::allo";
  let genVerifyDecl = 1;
}

def PartitionAttr : Allo_Attr<"Partition"> {
  let mnemonic = "partition";
  let parameters = (ins ArrayRefParameter<"allo::PartitionAxisAttr">:$partitions);
  let assemblyFormat = [{
    `<` `[` $partitions `]` `>`
  }];
  let cppNamespace = "mlir::allo";
  let genVerifyDecl = 1;
}

#endif // ALLO_ATTRS
