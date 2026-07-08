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

//===----------------------------------------------------------------------===//
// Dependence-hint enums (mirror `#pragma HLS dependence`; see allo.assume.nodep)
//===----------------------------------------------------------------------===//

def AssumeDepInter : I32EnumAttrCase<"Inter", 0, "inter">;
def AssumeDepIntra : I32EnumAttrCase<"Intra", 1, "intra">;
def AssumeDepTypeEnum : I32EnumAttr<"AssumeDepTypeEnum",
    "Dependence class of an assume.nodep hint: across vs. within iterations",
    [AssumeDepInter, AssumeDepIntra]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def AssumeDepTypeAttr : EnumAttr<Allo_Dialect, AssumeDepTypeEnum, "dep_type">;

def AssumeDirRAW : I32EnumAttrCase<"RAW", 0, "raw">;
def AssumeDirWAR : I32EnumAttrCase<"WAR", 1, "war">;
def AssumeDirWAW : I32EnumAttrCase<"WAW", 2, "waw">;
def AssumeDepDirEnum : I32EnumAttr<"AssumeDepDirEnum",
    "Dependence direction of an assume.nodep hint (read/write ordering)",
    [AssumeDirRAW, AssumeDirWAR, AssumeDirWAW]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def AssumeDepDirAttr : EnumAttr<Allo_Dialect, AssumeDepDirEnum, "dep_dir">;

//===----------------------------------------------------------------------===//
// Memory-model enums: the RTL backend's own storage vocabulary, decoupled from
// the Vitis `bind_storage` pragma strings. A bridge util maps the legacy strings
// (bram / ram_s2p / rom_1p / ...) onto these; the frontend will eventually attach
// them directly (see drafts/memory-model-design.md). The three axes are
// orthogonal: implementation primitive, port topology, and read-only-ness.
//===----------------------------------------------------------------------===//

def MemImplAuto     : I32EnumAttrCase<"Auto", 0, "auto">;
def MemImplRegister : I32EnumAttrCase<"Register", 1, "register">;
def MemImplLUTRAM   : I32EnumAttrCase<"LUTRAM", 2, "lutram">;
def MemImplBRAM     : I32EnumAttrCase<"BRAM", 3, "bram">;
def MemImplURAM     : I32EnumAttrCase<"URAM", 4, "uram">;
def MemoryImplEnum : I32EnumAttr<"MemoryImplEnum",
    "Storage implementation primitive a memref is bound to",
    [MemImplAuto, MemImplRegister, MemImplLUTRAM, MemImplBRAM, MemImplURAM]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def MemoryImplAttr : EnumAttr<Allo_Dialect, MemoryImplEnum, "mem_impl">;

// Port topology: how many concurrent read/write ports, and whether reads and
// writes contend. SinglePort = 1 shared R/W; SimpleDualPort = 1 read + 1 write
// (dedicated, no contention); TrueDualPort = 2 shared R/W.
def MemPortSingle     : I32EnumAttrCase<"SinglePort", 0, "1p">;
def MemPortSimpleDual : I32EnumAttrCase<"SimpleDualPort", 1, "s2p">;
def MemPortTrueDual   : I32EnumAttrCase<"TrueDualPort", 2, "t2p">;
def MemoryPortEnum : I32EnumAttr<"MemoryPortEnum",
    "Port topology of a memref (concurrent read/write ports)",
    [MemPortSingle, MemPortSimpleDual, MemPortTrueDual]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def MemoryPortAttr : EnumAttr<Allo_Dialect, MemoryPortEnum, "mem_port">;

// RAM vs ROM: a ROM has no write ports (a store to it is a design error).
def MemKindRAM : I32EnumAttrCase<"RAM", 0, "ram">;
def MemKindROM : I32EnumAttrCase<"ROM", 1, "rom">;
def MemoryKindEnum : I32EnumAttr<"MemoryKindEnum",
    "Whether a memref is writable (RAM) or read-only (ROM)",
    [MemKindRAM, MemKindROM]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def MemoryKindAttr : EnumAttr<Allo_Dialect, MemoryKindEnum, "mem_kind">;

#endif // ALLO_ATTRS
