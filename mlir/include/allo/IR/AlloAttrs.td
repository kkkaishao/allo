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
// A SKEWED partition: the bank is the sum of ALL subscripts modulo the factor,
// where block and cyclic each read one subscript. `dim` names the distribution
// dimension, the one divided down by the factor to make room. No HLS pragma
// spells this; it is the standard answer to an array read both by row and by
// column, where no per-axis layout can serve both.
def SkewPartition: I32EnumAttrCase<"SkewPartition", 3, "Skew">;

def PartitionKindEnum: I32EnumAttr<"PartitionKindEnum",
        "An example partition enum",
        [CompletePartition,BlockPartition,CyclicPartition,SkewPartition]> {
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
// Dependence-hint enums (`#pragma HLS dependence`) for allo.assume.nodep
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
// the Vitis `bind_storage` pragma strings. A bridge util maps those pragma
// strings (bram / ram_s2p / rom_1p / ...) onto these. The three axes are
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

//===----------------------------------------------------------------------===//
// Determinacy: the scheduling-determinacy class of a dcp region or kernel. It
// is the discriminant that selects the controller regime and the composition
// wiring (time-triggered vs handshake vs broadcast). The reifier declares it in
// the IR so that no consumer has to re-derive it.
//   * counted_static: exact compile-time latency (static trip / length);
//   * conditional:    a while loop or data-dependent guard (flushing exit);
//   * indeterminate:  dynamic-trip, bounded or unknown-length (no exact span);
//   * concurrent:     a dataflow container (async spawns; self-timed).
//===----------------------------------------------------------------------===//

def DetCountedStatic : I32EnumAttrCase<"CountedStatic", 0, "counted_static">;
def DetConditional   : I32EnumAttrCase<"Conditional",   1, "conditional">;
def DetIndeterminate : I32EnumAttrCase<"Indeterminate", 2, "indeterminate">;
def DetConcurrent    : I32EnumAttrCase<"Concurrent",    3, "concurrent">;
def DeterminacyEnum : I32EnumAttr<"DeterminacyEnum",
    "Scheduling-determinacy class of a dcp region / kernel",
    [DetCountedStatic, DetConditional, DetIndeterminate, DetConcurrent]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def DeterminacyAttr : EnumAttr<Allo_Dialect, DeterminacyEnum, "determinacy">;

//===----------------------------------------------------------------------===//
// CombOpKind: the combinational op a `dcp.compute` realizes when it takes the
// comb (not IP) path.
//===----------------------------------------------------------------------===//

def CombAddi      : I32EnumAttrCase<"Addi",      0,  "addi">;
def CombSubi      : I32EnumAttrCase<"Subi",      1,  "subi">;
def CombMuli      : I32EnumAttrCase<"Muli",      2,  "muli">;
def CombDivsi     : I32EnumAttrCase<"Divsi",     3,  "divsi">;
def CombDivui     : I32EnumAttrCase<"Divui",     4,  "divui">;
def CombRemsi     : I32EnumAttrCase<"Remsi",     5,  "remsi">;
def CombRemui     : I32EnumAttrCase<"Remui",     6,  "remui">;
def CombAndi      : I32EnumAttrCase<"Andi",      7,  "andi">;
def CombOri       : I32EnumAttrCase<"Ori",       8,  "ori">;
def CombXori      : I32EnumAttrCase<"Xori",      9,  "xori">;
def CombShli      : I32EnumAttrCase<"Shli",      10, "shli">;
def CombShrsi     : I32EnumAttrCase<"Shrsi",     11, "shrsi">;
def CombShrui     : I32EnumAttrCase<"Shrui",     12, "shrui">;
def CombCmpi      : I32EnumAttrCase<"Cmpi",      13, "cmpi">;
def CombSelect    : I32EnumAttrCase<"Select",    14, "select">;
def CombExtsi     : I32EnumAttrCase<"Extsi",     15, "extsi">;
def CombExtui     : I32EnumAttrCase<"Extui",     16, "extui">;
def CombTrunci    : I32EnumAttrCase<"Trunci",    17, "trunci">;
def CombIndexCast : I32EnumAttrCase<"IndexCast", 18, "index_cast">;
def CombApply     : I32EnumAttrCase<"Apply",     19, "apply">;
def CombNegf      : I32EnumAttrCase<"Negf",      20, "negf">;
def CombMinsi     : I32EnumAttrCase<"Minsi",     21, "minsi">;
def CombMaxsi     : I32EnumAttrCase<"Maxsi",     22, "maxsi">;
def CombMinui     : I32EnumAttrCase<"Minui",     23, "minui">;
def CombMaxui     : I32EnumAttrCase<"Maxui",     24, "maxui">;

def CombOpKindEnum : I32EnumAttr<"CombOpKindEnum",
    "The combinational op a comb-path dcp.compute realizes",
    [CombAddi, CombSubi, CombMuli, CombDivsi, CombDivui, CombRemsi, CombRemui,
     CombAndi, CombOri, CombXori, CombShli, CombShrsi, CombShrui, CombCmpi,
     CombSelect, CombExtsi, CombExtui, CombTrunci, CombIndexCast, CombApply,
     CombNegf, CombMinsi, CombMaxsi, CombMinui, CombMaxui]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def CombOpKindAttr : EnumAttr<Allo_Dialect, CombOpKindEnum, "comb_kind">;

//===----------------------------------------------------------------------===//
// StallContract: how a realized IP operator participates in the datapath's
// stall protocol
//===----------------------------------------------------------------------===//

// (data.., clk, ce) -> data   the FPGA hard-IP default (ce freezes the pipe)
def StallCe   : I32EnumAttrCase<"Ce",   0, "ce">;
// (data.., clk) -> data        no enable; the pipeline always advances
def StallFree : I32EnumAttrCase<"Free", 1, "free">;
// (s_*, m_*, clk) -> ...        a valid/ready handshake (variable latency)
def StallElastic : I32EnumAttrCase<"Elastic", 2, "elastic">;
def StallContractEnum : I32EnumAttr<"StallContractEnum",
    "How a realized IP operator participates in the stall protocol",
    [StallCe, StallFree, StallElastic]> {
  let cppNamespace = "mlir::allo";
  let genSpecializedAttr = 0;
}
def StallContractAttr : EnumAttr<Allo_Dialect, StallContractEnum, "stall">;

#endif // ALLO_ATTRS
