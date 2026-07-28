/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// The naming vocabulary: the one place a hardware identifier is composed. Every
// module, boundary port, storage cell and instance name the microarchitecture
// emits is built here, from one grammar:
//
//   name      := <owner> ( "_" <qualifier> )* [ "_" <field> ]
//   owner     := the source identifier (a NameLoc), else a structural token:
//                a<argNo> / m<memId> / u<unitId> / ch<chanId>
//   qualifier := <letters><number>, the number bound with no separator
//                (rd0, wr1, st, b3)
//   field     := addr | data | we | valid | ready | in | out
//
// Two rules keep the names stable, which is what the port manifest (the
// C++/Python contract) needs:
//
//   1. A group index is emitted unconditionally wherever the EMITTER decides
//      the count, i.e. a memory argument's port groups, and never where the
//      source signature fixes it (a scalar, a stream, a result). Indexing only
//      from the second port on would rename the first the moment a second
//      access appears, letting a scheduling decision rename a boundary.
//   2. A fallback keys on the owner's own id, never on a position in the port
//      list, so adding a port to one argument cannot rename another's.
//
// `verilogName` closes the loop against CIRCT: it escapes anything
// ExportVerilog would rewrite, so the manifest, authored before LegalizeNames
// runs, equals the emitted Verilog.
//
// The one thing stability does not cover: inserting an access BEFORE an
// existing access to the same argument still shifts its group index, so `A_rd0`
// and `A_rd1` swap. Keying the group on source line and column would fix it and
// cost more readability than it buys, and the manifest already shields every
// automated consumer.
//
// Emitters call these functions and never concatenate a name themselves. That
// invariant is what keeps the port declaration, the body's port accesses, the
// dataflow wiring and the cosim manifest on one string.
//===----------------------------------------------------------------------===//

#ifndef ALLO_MICROARCH_NAMING_H
#define ALLO_MICROARCH_NAMING_H

#include "allo/Microarch/Datapath.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::allo::uarch {

//===----------------------------------------------------------------------===//
// The fixed control ABI. Every emitted module carries exactly these four ports.
// They are spelled once here, declared and wired from here, and published in
// the port manifest, so the cosim harness reads them like any other port.
//===----------------------------------------------------------------------===//
constexpr llvm::StringLiteral kClk = "clk";
constexpr llvm::StringLiteral kRst = "rst";
constexpr llvm::StringLiteral kStart = "start";
constexpr llvm::StringLiteral kDone = "done";
/// An extern operator module's own fixed ports: data inputs `a`, `b`, ..., then
/// `clk`, then `ce` under a clock-enabled stall contract, then the result.
constexpr llvm::StringLiteral kCe = "ce";
constexpr llvm::StringLiteral kOpOut = "y";

/// A source-derived string made safe as a *final* SystemVerilog identifier.
/// Illegal characters become '_', and a name ExportVerilog would rewrite (a
/// keyword such as `input`, `wire`, `buf`) gets a trailing '_'. Applied to the
/// composed name, so an `output` array yields `output_wr0`, not `output__wr0`.
std::string verilogName(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// Owner tokens: the one structural-fallback vocabulary.
//===----------------------------------------------------------------------===//

/// The owner token of a boundary value: its source name (NameLoc), else
/// `a<argNo>` when it is a kernel argument, else \p fallback. Charset-sanitized
/// but not keyword-escaped, since the escape belongs to the composed name.
std::string ownerOf(Value v, llvm::StringRef fallback);
std::string ownerOf(Location loc, llvm::StringRef fallback);
/// `ownerOf` disambiguated against \p siblings, the values whose owner tokens
/// share a namespace (every memref of a module, every scalar argument). Two
/// values carrying one source name would give their cells or port groups one
/// set of names; each colliding value takes a token that is unique by
/// construction, its argument position for an argument and \p fallback
/// otherwise. Unrolling a body that declares an array is the second case: the
/// copies share a source name by design.
std::string uniqueOwnerOf(Value v, llvm::ArrayRef<Value> siblings,
                          llvm::StringRef fallback);
std::string argOwner(unsigned argNo); // a2
std::string memOwner(MemId m);        // m0
std::string unitOwner(UnitId u);      // u5
std::string chanOwner(StreamId s);    // ch1
std::string regOwner(RegId r);        // reg3

//===----------------------------------------------------------------------===//
// Field suffixes: the leaves of every port name.
//===----------------------------------------------------------------------===//

std::string portAddr(llvm::StringRef base);
std::string portData(llvm::StringRef base);
std::string portWe(llvm::StringRef base);
std::string portValid(llvm::StringRef base);
std::string portReady(llvm::StringRef base);

//===----------------------------------------------------------------------===//
// Port-group bases. The primitives compose a base from (owner, role, index);
// the resolvers derive those from the datapath.
//===----------------------------------------------------------------------===//

/// `<owner>_rd<i>` / `<owner>_wr<i>`: one memory access's port group.
std::string memBase(llvm::StringRef owner, bool write, unsigned group);
/// `<owner>_st`: a stream channel's handshake group.
std::string streamBase(llvm::StringRef owner);
/// `<owner>_in`: a scalar argument port (a whole port, no field).
std::string scalarBase(llvm::StringRef owner);
/// `<owner>_out`: a scalar result port (a whole port, no field).
std::string resultBase(llvm::StringRef owner);
/// `<base>_b<k>`: one bank of a partitioned array, port group or storage cell.
std::string bankBase(llvm::StringRef base, unsigned bank);
/// Which direction of a scattered element port a name is for. `Only` is the
/// unambiguous case, an argument the kernel just reads or just writes, and it
/// takes the bare name; an argument used BOTH ways needs its two ports told
/// apart. Vitis draws the same distinction (`b_0` vs `a_0_i` / `a_0_o`).
enum class ElemDir { Only, In, Out };

/// `<owner>_<k>`, plus `_in` / `_out` for a read-write argument: element \p k
/// of a scattered argument (`MemUnit::scattered`). A whole port on its own for
/// a read; the group base a write's `_we` hangs off for a write.
///
/// A BARE index, unlike every other qualifier here, and rule 1 above is why:
/// the count is fixed by the argument's TYPE, the way a scalar's or a stream's
/// is, not chosen by the emitter, so there is no scheduling decision that could
/// renumber it. It is also the name Vitis gives the same port.
std::string elemBase(llvm::StringRef owner, unsigned index,
                     ElemDir dir = ElemDir::Only);

/// The boundary interfaces of one external access, as (bank, base): one entry
/// for an unbanked or statically-routed access, one per bank for a
/// data-dependent one, whose crossbar drives every bank. The base itself is
/// `acc.portBase`, composed once by `enumerateBoundaryPorts`; this only expands
/// it across the banks the access reaches.
llvm::SmallVector<std::pair<unsigned, std::string>>
extPorts(const MemUnit &m, const MemUnit::Access &acc);
/// A stream channel's port base. Takes the whole \p dp because two stream
/// arguments of one module can share a source name (a systolic PE gets
/// `fifo[i,j]` and puts `fifo[i,j+1]`); a colliding group splits by direction,
/// then by channel id.
std::string streamPortBase(const Datapath &dp, const StreamChannel &s);
/// A scalar argument's port.
std::string scalarPortName(const Datapath &dp, const IOPort &io);
/// Result \p i of \p n: `ret_out`, or `ret<i>_out` for a multi-result kernel.
std::string resultPortName(unsigned i, unsigned n);

//===----------------------------------------------------------------------===//
// Internal cells and instances. These names reach waveforms and the netlist but
// never the manifest, so CIRCT is free to uniquify them. An unnamed value
// becomes `_GEN_37`, which is why every state cell below gets a name.
//===----------------------------------------------------------------------===//

/// On-chip storage for the buffer named \p owner: bank \p bank when it is one
/// of \p numBanks. The Datapath overload resolves a MemUnit's owner name first;
/// a caller that owns a buffer outside its own `Datapath` passes its own.
std::string memCellName(llvm::StringRef owner, unsigned numBanks,
                        unsigned bank);
std::string memCellName(const Datapath &dp, const MemUnit &m, unsigned bank);
/// `r<region>_<sig>`: a region's control-plane signal (`run`, `issue`, `iv`,
/// `phase`, `done`, `ce`). Region-scoped, so a waveform search for `r2_` pulls
/// up exactly one loop's controller. The StringRef form takes an already-formed
/// tag (`EmitContext::regionTag`).
std::string regionSignal(unsigned region, llvm::StringRef sig);
std::string regionSignal(llvm::StringRef tag, llvm::StringRef sig);
/// `<owner>_d<k>`: tap \p k of a delay chain, i.e. \p owner delayed k cycles.
/// The index carries the timing; CIRCT's uniquifying `_0`/`_1` suffixes only
/// look like they do.
std::string regTapName(llvm::StringRef owner, unsigned k);
/// `r<region>_sv<k>`: a region's survivor or loop-carried iter-arg latch.
std::string survivorName(unsigned region, unsigned k);
/// `<owner>_u<id>`, else `u<id>`: a compute-unit instance. ExportVerilog names
/// an instance's results `_<instance>_<port>` and ignores any namehint on the
/// instance, so folding the source name into the instance name is the only way
/// an IP result reaches the waveform as `_acc_u3_y` rather than `_u3_y`.
std::string unitInstanceName(const FuncUnit &u);
/// `<callee>_i<n>`: a child-kernel instance, indexed so two invocations of one
/// callee stay distinct instead of being uniquified apart by CIRCT.
std::string childInstanceName(llvm::StringRef callee, unsigned n);
/// `<chan>_<sig>`: a composed channel's own signal. Covers only the shim
/// built here; a `seq.fifo`'s own internals are named by CIRCT's lowering
/// (`fifo_mem`, `fifo_count`) with no name attribute to steer them.
std::string channelSignal(llvm::StringRef chan, llvm::StringRef sig);

/// The extern operator-module name for an IP-realized unit: its `impl` (the
/// operator's RTL module name), with a floating-point compare additionally
/// encoding its predicate, since `impl` alone (`fcmp_l1`) does not say which
/// comparison. Not passed through `verilogName`, because the simulation-model
/// generator joins this name back to the device operator's `sym_name` and it
/// must stay the device's own string.
std::string operatorModuleName(const FuncUnit &u);
/// The predicate an operator module name encodes (a floating-point compare's
/// `ogt`), empty otherwise. Published in the manifest so the simulation model
/// joins on it instead of re-splitting the module name.
std::string operatorPredicate(const FuncUnit &u);

/// Attach a readable Verilog name to \p v, derived from \p loc's NameLoc, so a
/// frontend variable (acc, buf, i) keeps its source name instead of CIRCT's
/// `_GEN` fallback. Picks the channel ExportVerilog reads: a register
/// (`seq.compreg`) names from its `name` attr, any other value from
/// `sv.namehint`. Best-effort, and a no-op when \p loc carries no name or when
/// \p v is a block argument, which the port interface names. CIRCT's
/// LegalizeNames uniquifies any collision.
void nameValue(Value v, Location loc);
/// Attach \p name directly. A no-op if empty or if \p v is not an op result.
/// For a name held as a string, such as a region's counter name, rather than on
/// a Location.
void nameValue(Value v, llvm::StringRef name);

} // namespace mlir::allo::uarch

#endif // ALLO_MICROARCH_NAMING_H
