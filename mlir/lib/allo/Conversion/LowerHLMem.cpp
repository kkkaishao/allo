/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------------------------------------------------------===//
// Lower `seq.hlmem` and the ports referring to it onto an `sv.reg` array.
//
// Ported from CIRCT's `lower-seq-hlmem`, which this tree used until the write
// half had to change. That pass materializes EVERY write of a memory inside one
// `always_ff`, so an array with two writers becomes a register file with a
// priority multiplexer in front of every word: no Xilinx template infers a
// block RAM from it, and measured at 512x32 the difference is one BRAM36
// against 43,361 LUTs. A true dual port infers only when each port is described
// in its own block, which is a shape the upstream pass cannot express.
//
// Owning the lowering also folds in the two passes that used to bracket it. A
// memory's power-on contents live on the `seq.hlmem`, which the lowering
// erases, and belong in the `sv.reg` the lowering creates, so they used to ride
// out onto the enclosing module as an attribute and be matched back by NAME
// afterwards. Here the register is in hand, and the initializer is written
// straight into it.
//===----------------------------------------------------------------------===//

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h" // kMemoryInitAttr

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::allo {
#define GEN_PASS_DEF_LOWERHLMEMPASS
#include "allo/Conversion/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;
using namespace circt;

namespace {

void lowerMemory(seq::HLMemOp mem) {
  hw::UnpackedArrayType arrayTy = hw::UnpackedArrayType::get(
      mem.getMemType().getElementType(), mem.getMemType().getShape()[0]);
  // Only the ports may hold an `!seq.hlmem` handle, so a user is one or the
  // other; both invariants below hold for every producer in this tree, the
  // datapath emitter and CIRCT's own FIFO lowering.
  assert(mem.getMemType().getShape().size() == 1 &&
         "an hlmem is emitted one dimensional");
  SmallVector<seq::ReadPortOp> reads;
  SmallVector<seq::WritePortOp> writes;
  for (Operation *user : mem.getHandle().getUsers()) {
    if (auto read = dyn_cast<seq::ReadPortOp>(user)) {
      reads.push_back(read);
      continue;
    }
    auto write = cast<seq::WritePortOp>(user);
    assert(write.getLatency() == 1 && "a write port is emitted at latency 1");
    writes.push_back(write);
  }

  OpBuilder b(mem);
  Location loc = mem.getLoc();
  Value clk = mem.getClk();
  StringRef name = mem.getName();
  Value array = sv::RegOp::create(b, loc, arrayTy, mem.getNameAttr());

  // The power-on contents. The assignments are BLOCKING, which is what a
  // simulator starts the array from and what synthesis reads back as a
  // block-RAM INIT; non-blocking ones still simulate, so nothing but the
  // emitted text catches the difference.
  if (auto init = mem->getAttrOfType<ArrayAttr>(kMemoryInitAttr)) {
    assert(init.size() == arrayTy.getNumElements() &&
           "an initializer must cover exactly the declared words");
    Type addrTy = b.getIntegerType(llvm::Log2_64_Ceil(init.size()));
    sv::InitialOp::create(b, loc, [&] {
      for (auto [i, word] : llvm::enumerate(init)) {
        Value idx = hw::ConstantOp::create(b, loc, addrTy, i);
        Value slot = sv::ArrayIndexInOutOp::create(b, loc, array, idx);
        sv::BPAssignOp::create(
            b, loc, slot,
            hw::ConstantOp::create(b, loc, cast<IntegerAttr>(word)));
      }
    });
  }

  // The writes. One block for all of them, which is what upstream did and what
  // defeats block-RAM inference past one writer; splitting them onto a port
  // each is the change this pass exists to make and is not made yet, because it
  // turns a same-address collision from last-writer-wins into a race and so
  // needs the addresses PROVEN disjoint.
  Value hwClk = seq::FromClockOp::create(b, clk.getLoc(), clk);
  sv::AlwaysFFOp::create(
      b, loc, sv::EventControl::AtPosEdge, hwClk, sv::ResetType::SyncReset,
      sv::EventControl::AtPosEdge, mem.getRst(), [&] {
        for (seq::WritePortOp write : writes) {
          Location wloc = write.getLoc();
          sv::IfOp::create(b, wloc, write.getWrEn(), [&] {
            Value slot = sv::ArrayIndexInOutOp::create(b, wloc, array,
                                                       write.getAddresses()[0]);
            sv::PAssignOp::create(b, wloc, slot, write.getInData());
          });
        }
      });

  // The reads. A latency above one delays the ADDRESS for all but the last
  // cycle and registers the datum for that one, so the combinational read never
  // starts a critical path. A read enable is not modelled, which is upstream's
  // behaviour too and is why a FIFO's `rdEn` reaches here and is dropped.
  for (auto [i, read] : llvm::enumerate(reads)) {
    OpBuilder rb(read);
    rb.setInsertionPointAfter(read);
    Location rloc = read.getLoc();
    Value addr = read.getAddresses()[0];
    unsigned latency = read.getLatency();
    for (unsigned d = 0; d + 1 < latency; ++d)
      addr = seq::CompRegOp::create(
          rb, rloc, addr, clk,
          rb.getStringAttr(name + "_rdaddr" + Twine(i) + "_dly" + Twine(d)));
    Value slot = sv::ArrayIndexInOutOp::create(rb, rloc, array, addr);
    Value data = sv::ReadInOutOp::create(rb, rloc, slot);
    if (latency > 0)
      data = seq::CompRegOp::create(
          rb, rloc, data, clk,
          rb.getStringAttr(name + "_rd" + Twine(i) + "_reg"));
    read.replaceAllUsesWith(data);
    read.erase();
  }

  for (seq::WritePortOp write : writes)
    write.erase();
  mem.erase();
}

struct LowerHLMemPass : public allo::impl::LowerHLMemPassBase<LowerHLMemPass> {
  void runOnOperation() override {
    SmallVector<seq::HLMemOp> mems;
    getOperation().walk([&](seq::HLMemOp mem) { mems.push_back(mem); });
    for (seq::HLMemOp mem : mems)
      lowerMemory(mem);
  }
};

} // namespace
