/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"
#include "allo/IR/AlloOps.h" // kMemoryInitAttr, kIndependentWritesAttr

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

  // BLOCKING assignments: only those does synthesis read back as a block-RAM
  // INIT, and only the emitted text tells the two apart.
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

  Value hwClk = seq::FromClockOp::create(b, clk.getLoc(), clk);
  auto emitBlock = [&](ArrayRef<seq::WritePortOp> group) {
    sv::AlwaysFFOp::create(
        b, loc, sv::EventControl::AtPosEdge, hwClk, sv::ResetType::SyncReset,
        sv::EventControl::AtPosEdge, mem.getRst(), [&] {
          for (seq::WritePortOp write : group) {
            Location wloc = write.getLoc();
            sv::IfOp::create(b, wloc, write.getWrEn(), [&] {
              Value slot = sv::ArrayIndexInOutOp::create(
                  b, wloc, array, write.getAddresses()[0]);
              sv::PAssignOp::create(b, wloc, slot, write.getInData());
            });
          }
        });
  };
  // A block per port only where the memory promises the ports never collide:
  // sharing one is what orders a same-address collision.
  if (writes.size() > 1 && mem->hasAttr(kIndependentWritesAttr))
    for (seq::WritePortOp write : writes)
      emitBlock(write);
  else
    emitBlock(writes);

  // A read enable is not modelled, so a FIFO's `rdEn` reaches here and is
  // dropped.
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
