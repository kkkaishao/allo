/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/VerilogEmitter.h"
#include "allo/IR/AlloOps.h" // kMemoryInitAttr (power-on memory contents)
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/VerifToSV.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace {
// The two halves of giving an on-chip memory power-on contents, bracketing the
// lowering that loses them. `seq.hlmem` has no initializer, and the `sv.reg`
// that replaces it, the handle an `initial` block needs, does not exist until
// that lowering has run. So the values ride the memory itself up to the
// boundary, cross it on the module, and are written afterwards.

// Before: move each memory's recorded contents onto the enclosing module,
// keyed by the memory's name, since the memory is about to be erased.
struct HoistMemoryInitPass
    : public PassWrapper<HoistMemoryInitPass,
                         OperationPass<circt::hw::HWModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistMemoryInitPass)
  StringRef getArgument() const final { return "allo-hoist-memory-init"; }

  void runOnOperation() override {
    circt::hw::HWModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SmallVector<NamedAttribute> entries;
    module.walk([&](circt::seq::HLMemOp mem) {
      if (auto init = mem->getAttrOfType<ArrayAttr>(allo::kMemoryInitAttr)) {
        entries.emplace_back(StringAttr::get(ctx, mem.getName()), init);
        mem->removeAttr(allo::kMemoryInitAttr);
      }
    });
    if (!entries.empty())
      module->setAttr(allo::kMemoryInitAttr, DictionaryAttr::get(ctx, entries));
  }
};

// After: write those contents into the backing `sv.reg`, which the hlmem
// lowering names after the memory it replaced. That name is the one link
// between the two halves. Blocking assignments in an `initial` block are what a
// simulator starts the array from and what a synthesis tool reads back as a
// BRAM INIT.
struct InitializeMemoriesPass
    : public PassWrapper<InitializeMemoriesPass,
                         OperationPass<circt::hw::HWModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitializeMemoriesPass)
  StringRef getArgument() const final { return "allo-initialize-memories"; }

  void runOnOperation() override {
    circt::hw::HWModuleOp module = getOperation();
    auto inits = module->getAttrOfType<DictionaryAttr>(allo::kMemoryInitAttr);
    if (!inits)
      return;
    llvm::StringMap<circt::sv::RegOp> regs;
    module.walk([&](circt::sv::RegOp reg) { regs[reg.getName()] = reg; });
    for (NamedAttribute entry : inits) {
      auto it = regs.find(entry.getName().strref());
      assert(it != regs.end() &&
             "no storage backs an initialized memory; seq.hlmem lowering must "
             "name its reg after the memory");
      circt::sv::RegOp reg = it->second;
      auto words = cast<ArrayAttr>(entry.getValue());
      assert(words.size() ==
                 cast<circt::hw::UnpackedArrayType>(reg.getElementType())
                     .getNumElements() &&
             "an initializer must cover exactly the declared words");
      OpBuilder b(reg);
      b.setInsertionPointAfter(reg);
      Location loc = reg.getLoc();
      Type addrTy = b.getIntegerType(llvm::Log2_64_Ceil(words.size()));
      circt::sv::InitialOp::create(b, loc, [&] {
        for (auto [i, w] : llvm::enumerate(words)) {
          Value idx = circt::hw::ConstantOp::create(b, loc, addrTy, i);
          Value slot = circt::sv::ArrayIndexInOutOp::create(b, loc, reg, idx);
          Value val =
              circt::hw::ConstantOp::create(b, loc, cast<IntegerAttr>(w));
          circt::sv::BPAssignOp::create(b, loc, slot, val);
        }
      });
    }
    module->removeAttr(allo::kMemoryInitAttr);
  }
};
} // namespace

// The seq->SV lowering pipeline shared by the two emitters. The seq lowerings
// and lower-hw-to-sv are anchored on `hw::HWModuleOp`, so they must nest under
// the module-level pass manager; only lower-seq-to-sv runs on the ModuleOp.
static void addLowerToSV(PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  OpPassManager &hwPM = pm.nest<circt::hw::HWModuleOp>();
  // FIFO lowering emits a backing seq.hlmem, so it must precede HLMem lowering.
  hwPM.addPass(circt::seq::createLowerSeqFIFO());
  // A memory's power-on contents cross the hlmem lowering on the module: the
  // memory that carries them is erased by it, the reg that receives them is
  // created by it.
  hwPM.addPass(std::make_unique<HoistMemoryInitPass>());
  hwPM.addPass(circt::seq::createLowerSeqHLMem());
  hwPM.addPass(std::make_unique<InitializeMemoriesPass>());
  hwPM.addPass(circt::seq::createLowerSeqShiftReg());
  hwPM.addPass(circt::seq::createLowerSeqCompRegCE());
  // FIFO lowering also emits verif over/underflow assertions; lower them to SV
  // (sim-only assertions ExportVerilog understands).
  hwPM.addPass(circt::createLowerVerifToSVPass());
  pm.addPass(circt::createLowerSeqToSVPass());
  pm.nest<circt::hw::HWModuleOp>().addPass(circt::createLowerHWToSVPass());
}

LogicalResult allo::emitVerilog(ModuleOp mod, llvm::raw_ostream &os) {
  PassManager pm(mod.getContext());
  addLowerToSV(pm);
  if (failed(pm.run(mod)))
    return failure();
  return circt::exportVerilog(mod, os);
}

LogicalResult allo::emitSplitVerilog(ModuleOp mod, StringRef directory) {
  PassManager pm(mod.getContext());
  addLowerToSV(pm);
  if (failed(pm.run(mod)))
    return failure();
  return circt::exportSplitVerilog(mod, directory);
}
