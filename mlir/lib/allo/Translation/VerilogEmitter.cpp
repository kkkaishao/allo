/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Translation/VerilogEmitter.h"
#include "allo/Conversion/Passes.h" // the hlmem lowering this tree owns
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/VerifToSV.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

// The seq->SV lowering pipeline shared by the two emitters. The seq lowerings
// and lower-hw-to-sv are anchored on `hw::HWModuleOp`, so they must nest under
// the module-level pass manager; only lower-seq-to-sv runs on the ModuleOp.
static void addLowerToSV(PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  OpPassManager &hwPM = pm.nest<circt::hw::HWModuleOp>();
  // FIFO lowering emits a backing seq.hlmem, so it must precede HLMem lowering.
  hwPM.addPass(circt::seq::createLowerSeqFIFO());
  hwPM.addPass(allo::createLowerHLMemPass());
  hwPM.addPass(circt::seq::createLowerSeqShiftReg());
  hwPM.addPass(circt::seq::createLowerSeqCompRegCE());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
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
