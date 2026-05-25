#include "ir.h"

#include "llvm/Support/Signals.h"

NB_MODULE(_liballo, m) {
  m.doc() = "Python bindings to the C++ Allo API";
  llvm::sys::PrintStackTraceOnErrorSignal("_liballo");

  auto ir = m.def_submodule("ir", "core IR");
  bindIR(ir);

  auto arith = m.def_submodule("arith", "arith dialect");
  bindArithOps(arith);
  auto math = m.def_submodule("math", "math dialect");
  bindMathOps(math);
  auto scf = m.def_submodule("scf", "scf dialect");
  bindSCFOps(scf);
  auto cf = m.def_submodule("cf", "cf dialect");
  bindCFOps(cf);
  auto ub = m.def_submodule("ub", "ub dialect");
  bindUBOps(ub);
  auto func = m.def_submodule("func", "func dialect");
  bindFuncOps(func);
  auto affine = m.def_submodule("affine", "affine dialect");
  bindAffineOps(affine);
  auto tensor = m.def_submodule("tensor", "tensor dialect");
  bindTensorOps(tensor);
  auto memref = m.def_submodule("memref", "memref dialect");
  bindMemRefOps(memref);
  auto linalg = m.def_submodule("linalg", "linalg dialect");
  bindLinalgOps(linalg);
  auto transform = m.def_submodule("transform", "transform dialect");
  bindTransform(transform);
  auto schedule = m.def_submodule("schedule", "schedule analysis");
  bindSchedule(schedule);
  auto allo = m.def_submodule("allo", "allo dialect");
  bindAlloOps(allo);
  auto passes = m.def_submodule("passes", "compiler passes");
  bindPasses(passes);
  auto executionEngine =
      m.def_submodule("execution_engine", "MLIR execution engine");
  bindExecutionEngine(executionEngine);
}
