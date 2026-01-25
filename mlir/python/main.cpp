#include "nanobind/nanobind.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"

namespace nb = nanobind;

void init_ir(nb::module_ &m);
void init_math_ops(nb::module_ &m);
void init_arith_ops(nb::module_ &m);
void init_scf_ops(nb::module_ &m);
void init_cf_ops(nb::module_ &m);
void init_ub_ops(nb::module_ &m);
void init_func_ops(nb::module_ &m);
void init_affine_ops(nb::module_ &m);
void init_tensor_ops(nb::module_ &m);
void init_memref_ops(nb::module_ &m);
void init_linalg_ops(nb::module_ &m);

NB_MODULE(_liballo, m) {
  m.doc() = "Python bindings to the C++ Allo API";
  llvm::sys::PrintStackTraceOnErrorSignal("_liballo");
  auto ir = m.def_submodule("ir");
  init_ir(ir);
  auto arith = m.def_submodule("arith");
  init_arith_ops(arith);
  auto math = m.def_submodule("math");
  init_math_ops(math);
  auto scf = m.def_submodule("scf");
  init_scf_ops(scf);
  auto cf = m.def_submodule("cf");
  init_cf_ops(cf);
  auto ub = m.def_submodule("ub");
  init_ub_ops(ub);
  auto func = m.def_submodule("func");
  init_func_ops(func);
  auto affine = m.def_submodule("affine");
  init_affine_ops(affine);
  auto tensor = m.def_submodule("tensor");
  init_tensor_ops(tensor);
  auto memref = m.def_submodule("memref");
  init_memref_ops(memref);
  auto linalg = m.def_submodule("linalg");
  init_linalg_ops(linalg);
}