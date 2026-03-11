#include "ir.h"

using InitFunc = void (*)(nb::module_ &);

struct SubmoduleDesc {
  std::string_view name;
  InitFunc init;
  const char *doc;
};

static constexpr SubmoduleDesc kSubmodules[] = {
    {"utils", init_utils, "Utility functions and classes for MLIR"},
    {"arith", init_arith_ops, "arith dialect"},
    {"math", init_math_ops, "math dialect"},
    {"scf", init_scf_ops, "scf dialect"},
    {"cf", init_cf_ops, "cf dialect"},
    {"ub", init_ub_ops, "ub dialect"},
    {"func", init_func_ops, "func dialect"},
    {"affine", init_affine_ops, "affine dialect"},
    {"tensor", init_tensor_ops, "tensor dialect"},
    {"memref", init_memref_ops, "memref dialect"},
    {"linalg", init_linalg_ops, "linalg dialect"},
    {"transform", init_transform, "transform dialect"},
};

static std::once_flag g_ir_once;
static std::once_flag g_submodule_once[std::size(kSubmodules)];

static nb::module_ ensure_ir_loaded(nb::module_ &parent) {
  std::call_once(g_ir_once, [&] {
    auto ir = parent.def_submodule("ir", "core IR");
    init_ir(ir);
  });
  return nb::borrow<nb::module_>(parent.attr("ir"));
}

static nb::object load_submodule(nb::module_ &parent, std::string_view target) {
  ensure_ir_loaded(parent);

  for (size_t i = 0; i < std::size(kSubmodules); ++i) {
    const auto &d = kSubmodules[i];
    if (d.name != target)
      continue;

    std::call_once(g_submodule_once[i], [&] {
      auto sm = parent.def_submodule(d.name.data(), d.doc);
      d.init(sm);
    });

    return nb::borrow<nb::object>(parent.attr(d.name.data()));
  }

  throw nb::attribute_error("unknown submodule");
}

NB_MODULE(_liballo, m) {
  m.doc() = "Python bindings to the C++ Allo API";
  llvm::sys::PrintStackTraceOnErrorSignal("_liballo");

  ensure_ir_loaded(m);

  m.def("_load_submodule", [](std::string_view name) {
    auto parent = nb::module_::import_("allo.bindings._liballo");
    return load_submodule(parent, name);
  });
}
