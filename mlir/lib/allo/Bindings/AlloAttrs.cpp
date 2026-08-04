/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CRTP nanobind subclasses of `allo._mlir.ir.Attribute` for the Allo dialect's
 * custom attributes (see mlir/examples/standalone for the upstream pattern),
 * funnelled through the Allo CAPI.
 */

#include "AlloBindings.h"

#include "allo-c/AlloAttrs.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "nanobind/stl/vector.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
namespace mpx = mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace {

/// PartitionAxisAttr: #allo.part_axis(dim, kind, factor)
///   kind: 0 = Complete, 1 = Block, 2 = Cyclic.
struct PyPartitionAxisAttr : mpx::PyConcreteAttribute<PyPartitionAxisAttr> {
  static constexpr IsAFunctionTy isaFunction = alloAttributeIsAPartitionAxis;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloPartitionAxisAttrGetTypeID;
  static constexpr const char *pyClassName = "PartitionAxisAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](uint32_t kind, int64_t factor, int64_t dim,
           mpx::DefaultingPyMlirContext ctx) {
          return PyPartitionAxisAttr(
              ctx->getRef(),
              alloPartitionAxisAttrGet(ctx.get()->get(), kind, factor, dim));
        },
        nb::arg("kind"), nb::arg("factor"), nb::arg("dim"),
        nb::arg("context").none() = nb::none());
    c.def_prop_ro("kind", [](PyPartitionAxisAttr &self) {
      return alloPartitionAxisAttrGetKind(self);
    });
    c.def_prop_ro("factor", [](PyPartitionAxisAttr &self) {
      return alloPartitionAxisAttrGetFactor(self);
    });
    c.def_prop_ro("dim", [](PyPartitionAxisAttr &self) {
      return alloPartitionAxisAttrGetDim(self);
    });
  }
};

/// PartitionAttr: #allo.partition<[ axes... ]>
struct PyPartitionAttr : mpx::PyConcreteAttribute<PyPartitionAttr> {
  static constexpr IsAFunctionTy isaFunction = alloAttributeIsAPartition;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      alloPartitionAttrGetTypeID;
  static constexpr const char *pyClassName = "PartitionAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<MlirAttribute> axes, mpx::DefaultingPyMlirContext ctx) {
          return PyPartitionAttr(
              ctx->getRef(),
              alloPartitionAttrGet(ctx.get()->get(),
                                   static_cast<intptr_t>(axes.size()),
                                   axes.data()));
        },
        nb::arg("axes"), nb::arg("context").none() = nb::none());
    c.def_prop_ro("num_axes", [](PyPartitionAttr &self) {
      return alloPartitionAttrGetNumAxes(self);
    });
    c.def(
        "axis",
        [](PyPartitionAttr &self, intptr_t pos) {
          return alloPartitionAttrGetAxis(self, pos);
        },
        nb::arg("pos"));
  }
};

/// Enum-backed attributes all share the same `get(value, context)` /
/// `value` shape, so generate a CRTP subclass per attr from its CAPI hooks.
#define ALLO_ENUM_ATTR(PyClass, PyName, IsAFn, GetFn, GetValueFn, GetIdFn)     \
  struct PyClass : mpx::PyConcreteAttribute<PyClass> {                         \
    static constexpr IsAFunctionTy isaFunction = IsAFn;                        \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction = GetIdFn;          \
    static constexpr const char *pyClassName = PyName;                         \
    using PyConcreteAttribute::PyConcreteAttribute;                            \
    static void bindDerived(ClassTy &c) {                                      \
      c.def_static(                                                            \
          "get",                                                               \
          [](uint32_t value, mpx::DefaultingPyMlirContext ctx) {               \
            return PyClass(ctx->getRef(), GetFn(ctx.get()->get(), value));     \
          },                                                                   \
          nb::arg("value"), nb::arg("context").none() = nb::none());           \
      c.def_prop_ro("value", [](PyClass &self) { return GetValueFn(self); });  \
    }                                                                          \
  };

ALLO_ENUM_ATTR(PyAssumeDepTypeAttr, "AssumeDepTypeAttr",
               alloAttributeIsAAssumeDepType, alloAssumeDepTypeAttrGet,
               alloAssumeDepTypeAttrGetValue, alloAssumeDepTypeAttrGetTypeID)
ALLO_ENUM_ATTR(PyAssumeDepDirAttr, "AssumeDepDirAttr",
               alloAttributeIsAAssumeDepDir, alloAssumeDepDirAttrGet,
               alloAssumeDepDirAttrGetValue, alloAssumeDepDirAttrGetTypeID)
ALLO_ENUM_ATTR(PyMemoryPortAttr, "MemoryPortAttr", alloAttributeIsAMemoryPort,
               alloMemoryPortAttrGet, alloMemoryPortAttrGetValue,
               alloMemoryPortAttrGetTypeID)
ALLO_ENUM_ATTR(PyMemoryKindAttr, "MemoryKindAttr", alloAttributeIsAMemoryKind,
               alloMemoryKindAttrGet, alloMemoryKindAttrGetValue,
               alloMemoryKindAttrGetTypeID)
ALLO_ENUM_ATTR(PyDeterminacyAttr, "DeterminacyAttr",
               alloAttributeIsADeterminacy, alloDeterminacyAttrGet,
               alloDeterminacyAttrGetValue, alloDeterminacyAttrGetTypeID)
ALLO_ENUM_ATTR(PyCombOpKindAttr, "CombOpKindAttr", alloAttributeIsACombOpKind,
               alloCombOpKindAttrGet, alloCombOpKindAttrGetValue,
               alloCombOpKindAttrGetTypeID)
ALLO_ENUM_ATTR(PyStallContractAttr, "StallContractAttr",
               alloAttributeIsAStallContract, alloStallContractAttrGet,
               alloStallContractAttrGetValue, alloStallContractAttrGetTypeID)

#undef ALLO_ENUM_ATTR

} // namespace

void allo::populateAlloAttrs(nb::module_ &m) {
  PyPartitionAxisAttr::bind(m);
  PyPartitionAttr::bind(m);
  PyAssumeDepTypeAttr::bind(m);
  PyAssumeDepDirAttr::bind(m);
  PyMemoryPortAttr::bind(m);
  PyMemoryKindAttr::bind(m);
  PyDeterminacyAttr::bind(m);
  PyCombOpKindAttr::bind(m);
  PyStallContractAttr::bind(m);
}
