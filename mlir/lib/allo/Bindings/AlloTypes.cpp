/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * nanobind subclasses of `allo._mlir.ir.Type` for the Allo dialect's custom
 * types. Construction/introspection is funnelled through the Allo CAPI so the
 * extension links no MLIR C++ statically.
 */

#include "AlloBindings.h"

#include "allo-c/AlloTypes.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "nanobind/stl/vector.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

void allo::populateAlloTypes(nb::module_ &m) {
  //===--------------------------------------------------------------------===//
  // StreamType: !allo.stream<baseType, depth, [shape...]>
  //===--------------------------------------------------------------------===//
  mlir_type_subclass(m, "StreamType", alloTypeIsAStream)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirType baseType, uint64_t depth,
             std::vector<int64_t> shape) {
            return cls(alloStreamTypeGet(
                mlirTypeGetContext(baseType), baseType, depth,
                static_cast<intptr_t>(shape.size()), shape.data()));
          },
          "Build an !allo.stream type carrying `base_type` with the given "
          "buffer `depth` and array `shape`.",
          nb::arg("cls"), nb::arg("base_type"), nb::arg("depth"),
          nb::arg("shape"))
      .def_property_readonly(
          "base_type",
          [](MlirType self) { return alloStreamTypeGetBaseType(self); })
      .def_property_readonly(
          "depth", [](MlirType self) { return alloStreamTypeGetDepth(self); })
      .def_property_readonly("shape", [](MlirType self) {
        std::vector<int64_t> shape;
        for (intptr_t i = 0, e = alloStreamTypeGetRank(self); i < e; ++i)
          shape.push_back(alloStreamTypeGetDimSize(self, i));
        return shape;
      });
}
