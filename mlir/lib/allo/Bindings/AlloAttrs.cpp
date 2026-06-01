/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * nanobind subclasses of `allo._mlir.ir.Attribute` for the Allo dialect's
 * custom attributes, funnelled through the Allo CAPI.
 */

#include "AlloBindings.h"

#include "allo-c/AlloAttrs.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "nanobind/stl/vector.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace mlir::python::nanobind_adaptors;

void allo::populateAlloAttrs(nb::module_ &m) {
  //===--------------------------------------------------------------------===//
  // PartitionAxisAttr: #allo.part_axis(dim, kind, factor)
  //   kind: 0 = Complete, 1 = Block, 2 = Cyclic.
  //===--------------------------------------------------------------------===//
  mlir_attribute_subclass(m, "PartitionAxisAttr", alloAttributeIsAPartitionAxis)
      .def_classmethod(
          "get",
          [](nb::object cls, uint32_t kind, int64_t factor, int64_t dim,
             MlirContext ctx) {
            return cls(alloPartitionAxisAttrGet(ctx, kind, factor, dim));
          },
          nb::arg("cls"), nb::arg("kind"), nb::arg("factor"), nb::arg("dim"),
          nb::arg("context"))
      .def_property_readonly(
          "kind",
          [](MlirAttribute self) { return alloPartitionAxisAttrGetKind(self); })
      .def_property_readonly("factor",
                             [](MlirAttribute self) {
                               return alloPartitionAxisAttrGetFactor(self);
                             })
      .def_property_readonly("dim", [](MlirAttribute self) {
        return alloPartitionAxisAttrGetDim(self);
      });

  //===--------------------------------------------------------------------===//
  // PartitionAttr: #allo.partition<[ axes... ]>
  //===--------------------------------------------------------------------===//
  mlir_attribute_subclass(m, "PartitionAttr", alloAttributeIsAPartition)
      .def_classmethod(
          "get",
          [](nb::object cls, std::vector<MlirAttribute> axes, MlirContext ctx) {
            return cls(alloPartitionAttrGet(
                ctx, static_cast<intptr_t>(axes.size()), axes.data()));
          },
          nb::arg("cls"), nb::arg("axes"), nb::arg("context"))
      .def_property_readonly(
          "num_axes",
          [](MlirAttribute self) { return alloPartitionAttrGetNumAxes(self); })
      .def(
          "axis",
          [](MlirAttribute self, intptr_t pos) {
            return alloPartitionAttrGetAxis(self, pos);
          },
          nb::arg("pos"));
}
