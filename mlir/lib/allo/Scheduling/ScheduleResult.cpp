/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Scheduling/ScheduleResult.h"
#include "allo/Scheduling/ScheduleAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace circt::scheduling;

namespace mlir::allo {

void annotateRegion(Problem &problem, func::FuncOp func, int64_t regionId,
                    StringRef kind, std::optional<unsigned> ii, int64_t order) {
  Builder b(func.getContext());

  int64_t maxStart = 0;
  for (Operation *op : problem.getOperations()) {
    std::optional<unsigned> start = problem.getStartTime(op);
    if (!start)
      continue;
    op->setAttr(sched::kStartTimeAttr, b.getI64IntegerAttr(*start));
    op->setAttr(sched::kRegionIdAttr, b.getI64IntegerAttr(regionId));
    maxStart = std::max<int64_t>(maxStart, static_cast<int64_t>(*start));
  }
  int64_t length = maxStart + 1;

  // Build the per-region descriptor.
  SmallVector<NamedAttribute> fields;
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyId),
                      b.getI64IntegerAttr(regionId));
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyKind),
                      b.getStringAttr(kind));
  if (ii)
    fields.emplace_back(b.getStringAttr(sched::kRegionKeyII),
                        b.getI64IntegerAttr(*ii));
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyLength),
                      b.getI64IntegerAttr(length));
  fields.emplace_back(b.getStringAttr(sched::kRegionKeyOrder),
                      b.getI64IntegerAttr(order));
  auto descriptor = b.getDictionaryAttr(fields);

  // Append to the func-level regions array.
  SmallVector<Attribute> regions;
  if (auto existing = func->getAttrOfType<ArrayAttr>(sched::kRegionsAttr))
    regions.append(existing.begin(), existing.end());
  regions.push_back(descriptor);
  func->setAttr(sched::kRegionsAttr, b.getArrayAttr(regions));
}

} // namespace mlir::allo
