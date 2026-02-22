#ifndef ALLO_ATTRS_H
#define ALLO_ATTRS_H

#include "mlir/IR/BuiltinAttributes.h"

#include "allo/IR/AlloEnums.h.inc"

#include <map>

namespace mlir::allo {
struct PartitionInfo {
  int64_t dim;
  PartitionKindEnum kind;
  int64_t factor;
};
using PartitionInfoVec = llvm::SmallVector<PartitionInfo>;
// use an ordered map
using PartitionInfoMap =
    std::map<int64_t, std::pair<PartitionKindEnum, int64_t>>;
} // namespace mlir::allo

#define GET_ATTRDEF_CLASSES
#include "allo/IR/AlloAttrs.h.inc"

#endif // ALLO_ATTRS_H