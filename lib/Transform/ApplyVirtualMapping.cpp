#include "allo/Transform/Passes.h"

using namespace mlir;
using namespace mlir::allo;

namespace mlir::allo {
#define GEN_PASS_DEF_APPLYVIRTUALMAPPING
#include "allo/Transform/Passes.h.inc"
namespace impl {
struct ApplyVirtualMapping
    : public impl::ApplyVirtualMappingBase<ApplyVirtualMapping> {
  void runOnOperation() override { return; }
};
} // namespace impl
} // namespace mlir::allo