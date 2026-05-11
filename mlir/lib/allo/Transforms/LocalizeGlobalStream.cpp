#include "allo/Transforms/Passes.h"

namespace mlir::allo {
#define GEN_PASS_DEF_LOCALIZEGLOBALSTREAMPASS
#include "allo/Transforms/Passes.h.inc"
} // namespace mlir::allo

using namespace mlir;
using namespace mlir::allo;

namespace {
struct LocalizeGlobalStreamPass
    : public allo::impl::LocalizeGlobalStreamPassBase<
          LocalizeGlobalStreamPass> {
  void runOnOperation() override {}
};
} // namespace
