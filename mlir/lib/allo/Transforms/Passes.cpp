#include "allo/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::allo;

void allo::registerAlloTransforms() { registerLocalizeGlobalStreamPass(); }
