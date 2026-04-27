#ifndef ACT_SUPPORT_CODEEMISSION_H
#define ACT_SUPPORT_CODEEMISSION_H

#include "act/Support/Planning.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::act {

LogicalResult emitInstructionSequence(RewriterBase &rewriter,
                                      ExecutionPlan &plan);

} // namespace mlir::act

#endif // ACT_SUPPORT_CODEEMISSION_H
