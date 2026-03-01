#include "allo/Transforms/ShardingInterfaceImpl.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::allo {
void registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, AlloDialect *dialect) {
    ReturnOp::attachInterface<
        shard::IndependentParallelIteratorDomainShardingInterface<ReturnOp>>(
        *ctx);
  });
}
} // namespace mlir::allo