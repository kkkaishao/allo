#ifndef ALLO_SHARDING_EXTENSION_H
#define ALLO_SHARDING_EXTENSION_H

namespace mlir {
class DialectRegistry;
namespace allo {
void registerShardingInterfaceExternalModels(DialectRegistry &registry);
} // namespace allo
} // namespace mlir

#endif // ALLO_SHARDING_EXTENSION_H
