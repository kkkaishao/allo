#include "allo/Transforms/ShardingInterfaceImpl.h"
#include "allo/IR/AlloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "mlir/Dialect/Shard/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::shard;
using namespace mlir::allo;

namespace {
template <typename OpTy>
struct ProgramIDShardingInterface
    : public ShardingInterface::ExternalModel<ProgramIDShardingInterface<OpTy>,
                                              OpTy> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {};
  }
  SmallVector<AffineMap> getIndexingMaps(Operation *) const { return {}; }
  FailureOr<ShardingOption>
  getShardingOption(Operation *, ArrayRef<Sharding>,
                    ArrayRef<Sharding> resultShardings) const {
    assert(resultShardings.size() == 1 &&
           "expected exactly one result sharding");
    auto &resultSharding = resultShardings[0];
    if (!resultSharding) {
      return failure();
    }
    return ShardingOption({}, resultSharding.getGridAttr());
  }
  LogicalResult partition(Operation *op, ArrayRef<Value>, ArrayRef<Sharding>,
                          ArrayRef<Sharding>, IRMapping &partitionMap,
                          SymbolTableCollection &, OpBuilder &builder) const {
    (void)builder.clone(*op, partitionMap);
    return success();
  }
};
} // namespace

namespace {
// inherit from ElementwiseShardingInterface
// rewrite the getIndexingMaps to handle scalar case (rank 0)
template <typename OpTy>
struct CustomElementwiseShardingInterface
    : public ShardingInterface::ExternalModel<
          CustomElementwiseShardingInterface<OpTy>, OpTy> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    SmallVector<utils::IteratorType> types(type.getRank(),
                                           utils::IteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    int64_t rank;
    if (type)
      rank = type.getRank();
    else if (val.getType().isIntOrIndexOrFloat())
      rank = 0; // scalar case
    else
      return {};
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    partitionTriviallyShardableOperation(*op, partitionedOperands,
                                         operandShardings, resultShardings,
                                         partitionMap, symbolTable, builder);
    return success();
  }
};
} // namespace

template <typename OpTy> static void registerElementwiseOne(MLIRContext &ctx) {
  OpTy::template attachInterface<CustomElementwiseShardingInterface<OpTy>>(ctx);
}

/// Variadic helper function.
template <typename... OpTys>
static void registerElementwiseAll(MLIRContext &ctx) {
  (registerElementwiseOne<OpTys>(ctx), ...);
}

// traces the value back to its defining operations
// returns true if all the defining operations are either constant/get_pid ops
// and the tracing does not encounter any ops other than arith dialect ops.
static bool traceToConstants(Value value) {
  llvm::SmallDenseSet<Value, 16> visited;
  SmallVector<Value, 8> worklist;
  worklist.push_back(value);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    if (isa<BlockArgument>(v)) {
      // iterated to a block argument, reject
      return false;
    }
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return false;

    if (isa<arith::ConstantOp, allo::GetProgramIdOp, GetNumProgramsOp>(defOp)) {
      // allowed boundaries, stop this path of tracing
      continue;
    }
    if (!isa<arith::ArithDialect>(defOp->getDialect())) {
      // only allow tracing through arith ops, reject otherwise
      return false;
    }
    llvm::append_range(worklist, defOp->getOperands());
  }
  return true;
}

namespace {
struct IfOpShardingInterface
    : public ShardingInterface::ExternalModel<IfOpShardingInterface,
                                              scf::IfOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {};
  }
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    // cond must be a single scalar, so create an empty map for it.
    return {AffineMap::getMultiDimIdentityMap(0, op->getContext())};
  }
  FailureOr<ShardingOption>
  getShardingOption(Operation *op, ArrayRef<Sharding>,
                    ArrayRef<Sharding> resultShardings) const {
    if (!resultShardings.empty())
      return failure(); // reject if with return values
    Value cond = llvm::cast<scf::IfOp>(op).getCondition();
    if (!traceToConstants(cond))
      return failure(); // reject if cond is not traced to constants
    return ShardingOption::makeEmpty();
  }
  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    // clone the if op and update the partitionMap recursively
    // will trigger an assertion error
    // we only want to map the original if op but not the inner cloned
    builder.clone(*op, partitionMap);
    return success();
  }
};
} // namespace

namespace {
template <typename OpTy>
struct ChanAccessShardingInterface
    : public ShardingInterface::ExternalModel<ChanAccessShardingInterface<OpTy>,
                                              OpTy> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    SmallVector<utils::IteratorType> iters;
    for (auto type : op->getOperandTypes())
      populateIteratorTypes(type, iters);
    for (auto type : op->getResultTypes())
      populateIteratorTypes(type, iters);
    return iters;
  }
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    int64_t rank;
    if (type)
      rank = type.getRank();
    else if (val.getType().isIntOrIndexOrFloat())
      rank = 0; // scalar case
    else
      return {};
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }
  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    builder.clone(*op, partitionMap);
    return success();
  }

private:
  void
  populateIteratorTypes(Type t,
                        SmallVector<utils::IteratorType> &iterTypes) const {
    RankedTensorType rankedTensorType = dyn_cast<RankedTensorType>(t);
    if (!rankedTensorType) {
      return;
    }

    iterTypes.reserve(iterTypes.size() + rankedTensorType.getRank());
    for (int64_t i = 0; i < rankedTensorType.getRank(); ++i) {
      iterTypes.push_back(utils::IteratorType::parallel);
    }
  }
};
} // namespace

namespace mlir::allo {
void registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, AlloDialect *dialect) {
    ReturnOp::attachInterface<
        shard::IndependentParallelIteratorDomainShardingInterface<ReturnOp>>(
        *ctx);
    GetProgramIdOp::attachInterface<ProgramIDShardingInterface<GetProgramIdOp>>(
        *ctx);
    GetNumProgramsOp::attachInterface<
        ProgramIDShardingInterface<GetNumProgramsOp>>(*ctx);
    registerElementwiseAll<arith::AddIOp, arith::AddFOp, arith::SubIOp,
                           arith::SubFOp, arith::MulIOp, arith::MulFOp,
                           arith::DivUIOp, arith::DivSIOp, arith::DivFOp,
                           arith::RemFOp, arith::RemSIOp, arith::RemUIOp,
                           arith::CmpFOp, arith::CmpIOp, arith::AndIOp,
                           arith::OrIOp, arith::XOrIOp>(*ctx);
    scf::IfOp::attachInterface<IfOpShardingInterface>(*ctx);
    ChanGetOp::attachInterface<ChanAccessShardingInterface<ChanGetOp>>(*ctx);
    ChanPutOp::attachInterface<ChanAccessShardingInterface<ChanPutOp>>(*ctx);
  });
}
} // namespace mlir::allo