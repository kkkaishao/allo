#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <string>
#include <unordered_set>

using namespace mlir;

static constexpr llvm::StringLiteral kScheduleIdAttr = "allo.schedule.id";

enum ScheduleOpTrait : uint64_t {
  ScheduleOpTraitLoopLike = 1ULL << 0,
  ScheduleOpTraitAffineLoop = 1ULL << 1,
  ScheduleOpTraitScfLoop = 1ULL << 2,
  ScheduleOpTraitRegionBranch = 1ULL << 3,
  ScheduleOpTraitFunctionLike = 1ULL << 4,
  ScheduleOpTraitSymbol = 1ULL << 5,
  ScheduleOpTraitMemoryAllocate = 1ULL << 6,
  ScheduleOpTraitMemoryFree = 1ULL << 7,
  ScheduleOpTraitMemoryRead = 1ULL << 8,
  ScheduleOpTraitMemoryWrite = 1ULL << 9,
  ScheduleOpTraitAffineFor = 1ULL << 10,
};

static bool isScheduled(Operation *op) {
  return op != nullptr && !op->hasTrait<OpTrait::IsTerminator>();
}

static void collectScheduledOps(Operation *op,
                                SmallVectorImpl<Operation *> &out) {
  if (!isScheduled(op))
    return;
  out.push_back(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        collectScheduledOps(&nested, out);
      }
    }
  }
}

static std::string freshScheduleId(uint64_t &counter,
                                   std::unordered_set<std::string> &used) {
  while (true) {
    std::string candidate = "s" + std::to_string(counter++);
    if (used.insert(candidate).second)
      return candidate;
  }
}

static std::string requireScheduleId(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr);
  assert(attr && "operation must be annotated before snapshot collection");
  return attr.str();
}

static std::optional<std::string> nameFromLoc(Location loc) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return nameLoc.getName().str();
  return std::nullopt;
}

static std::optional<std::string> bestOperationName(Operation *op) {
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return attr.str();
  return nameFromLoc(op->getLoc());
}

static std::optional<std::string> bestValueName(Value value, Operation *owner) {
  if (auto name = nameFromLoc(value.getLoc()))
    return name;
  if (owner)
    return bestOperationName(owner);
  return std::nullopt;
}

static std::optional<FileLineColLoc> findFileLineCol(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return findFileLineCol(nameLoc.getChildLoc());
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
    if (auto calleeLoc = findFileLineCol(callLoc.getCallee()))
      return calleeLoc;
    return findFileLineCol(callLoc.getCaller());
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fusedLoc.getLocations()) {
      if (auto childLoc = findFileLineCol(child))
        return childLoc;
    }
  }
  return std::nullopt;
}

static nb::object locationDict(Location loc) {
  auto fileLoc = findFileLineCol(loc);
  if (!fileLoc)
    return nb::none();

  nb::dict out;
  out["file"] = fileLoc->getFilename().str();
  out["line"] = fileLoc->getLine();
  out["col"] = fileLoc->getColumn();
  return out;
}

static std::string typeString(Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return os.str();
}

static bool isBufferLike(Value value) {
  Type type = value.getType();
  return isa<BaseMemRefType, TensorType>(type);
}

static std::string childSegment(Operation *op, unsigned loopIdx,
                                unsigned branchIdx, unsigned opIdx) {
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return attr.str();
  if (isa<LoopLikeOpInterface>(op))
    return "L" + std::to_string(loopIdx);
  if (isa<RegionBranchOpInterface>(op))
    return "B" + std::to_string(branchIdx);
  return "O" + std::to_string(opIdx);
}

static uint64_t memoryEffectTraits(Operation *op) {
  uint64_t traits = 0;
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return traits;

  for (const MemoryEffects::EffectInstance &effect : *effects) {
    if (isa<MemoryEffects::Allocate>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryAllocate;
    if (isa<MemoryEffects::Free>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryFree;
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryRead;
    if (isa<MemoryEffects::Write>(effect.getEffect()))
      traits |= ScheduleOpTraitMemoryWrite;
  }
  return traits;
}

static uint64_t operationTraits(Operation *op) {
  uint64_t traits = 0;
  if (isa<LoopLikeOpInterface>(op))
    traits |= ScheduleOpTraitLoopLike;
  if (isa<affine::AffineForOp>(op))
    traits |= ScheduleOpTraitAffineFor;
  if (isa<affine::AffineForOp, affine::AffineParallelOp>(op))
    traits |= ScheduleOpTraitAffineLoop;
  if (isa<scf::ForOp, scf::ParallelOp>(op))
    traits |= ScheduleOpTraitScfLoop;
  if (isa<RegionBranchOpInterface>(op))
    traits |= ScheduleOpTraitRegionBranch;
  if (isa<FunctionOpInterface>(op))
    traits |= ScheduleOpTraitFunctionLike;
  if (isa<SymbolOpInterface>(op))
    traits |= ScheduleOpTraitSymbol;
  traits |= memoryEffectTraits(op);
  return traits;
}

static nb::dict valueRecord(Value value, Operation *owner,
                            llvm::StringRef ownerId, llvm::StringRef ownerPath,
                            unsigned number, llvm::StringRef source) {
  std::string id =
      (llvm::Twine(ownerId) + ":" + source + std::to_string(number)).str();
  std::string path =
      (llvm::Twine(ownerPath) + ":" + source + std::to_string(number)).str();

  nb::dict out;
  out["id"] = id;
  out["owner_id"] = ownerId.str();
  if (auto name = bestValueName(value, owner))
    out["name"] = *name;
  else
    out["name"] = nb::none();
  out["type"] = typeString(value.getType());
  out["number"] = number;
  out["source"] = source.str();
  out["path"] = path;
  out["loc"] = locationDict(value.getLoc());
  return out;
}

static void collectValues(Operation *op, llvm::StringRef opId,
                          llvm::StringRef opPath, nb::list &values) {
  for (OpResult result : op->getResults()) {
    if (!isBufferLike(result))
      continue;
    values.append(
        valueRecord(result, op, opId, opPath, result.getResultNumber(), "res"));
  }

  auto func = dyn_cast<FunctionOpInterface>(op);
  if (!func)
    return;

  for (BlockArgument arg : func.getArguments()) {
    if (!isBufferLike(arg))
      continue;
    values.append(
        valueRecord(arg, op, opId, opPath, arg.getArgNumber(), "arg"));
  }
}

static void collectSnapshotNode(Operation *op, llvm::StringRef parentId,
                                llvm::StringRef path, nb::list &ops,
                                nb::list &values) {
  std::string id = requireScheduleId(op);
  nb::list childIds;
  SmallVector<std::pair<Operation *, std::string>> children;
  unsigned loopIdx = 0;
  unsigned branchIdx = 0;
  unsigned opIdx = 0;

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block) {
        if (!isScheduled(&nested))
          continue;
        unsigned currentLoop = loopIdx;
        unsigned currentBranch = branchIdx;
        unsigned currentOp = opIdx;
        if (isa<LoopLikeOpInterface>(nested))
          ++loopIdx;
        else if (isa<RegionBranchOpInterface>(nested))
          ++branchIdx;
        else
          ++opIdx;

        std::string segment =
            childSegment(&nested, currentLoop, currentBranch, currentOp);
        std::string childPath =
            path.empty() ? segment : (llvm::Twine(path) + "/" + segment).str();
        childIds.append(requireScheduleId(&nested));
        children.push_back({&nested, childPath});
      }
    }
  }

  nb::dict record;
  record["id"] = id;
  record["kind"] = op->getName().getStringRef().str();
  if (auto name = bestOperationName(op))
    record["name"] = *name;
  else
    record["name"] = nb::none();
  record["path"] = path.str();
  if (parentId.empty())
    record["parent_id"] = nb::none();
  else
    record["parent_id"] = nb::str(parentId.data(), parentId.size());
  record["children"] = childIds;
  record["loc"] = locationDict(op->getLoc());
  record["traits"] = operationTraits(op);
  ops.append(record);
  collectValues(op, id, path, values);

  for (auto &[child, childPath] : children)
    collectSnapshotNode(child, id, childPath, ops, values);
}

void bindSchedule(nb::module_ &m) {
  m.attr("SCHEDULE_ID_ATTR_NAME") = nb::str(kScheduleIdAttr.data());
  nb::enum_<ScheduleOpTrait>(m, "ScheduleOpTrait", nb::is_arithmetic(),
                             nb::is_flag())
      .value("OP_TRAIT_LOOP_LIKE", ScheduleOpTraitLoopLike)
      .value("OP_TRAIT_AFFINE_LOOP", ScheduleOpTraitAffineLoop)
      .value("OP_TRAIT_SCF_LOOP", ScheduleOpTraitScfLoop)
      .value("OP_TRAIT_REGION_BRANCH", ScheduleOpTraitRegionBranch)
      .value("OP_TRAIT_FUNCTION_LIKE", ScheduleOpTraitFunctionLike)
      .value("OP_TRAIT_SYMBOL", ScheduleOpTraitSymbol)
      .value("OP_TRAIT_MEMORY_ALLOCATE", ScheduleOpTraitMemoryAllocate)
      .value("OP_TRAIT_MEMORY_FREE", ScheduleOpTraitMemoryFree)
      .value("OP_TRAIT_MEMORY_READ", ScheduleOpTraitMemoryRead)
      .value("OP_TRAIT_MEMORY_WRITE", ScheduleOpTraitMemoryWrite)
      .value("OP_TRAIT_AFFINE_FOR", ScheduleOpTraitAffineFor);

  m.def("annotate_schedule_ids", [](ModuleOp module) {
    SmallVector<Operation *> ops;
    collectScheduledOps(module.getOperation(), ops);

    llvm::DenseMap<Attribute, unsigned> counts;
    for (Operation *op : ops) {
      if (auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr))
        ++counts[attr];
    }

    std::unordered_set<std::string> used;
    for (Operation *op : ops) {
      auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr);
      if (attr && counts[attr] == 1)
        used.insert(attr.str());
    }

    uint64_t counter = 0;
    MLIRContext *ctx = module.getContext();
    for (Operation *op : ops) {
      auto attr = op->getAttrOfType<StringAttr>(kScheduleIdAttr);
      if (attr && counts[attr] == 1)
        continue;
      op->setAttr(kScheduleIdAttr,
                  StringAttr::get(ctx, freshScheduleId(counter, used)));
    }
  });

  m.def("cleanup_schedule_ids", [](ModuleOp module) {
    module->walk([](Operation *op) { op->removeAttr(kScheduleIdAttr); });
  });

  m.def("collect_schedule_snapshot", [](ModuleOp module) {
    auto root = module.getOperation();
    assert(root->getAttrOfType<StringAttr>(kScheduleIdAttr) &&
           "call annotate_schedule_ids before collect_schedule_snapshot");

    nb::list ops;
    nb::list values;
    collectSnapshotNode(root, "", "module", ops, values);

    nb::dict out;
    out["root_id"] = requireScheduleId(root);
    out["ops"] = ops;
    out["values"] = values;
    return out;
  });
}
