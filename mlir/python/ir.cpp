#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "allo/Dialect/AlloDialect.h"
#include "mlir/InitAllDialects.h"

#include "llvm/Support/Signals.h"

#include <optional>

using namespace mlir;
namespace nb = nanobind;

class AlloOpBuilder : public OpBuilder {
public:
  using OpBuilder::OpBuilder;
  [[nodiscard]] Location get_loc() const { return loc; }
  void set_loc(Location newLoc) { loc = newLoc; }
  void set_unknown_loc() { loc = getUnknownLoc(); }
  [[nodiscard]] std::pair<OpBuilder::InsertPoint, Location>
  get_insertion_point_and_loc() const {
    return {saveInsertionPoint(), loc};
  }
  void set_insertion_point_and_loc(const OpBuilder::InsertPoint &ip,
                                   Location newLoc) {
    restoreInsertionPoint(ip);
    loc = newLoc;
  }

private:
  // default init to unknown
  Location loc = getUnknownLoc();
};

OpPrintingFlags getOpPrintingFlags(bool enable_debug = false) {
  auto printingFlags = OpPrintingFlags();
  printingFlags.enableDebugInfo(enable_debug);
  printingFlags.printNameLocAsPrefix(true);
  printingFlags.printGenericOpForm(false);
  return printingFlags;
}

static void init_context(nb::module_ &m) {
  nb::class_<MLIRContext>(m, "Context")
      .def("__init__",
           [](MLIRContext &self) {
             new (&self) MLIRContext(MLIRContext::Threading::DISABLED);
           })
      .def("printOpOnDiagnostic", &MLIRContext::printOpOnDiagnostic,
           nb::arg("enable"))
      .def("printStackTraceOnDiagnostic",
           &MLIRContext::printStackTraceOnDiagnostic, nb::arg("enable"))
      .def("get_loaded_dialects", [](MLIRContext &self) {
        std::vector<std::string> dialects;
        for (auto *dialect : self.getLoadedDialects()) {
          dialects.push_back(dialect->getNamespace().str());
        }
        return dialects;
      });

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    mlir::registerAllDialects(registry);
    // registry.insert<allo::AlloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}

static void init_builder(nb::module_ &m) {
  nb::class_<OpBuilder::InsertPoint>(m, "InsertPoint")
      .def(nb::init<>())
      .def(
          "get_block",
          [](OpBuilder::InsertPoint &self) { return self.getBlock(); },
          nb::rv_policy::reference);

  nb::class_<OpBuilder>(m, "OpBuilder")
      .def(nb::init<MLIRContext *>())
      .def(nb::init<Operation *>())
      .def(nb::init<Region *>())
      .def("get_context", &OpBuilder::getContext)
      .def(
          "set_insertion_point",
          [](OpBuilder &self, Operation *op) { self.setInsertionPoint(op); },
          nb::arg("op"))
      .def(
          "set_insertion_point_after",
          [](OpBuilder &self, Operation *op) {
            self.setInsertionPointAfter(op);
          },
          nb::arg("op"))
      .def(
          "set_insertion_point_to_start",
          [](OpBuilder &self, Block *block) {
            self.setInsertionPointToStart(block);
          },
          nb::arg("block"))
      .def(
          "set_insertion_point_to_end",
          [](OpBuilder &self, Block *block) {
            self.setInsertionPointToEnd(block);
          },
          nb::arg("block"))
      .def("save_insertion_point",
           [](OpBuilder &self) { return self.saveInsertionPoint(); })
      .def(
          "restore_insertion_point",
          [](OpBuilder &self, OpBuilder::InsertPoint &ip) {
            self.restoreInsertionPoint(ip);
          },
          nb::arg("ip"))
      .def(
          "get_affine_dim",
          [](OpBuilder &self, unsigned dim) {
            return self.getAffineDimExpr(dim);
          },
          nb::arg("dim"))
      .def(
          "get_affine_symbol",
          [](OpBuilder &self, unsigned sym) {
            return self.getAffineSymbolExpr(sym);
          },
          nb::arg("sym"))
      .def(
          "get_affine_constant",
          [](OpBuilder &self, int64_t value) {
            return self.getAffineConstantExpr(value);
          },
          nb::arg("value"))
      .def("get_unknown_loc", &OpBuilder::getUnknownLoc)
      .def(
          "create_block",
          [](OpBuilder &self) {
            return self.createBlock(self.getBlock()->getParent());
          },
          nb::rv_policy::reference)
      .def("create_block_in_region",
           [](OpBuilder &self, Region &region,
              const std::vector<Type> &argTypes, Location &loc) {
             llvm::SmallVector<Location, 4> locs(argTypes.size(), loc);
             return self.createBlock(&region, {}, argTypes, locs);
           });

  nb::class_<AlloOpBuilder, OpBuilder>(m, "AlloOpBuilder", nb::dynamic_attr())
      .def(nb::init<MLIRContext *>())
      .def(nb::init<Operation *>())
      .def("get_loc", &AlloOpBuilder::get_loc)
      .def("set_loc", &AlloOpBuilder::set_loc, nb::arg("new_loc"))
      .def("set_unknown_loc", &AlloOpBuilder::set_unknown_loc)
      .def("get_insertion_point_and_loc",
           &AlloOpBuilder::get_insertion_point_and_loc)
      .def("set_insertion_point_and_loc",
           &AlloOpBuilder::set_insertion_point_and_loc, nb::arg("ip"),
           nb::arg("new_loc"));
}

static void init_core_ir(nb::module_ &m) {
  nb::class_<Location>(m, "Location")
      // UnknownLoc init
      .def(
          "__init__",
          [](Location &self, MLIRContext &context) {
            self = Location(UnknownLoc::get(&context));
          },
          nb::arg("context"))
      // FileLineColLoc init
      .def(
          "__init__",
          [](Location &self, const std::string &filename, unsigned line,
             unsigned col, MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, filename);
            self = dyn_cast<Location>(FileLineColLoc::get(attr, line, col));
          },
          nb::arg("filename"), nb::arg("line"), nb::arg("col"),
          nb::arg("context"))
      // NamedLoc init
      .def(
          "__init__",
          [](Location &self, Location &childLoc, const std::string &name,
             MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, name);
            self = dyn_cast<LocationAttr>(NameLoc::get(attr, childLoc));
          },
          nb::arg("child_loc"), nb::arg("name"), nb::arg("context"))
      .def("get_context", &Location::getContext)
      .def("__str__",
           [](Location &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def(
          "set_name",
          [](Location &self, std::string &name) {
            StringAttr attr = StringAttr::get(self.getContext(), name);
            self = dyn_cast<Location>(NameLoc::get(attr, self));
          },
          nb::arg("name"))
      .def("get_col",
           [](Location &self) {
             if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
               return fileLineColLoc.getColumn();
             }
             throw nb::value_error("Location is not a FileLineColLoc");
           })
      .def("get_line",
           [](Location &self) {
             if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
               return fileLineColLoc.getLine();
             }
             throw nb::value_error("Location is not a FileLineColLoc");
           })
      .def("get_filename",
           [](Location &self) {
             if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(self)) {
               return fileLineColLoc.getFilename().str();
             }
             throw nb::value_error("Location is not a FileLineColLoc");
           })
      .def("get_name", [](Location &self) {
        if (auto nameLoc = dyn_cast<NameLoc>(self)) {
          return nameLoc.getName().str();
        }
        throw nb::value_error("Location is not a NameLoc");
      });

  nb::class_<Type>(m, "Type")
      .def("__init__",
           [](Type &self) {
             throw nb::type_error(
                 "Type cannot be directly instantiated, to get a Type, use a "
                 "specific Type's get() method");
           })
      .def("__eq__",
           [](Type &self, nb::object &other) {
             Type *otherTy = nb::cast<Type *>(other);
             return (otherTy != nullptr) && self == *otherTy;
           })
      .def("__ne__",
           [](Type &self, nb::object &other) {
             Type *otherTy = nb::cast<Type *>(other);
             return (otherTy == nullptr) || self != *otherTy;
           })
      .def("__str__",
           [](Type &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("get_as_tensor",
           [](Type &self) { return dyn_cast<RankedTensorType>(self); })
      .def("get_as_memref",
           [](Type &self) { return dyn_cast<MemRefType>(self); })
      .def("get_base_type", [](Type &self) {
        if (auto rankedTensorType = dyn_cast<RankedTensorType>(self)) {
          return rankedTensorType.getElementType();
        }
        if (auto memRefType = dyn_cast<MemRefType>(self)) {
          return memRefType.getElementType();
        }
        return self;
      });

  nb::class_<Value>(m, "Value")
      .def("__init__",
           [](Value &self) {
             throw nb::type_error(
                 "Value cannot be directly instantiated, to get a Value, use a "
                 "specific Value's defining operation or block argument");
           })
      .def("__str__",
           [](Value &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def(
          "set_attr",
          [](Value &self, const std::string &name, Attribute &attr) {
            if (auto defOp = self.getDefiningOp()) {
              defOp->setAttr(name, attr);
            } else {
              auto arg = cast<BlockArgument>(self);
              unsigned id = arg.getArgNumber();
              std::string argName = name + "_arg" + std::to_string(id);
              Block *owner = arg.getOwner();
              if (owner->isEntryBlock() &&
                  !isa<func::FuncOp>(owner->getParentOp())) {
                owner->getParentOp()->setAttr(name, attr);
              }
            }
          },
          nb::arg("name"), nb::arg("attr"))
      .def("get_context", &Value::getContext)
      .def("get_loc", &Value::getLoc)
      .def(
          "set_loc", [](Value &self, Location loc) { self.setLoc(loc); },
          nb::arg("loc"))
      .def("get_type", &Value::getType)
      .def(
          "set_type", [](Value &self, Type ty) { self.setType(ty); },
          nb::arg("type"))
      .def(
          "replace_all_uses_with",
          [](Value &self, Value &val) { self.replaceAllUsesWith(val); },
          nb::arg("val"))
      .def("id",
           [](Value &self) {
             return reinterpret_cast<uint64_t>(self.getImpl());
           })
      .def("erase", [](Value &self) {
        if (auto defOp = self.getDefiningOp()) {
          defOp->erase();
        } else {
          auto arg = cast<BlockArgument>(self);
          Block *owner = arg.getOwner();
          owner->eraseArgument(arg.getArgNumber());
        }
      });

  nb::class_<Attribute>(m, "Attribute")
      .def("__init__",
           [](Attribute &self) {
             throw nb::type_error("Attribute cannot be directly instantiated, "
                                  "to get an Attribute, "
                                  "use a specific Attribute's get() method");
           })
      .def("__str__",
           [](Attribute &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("get_context", &Attribute::getContext);

  nb::class_<Region>(m, "Region")
      .def("get_context", &Region::getContext)
      .def("get_parent_region", &Region::getParentRegion,
           nb::rv_policy::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def("empty", &Region::empty)
      .def(
          "front", [](Region &self) { return &self.front(); },
          nb::rv_policy::reference)
      .def(
          "back", [](Region &self) { return &self.back(); },
          nb::rv_policy::reference)
      .def(
          "push_back",
          [](Region &self, Block *block) { self.push_back(block); },
          nb::arg("block"))
      .def(
          "push_front",
          [](Region &self, Block *block) { self.push_front(block); },
          nb::arg("block"))
      .def(
          "emplace_block", [](Region &self) { return &self.emplaceBlock(); },
          nb::rv_policy::reference)
      .def("id",
           [](Region &self) { return reinterpret_cast<uint64_t>(&self); });

  nb::class_<Block>(m, "Block")
      .def("__init__",
           [](Block &self) {
             throw nb::type_error(
                 "Block cannot be directly instantiated, to get a Block, use "
                 "Region's emplace_block() or other methods");
           })
      .def("get_arg_at",
           [](Block &self, unsigned idx) {
             if (idx >= self.getNumArguments()) {
               throw nb::index_error("block argument index out of range");
             }
             return self.getArgument(idx);
           })
      .def(
          "add_arg",
          [](Block &self, Type type) {
            Location loc = UnknownLoc::get(type.getContext());
            return self.addArgument(type, loc);
          },
          nb::arg("type"))
      .def(
          "add_arg_at_loc",
          [](Block &self, Type type, Location loc) {
            return self.addArgument(type, loc);
          },
          nb::arg("type"), nb::arg("loc"))
      .def("num_args", &Block::getNumArguments)
      .def("move_before",
           [](Block &self, Block &dst) { self.moveBefore(&dst); })
      .def("insert_before", &Block::insertBefore, nb::arg("block"))
      .def("get_parent_region", &Block::getParent, nb::rv_policy::reference)
      .def(
          "merge_before",
          [](Block &self, Block &dst) {
            // ref: RewriterBase::mergeBlocks()
            if (self.getNumArguments() != 0)
              throw std::runtime_error("This block has arguments, don't merge");
            dst.getOperations().splice(dst.begin(), self.getOperations());
            self.dropAllUses();
            self.erase();
          },
          nb::arg("dst"))
      .def(
          "replace_use_in_block_with",
          [](Block &self, Value &oldVal, Value &newVal) {
            oldVal.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              Block *currentBlock = user->getBlock();
              while (currentBlock) {
                if (currentBlock == &self)
                  return true;
                // Move up one level
                currentBlock =
                    currentBlock->getParent()->getParentOp()->getBlock();
              }
              return false;
            });
          },
          nb::arg("old_val"), nb::arg("new_val"))
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("has_terminator",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::IsTerminator>();
           })
      .def("remove_terminator",
           [](Block &self) {
             if (self.empty() ||
                 !self.back().hasTrait<OpTrait::IsTerminator>()) {
               return;
             }
             self.back().erase();
           })
      .def("has_return",
           [](Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<OpTrait::ReturnLike>();
           })
      .def("erase", [](Block &self) { self.erase(); })
      .def("id", [](Block &self) { return reinterpret_cast<int64_t>(&self); });

  // Base Operation class
  nb::class_<Operation>(m, "Operation")
      .def("__init__",
           [](Operation &self) {
             throw nb::type_error(
                 "Operation cannot be directly instantiated, to get an "
                 "Operation, use "
                 "the OpBuilder or specific Op's create() method");
           })
      .def("get_context", &Operation::getContext)
      .def("get_loc", &Operation::getLoc)
      .def("get_name",
           [](Operation &self) { return self.getName().getStringRef().str(); });

  nb::class_<OpState>(m, "OpState")
      .def("__init__",
           [](OpState &self) {
             throw nb::type_error(
                 "OpState cannot be directly instantiated, to get an OpState, "
                 "use "
                 "the OpBuilder or specific Op's create() method");
           })
      .def("get_context", &OpState::getContext)
      .def("get_loc", &OpState::getLoc)
      .def("set_attr", [](OpState &self, std::string &name,
                          Attribute &attr) { self->setAttr(name, attr); })
      .def("get_num_operands",
           [](OpState &self) { return self->getNumOperands(); })
      .def(
          "get_operand_at",
          [](OpState &self, unsigned idx) {
            if (idx >= self->getNumOperands()) {
              throw nb::index_error("Op operand index out of range");
            }
            return self->getOperand(idx);
          },
          nb::arg("idx"))
      .def("get_num_results",
           [](OpState &self) { return self->getNumResults(); })
      .def("get_result_at",
           [](OpState &self, unsigned idx) {
             if (idx >= self->getNumResults())
               throw nb::index_error("Op result index out of range");
             return self->getResult(idx);
           })
      .def("get_num_regions",
           [](OpState &self) { return self->getNumRegions(); })
      .def(
          "get_region_at",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw nb::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          nb::rv_policy::reference)
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags();
             self->print(os, printingFlags);
             return os.str();
           })
      .def("append_operand",
           [](OpState &self, Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify",
           [](OpState &self) -> bool {
             return succeeded(verify(self.getOperation()));
           })
      .def(
          "get_operation", [](OpState &self) { return self.getOperation(); },
          nb::rv_policy::reference)
      .def(
          "get_block",
          [](OpState &self) { return self.getOperation()->getBlock(); },
          nb::rv_policy::reference);

  nb::class_<ModuleOp, OpState>(m, "ModuleOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc) {
            return ModuleOp::create(loc);
          },
          nb::arg("builder"), nb::arg("loc"))
      .def(
          "get_body", [](ModuleOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def("push_back",
           [](ModuleOp &self, Operation *op) { self.getBody()->push_back(op); })
      .def(
          "lookup_function",
          [](ModuleOp &self, const std::string &name) {
            return self.lookupSymbol<func::FuncOp>(name);
          },
          nb::arg("name"));
}

static void init_types(nb::module_ &m) {
  nb::class_<FunctionType, Type>(m, "FunctionType")
      .def_static(
          "get",
          [](const std::vector<Type> &argTypes,
             const std::vector<Type> &retTypes, MLIRContext &context) {
            return FunctionType::get(&context, argTypes, retTypes);
          },
          nb::arg("arg_types"), nb::arg("ret_types"), nb::arg("context"))
      .def("get_arg_types",
           [](FunctionType &self) {
             std::vector<Type> argTypes;
             for (Type ty : self.getInputs()) {
               argTypes.push_back(ty);
             }
             return argTypes;
           })
      .def("get_res_types",
           [](FunctionType &self) {
             std::vector<Type> retTypes;
             for (Type ty : self.getResults()) {
               retTypes.push_back(ty);
             }
             return retTypes;
           })
      .def("get_num_args", &FunctionType::getNumInputs)
      .def("get_num_results", &FunctionType::getNumResults);

  nb::class_<NoneType, Type>(m, "NoneType")
      .def_static(
          "get", [](MLIRContext &context) { return NoneType::get(&context); },
          nb::arg("context"));

  nb::class_<IntegerType, Type>(m, "IntegerType")
      .def_static(
          "get",
          [](unsigned width, MLIRContext &context) {
            return IntegerType::get(&context, width);
          },
          nb::arg("width"), nb::arg("context"))
      .def_static("get_width", [](Type &ty) {
        if (auto intType = dyn_cast<IntegerType>(ty)) {
          return intType.getWidth();
        }
        throw nb::type_error("Type is not an IntegerType");
      });

  nb::class_<IndexType, Type>(m, "IndexType")
      .def_static(
          "get", [](MLIRContext &context) { return IndexType::get(&context); },
          nb::arg("context"))
      .def_static(
          "isa", [](Type &ty) { return isa<IndexType>(ty); }, nb::arg("type"));

  // Float Types
  (void)nb::class_<FloatType, Type>(m, "FloatType");

  nb::class_<Float16Type, Type>(m, "F16Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return Float16Type::get(&context); },
          nb::arg("context"));

  nb::class_<Float32Type, Type>(m, "F32Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return Float32Type::get(&context); },
          nb::arg("context"));

  nb::class_<Float64Type, Type>(m, "F64Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return Float64Type::get(&context); },
          nb::arg("context"));

  nb::class_<BFloat16Type, Type>(m, "BF16Type")
      .def_static(
          "get",
          [](MLIRContext &context) { return BFloat16Type::get(&context); },
          nb::arg("context"));

  // RankedTensorType
  nb::class_<RankedTensorType, Type>(m, "RankedTensorType")
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType,
             std::optional<Attribute> encoding) {
            return RankedTensorType::get(shape, elementType,
                                         encoding.value_or(Attribute()));
          },
          nb::arg("shape"), nb::arg("element_type"),
          nb::arg("encoding").none() = nb::none())
      .def("get_encoding", &RankedTensorType::getEncoding)
      .def("get_element_type",
           [](RankedTensorType &self) { return self.getElementType(); })
      .def("get_shape",
           [](RankedTensorType &self) {
             auto shape = self.getShape();
             std::vector<int64_t> ret;
             for (auto dim : shape) {
               ret.push_back(dim);
             }
             return ret;
           })
      .def("get_rank", &RankedTensorType::getRank)
      .def(
          "get_dim_size_at",
          [](RankedTensorType &self, unsigned index) {
            if (index >= self.getRank()) {
              throw nb::index_error("tensor type dimension index out of range");
            }
            return self.getDimSize(index);
          },
          nb::arg("index"))
      .def(
          "set_element_type",
          [](RankedTensorType &self, Type newElementType) {
            return RankedTensorType::get(self.getShape(), newElementType,
                                         self.getEncoding());
          },
          nb::arg("type"));

  nb::class_<MemRefType, Type>(m, "MemRefType")
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType, AffineMap map,
             std::optional<Attribute> memorySpace) {
            return MemRefType::get(shape, elementType, map,
                                   memorySpace.value_or(IntegerAttr()));
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("affine_maps"),
          nb::arg("memory_space").none() = nb::none())
      .def("get_element_type",
           [](MemRefType &self) { return self.getElementType(); })
      .def("get_shape",
           [](MemRefType &self) {
             auto shape = self.getShape();
             std::vector<int64_t> ret;
             for (auto dim : shape) {
               ret.push_back(dim);
             }
             return ret;
           })
      .def("get_rank", &MemRefType::getRank);
}

static void init_values(nb::module_ &m) {
  nb::class_<BlockArgument, Value>(m, "BlockArgument")
      .def("get_arg_number", &BlockArgument::getArgNumber)
      .def("get_owner", &BlockArgument::getOwner, nb::rv_policy::reference);

  nb::class_<OpResult, Value>(m, "OpResult")
      .def("get_owner", &OpResult::getOwner, nb::rv_policy::reference)
      .def("get_res_no", &OpResult::getResultNumber);
}

static void init_attributes(nb::module_ &m) {
  nb::class_<IntegerAttr, Attribute>(m, "IntegerAttr")
      .def_static(
          "get",
          [](Type ty, int64_t value) { return IntegerAttr::get(ty, value); },
          nb::arg("ty"), nb::arg("value"))
      .def("get_signless", &IntegerAttr::getInt)
      .def("get_signed", &IntegerAttr::getSInt)
      .def("get_unsigned", &IntegerAttr::getUInt);

  nb::class_<UnitAttr, Attribute>(m, "UnitAttr")
      .def_static(
          "get", [](MLIRContext &context) { return UnitAttr::get(&context); },
          nb::arg("context"));

  nb::class_<StringAttr, Attribute>(m, "StringAttr")
      .def_static(
          "get",
          [](const std::string &value, MLIRContext &context) {
            return StringAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"))
      .def("get_str", [](StringAttr &self) { return self.getValue().str(); });

  nb::class_<BoolAttr, Attribute>(m, "BoolAttr")
      .def_static(
          "get",
          [](bool value, MLIRContext &context) {
            return BoolAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"));

  nb::class_<DenseI32ArrayAttr, Attribute>(m, "DenseI32ArrayAttr")
      .def_static(
          "get",
          [](MLIRContext &context, const std::vector<int32_t> &values) {
            return DenseI32ArrayAttr::get(&context, values);
          },
          nb::arg("context"), nb::arg("values"))
      .def("size", [](DenseI32ArrayAttr &self) { return self.getSize(); });
}

static void init_affine_objects(nb::module_ &m) {

  nb::class_<IntegerSet>(m, "IntegerSet")
      .def_static(
          "get",
          [](unsigned numDims, unsigned numSymbols,
             const std::vector<AffineExpr> &constraints,
             const std::vector<int> &eqFlags) {
            // convert eqFlags to SmallVector
            llvm::SmallVector<bool, 4> eqFlagsSmall;
            for (int flag : eqFlags) {
              eqFlagsSmall.push_back(flag != 0);
            }
            return IntegerSet::get(numDims, numSymbols, constraints,
                                   eqFlagsSmall);
          },
          nb::arg("num_dims"), nb::arg("num_symbols"), nb::arg("constraints"),
          nb::arg("context"))
      .def("get_num_dims", &IntegerSet::getNumDims)
      .def("get_num_symbols", &IntegerSet::getNumSymbols)
      .def("get_num_constraints", &IntegerSet::getNumConstraints)
      .def("__str__", [](IntegerSet &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<AffineExpr>(m, "AffineExpr")
      .def("__init__",
           [](AffineExpr &self) {
             throw nb::type_error(
                 "AffineExpr cannot be directly instantiated, "
                 "to get an AffineExpr, use OpBuilder methods");
           })
      .def("__str__",
           [](AffineExpr &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      // operator overloading
      .def("__add__",
           [](AffineExpr &self, AffineExpr &other) { return self + other; })
      .def("__sub__",
           [](AffineExpr &self, AffineExpr &other) { return self - other; })
      .def("__mul__",
           [](AffineExpr &self, AffineExpr &other) { return self * other; })
      .def("__floordiv__",
           [](AffineExpr &self, AffineExpr &other) {
             return self.floorDiv(other);
           })
      .def("__truediv__", [](AffineExpr &self,
                             AffineExpr &other) { return self.ceilDiv(other); })
      .def("__mod__",
           [](AffineExpr &self, AffineExpr &other) { return self % other; });

  nb::class_<AffineMap>(m, "AffineMap")
      .def("__init__",
           [](AffineMap &self) {
             throw nb::type_error(
                 "AffineMap cannot be directly instantiated, to get an "
                 "AffineMap, use AffineMap.get()");
           })
      .def_static(
          "get",
          [](const std::vector<unsigned> &dimSizes,
             const std::vector<unsigned> &symbolSizes,
             const std::vector<AffineExpr> &results, MLIRContext &context) {
            return AffineMap::get(dimSizes.size(), symbolSizes.size(), results,
                                  &context);
          },
          nb::arg("dim_sizes"), nb::arg("symbol_sizes"), nb::arg("results"),
          nb::arg("context"))
      .def_static(
          "get_identity",
          [](unsigned dimCount, MLIRContext &context) {
            return AffineMap::getMultiDimIdentityMap(dimCount, &context);
          },
          nb::arg("dim_count"), nb::arg("context"))
      .def("get_num_dims", &AffineMap::getNumDims)
      .def("get_num_symbols", &AffineMap::getNumSymbols)
      .def("get_num_results", &AffineMap::getNumResults)
      .def("__str__", [](AffineMap &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<AffineMapAttr, Attribute>(m, "AffineMapAttr")
      .def_static(
          "get",
          [](AffineMap map, MLIRContext &context) {
            return AffineMapAttr::get(map);
          },
          nb::arg("map"), nb::arg("context"))
      .def("get_map", &AffineMapAttr::getValue);

  nb::class_<affine::AffineBound>(m, "AffineBound")
      .def("__init__",
           [](affine::AffineBound &self) {
             throw nb::type_error(
                 "AffineBound cannot be directly instantiated, it should only "
                 "be obtained from AffineAnalysis");
           })
      .def("get_map", &affine::AffineBound::getMap)
      .def("get_operands", [](affine::AffineBound &self) {
        std::vector<Value> vals;
        for (Value val : self.getOperands()) {
          vals.push_back(val);
        }
        return vals;
      });
}

void init_ir(nb::module_ &m) {
  init_context(m);
  init_builder(m);
  init_core_ir(m);
  init_types(m);
  init_values(m);
  init_attributes(m);
  init_affine_objects(m);
}