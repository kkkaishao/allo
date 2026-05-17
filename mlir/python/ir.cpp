#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

#include "allo/InitAllDialects.h"
#include "allo/InitAllExtensions.h"

using namespace mlir;

static void bindContext(nb::module_ &m) {
  nb::class_<MLIRContext>(m, "Context")
      .def("__init__",
           [](MLIRContext &self) {
             new (&self) MLIRContext(MLIRContext::Threading::DISABLED);
           })
      .def("load_dialects",
           [](MLIRContext &self) {
             DialectRegistry registry;
             allo::registerAllDialects(registry);
             self.appendDialectRegistry(registry);
             self.loadAllAvailableDialects();
           })
      .def("load_transform_dialects",
           [](MLIRContext &self) {
             DialectRegistry registry;
             registry.insert<transform::TransformDialect>();
             allo::registerAllExtensions(registry);
             self.appendDialectRegistry(registry);
             self.loadAllAvailableDialects();
           })
      .def("get_loaded_dialects", [](MLIRContext &self) {
        std::vector<std::string> dialects;
        for (auto *dialect : self.getLoadedDialects()) {
          dialects.push_back(dialect->getNamespace().str());
        }
        return dialects;
      });
}

static void bindBuilder(nb::module_ &m) {
  nb::class_<OpBuilder::InsertPoint>(m, "InsertPoint")
      .def(nb::init<>())
      .def(
          "get_block",
          [](OpBuilder::InsertPoint &self) { return self.getBlock(); },
          nb::rv_policy::reference);

  nb::class_<OpBuilder>(m, "OpBuilder")
      .def(nb::init<MLIRContext *>(), nb::arg("context"))
      .def(nb::init<Operation *>(), nb::arg("operation"))
      .def(nb::init<Region *>(), nb::arg("region"))
      .def_prop_ro("context", &OpBuilder::getContext)
      // insertion point management
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
      // affine attributes
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
          "get_dict_attr",
          [](OpBuilder &self, nb::dict &dict) {
            llvm::SmallVector<NamedAttribute, 4> attrs;
            for (const auto &[k, v] : dict) {
              auto key = nb::cast<std::string>(k);
              auto value = nb::cast<Attribute>(v);
              attrs.push_back(self.getNamedAttr(key, value));
            }
            return self.getDictionaryAttr(attrs);
          },
          nb::arg("dict"))
      .def(
          "get_string_attr",
          [](OpBuilder &self, std::string_view value) {
            return self.getStringAttr(value);
          },
          nb::arg("value"));

  nb::class_<AlloOpBuilder, OpBuilder>(m, "AlloOpBuilder")
      .def(nb::init<MLIRContext *>(), nb::arg("context"))
      .def(nb::init<Operation *>(), nb::arg("operation"))
      .def("get_loc", &AlloOpBuilder::getLocation)
      .def("set_loc", &AlloOpBuilder::setLocation, nb::arg("new_loc"))
      .def("set_unknown_loc", &AlloOpBuilder::setUnknownLoc)
      .def("get_insertion_point_and_loc",
           &AlloOpBuilder::getInsertionPointAndLoc)
      .def("set_insertion_point_and_loc",
           &AlloOpBuilder::setInsertionPointAndLoc, nb::arg("ip"),
           nb::arg("new_loc"))
      .def(
          "create_block",
          [](AlloOpBuilder &self, Region &region,
             const std::vector<Type> &argTypes = {}) {
            if (!argTypes.empty()) {
              llvm::SmallVector<Location, 4> locs(argTypes.size(),
                                                  self.getLocation());
              return self.createBlock(&region, {}, argTypes, locs);
            }
            return self.createBlock(&region);
          },
          nb::rv_policy::reference, nb::arg("region"),
          nb::arg("arg_types") = std::vector<Type>());
}

static void bindCoreIR(nb::module_ &m) {
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
          [](Location &self, std::string_view filename, unsigned line,
             unsigned col, MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, filename);
            self = FileLineColLoc::get(attr, line, col);
          },
          nb::arg("filename"), nb::arg("line"), nb::arg("col"),
          nb::arg("context"))
      // NamedLoc init
      .def(
          "__init__",
          [](Location &self, Location &childLoc, std::string_view name,
             MLIRContext &context) {
            StringAttr attr = StringAttr::get(&context, name);
            self = dyn_cast<LocationAttr>(NameLoc::get(attr, childLoc));
          },
          nb::arg("child_loc"), nb::arg("name"), nb::arg("context"))
      .def("__str__",
           [](Location &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def(
          "set_name",
          [](Location &self, std::string_view name) {
            StringAttr attr = StringAttr::get(self.getContext(), name);
            self = NameLoc::get(attr, self);
          },
          nb::arg("name"));

  nb::class_<Type>(m, "Type")
      .def("__init__",
           [](Type &self) {
             throw nb::type_error(
                 "Type cannot be directly instantiated, to get a Type, use a "
                 "specific Type's get() method");
           })
      .def(
          "__eq__",
          [](Type &self, nb::object &other) {
            Type *otherTy = nb::cast<Type *>(other);
            return (otherTy != nullptr) && self == *otherTy;
          },
          nb::arg("other"))
      .def(
          "__ne__",
          [](Type &self, nb::object &other) {
            Type *otherTy = nb::cast<Type *>(other);
            return (otherTy == nullptr) || self != *otherTy;
          },
          nb::arg("other"))
      .def("__str__", [](Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<Value>(m, "Value")
      .def("__str__",
           [](Value &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("get_loc", &Value::getLoc)
      .def(
          "set_loc", [](Value &self, Location loc) { self.setLoc(loc); },
          nb::arg("loc"))
      .def(
          "replace_all_uses_with",
          [](Value &self, Value &val) { self.replaceAllUsesWith(val); },
          nb::arg("val"))
      .def("get_type", &Value::getType);

  nb::class_<Attribute>(m, "Attribute").def("__str__", [](Attribute &self) {
    std::string str;
    llvm::raw_string_ostream os(str);
    self.print(os);
    return os.str();
  });

  (void)nb::class_<Region>(m, "Region");

  nb::class_<Block>(m, "Block")
      .def(
          "get_arg_at",
          [](Block &self, unsigned idx) {
            if (idx >= self.getNumArguments()) {
              throw nb::index_error("block argument index out of range");
            }
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def("get_args",
           [](Block &self) {
             std::vector<Value> args;
             for (auto arg : self.getArguments())
               args.push_back(arg);
             return args;
           })
      .def("get_arg_types",
           [](Block &self) {
             std::vector<Type> argTypes;
             for (auto arg : self.getArguments())
               argTypes.push_back(arg.getType());
             return argTypes;
           })
      .def("get_num_args", &Block::getNumArguments)
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
      .def("get_parent_region", &Block::getParent, nb::rv_policy::reference)
      .def("get_parent_op", &Block::getParentOp, nb::rv_policy::reference)
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def(
          "get_terminator", [](Block &self) { return self.getTerminator(); },
          nb::rv_policy::reference)
      .def("erase", [](Block &self) { self.erase(); })
      .def("merge_before",
           [](Block &self, Block &dst) {
             if (!self.hasNoPredecessors())
               throw nb::value_error(
                   "Only blocks with no predecessors can be merged");
             if (self.getNumArguments() != 0)
               throw nb::value_error(
                   "Only blocks with no arguments can be merged");
             auto insertPt = dst.empty() ? dst.end() : std::prev(dst.end());
             dst.getOperations().splice(insertPt, self.getOperations());
             self.erase();
           })
      .def("remove_terminator", [](Block &self) {
        if (!self.empty() && self.back().hasTrait<OpTrait::IsTerminator>())
          self.getTerminator()->erase();
      });

  // Base Operation class
  nb::class_<Operation>(m, "Operation")
      .def("get_loc", &Operation::getLoc)
      .def("get_name",
           [](Operation &self) { return self.getName().getStringRef().str(); })
      .def("erase", &Operation::erase);

  nb::class_<OpState>(m, "OpState")
      .def("get_loc", &OpState::getLoc)
      .def(
          "set_attr",
          [](OpState &self, std::string_view name, Attribute &attr) {
            self->setAttr(name, attr);
          },
          nb::arg("name"), nb::arg("attr"))
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
      .def("get_operands",
           [](OpState &self) {
             std::vector<Value> operands;
             for (auto operand : self->getOperands())
               operands.push_back(operand);
             return operands;
           })
      .def("get_num_results",
           [](OpState &self) { return self->getNumResults(); })
      .def(
          "get_result_at",
          [](OpState &self, unsigned idx) {
            if (idx >= self->getNumResults())
              throw nb::index_error("Op result index out of range");
            return self->getResult(idx);
          },
          nb::arg("idx"))
      .def("get_results",
           [](OpState &self) {
             std::vector<Value> results;
             for (auto result : self->getResults())
               results.push_back(result);
             return results;
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
          nb::rv_policy::reference, nb::arg("idx"))
      .def(
          "get_block",
          [](OpState &self) { return self.getOperation()->getBlock(); },
          nb::rv_policy::reference)
      .def(
          "get_operation", [](OpState &self) { return self.getOperation(); },
          nb::rv_policy::reference)
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = getOpPrintingFlags();
             self->print(os, printingFlags);
             return os.str();
           })
      .def("verify",
           [](OpState &self) -> bool {
             return succeeded(verify(self.getOperation()));
           })
      .def("erase", &OpState::erase);

  nb::class_<OwningOpRef<ModuleOp>>(m, "OwningModuleOp")
      .def("get", [](OwningOpRef<ModuleOp> &self) { return self.get(); });

  nb::class_<ModuleOp, OpState>(m, "ModuleOp")
      .def(
          "__init__",
          [](ModuleOp &self, AlloOpBuilder &builder) {
            self = ModuleOp::create(builder.getLocation());
          },
          nb::arg("builder"))
      .def(
          "get_body", [](ModuleOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def(
          "push_back",
          [](ModuleOp &self, Operation *op) { self.getBody()->push_back(op); },
          nb::arg("op"))
      .def("get_context", &ModuleOp::getContext, nb::rv_policy::reference)
      .def("run_canonicalize",
           [](ModuleOp &self) {
             PassManager pm(self.getContext());
             pm.addPass(mlir::createCanonicalizerPass());
             pm.addPass(mlir::createCSEPass());
             (void)pm.run(self);
           })
      .def("clone",
           [](ModuleOp &self) {
             return OwningOpRef<ModuleOp>(cast<ModuleOp>(self.clone()));
           })
      .def("lookup_kernel",
           [](ModuleOp &self,
              std::string_view name) -> std::optional<allo::KernelOp> {
             auto sym = self.lookupSymbol<allo::KernelOp>(name);
             if (!sym)
               return std::nullopt;
             return sym;
           })
      .def_static("from_string",
                  [](MLIRContext *ctx, std::string_view source) {
                    ParserConfig config(ctx, true, nullptr);
                    auto module =
                        mlir::parseSourceString<ModuleOp>(source, config);
                    if (!module)
                      throw std::runtime_error(
                          "failed to parse MLIR module from string");
                    return module.release();
                  })
      .def_static("from_file", [](MLIRContext *ctx, std::string_view filename) {
        std::string errorMessage;
        auto file = mlir::openInputFile(filename, &errorMessage);
        if (!file)
          throw std::runtime_error("failed to open file: " + errorMessage);

        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
        ParserConfig config(ctx, true, nullptr);
        auto module = mlir::parseSourceFile<ModuleOp>(sourceMgr, config);
        if (!module)
          throw std::runtime_error("failed to parse MLIR module from file: " +
                                   std::string(filename));
        return module.release();
      });
}

static void bindTypes(nb::module_ &m) {
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
          nb::arg("width"), nb::arg("context"));

  nb::class_<IndexType, Type>(m, "IndexType")
      .def_static(
          "get", [](MLIRContext &context) { return IndexType::get(&context); },
          nb::arg("context"));

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
          [](const std::vector<int64_t> &shape, Type elementType) {
            return RankedTensorType::get(shape, elementType);
          },
          nb::arg("shape"), nb::arg("element_type"))
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType,
             Attribute encoding) {
            return RankedTensorType::get(shape, elementType, encoding);
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("encoding"));

  nb::class_<MemRefType, Type>(m, "MemRefType")
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType,
             AffineMap map) {
            return MemRefType::get(shape, elementType, map);
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("affine_maps"))
      .def_static(
          "get",
          [](const std::vector<int64_t> &shape, Type elementType, AffineMap map,
             Attribute memorySpace) {
            return MemRefType::get(shape, elementType, map, memorySpace);
          },
          nb::arg("shape"), nb::arg("element_type"), nb::arg("affine_maps"),
          nb::arg("memory_space"));
}

static void bindValues(nb::module_ &m) {
  (void)nb::class_<BlockArgument, Value>(m, "BlockArgument");
  (void)nb::class_<OpResult, Value>(m, "OpResult");
}

static void bindAttributes(nb::module_ &m) {
  nb::class_<IntegerAttr, Attribute>(m, "IntegerAttr")
      .def_static(
          "get",
          [](Type ty, int64_t value) { return IntegerAttr::get(ty, value); },
          nb::arg("ty"), nb::arg("value"));

  nb::class_<FloatAttr, Attribute>(m, "FloatAttr")
      .def_static(
          "get",
          [](Type ty, double value) { return FloatAttr::get(ty, value); },
          nb::arg("ty"), nb::arg("value"));

  nb::class_<UnitAttr, Attribute>(m, "UnitAttr")
      .def_static(
          "get", [](MLIRContext &context) { return UnitAttr::get(&context); },
          nb::arg("context"));

  nb::class_<StringAttr, Attribute>(m, "StringAttr")
      .def_static(
          "get",
          [](std::string_view value, MLIRContext &context) {
            return StringAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"));

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
          [](MLIRContext &context,
             const std::vector<int32_t> &values) -> DenseI32ArrayAttr {
            return DenseI32ArrayAttr::get(&context, values);
          },
          nb::arg("context"), nb::arg("values"));

  nb::class_<DenseI64ArrayAttr, Attribute>(m, "DenseI64ArrayAttr")
      .def_static(
          "get",
          [](MLIRContext &context,
             const std::vector<int64_t> &values) -> DenseI64ArrayAttr {
            return DenseI64ArrayAttr::get(&context, values);
          },
          nb::arg("context"), nb::arg("values"));

  nb::class_<DenseElementsAttr, Attribute>(m, "DenseElementsAttr")
      .def_static(
          "get",
          [](Type type, const std::vector<Attribute> &values) {
            auto shapedType = llvm::dyn_cast<ShapedType>(type);
            if (!shapedType)
              throw nb::type_error("DenseElementsAttr requires a shaped type");
            return DenseElementsAttr::get(shapedType, values);
          },
          nb::arg("type"), nb::arg("values"));

  nb::class_<FlatSymbolRefAttr, Attribute>(m, "FlatSymbolRefAttr")
      .def_static(
          "get",
          [](std::string_view value, MLIRContext &context) {
            return FlatSymbolRefAttr::get(&context, value);
          },
          nb::arg("value"), nb::arg("context"));

  nb::class_<StridedLayoutAttr, Attribute>(m, "StridedLayoutAttr")
      .def_static(
          "get",
          [](MLIRContext &context, int64_t offset,
             const std::vector<int64_t> &strides) {
            return StridedLayoutAttr::get(&context, offset, strides);
          },
          nb::arg("context"), nb::arg("offset"), nb::arg("strides"));

  nb::class_<DictionaryAttr, Attribute>(m, "DictionaryAttr")
      .def_static(
          "get",
          [](MLIRContext &context, nb::dict &dict) {
            llvm::SmallVector<NamedAttribute, 4> attrs;
            for (const auto &[k, v] : dict) {
              std::string key = nb::cast<std::string>(k);
              Attribute value = nb::cast<Attribute>(v);
              attrs.push_back(
                  NamedAttribute(StringAttr::get(&context, key), value));
            }
            return DictionaryAttr::get(&context, attrs);
          },
          nb::arg("context"), nb::arg("d"));

  nb::class_<TypeAttr, Attribute>(m, "TypeAttr")
      .def_static(
          "get", [](Type ty) { return TypeAttr::get(ty); }, nb::arg("type"));
}

static void bindAffineObjects(nb::module_ &m) {
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
      .def("__str__", [](IntegerSet &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  nb::class_<AffineExpr>(m, "AffineExpr")
      .def("__str__",
           [](AffineExpr &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      // operator overloading
      .def(
          "__add__",
          [](AffineExpr &self, AffineExpr &other) { return self + other; },
          nb::arg("other"))
      .def(
          "__sub__",
          [](AffineExpr &self, AffineExpr &other) { return self - other; },
          nb::arg("other"))
      .def(
          "__mul__",
          [](AffineExpr &self, AffineExpr &other) { return self * other; },
          nb::arg("other"))
      .def(
          "__floordiv__",
          [](AffineExpr &self, AffineExpr &other) {
            return self.floorDiv(other);
          },
          nb::arg("other"))
      .def(
          "__truediv__",
          [](AffineExpr &self, AffineExpr &other) {
            return self.ceilDiv(other);
          },
          nb::arg("other"))
      .def(
          "__mod__",
          [](AffineExpr &self, AffineExpr &other) { return self % other; },
          nb::arg("other"));

  nb::class_<AffineMap>(m, "AffineMap")
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
      .def("get_sub_map", &AffineMap::getSubMap)
      .def("__str__", [](AffineMap &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });
}

void bindIR(nb::module_ &m) {
  bindContext(m);
  bindBuilder(m);
  bindCoreIR(m);
  bindTypes(m);
  bindValues(m);
  bindAttributes(m);
  bindAffineObjects(m);
}
