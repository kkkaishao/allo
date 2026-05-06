#include "ir.h"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"

#include "allo/IR/AlloOps.h"

using namespace mlir;

void bindFuncOps(nb::module_ &m) {
  nb::class_<func::FuncOp, OpState>(m, "FuncOp")
      .def(
          "__init__",
          [](func::FuncOp &self, AlloOpBuilder &builder, std::string_view name,
             FunctionType &type) {
            self = func::FuncOp::create(builder, builder.getLocation(), name,
                                        type);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("type"))
      .def(
          "get_arg_at",
          [](func::FuncOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def("get_args",
           [](func::FuncOp &self) {
             std::vector<BlockArgument> args;
             for (auto arg : self.getArguments())
               args.push_back(arg);
             return args;
           })
      .def("get_num_args", &func::FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); },
          nb::rv_policy::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, unsigned argNo, std::string_view name,
             Attribute &attr) {
            if (argNo >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            // set arg attributes "name" to Value &"val"
            self.setArgAttr(argNo, name, attr);
          },
          nb::arg("arg_no"), nb::arg("name"), nb::arg("attr"))
      .def("get_func_type", &func::FuncOp::getFunctionType)
      .def("set_type", &func::FuncOp::setType, nb::arg("type"))
      .def("get_func_name",
           [](func::FuncOp &self) { return self.getName().str(); });

  nb::class_<func::ReturnOp, OpState>(m, "ReturnOp")
      .def(
          "__init__",
          [](func::ReturnOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &operands) {
            self = func::ReturnOp::create(builder, builder.getLocation(),
                                          operands);
          },
          nb::arg("builder"), nb::arg("operands"));

  nb::class_<func::CallOp, OpState>(m, "CallOp")
      .def(
          "__init__",
          [](func::CallOp &self, AlloOpBuilder &builder, func::FuncOp &func,
             const std::vector<Value> &args) {
            self = func::CallOp::create(builder, builder.getLocation(), func,
                                        args);
          },
          nb::arg("builder"), nb::arg("func"), nb::arg("args"));
}

void bindAffineOps(nb::module_ &m) {
  // affine ops
  nb::class_<affine::AffineForOp, OpState>(m, "AffineForOp")
      .def(
          "__init__",
          [](affine::AffineForOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &lb, AffineMap lbMap,
             const std::vector<Value> &ub, AffineMap ubMap, int64_t step) {
            self = affine::AffineForOp::create(builder, builder.getLocation(),
                                               lb, lbMap, ub, ubMap, step);
          },
          nb::arg("builder"), nb::arg("lb_operands"), nb::arg("lb_map"),
          nb::arg("ub_operands"), nb::arg("ub_map"), nb::arg("step") = 1)
      .def(
          "__init__",
          [](affine::AffineForOp &self, AlloOpBuilder &builder, int64_t lb,
             int64_t ub, int64_t step = 1) {
            self = affine::AffineForOp::create(builder, builder.getLocation(),
                                               lb, ub, step);
          },
          nb::arg("builder"), nb::arg("lb"), nb::arg("ub"), nb::arg("step") = 1)
      .def("get_induction_var", &affine::AffineForOp::getInductionVar)
      .def(
          "get_body", [](affine::AffineForOp &self) { return self.getBody(); },
          nb::rv_policy::reference);

  nb::class_<affine::AffineIfOp, OpState>(m, "AffineIfOp")
      .def(
          "__init__",
          [](affine::AffineIfOp &self, AlloOpBuilder &builder, IntegerSet set,
             const std::vector<Value> &operands, bool withElse) {
            self = affine::AffineIfOp::create(builder, builder.getLocation(),
                                              set, operands, withElse);
          },
          nb::arg("builder"), nb::arg("set"), nb::arg("operands"),
          nb::arg("with_else") = false)
      .def("get_integer_set", &affine::AffineIfOp::getIntegerSet)
      .def("get_then_block", &affine::AffineIfOp::getThenBlock,
           nb::rv_policy::reference)
      .def("get_else_block", &affine::AffineIfOp::getElseBlock,
           nb::rv_policy::reference);

  nb::class_<affine::AffineLoadOp, OpState>(m, "AffineLoadOp")
      .def(
          "__init__",
          [](affine::AffineLoadOp &self, AlloOpBuilder &builder, Value &memref,
             AffineMap &map, const std::vector<Value> &operands) {
            self = affine::AffineLoadOp::create(builder, builder.getLocation(),
                                                memref, map, operands);
          },
          nb::arg("builder"), nb::arg("memref"), nb::arg("map"),
          nb::arg("operands"));

  nb::class_<affine::AffineStoreOp, OpState>(m, "AffineStoreOp")
      .def(
          "__init__",
          [](affine::AffineStoreOp &self, AlloOpBuilder &builder, Value &value,
             Value &memref, AffineMap &map,
             const std::vector<Value> &operands) {
            self = affine::AffineStoreOp::create(builder, builder.getLocation(),
                                                 value, memref, map, operands);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("memref"),
          nb::arg("map"), nb::arg("operands"));

  nb::class_<affine::AffineApplyOp, OpState>(m, "AffineApplyOp")
      .def(
          "__init__",
          [](affine::AffineApplyOp &self, AlloOpBuilder &builder,
             AffineMap &map, const std::vector<Value> &operands) {
            self = affine::AffineApplyOp::create(builder, builder.getLocation(),
                                                 map, operands);
          },
          nb::arg("builder"), nb::arg("map"), nb::arg("operands"));
}

void bindSCFOps(nb::module_ &m) {
  // scf ops
  nb::class_<scf::ForOp, OpState>(m, "ForOp")
      .def(
          "__init__",
          [](scf::ForOp &self, AlloOpBuilder &builder, Value &lb, Value &ub,
             Value &step, const std::vector<Value> &initArgs) {
            self = scf::ForOp::create(builder, builder.getLocation(), lb, ub,
                                      step, initArgs);
          },
          nb::arg("builder"), nb::arg("lb"), nb::arg("ub"), nb::arg("step"),
          nb::arg("init_args") = std::vector<Value>())
      .def("get_induction_var", &scf::ForOp::getInductionVar)
      .def(
          "get_body", [](scf::ForOp &self) { return self.getBody(0); },
          nb::rv_policy::reference);

  nb::class_<scf::IfOp, OpState>(m, "IfOp")
      .def(
          "__init__",
          [](scf::IfOp &self, AlloOpBuilder &builder,
             const std::vector<Type> &resultTypes, Value &cond, bool withElse) {
            self = scf::IfOp::create(builder, builder.getLocation(),
                                     resultTypes, cond, withElse);
          },
          nb::arg("builder"), nb::arg("res_types"), nb::arg("cond"),
          nb::arg("with_else") = false)
      .def("get_then_block", &scf::IfOp::thenBlock, nb::rv_policy::reference)
      .def("get_else_block", &scf::IfOp::elseBlock, nb::rv_policy::reference)
      .def("get_then_yield", &scf::IfOp::thenYield)
      .def("get_else_yield", &scf::IfOp::elseYield);

  nb::class_<scf::YieldOp, OpState>(m, "YieldOp")
      .def(
          "__init__",
          [](scf::YieldOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &results) {
            self =
                scf::YieldOp::create(builder, builder.getLocation(), results);
          },
          nb::arg("builder"), nb::arg("results"));

  nb::class_<scf::WhileOp, OpState>(m, "WhileOp")
      .def(
          "__init__",
          [](scf::WhileOp &self, AlloOpBuilder &builder,
             const std::vector<Type> &resultTypes,
             const std::vector<Value> &operands) {
            self = scf::WhileOp::create(builder, builder.getLocation(),
                                        resultTypes, operands);
          },
          nb::arg("builder"), nb::arg("result_types"), nb::arg("operands"))
      .def("get_before", &scf::WhileOp::getBefore, nb::rv_policy::reference)
      .def("get_after", &scf::WhileOp::getAfter, nb::rv_policy::reference);

  nb::class_<scf::ConditionOp, OpState>(m, "ConditionOp")
      .def(
          "__init__",
          [](scf::ConditionOp &self, AlloOpBuilder &builder, Value &cond,
             const std::vector<Value> &args) {
            self = scf::ConditionOp::create(builder, builder.getLocation(),
                                            cond, args);
          },
          nb::arg("builder"), nb::arg("cond"), nb::arg("args"));

  nb::class_<scf::ParallelOp, OpState>(m, "ParallelOp")
      .def(
          "__init__",
          [](scf::ParallelOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &lbs, const std::vector<Value> &ubs,
             const std::vector<Value> &steps,
             const std::vector<Value> &initArgs) {
            self = scf::ParallelOp::create(builder, builder.getLocation(), lbs,
                                           ubs, steps, initArgs);
          },
          nb::arg("builder"), nb::arg("lbs"), nb::arg("ubs"), nb::arg("steps"),
          nb::arg("init_args") = std::vector<Value>())
      .def(
          "get_body", [](scf::ParallelOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def("get_induction_vars", [](scf::ParallelOp &self) {
        auto ivs = self.getInductionVars();
        return std::vector<Value>(ivs.begin(), ivs.end());
      });
}

void bindCFOps(nb::module_ &m) {
  nb::class_<cf::CondBranchOp, OpState>(m, "CondBranchOp")
      .def(
          "__init__",
          [](cf::CondBranchOp &self, AlloOpBuilder &builder, Value &cond,
             Block *trueDst, Block *falseDst) {
            self = cf::CondBranchOp::create(builder, builder.getLocation(),
                                            cond, trueDst, falseDst);
          },
          nb::arg("builder"), nb::arg("cond"), nb::arg("true_dest"),
          nb::arg("false_dest"));

  nb::class_<cf::BranchOp, OpState>(m, "BranchOp")
      .def(
          "__init__",
          [](cf::BranchOp &self, AlloOpBuilder &builder, Block *dest,
             const std::vector<Value> &args) {
            self = cf::BranchOp::create(builder, builder.getLocation(), dest,
                                        args);
          },
          nb::arg("builder"), nb::arg("dest"), nb::arg("args"));
}

void bindArithOps(nb::module_ &m) {
  // constant ops
  nb::class_<arith::ConstantOp, OpState>(m, "ConstantOp")
      .def(
          "__init__",
          [](arith::ConstantOp &self, AlloOpBuilder &builder,
             Attribute &value) {
            auto typedValue = llvm::dyn_cast<TypedAttr>(value);
            if (!typedValue)
              throw nb::type_error(
                  "arith.ConstantOp requires a typed attribute");
            self = arith::ConstantOp::create(builder, builder.getLocation(),
                                             typedValue);
          },
          nb::arg("builder"), nb::arg("value"));

  nb::class_<arith::ConstantIntOp, arith::ConstantOp>(m, "ConstantIntOp")
      .def(
          "__init__",
          [](arith::ConstantIntOp &self, AlloOpBuilder &builder,
             IntegerType &type, int64_t value) {
            self = arith::ConstantIntOp::create(builder, builder.getLocation(),
                                                type, value);
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"));

  nb::class_<arith::ConstantFloatOp, arith::ConstantOp>(m, "ConstantFloatOp")
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             Float32Type &type, float value) {
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             Float64Type &type, double value) {
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             Float16Type &type, float value) {
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, APFloat(value));
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"))
      .def(
          "__init__",
          [](arith::ConstantFloatOp &self, AlloOpBuilder &builder,
             BFloat16Type &type, float value) {
            // bf16 does not satisfy IEEE754, so we need to convert manually
            const llvm::fltSemantics &sem = type.getFloatSemantics();
            llvm::APFloat val(value);
            bool lost;
            val.convert(sem, llvm::APFloat::rmNearestTiesToEven, &lost);
            self = arith::ConstantFloatOp::create(
                builder, builder.getLocation(), type, val);
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("value"));

  nb::class_<arith::ConstantIndexOp, arith::ConstantOp>(m, "ConstantIndexOp")
      .def(
          "__init__",
          [](arith::ConstantIndexOp &self, AlloOpBuilder &builder,
             int64_t value) {
            self = arith::ConstantIndexOp::create(builder,
                                                  builder.getLocation(), value);
          },
          nb::arg("builder"), nb::arg("value"));

  // casts / conversions
  nb::class_<arith::SIToFPOp, OpState>(m, "SIToFPOp")
      .def(
          "__init__",
          [](arith::SIToFPOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::SIToFPOp::create(builder, builder.getLocation(),
                                           dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::UIToFPOp, OpState>(m, "UIToFPOp")
      .def(
          "__init__",
          [](arith::UIToFPOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::UIToFPOp::create(builder, builder.getLocation(),
                                           dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::FPToSIOp, OpState>(m, "FPToSIOp")
      .def(
          "__init__",
          [](arith::FPToSIOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::FPToSIOp::create(builder, builder.getLocation(),
                                           dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::FPToUIOp, OpState>(m, "FPToUIOp")
      .def(
          "__init__",
          [](arith::FPToUIOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::FPToUIOp::create(builder, builder.getLocation(),
                                           dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::ExtFOp, OpState>(m, "ExtFOp")
      .def(
          "__init__",
          [](arith::ExtFOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::ExtFOp::create(builder, builder.getLocation(),
                                         dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::TruncFOp, OpState>(m, "TruncFOp")
      .def(
          "__init__",
          [](arith::TruncFOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::TruncFOp::create(builder, builder.getLocation(),
                                           dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::IndexCastOp, OpState>(m, "IndexCastOp")
      .def(
          "__init__",
          [](arith::IndexCastOp &self, AlloOpBuilder &builder, Type &dstType,
             Value &src) {
            self = arith::IndexCastOp::create(builder, builder.getLocation(),
                                              dstType, src);
          },
          nb::arg("builder"), nb::arg("dst_type"), nb::arg("src"));

  // integer extension / truncation / bitcast
  nb::class_<arith::ExtSIOp, OpState>(m, "ExtSIOp")
      .def(
          "__init__",
          [](arith::ExtSIOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::ExtSIOp::create(builder, builder.getLocation(),
                                          dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::ExtUIOp, OpState>(m, "ExtUIOp")
      .def(
          "__init__",
          [](arith::ExtUIOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::ExtUIOp::create(builder, builder.getLocation(),
                                          dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::BitcastOp, OpState>(m, "BitcastOp")
      .def(
          "__init__",
          [](arith::BitcastOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::BitcastOp::create(builder, builder.getLocation(),
                                            dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  nb::class_<arith::TruncIOp, OpState>(m, "TruncIOp")
      .def(
          "__init__",
          [](arith::TruncIOp &self, AlloOpBuilder &builder, Value &src,
             Type &dstType) {
            self = arith::TruncIOp::create(builder, builder.getLocation(),
                                           dstType, src);
          },
          nb::arg("builder"), nb::arg("src"), nb::arg("dst_type"));

  // floating ops
  nb::class_<arith::AddFOp, OpState>(m, "AddFOp")
      .def(
          "__init__",
          [](arith::AddFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::AddFOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SubFOp, OpState>(m, "SubFOp")
      .def(
          "__init__",
          [](arith::SubFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::SubFOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MulFOp, OpState>(m, "MulFOp")
      .def(
          "__init__",
          [](arith::MulFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::MulFOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivFOp, OpState>(m, "DivFOp")
      .def(
          "__init__",
          [](arith::DivFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::DivFOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemFOp, OpState>(m, "RemFOp")
      .def(
          "__init__",
          [](arith::RemFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::RemFOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::NegFOp, OpState>(m, "NegFOp")
      .def(
          "__init__",
          [](arith::NegFOp &self, AlloOpBuilder &builder, Value &input) {
            self = arith::NegFOp::create(builder, builder.getLocation(), input);
          },
          nb::arg("builder"), nb::arg("input"));

  // integer arithmetic
  nb::class_<arith::AddIOp, OpState>(m, "AddIOp")
      .def(
          "__init__",
          [](arith::AddIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::AddIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SubIOp, OpState>(m, "SubIOp")
      .def(
          "__init__",
          [](arith::SubIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::SubIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MulIOp, OpState>(m, "MulIOp")
      .def(
          "__init__",
          [](arith::MulIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::MulIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivSIOp, OpState>(m, "DivSIOp")
      .def(
          "__init__",
          [](arith::DivSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::DivSIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivUIOp, OpState>(m, "DivUIOp")
      .def(
          "__init__",
          [](arith::DivUIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::DivUIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::CeilDivSIOp, OpState>(m, "CeilDivSIOp")
      .def(
          "__init__",
          [](arith::CeilDivSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::CeilDivSIOp::create(builder, builder.getLocation(),
                                              lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::CeilDivUIOp, OpState>(m, "CeilDivUIOp")
      .def(
          "__init__",
          [](arith::CeilDivUIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::CeilDivUIOp::create(builder, builder.getLocation(),
                                              lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::FloorDivSIOp, OpState>(m, "FloorDivSIOp")
      .def(
          "__init__",
          [](arith::FloorDivSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::FloorDivSIOp::create(builder, builder.getLocation(),
                                               lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemSIOp, OpState>(m, "RemSIOp")
      .def(
          "__init__",
          [](arith::RemSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::RemSIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemUIOp, OpState>(m, "RemUIOp")
      .def(
          "__init__",
          [](arith::RemUIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::RemUIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  // fused / special ops
  nb::class_<math::FmaOp, OpState>(m, "FmaOp")
      .def(
          "__init__",
          [](math::FmaOp &self, AlloOpBuilder &builder, Value &a, Value &b,
             Value &c) {
            self = math::FmaOp::create(builder, builder.getLocation(), a, b, c);
          },
          nb::arg("builder"), nb::arg("a"), nb::arg("b"), nb::arg("c"));

  // shifts
  nb::class_<arith::ShLIOp, OpState>(m, "ShLIOp")
      .def(
          "__init__",
          [](arith::ShLIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::ShLIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::ShRUIOp, OpState>(m, "ShRUIOp")
      .def(
          "__init__",
          [](arith::ShRUIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::ShRUIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::ShRSIOp, OpState>(m, "ShRSIOp")
      .def(
          "__init__",
          [](arith::ShRSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::ShRSIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  // mins / maxs
  nb::class_<arith::MinSIOp, OpState>(m, "MinSIOp")
      .def(
          "__init__",
          [](arith::MinSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MinSIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinUIOp, OpState>(m, "MinUIOp")
      .def(
          "__init__",
          [](arith::MinUIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MinUIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinimumFOp, OpState>(m, "MinimumFOp")
      .def(
          "__init__",
          [](arith::MinimumFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MinimumFOp::create(builder, builder.getLocation(),
                                             lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinNumFOp, OpState>(m, "MinNumFOp")
      .def(
          "__init__",
          [](arith::MinNumFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MinNumFOp::create(builder, builder.getLocation(), lhs,
                                            rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxSIOp, OpState>(m, "MaxSIOp")
      .def(
          "__init__",
          [](arith::MaxSIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MaxSIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxUIOp, OpState>(m, "MaxUIOp")
      .def(
          "__init__",
          [](arith::MaxUIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MaxUIOp::create(builder, builder.getLocation(), lhs,
                                          rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaximumFOp, OpState>(m, "MaximumFOp")
      .def(
          "__init__",
          [](arith::MaximumFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MaximumFOp::create(builder, builder.getLocation(),
                                             lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxNumFOp, OpState>(m, "MaxNumFOp")
      .def(
          "__init__",
          [](arith::MaxNumFOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self = arith::MaxNumFOp::create(builder, builder.getLocation(), lhs,
                                            rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  // comparisons (int)
  nb::class_<arith::CmpIOp, OpState>(m, "CmpIOp")
      .def(
          "__init__",
          [](arith::CmpIOp &self, AlloOpBuilder &builder, std::size_t pred,
             Value &lhs, Value &rhs) {
            self = arith::CmpIOp::create(
                builder, builder.getLocation(),
                static_cast<arith::CmpIPredicate>(pred), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("pred"), nb::arg("lhs"), nb::arg("rhs"));

  // comparisons (float)
  nb::class_<arith::CmpFOp, OpState>(m, "CmpFOp")
      .def(
          "__init__",
          [](arith::CmpFOp &self, AlloOpBuilder &builder, std::size_t pred,
             Value &lhs, Value &rhs) {
            self = arith::CmpFOp::create(
                builder, builder.getLocation(),
                static_cast<arith::CmpFPredicate>(pred), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("pred"), nb::arg("lhs"), nb::arg("rhs"));

  // logical
  nb::class_<arith::AndIOp, OpState>(m, "AndIOp")
      .def(
          "__init__",
          [](arith::AndIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::AndIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::XOrIOp, OpState>(m, "XOrIOp")
      .def(
          "__init__",
          [](arith::XOrIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::XOrIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::OrIOp, OpState>(m, "OrIOp")
      .def(
          "__init__",
          [](arith::OrIOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs) {
            self =
                arith::OrIOp::create(builder, builder.getLocation(), lhs, rhs);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SelectOp, OpState>(m, "SelectOp")
      .def(
          "__init__",
          [](arith::SelectOp &self, AlloOpBuilder &builder, Value &condition,
             Value &trueValue, Value &falseValue) {
            self = arith::SelectOp::create(builder, builder.getLocation(),
                                           condition, trueValue, falseValue);
          },
          nb::arg("builder"), nb::arg("condition"), nb::arg("true_value"),
          nb::arg("false_value"));
}

void bindMathOps(nb::module_ &m) {
  nb::class_<math::FloorOp, OpState>(m, "FloorOp")
      .def(
          "__init__",
          [](math::FloorOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::FloorOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::CeilOp, OpState>(m, "CeilOp")
      .def(
          "__init__",
          [](math::CeilOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::CeilOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::ExpOp, OpState>(m, "ExpOp")
      .def(
          "__init__",
          [](math::ExpOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::ExpOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::Exp2Op, OpState>(m, "Exp2Op")
      .def(
          "__init__",
          [](math::Exp2Op &self, AlloOpBuilder &builder, Value &val) {
            self = math::Exp2Op::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::CosOp, OpState>(m, "CosOp")
      .def(
          "__init__",
          [](math::CosOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::CosOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::SinOp, OpState>(m, "SinOp")
      .def(
          "__init__",
          [](math::SinOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::SinOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::LogOp, OpState>(m, "LogOp")
      .def(
          "__init__",
          [](math::LogOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::LogOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::Log2Op, OpState>(m, "Log2Op")
      .def(
          "__init__",
          [](math::Log2Op &self, AlloOpBuilder &builder, Value &val) {
            self = math::Log2Op::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::ErfOp, OpState>(m, "ErfOp")
      .def(
          "__init__",
          [](math::ErfOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::ErfOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::SqrtOp, OpState>(m, "SqrtOp")
      .def(
          "__init__",
          [](math::SqrtOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::SqrtOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::RsqrtOp, OpState>(m, "RsqrtOp")
      .def(
          "__init__",
          [](math::RsqrtOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::RsqrtOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::AbsFOp, OpState>(m, "AbsFOp")
      .def(
          "__init__",
          [](math::AbsFOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::AbsFOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::AbsIOp, OpState>(m, "AbsIOp")
      .def(
          "__init__",
          [](math::AbsIOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::AbsIOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::PowFOp, OpState>(m, "PowFOp")
      .def(
          "__init__",
          [](math::PowFOp &self, AlloOpBuilder &builder, Value &base,
             Value &exponent) {
            self = math::PowFOp::create(builder, builder.getLocation(), base,
                                        exponent);
          },
          nb::arg("builder"), nb::arg("base"), nb::arg("exponent"));

  nb::class_<math::TanOp, OpState>(m, "TanOp")
      .def(
          "__init__",
          [](math::TanOp &self, AlloOpBuilder &builder, Value &val) {
            self = math::TanOp::create(builder, builder.getLocation(), val);
          },
          nb::arg("builder"), nb::arg("val"));

  nb::class_<math::IPowIOp, OpState>(m, "IPowIOp")
      .def(
          "__init__",
          [](math::IPowIOp &self, AlloOpBuilder &builder, Value &base,
             Value &exponent) {
            self = math::IPowIOp::create(builder, builder.getLocation(), base,
                                         exponent);
          },
          nb::arg("builder"), nb::arg("base"), nb::arg("exponent"));

  nb::class_<math::FPowIOp, OpState>(m, "FPowIOp")
      .def(
          "__init__",
          [](math::FPowIOp &self, AlloOpBuilder &builder, Value &base,
             Value &exponent) {
            self = math::FPowIOp::create(builder, builder.getLocation(), base,
                                         exponent);
          },
          nb::arg("builder"), nb::arg("base"), nb::arg("exponent"));
}

void bindTensorOps(nb::module_ &m) {
  nb::class_<tensor::ExtractOp, OpState>(m, "ExtractOp")
      .def(
          "__init__",
          [](tensor::ExtractOp &self, AlloOpBuilder &builder, Value &tensor,
             const std::vector<Value> &indices) {
            self = tensor::ExtractOp::create(builder, builder.getLocation(),
                                             tensor, indices);
          },
          nb::arg("builder"), nb::arg("tensor"), nb::arg("indices"));

  nb::class_<tensor::InsertOp, OpState>(m, "InsertOp")
      .def(
          "__init__",
          [](tensor::InsertOp &self, AlloOpBuilder &builder, Value &value,
             Value &tensor, const std::vector<Value> &indices) {
            self = tensor::InsertOp::create(builder, builder.getLocation(),
                                            value, tensor, indices);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("tensor"),
          nb::arg("indices"));

  nb::class_<tensor::SplatOp, OpState>(m, "SplatOp")
      .def(
          "__init__",
          [](tensor::SplatOp &self, AlloOpBuilder &builder, Value &value,
             const std::vector<int64_t> &shape) {
            self = tensor::SplatOp::create(builder, builder.getLocation(),
                                           value, shape);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("shape"));

  nb::class_<tensor::CastOp, OpState>(m, "CastOp")
      .def(
          "__init__",
          [](tensor::CastOp &self, AlloOpBuilder &builder, Value &input,
             Type &dstType) {
            self = tensor::CastOp::create(builder, builder.getLocation(),
                                          dstType, input);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("dst_type"));

  nb::class_<tensor::EmptyOp, OpState>(m, "EmptyOp")
      .def(
          "__init__",
          [](tensor::EmptyOp &self, AlloOpBuilder &builder,
             const std::vector<int64_t> &shape, Type &elementType) {
            self = tensor::EmptyOp::create(builder, builder.getLocation(),
                                           shape, elementType);
          },
          nb::arg("builder"), nb::arg("shape"), nb::arg("element_type"))
      .def(
          "__init__",
          [](tensor::EmptyOp &self, AlloOpBuilder &builder, Type &type) {
            if (auto tensor = dyn_cast<RankedTensorType>(type)) {
              self = tensor::EmptyOp::create(
                  builder, builder.getLocation(), tensor.getShape(),
                  tensor.getElementType(), tensor.getEncoding());
              return;
            }
            if (auto memref = dyn_cast<MemRefType>(type)) {
              self = tensor::EmptyOp::create(
                  builder, builder.getLocation(), memref.getShape(),
                  memref.getElementType(), memref.getMemorySpace());
              return;
            }
            throw nb::type_error("Unsupported type for tensor.EmptyOp");
          },
          nb::arg("builder"), nb::arg("type"));

  nb::class_<tensor::ExtractSliceOp, OpState>(m, "ExtractSliceOp")
      .def(
          "__init__",
          [](tensor::ExtractSliceOp &self, AlloOpBuilder &builder,
             Value &source, const std::vector<Value> &offsets,
             const std::vector<Value> &sizes,
             const std::vector<Value> &strides) {
            self =
                tensor::ExtractSliceOp::create(builder, builder.getLocation(),
                                               source, offsets, sizes, strides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("offsets"),
          nb::arg("sizes"), nb::arg("strides"))
      .def(
          "__init__",
          [](tensor::ExtractSliceOp &self, AlloOpBuilder &builder,
             Type &resType, Value &source, const std::vector<Value> &offsets,
             const std::vector<Value> &sizes, const std::vector<Value> &strides,
             const std::vector<int64_t> &staticOffsets,
             const std::vector<int64_t> &staticSizes,
             const std::vector<int64_t> &staticStrides) {
            self = tensor::ExtractSliceOp::create(
                builder, builder.getLocation(), resType, source, offsets, sizes,
                strides, staticOffsets, staticSizes, staticStrides);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("source"),
          nb::arg("offsets"), nb::arg("sizes"), nb::arg("strides"),
          nb::arg("static_offsets"), nb::arg("static_sizes"),
          nb::arg("static_strides"));

  nb::class_<tensor::InsertSliceOp, OpState>(m, "InsertSliceOp")
      .def(
          "__init__",
          [](tensor::InsertSliceOp &self, AlloOpBuilder &builder, Value &source,
             Value &dest, const std::vector<Value> &offsets,
             const std::vector<Value> &sizes,
             const std::vector<Value> &strides) {
            self = tensor::InsertSliceOp::create(builder, builder.getLocation(),
                                                 source, dest, offsets, sizes,
                                                 strides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("dest"),
          nb::arg("offsets"), nb::arg("sizes"), nb::arg("strides"))
      .def(
          "__init__",
          [](tensor::InsertSliceOp &self, AlloOpBuilder &builder, Value &source,
             Value &dest, const std::vector<Value> &offsets,
             const std::vector<Value> &sizes, const std::vector<Value> &strides,
             const std::vector<int64_t> &staticOffsets,
             const std::vector<int64_t> &staticSizes,
             const std::vector<int64_t> &staticStrides) {
            self = tensor::InsertSliceOp::create(
                builder, builder.getLocation(), source, dest, offsets, sizes,
                strides, staticOffsets, staticSizes, staticStrides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("dest"),
          nb::arg("offsets"), nb::arg("sizes"), nb::arg("strides"),
          nb::arg("static_offsets"), nb::arg("static_sizes"),
          nb::arg("static_strides"));

  nb::class_<tensor::GatherOp, OpState>(m, "GatherOp")
      .def(
          "__init__",
          [](tensor::GatherOp &self, AlloOpBuilder &builder, Type &resType,
             Value &source, Value &indices, const std::vector<int64_t> &dims,
             bool unique) {
            self = tensor::GatherOp::create(builder, builder.getLocation(),
                                            resType, source, indices, dims,
                                            unique);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("source"),
          nb::arg("indices"), nb::arg("dims"), nb::arg("unique") = false);

  nb::class_<tensor::ScatterOp, OpState>(m, "ScatterOp")
      .def(
          "__init__",
          [](tensor::ScatterOp &self, AlloOpBuilder &builder, Type &resType,
             Value &source, Value &dest, Value &indices,
             const std::vector<int64_t> &dims, bool unique) {
            self = tensor::ScatterOp::create(builder, builder.getLocation(),
                                             resType, source, dest, indices,
                                             dims, unique);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("source"),
          nb::arg("dest"), nb::arg("indices"), nb::arg("dims"),
          nb::arg("unique") = false);
}

void bindMemRefOps(nb::module_ &m) {
  nb::class_<memref::LoadOp, OpState>(m, "LoadOp")
      .def(
          "__init__",
          [](memref::LoadOp &self, AlloOpBuilder &builder, Value &memref,
             const std::vector<Value> &indices) {
            self = memref::LoadOp::create(builder, builder.getLocation(),
                                          memref, indices);
          },
          nb::arg("builder"), nb::arg("memref"), nb::arg("indices"));

  nb::class_<memref::StoreOp, OpState>(m, "StoreOp")
      .def(
          "__init__",
          [](memref::StoreOp &self, AlloOpBuilder &builder, Value &value,
             Value &memref, const std::vector<Value> &indices) {
            self = memref::StoreOp::create(builder, builder.getLocation(),
                                           value, memref, indices);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("memref"),
          nb::arg("indices"));

  nb::class_<memref::AllocOp, OpState>(m, "AllocOp")
      .def(
          "__init__",
          [](memref::AllocOp &self, AlloOpBuilder &builder, MemRefType &type) {
            self =
                memref::AllocOp::create(builder, builder.getLocation(), type);
          },
          nb::arg("builder"), nb::arg("type"));

  nb::class_<memref::SubViewOp, OpState>(m, "SubViewOp")
      .def(
          "__init__",
          [](memref::SubViewOp &self, AlloOpBuilder &builder, Value &source,
             const std::vector<Value> &offsets, const std::vector<Value> &sizes,
             const std::vector<Value> &strides) {
            self = memref::SubViewOp::create(builder, builder.getLocation(),
                                             source, offsets, sizes, strides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("offsets"),
          nb::arg("sizes"), nb::arg("strides"))
      .def(
          "__init__",
          [](memref::SubViewOp &self, AlloOpBuilder &builder, Value &source,
             const std::vector<int64_t> &offsets,
             const std::vector<int64_t> &sizes,
             const std::vector<int64_t> &strides) {
            self = memref::SubViewOp::create(builder, builder.getLocation(),
                                             source, offsets, sizes, strides);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("offsets"),
          nb::arg("sizes"), nb::arg("strides"))
      .def(
          "__init__",
          [](memref::SubViewOp &self, AlloOpBuilder &builder, Type &type,
             Value &source, Value &offset, Value &size, Value &stride,
             const std::vector<int64_t> &staticOffsets,
             const std::vector<int64_t> &staticSizes,
             const std::vector<int64_t> &staticStrides) {
            self = memref::SubViewOp::create(
                builder, builder.getLocation(), type, source, offset, size,
                stride, staticOffsets, staticSizes, staticStrides);
          },
          nb::arg("builder"), nb::arg("type"), nb::arg("source"),
          nb::arg("offset"), nb::arg("size"), nb::arg("stride"),
          nb::arg("static_offsets"), nb::arg("static_sizes"),
          nb::arg("static_strides"));

  nb::class_<memref::CopyOp, OpState>(m, "CopyOp")
      .def(
          "__init__",
          [](memref::CopyOp &self, AlloOpBuilder &builder, Value &source,
             Value &dest) {
            self = memref::CopyOp::create(builder, builder.getLocation(),
                                          source, dest);
          },
          nb::arg("builder"), nb::arg("source"), nb::arg("dest"));

  nb::class_<memref::GlobalOp, OpState>(m, "GlobalOp")
      .def(
          "__init__",
          [](memref::GlobalOp &self, AlloOpBuilder &builder,
             std::string_view name, std::string_view visibility,
             MemRefType &type, Attribute &initValue, bool isConstant,
             int64_t alignment) {
            auto visAttr = builder.getStringAttr(visibility);
            auto alignAttr =
                builder.getIntegerAttr(builder.getI64Type(), alignment);
            self = memref::GlobalOp::create(builder, builder.getLocation(),
                                            name, visAttr, type, initValue,
                                            isConstant, alignAttr);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("visibility"),
          nb::arg("res_type"), nb::arg("init_value"), nb::arg("is_constant"),
          nb::arg("alignment"))
      .def(
          "__init__",
          [](memref::GlobalOp &self, AlloOpBuilder &builder,
             std::string_view name, std::string_view visibility,
             MemRefType &type, Attribute &initValue, bool isConstant) {
            auto visAttr = builder.getStringAttr(visibility);
            self = memref::GlobalOp::create(builder, builder.getLocation(),
                                            name, visAttr, type, initValue,
                                            isConstant, IntegerAttr());
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("visibility"),
          nb::arg("res_type"), nb::arg("init_value"), nb::arg("is_constant"));

  nb::class_<memref::GetGlobalOp, OpState>(m, "GetGlobalOp")
      .def(
          "__init__",
          [](memref::GetGlobalOp &self, AlloOpBuilder &builder, Type &resType,
             std::string_view name) {
            self = memref::GetGlobalOp::create(builder, builder.getLocation(),
                                               resType, name);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("name"));

  nb::class_<memref::TransposeOp, OpState>(m, "TransposeOp")
      .def(
          "__init__",
          [](memref::TransposeOp &self, AlloOpBuilder &builder, Value &input,
             AffineMap &permutation) {
            auto permAttr = AffineMapAttr::get(permutation);
            self = memref::TransposeOp::create(builder, builder.getLocation(),
                                               input, permAttr);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("permutation"));

  nb::class_<memref::ReshapeOp, OpState>(m, "ReshapeOp")
      .def(
          "__init__",
          [](memref::ReshapeOp &self, AlloOpBuilder &builder, Type &resType,
             Value &input, Value &shape) {
            self = memref::ReshapeOp::create(builder, builder.getLocation(),
                                             resType, input, shape);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("input"),
          nb::arg("shape"));
}

void bindLinalgOps(nb::module_ &m) {
  nb::class_<linalg::MatmulOp, OpState>(m, "MatmulOp")
      .def(
          "__init__",
          [](linalg::MatmulOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &result) {
            self = linalg::MatmulOp::create(
                builder, builder.getLocation(),
                std::initializer_list<Value>{lhs, rhs}, result);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("result"));

  nb::class_<linalg::FillOp, OpState>(m, "FillOp")
      .def(
          "__init__",
          [](linalg::FillOp &self, AlloOpBuilder &builder, Value &value,
             Value &output) {
            self = linalg::FillOp::create(builder, builder.getLocation(), value,
                                          output);
          },
          nb::arg("builder"), nb::arg("value"), nb::arg("output"));

  nb::class_<linalg::BroadcastOp, OpState>(m, "BroadcastOp")
      .def(
          "__init__",
          [](linalg::BroadcastOp &self, AlloOpBuilder &builder, Value &input,
             Value &init, const std::vector<int64_t> &dims) {
            self = linalg::BroadcastOp::create(builder, builder.getLocation(),
                                               input, init, dims);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"),
          nb::arg("dims"));

  nb::class_<linalg::AddOp, OpState>(m, "AddOp")
      .def(
          "__init__",
          [](linalg::AddOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &init) {
            self = linalg::AddOp::create(builder, builder.getLocation(),
                                         std::initializer_list<Value>{lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::SubOp, OpState>(m, "SubOp")
      .def(
          "__init__",
          [](linalg::SubOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &init) {
            self = linalg::SubOp::create(builder, builder.getLocation(),
                                         std::initializer_list<Value>{lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::MulOp, OpState>(m, "MulOp")
      .def(
          "__init__",
          [](linalg::MulOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &init) {
            self = linalg::MulOp::create(builder, builder.getLocation(),
                                         std::initializer_list<Value>{lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::DivOp, OpState>(m, "DivOp")
      .def(
          "__init__",
          [](linalg::DivOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &init) {
            self = linalg::DivOp::create(builder, builder.getLocation(),
                                         std::initializer_list<Value>{lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::DivUnsignedOp, OpState>(m, "DivUnsignedOp")
      .def(
          "__init__",
          [](linalg::DivUnsignedOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &init) {
            self = linalg::DivUnsignedOp::create(
                builder, builder.getLocation(),
                std::initializer_list<Value>{lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::class_<linalg::PowFOp, OpState>(m, "PowFOp")
      .def(
          "__init__",
          [](linalg::PowFOp &self, AlloOpBuilder &builder, Value &base,
             Value &exponent, Value &init) {
            self = linalg::PowFOp::create(
                builder, builder.getLocation(),
                std::initializer_list<Value>{base, exponent}, init);
          },
          nb::arg("builder"), nb::arg("base"), nb::arg("exponent"),
          nb::arg("init"));

  nb::class_<linalg::FloorOp, OpState>(m, "FloorOp")
      .def(
          "__init__",
          [](linalg::FloorOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::FloorOp::create(builder, builder.getLocation(),
                                           input, init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::ExpOp, OpState>(m, "ExpOp")
      .def(
          "__init__",
          [](linalg::ExpOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::ExpOp::create(builder, builder.getLocation(), input,
                                         init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::LogOp, OpState>(m, "LogOp")
      .def(
          "__init__",
          [](linalg::LogOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::LogOp::create(builder, builder.getLocation(), input,
                                         init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::SqrtOp, OpState>(m, "SqrtOp")
      .def(
          "__init__",
          [](linalg::SqrtOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::SqrtOp::create(builder, builder.getLocation(), input,
                                          init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::ReciprocalOp, OpState>(m, "ReciprocalOp")
      .def(
          "__init__",
          [](linalg::ReciprocalOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::ReciprocalOp::create(builder, builder.getLocation(),
                                                input, init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::RsqrtOp, OpState>(m, "RsqrtOp")
      .def(
          "__init__",
          [](linalg::RsqrtOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::RsqrtOp::create(builder, builder.getLocation(),
                                           input, init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::SquareOp, OpState>(m, "SquareOp")
      .def(
          "__init__",
          [](linalg::SquareOp &self, AlloOpBuilder &builder, Value &input,
             Value &init) {
            self = linalg::SquareOp::create(builder, builder.getLocation(),
                                            input, init);
          },
          nb::arg("builder"), nb::arg("input"), nb::arg("init"));

  nb::class_<linalg::DotOp, OpState>(m, "DotOp")
      .def(
          "__init__",
          [](linalg::DotOp &self, AlloOpBuilder &builder, Value &lhs,
             Value &rhs, Value &init) {
            self = linalg::DotOp::create(builder, builder.getLocation(),
                                         std::initializer_list<Value>{lhs, rhs},
                                         init);
          },
          nb::arg("builder"), nb::arg("lhs"), nb::arg("rhs"), nb::arg("init"));

  nb::enum_<utils::IteratorType>(m, "IteratorType")
      .value("PAR", utils::IteratorType::parallel)
      .value("RED", utils::IteratorType::reduction)
      .export_values();

  nb::class_<linalg::GenericOp, OpState>(m, "GenericOp")
      .def(
          "__init__",
          [](linalg::GenericOp &self, AlloOpBuilder &builder,
             const std::vector<Type> &resTypes,
             const std::vector<Value> &inputs,
             const std::vector<Value> &outputs,
             const std::vector<AffineMap> &indexingMaps,
             const std::vector<utils::IteratorType> &iteratorTypes) {
            self = linalg::GenericOp::create(builder, builder.getLocation(),
                                             resTypes, inputs, outputs,
                                             indexingMaps, iteratorTypes);
          },
          nb::arg("builder"), nb::arg("result_types"), nb::arg("inputs"),
          nb::arg("outputs"), nb::arg("indexing_maps"),
          nb::arg("iterator_types"))
      .def("get_body", [](linalg::GenericOp &self) { return self.getBody(); })
      .def(
          "add_entry_block",
          [](linalg::GenericOp &self) {
            SmallVector<Type, 4> blockArgTypes;
            SmallVector<Location, 4> blockArgLocs;
            for (ValueRange container : {self.getInputs(), self.getOutputs()}) {
              for (Value v : container) {
                Type t = v.getType();
                blockArgTypes.push_back(isa<MemRefType, RankedTensorType>(t)
                                            ? getElementTypeOrSelf(t)
                                            : t);
                blockArgLocs.push_back(v.getLoc());
              }
            }
            Block *block = &self->getRegion(0).emplaceBlock();
            block->addArguments(blockArgTypes, blockArgLocs);
            return block;
          },
          nb::rv_policy::reference);

  nb::class_<linalg::YieldOp, OpState>(m, "YieldOp")
      .def(
          "__init__",
          [](linalg::YieldOp &self, AlloOpBuilder &builder,
             const std::vector<Value> &values) {
            self =
                linalg::YieldOp::create(builder, builder.getLocation(), values);
          },
          nb::arg("builder"), nb::arg("values"));
}

void bindUBOps(nb::module_ &m) {
  nb::class_<ub::PoisonOp, OpState>(m, "PoisonOp")
      .def(
          "__init__",
          [](ub::PoisonOp &self, AlloOpBuilder &builder, Type &resType) {
            self =
                ub::PoisonOp::create(builder, builder.getLocation(), resType);
          },
          nb::arg("builder"), nb::arg("res_type"));
}

void bindAlloOps(nb::module_ &m) {
  nb::class_<allo::StreamType, Type>(m, "StreamType")
      .def_static(
          "get",
          [](MLIRContext &ctx, Type &baseType, size_t depth,
             const std::vector<int64_t> &shape = {}) {
            return allo::StreamType::get(&ctx, baseType, depth, shape);
          },
          nb::arg("context"), nb::arg("base_type"), nb::arg("depth"),
          nb::arg("shape") = std::vector<int64_t>());

  nb::class_<allo::GetWorkerIdOp, OpState>(m, "GetWorkerIdOp")
      .def(
          "__init__",
          [](allo::GetWorkerIdOp &self, AlloOpBuilder &builder, uint32_t axis) {
            self = allo::GetWorkerIdOp::create(builder, builder.getLocation(),
                                               axis);
          },
          nb::arg("builder"), nb::arg("axis"));

  nb::class_<allo::GetNumWorkersOp, OpState>(m, "GetNumWorkersOp")
      .def(
          "__init__",
          [](allo::GetNumWorkersOp &self, AlloOpBuilder &builder,
             uint32_t axis) {
            self = allo::GetNumWorkersOp::create(builder, builder.getLocation(),
                                                 axis);
          },
          nb::arg("builder"), nb::arg("axis"));

  nb::class_<allo::StreamGetOp, OpState>(m, "StreamGetOp")
      .def(
          "__init__",
          [](allo::StreamGetOp &self, AlloOpBuilder &builder, Value &stream,
             const std::vector<Value> &indices) {
            self = allo::StreamGetOp::create(builder, builder.getLocation(),
                                             stream, indices);
          },
          nb::arg("builder"), nb::arg("stream"), nb::arg("indices"));

  nb::class_<allo::StreamPutOp, OpState>(m, "StreamPutOp")
      .def(
          "__init__",
          [](allo::StreamPutOp &self, AlloOpBuilder &builder, Value &stream,
             const std::vector<Value> &indices, Value &value) {
            self = allo::StreamPutOp::create(builder, builder.getLocation(),
                                             stream, indices, value);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("indices"),
          nb::arg("value"));

  nb::class_<allo::GlobalStreamGetOp, OpState>(m, "GlobalStreamGetOp")
      .def(
          "__init__",
          [](allo::GlobalStreamGetOp &self, AlloOpBuilder &builder,
             Type &resType, std::string_view name,
             const std::vector<Value> &indices) {
            self = allo::GlobalStreamGetOp::create(
                builder, builder.getLocation(), resType, name, indices);
          },
          nb::arg("builder"), nb::arg("res_type"), nb::arg("name"),
          nb::arg("indices"));

  nb::class_<allo::GlobalStreamPutOp, OpState>(m, "GlobalStreamPutOp")
      .def(
          "__init__",
          [](allo::GlobalStreamPutOp &self, AlloOpBuilder &builder,
             std::string_view name, const std::vector<Value> &indices,
             Value &value) {
            self = allo::GlobalStreamPutOp::create(
                builder, builder.getLocation(), name, indices, value);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("indices"),
          nb::arg("value"));

  nb::class_<allo::StreamCreateOp, OpState>(m, "StreamCreateOp")
      .def(
          "__init__",
          [](allo::StreamCreateOp &self, AlloOpBuilder &builder,
             allo::StreamType &streamType) {
            self = allo::StreamCreateOp::create(builder, builder.getLocation(),
                                                streamType);
          },
          nb::arg("builder"), nb::arg("element_type"));

  nb::class_<allo::GlobalStreamCreateOp, OpState>(m, "GlobalStreamCreateOp")
      .def(
          "__init__",
          [](allo::GlobalStreamCreateOp &self, AlloOpBuilder &builder,
             std::string_view name, allo::StreamType &streamType) {
            self = allo::GlobalStreamCreateOp::create(
                builder, builder.getLocation(), streamType, name);
          },
          nb::arg("builder"), nb::arg("name"), nb::arg("stream_type"));
}
