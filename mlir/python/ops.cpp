#include "nanobind/nanobind.h"
#include "nanobind/stl/function.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace nb = nanobind;
using namespace mlir;

OpPrintingFlags getOpPrintingFlags();

void init_ub_ops(nb::module_ &m) {
  nb::class_<ub::PoisonOp, OpState>(m, "PoisonOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Type &type) {
            return ub::PoisonOp::create(builder, loc, type);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("type"));
}

void init_func_ops(nb::module_ &m) {
  nb::class_<func::FuncOp, OpState>(m, "FuncOp")
      .def_static("create",
                  [](OpBuilder &builder, Location &loc, const std::string &name,
                     FunctionType &type) {
                    return func::FuncOp::create(builder, loc, name, type);
                  })
      .def(
          "get_arg_at",
          [](func::FuncOp &self, unsigned idx) -> BlockArgument {
            if (idx >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            return self.getArgument(idx);
          },
          nb::arg("idx"))
      .def("get_num_args", &func::FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); },
          nb::rv_policy::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, unsigned arg_no, const std::string &name,
             Attribute &attr) {
            if (arg_no >= self.getNumArguments())
              throw nb::index_error("Function argument index out of range");
            // set arg attributes "name" to Value &"val"
            self.setArgAttr(arg_no, name, attr);
          },
          nb::arg("arg_no"), nb::arg("name"), nb::arg("attr"))
      .def("get_func_type", &func::FuncOp::getFunctionType)
      .def("set_type", &func::FuncOp::setType, nb::arg("type"))
      .def("get_func_name",
           [](func::FuncOp &self) { return self.getName().str(); });

  nb::class_<func::ReturnOp, OpState>(m, "ReturnOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc,
             const std::vector<Value> &operands) {
            return func::ReturnOp::create(builder, loc, operands);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("operands"));

  nb::class_<func::CallOp, OpState>(m, "CallOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, func::FuncOp &func,
             const std::vector<Value> &args) {
            return func::CallOp::create(builder, loc, func, args);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("func"), nb::arg("args"));
}

void init_affine_ops(nb::module_ &m) {
  // affine ops
  nb::class_<affine::AffineForOp, OpState>(m, "AffineForOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, const std::vector<Value> &lb,
             AffineMap lbMap, const std::vector<Value> &ub, AffineMap ubMap,
             int64_t step = 1) {
            return affine::AffineForOp::create(builder, loc, lb, lbMap, ub,
                                               ubMap, step);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lb_operands"),
          nb::arg("lb_map"), nb::arg("ub_operands"), nb::arg("ub_map"),
          nb::arg("step") = 1)
      .def_static("create",
                  [](OpBuilder &builder, Location &loc, int64_t lb, int64_t ub,
                     int64_t step = 1) {
                    return affine::AffineForOp::create(builder, loc, lb, ub,
                                                       step);
                  })
      .def("get_induction_var", &affine::AffineForOp::getInductionVar)
      .def(
          "body", [](affine::AffineForOp &self) { return self.getBody(); },
          nb::rv_policy::reference)
      .def("get_upper_bound", &affine::AffineForOp::getUpperBound)
      .def("get_lower_bound", &affine::AffineForOp::getLowerBound)
      .def("get_constant_upper_bound",
           &affine::AffineForOp::getConstantUpperBound)
      .def("get_constant_lower_bound",
           &affine::AffineForOp::getConstantLowerBound)
      .def("has_constant_lower_bound",
           &affine::AffineForOp::hasConstantLowerBound)
      .def("has_constant_upper_bound",
           &affine::AffineForOp::hasConstantUpperBound)
      .def("get_step", &affine::AffineForOp::getStepAsInt)
      .def("has_constant_bounds", &affine::AffineForOp::hasConstantBounds);

  nb::class_<affine::AffineIfOp, OpState>(m, "AffineIfOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, IntegerSet set,
             const std::vector<Value> &operands, bool withElse = false) {
            return affine::AffineIfOp::create(builder, loc, set, operands,
                                              withElse);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("set"),
          nb::arg("operands"), nb::arg("with_else") = false)
      .def("get_integer_set", &affine::AffineIfOp::getIntegerSet)
      .def("get_then_block", &affine::AffineIfOp::getThenBlock,
           nb::rv_policy::reference)
      .def("get_else_block", &affine::AffineIfOp::getElseBlock,
           nb::rv_policy::reference);
}

void init_scf_ops(nb::module_ &m) {
  // scf ops
  nb::class_<scf::ForOp, OpState>(m, "ForOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lb, Value &ub,
             Value &step, const std::vector<Value> &initArgs = {}) {
            return scf::ForOp::create(builder, loc, lb, ub, step, initArgs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lb"), nb::arg("ub"),
          nb::arg("step"), nb::arg("init_args") = std::vector<Value>())
      .def("get_induction_var", &scf::ForOp::getInductionVar)
      .def("get_body", &scf::ForOp::getBody, nb::rv_policy::reference);

  nb::class_<scf::IfOp, OpState>(m, "IfOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc,
             const std::vector<Type> &resultTypes, Value &cond,
             bool withElse = false) {
            return scf::IfOp::create(builder, loc, resultTypes, cond, withElse);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("res_types"),
          nb::arg("cond"), nb::arg("with_else") = false)
      .def("get_then_block", &scf::IfOp::thenBlock, nb::rv_policy::reference)
      .def("get_else_block", &scf::IfOp::elseBlock, nb::rv_policy::reference)
      .def("get_then_yield", &scf::IfOp::thenYield)
      .def("get_else_yield", &scf::IfOp::elseYield);
  nb::class_<scf::YieldOp, OpState>(m, "YieldOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc,
             const std::vector<Value> &results) {
            return scf::YieldOp::create(builder, loc, results);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("results"));
  nb::class_<scf::WhileOp, OpState>(m, "WhileOp")
      .def_static("create",
                  [](OpBuilder &builder, Location &loc,
                     const std::vector<Type> &resultTypes,
                     const std::vector<Value> &operands) {
                    return scf::WhileOp::create(builder, loc, resultTypes,
                                                operands);
                  })
      .def("get_before", &scf::WhileOp::getBefore, nb::rv_policy::reference)
      .def("get_after", &scf::WhileOp::getAfter, nb::rv_policy::reference);
  nb::class_<scf::ConditionOp, OpState>(m, "ConditionOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &cond,
             const std::vector<Value> &args) {
            return scf::ConditionOp::create(builder, loc, cond, args);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("cond"), nb::arg("args"));
}

void init_cf_ops(nb::module_ &m) {
  nb::class_<cf::CondBranchOp, OpState>(m, "CondBranchOp")
      .def_static("create", [](OpBuilder &builder, Location &loc, Value &cond,
                               Block *trueDest, Block *falseDest) {
        return cf::CondBranchOp::create(builder, loc, cond, trueDest,
                                        falseDest);
      });

  nb::class_<cf::BranchOp, OpState>(m, "BranchOp")
      .def_static("create", [](OpBuilder &builder, Location &loc, Block *dest,
                               const std::vector<Value> &args) {
        return cf::BranchOp::create(builder, loc, dest, args);
      });
}

void init_arith_ops(nb::module_ &m) {
  // constant ops
  (void)nb::class_<arith::ConstantOp, OpState>(m, "ConstantOp");

  nb::class_<arith::ConstantIntOp, arith::ConstantOp>(m, "ConstantIntOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, IntegerType &type,
             int64_t value) {
            return arith::ConstantIntOp::create(builder, loc, type, value);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("type"),
          nb::arg("value"));

  nb::class_<arith::ConstantFloatOp, arith::ConstantOp>(m, "ConstantFloatOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Float32Type &type,
             float value) {
            return arith::ConstantFloatOp::create(builder, loc, type,
                                                  APFloat(value));
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("type"), nb::arg("value"))
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Float64Type &type,
             double value) {
            return arith::ConstantFloatOp::create(builder, loc, type,
                                                  APFloat(value));
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("type"), nb::arg("value"))
      .def_static("create",
                  [](OpBuilder &builder, Location &loc, Float16Type &type,
                     float value) {
                    return arith::ConstantFloatOp::create(builder, loc, type,
                                                          APFloat(value));
                  })
      .def_static("create", [](OpBuilder &builder, Location &loc,
                               BFloat16Type &type, float value) {
        // bf16 does not satisfy IEEE754, so we need to convert manually
        const llvm::fltSemantics &sem = type.getFloatSemantics();
        llvm::APFloat val(value);
        bool lost;
        val.convert(sem, llvm::APFloat::rmNearestTiesToEven, &lost);
        return arith::ConstantFloatOp::create(builder, loc, type, val);
      });

  nb::class_<arith::ConstantIndexOp, arith::ConstantOp>(m, "ConstantIndexOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, int64_t value) {
            return arith::ConstantIndexOp::create(builder, loc, value);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("value"));

  // casts / conversions
  nb::class_<arith::SIToFPOp, OpState>(m, "SIToFPOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::SIToFPOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::UIToFPOp, OpState>(m, "UIToFPOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::UIToFPOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::FPToSIOp, OpState>(m, "FPToSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::FPToSIOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::FPToUIOp, OpState>(m, "FPToUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::FPToUIOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::ExtFOp, OpState>(m, "ExtFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::ExtFOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::TruncFOp, OpState>(m, "TruncFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::TruncFOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::IndexCastOp, OpState>(m, "IndexCastOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Type &dstType, Value &src) {
            return arith::IndexCastOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("dst_type"),
          nb::arg("src"));

  // integer extension / truncation / bitcast
  nb::class_<arith::ExtSIOp, OpState>(m, "ExtSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::ExtSIOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::ExtUIOp, OpState>(m, "ExtUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::ExtUIOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::BitcastOp, OpState>(m, "BitcastOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::BitcastOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  nb::class_<arith::TruncIOp, OpState>(m, "TruncIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &src, Type &dstType) {
            return arith::TruncIOp::create(builder, loc, dstType, src);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("src"),
          nb::arg("dst_type"));

  // floating ops
  nb::class_<arith::AddFOp, OpState>(m, "AddFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::AddFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SubFOp, OpState>(m, "SubFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::SubFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MulFOp, OpState>(m, "MulFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MulFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivFOp, OpState>(m, "DivFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::DivFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemFOp, OpState>(m, "RemFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::RemFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::NegFOp, OpState>(m, "NegFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input) {
            return arith::NegFOp::create(builder, loc, input);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"));

  // integer arithmetic
  nb::class_<arith::AddIOp, OpState>(m, "AddIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::AddIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SubIOp, OpState>(m, "SubIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::SubIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MulIOp, OpState>(m, "MulIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MulIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivSIOp, OpState>(m, "DivSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::DivSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::DivUIOp, OpState>(m, "DivUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::DivUIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::CeilDivSIOp, OpState>(m, "CeilDivSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CeilDivSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::CeilDivUIOp, OpState>(m, "CeilDivUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CeilDivUIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::FloorDivSIOp, OpState>(m, "FloorDivSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::FloorDivSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemSIOp, OpState>(m, "RemSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::RemSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::RemUIOp, OpState>(m, "RemUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::RemUIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  // fused / special ops
  nb::class_<math::FmaOp, OpState>(m, "FmaOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &a, Value &b, Value &c) {
            return math::FmaOp::create(builder, loc, a, b, c);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("a"), nb::arg("b"),
          nb::arg("c"));

  // shifts
  nb::class_<arith::ShLIOp, OpState>(m, "ShLIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::ShLIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::ShRUIOp, OpState>(m, "ShRUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::ShRUIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::ShRSIOp, OpState>(m, "ShRSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::ShRSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  // mins / maxs
  nb::class_<arith::MinSIOp, OpState>(m, "MinSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MinSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinUIOp, OpState>(m, "MinUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MinUIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinimumFOp, OpState>(m, "MinimumFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MinimumFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MinNumFOp, OpState>(m, "MinNumFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MinNumFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxSIOp, OpState>(m, "MaxSIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MaxSIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxUIOp, OpState>(m, "MaxUIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MaxUIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaximumFOp, OpState>(m, "MaximumFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MaximumFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::MaxNumFOp, OpState>(m, "MaxNumFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::MaxNumFOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  // comparisons (int)
  nb::class_<arith::CmpIOp, OpState>(m, "CmpIOp")
      .def_static(
          "create_icmpSLE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::sle, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpSLT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::slt, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpSGE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::sge, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpSGT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::sgt, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpULE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::ule, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpULT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::ult, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpUGE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::uge, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpUGT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc,
                                         arith::CmpIPredicate::ugt, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpEQ",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_icmpNE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                         lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, unsigned pred, Value &lhs,
             Value &rhs) {
            return arith::CmpIOp::create(
                builder, loc, static_cast<arith::CmpIPredicate>(pred), lhs,
                rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("pred"), nb::arg("lhs"),
          nb::arg("rhs"));

  // comparisons (float)
  nb::class_<arith::CmpFOp, OpState>(m, "CmpFOp")
      .def_static(
          "create_fcmpOLT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::OLT, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpOGT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::OGT, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpOLE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::OLE, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpOGE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::OGE, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpOEQ",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::OEQ, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpONE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::ONE, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpULT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::ULT, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpUGT",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::UGT, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpULE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::ULE, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpUGE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::UGE, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpUEQ",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::UEQ, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create_fcmpUNE",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::CmpFOp::create(builder, loc,
                                         arith::CmpFPredicate::UNE, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"))
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, unsigned pred, Value &lhs,
             Value &rhs) {
            return arith::CmpFOp::create(
                builder, loc, static_cast<arith::CmpFPredicate>(pred), lhs,
                rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("pred"), nb::arg("lhs"),
          nb::arg("rhs"));

  // logical
  nb::class_<arith::AndIOp, OpState>(m, "AndIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::AndIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::XOrIOp, OpState>(m, "XOrIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::XOrIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::OrIOp, OpState>(m, "OrIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs) {
            return arith::OrIOp::create(builder, loc, lhs, rhs);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"));

  nb::class_<arith::SelectOp, OpState>(m, "SelectOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &condition,
             Value &trueValue, Value &falseValue) {
            return arith::SelectOp::create(builder, loc, condition, trueValue,
                                           falseValue);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("condition"),
          nb::arg("true_value"), nb::arg("false_value"));

  m.def(
      "create_int_cast",
      [](OpBuilder &builder, Location &loc, Value &src, Type &dstType,
         bool isSigned) -> Value {
        // get element type if necessary
        Type srcType = src.getType();
        auto srcTensorType = dyn_cast<RankedTensorType>(srcType);
        auto dstTensorType = dyn_cast<RankedTensorType>(dstType);
        Type srcEltType = srcType;
        Type dstEltType = dstType;
        if (dstTensorType && srcTensorType) {
          dstEltType = dstTensorType.getElementType();
          srcEltType = srcTensorType.getElementType();
        }
        unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
        unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
        if (srcWidth == dstWidth)
          return arith::BitcastOp::create(builder, loc, dstType, src);
        else if (srcWidth > dstWidth)
          return arith::TruncIOp::create(builder, loc, dstType, src);
        else if (isSigned)
          return arith::ExtSIOp::create(builder, loc, dstType, src);
        else
          return arith::ExtUIOp::create(builder, loc, dstType, src);
      },
      nb::arg("builder"), nb::arg("loc"), nb::arg("src"), nb::arg("dst_type"),
      nb::arg("is_signed"));
}

void init_math_ops(nb::module_ &m) {
  nb::class_<math::FloorOp, OpState>(m, "FloorOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::FloorOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::CeilOp, OpState>(m, "CeilOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::CeilOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));

  nb::class_<math::ExpOp, OpState>(m, "ExpOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::ExpOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::Exp2Op, OpState>(m, "Exp2Op")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::Exp2Op::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::CosOp, OpState>(m, "CosOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::CosOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::SinOp, OpState>(m, "SinOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::SinOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::LogOp, OpState>(m, "LogOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::LogOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::Log2Op, OpState>(m, "Log2Op")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::Log2Op::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::ErfOp, OpState>(m, "ErfOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::ErfOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::SqrtOp, OpState>(m, "SqrtOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::SqrtOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::RsqrtOp, OpState>(m, "RsqrtOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::RsqrtOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::AbsFOp, OpState>(m, "AbsFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::AbsFOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
  nb::class_<math::AbsIOp, OpState>(m, "AbsIOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::AbsIOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));

  nb::class_<math::PowFOp, OpState>(m, "PowFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &base, Value &exponent) {
            return math::PowFOp::create(builder, loc, base, exponent);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("base"),
          nb::arg("exponent"));

  nb::class_<math::TanOp, OpState>(m, "TanOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &val) {
            return math::TanOp::create(builder, loc, val);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("val"));
}

void init_tensor_ops(nb::module_ &m) {
  nb::class_<tensor::ExtractOp, OpState>(m, "ExtractOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &tensor,
             const std::vector<Value> &indices) {
            return tensor::ExtractOp::create(builder, loc, tensor, indices);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("tensor"),
          nb::arg("indices"));

  nb::class_<tensor::InsertOp, OpState>(m, "InsertOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &value, Value &tensor,
             const std::vector<Value> &indices) {
            return tensor::InsertOp::create(builder, loc, value, tensor,
                                            indices);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("value"),
          nb::arg("tensor"), nb::arg("indices"));

  nb::class_<tensor::SplatOp, OpState>(m, "SplatOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &value,
             const std::vector<int64_t> &shape) {
            return tensor::SplatOp::create(builder, loc, value, shape);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("value"),
          nb::arg("shape"));

  nb::class_<tensor::CastOp, OpState>(m, "CastOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Type &dstType) {
            return tensor::CastOp::create(builder, loc, dstType, input);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("dst_type"));

  nb::class_<tensor::EmptyOp, OpState>(m, "EmptyOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc,
             const std::vector<int64_t> &shape, Type &elementType) {
            return tensor::EmptyOp::create(builder, loc, shape, elementType);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("shape"),
          nb::arg("element_type"))
      .def_static("create", [](OpBuilder &builder, Location &loc, Type &type) {
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
          return tensor::EmptyOp::create(builder, loc, tensor.getShape(),
                                         tensor.getElementType(),
                                         tensor.getEncoding());
        }
        if (auto memref = dyn_cast<MemRefType>(type)) {
          return tensor::EmptyOp::create(builder, loc, memref.getShape(),
                                         memref.getElementType(),
                                         memref.getMemorySpace());
        }
        throw nb::type_error("Unsupported type for tensor.EmptyOp");
      });
}

void init_memref_ops(nb::module_ &m) {
  nb::class_<memref::LoadOp, OpState>(m, "LoadOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &memref,
             const std::vector<Value> &indices) {
            return memref::LoadOp::create(builder, loc, memref, indices);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("memref"),
          nb::arg("indices"));
  nb::class_<memref::StoreOp, OpState>(m, "StoreOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &value, Value &memref,
             const std::vector<Value> &indices) {
            return memref::StoreOp::create(builder, loc, value, memref,
                                           indices);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("value"),
          nb::arg("memref"), nb::arg("indices"));
}

void init_linalg_ops(nb::module_ &m) {
  nb::class_<linalg::MatmulOp, OpState>(m, "MatmulOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &result) {
            return linalg::MatmulOp::create(builder, loc, {lhs, rhs}, result);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("result"));

  nb::class_<linalg::FillOp, OpState>(m, "FillOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &value, Value &output) {
            return linalg::FillOp::create(builder, loc, value, output);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("value"),
          nb::arg("output"));

  nb::class_<linalg::BroadcastOp, OpState>(m, "BroadcastOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init,
             const std::vector<int64_t> &dims) {
            return linalg::BroadcastOp::create(builder, loc, input, init, dims);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"), nb::arg("init"),
          nb::arg("dims"));

  nb::class_<linalg::AddOp, OpState>(m, "AddOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &init) {
            return linalg::AddOp::create(builder, loc, {lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("init"));

  nb::class_<linalg::SubOp, OpState>(m, "SubOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &init) {
            return linalg::SubOp::create(builder, loc, {lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("init"));

  nb::class_<linalg::MulOp, OpState>(m, "MulOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &init) {
            return linalg::MulOp::create(builder, loc, {lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("init"));

  nb::class_<linalg::DivOp, OpState>(m, "DivOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &init) {
            return linalg::DivOp::create(builder, loc, {lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("init"));

  nb::class_<linalg::DivUnsignedOp, OpState>(m, "DivUnsignedOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &init) {
            return linalg::DivUnsignedOp::create(builder, loc, {lhs, rhs},
                                                 init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("init"));

  nb::class_<linalg::PowFOp, OpState>(m, "PowFOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &base, Value &exponent,
             Value &init) {
            return linalg::PowFOp::create(builder, loc, {base, exponent}, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("base"),
          nb::arg("exponent"), nb::arg("init"));

  nb::class_<linalg::FloorOp, OpState>(m, "FloorOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::FloorOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::ExpOp, OpState>(m, "ExpOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::ExpOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::LogOp, OpState>(m, "LogOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::LogOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::SqrtOp, OpState>(m, "SqrtOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::SqrtOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::ReciprocalOp, OpState>(m, "ReciprocalOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::ReciprocalOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::RsqrtOp, OpState>(m, "RsqrtOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::RsqrtOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::SquareOp, OpState>(m, "SquareOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &input, Value &init) {
            return linalg::SquareOp::create(builder, loc, input, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("input"),
          nb::arg("init"));

  nb::class_<linalg::DotOp, OpState>(m, "DotOp")
      .def_static(
          "create",
          [](OpBuilder &builder, Location &loc, Value &lhs, Value &rhs,
             Value &init) {
            return linalg::DotOp::create(builder, loc, {lhs, rhs}, init);
          },
          nb::arg("builder"), nb::arg("loc"), nb::arg("lhs"), nb::arg("rhs"),
          nb::arg("init"));
}