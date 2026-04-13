#ifndef ACT_SUPPORT_SYMBOLIC_EXPR_H
#define ACT_SUPPORT_SYMBOLIC_EXPR_H

#include "act/IR/ActOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>

namespace mlir::act {

/// A tree representing integer expressions over addr parameters.
/// Supports Constant, Param (addr block arg), Add, and Mul.
struct SymExpr {
  enum Kind { Constant, Param, Add, Mul };
  Kind kind;
  int64_t value = 0;                 // Constant
  unsigned paramIdx = 0;             // Param: index into addr block args
  std::shared_ptr<SymExpr> lhs, rhs; // Add, Mul

  static SymExpr constant(int64_t v) {
    SymExpr e;
    e.kind = Constant;
    e.value = v;
    return e;
  }

  static SymExpr param(unsigned idx) {
    SymExpr e;
    e.kind = Param;
    e.paramIdx = idx;
    return e;
  }

  static SymExpr add(SymExpr a, SymExpr b) {
    // Constant fold
    if (a.kind == Constant && b.kind == Constant)
      return constant(a.value + b.value);
    SymExpr e;
    e.kind = Add;
    e.lhs = std::make_shared<SymExpr>(std::move(a));
    e.rhs = std::make_shared<SymExpr>(std::move(b));
    return e;
  }

  static SymExpr mul(SymExpr a, SymExpr b) {
    // Constant fold
    if (a.kind == Constant && b.kind == Constant)
      return constant(a.value * b.value);
    // Multiply by 1 identity
    if (a.kind == Constant && a.value == 1)
      return b;
    if (b.kind == Constant && b.value == 1)
      return a;
    SymExpr e;
    e.kind = Mul;
    e.lhs = std::make_shared<SymExpr>(std::move(a));
    e.rhs = std::make_shared<SymExpr>(std::move(b));
    return e;
  }

  /// Evaluate with concrete parameter values.
  int64_t evaluate(ArrayRef<int64_t> paramValues) const {
    switch (kind) {
    case Constant:
      return value;
    case Param:
      assert(paramIdx < paramValues.size() && "param index out of bounds");
      return paramValues[paramIdx];
    case Add:
      return lhs->evaluate(paramValues) + rhs->evaluate(paramValues);
    case Mul:
      return lhs->evaluate(paramValues) * rhs->evaluate(paramValues);
    }
    llvm_unreachable("unknown SymExpr kind");
  }

  /// Collect all param indices that appear in this expression.
  void collectParams(DenseSet<unsigned> &out) const {
    switch (kind) {
    case Constant:
      return;
    case Param:
      out.insert(paramIdx);
      return;
    case Add:
    case Mul:
      lhs->collectParams(out);
      rhs->collectParams(out);
      return;
    }
  }

  bool isConstant() const { return kind == Constant; }
  bool isParam() const { return kind == Param; }

  /// Return the constant value, or std::nullopt if not a constant.
  std::optional<int64_t> getConstantValue() const {
    if (kind == Constant)
      return value;
    return std::nullopt;
  }

  /// Return the param index, or std::nullopt if not a simple param.
  std::optional<unsigned> getParamIdx() const {
    if (kind == Param)
      return paramIdx;
    return std::nullopt;
  }

  /// Pretty-print for debugging.
  std::string toString() const {
    switch (kind) {
    case Constant:
      return "Const(" + std::to_string(value) + ")";
    case Param:
      return "Param<" + std::to_string(paramIdx) + ">";
    case Add:
      return "(" + lhs->toString() + " + " + rhs->toString() + ")";
    case Mul:
      return "(" + lhs->toString() + " * " + rhs->toString() + ")";
    }
    llvm_unreachable("unknown SymExpr kind");
  }
};

/// Symbolic tensor shape: one SymExpr per dimension.
using SymShape = SmallVector<SymExpr>;

/// Build a SymExpr from an SSA Value in the addr region.
/// Traces backward through arith ops and block arguments.
FailureOr<SymExpr> buildSymExpr(Value v);

/// Build a SymExpr from a mixed static/dynamic value (OpFoldResult).
FailureOr<SymExpr> buildSymExpr(OpFoldResult ofr);

/// Generate the symbolic shape expression for an access pattern op.
/// Recursively walks the access chain (relayout ops delegate to their source).
/// Returns the shape of the tensor that the compute region will see.
FailureOr<SymShape> generateShapeExpr(Operation *accessOp,
                                      BufferTypeInterface bufferType);

/// Debug: print a SymShape.
inline std::string symShapeToString(const SymShape &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      result += ", ";
    result += shape[i].toString();
  }
  result += "]";
  return result;
}

} // namespace mlir::act

#endif // ACT_SUPPORT_SYMBOLIC_EXPR_H
