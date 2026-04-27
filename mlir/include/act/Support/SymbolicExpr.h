#ifndef ACT_SUPPORT_SYMBOLIC_EXPR_H
#define ACT_SUPPORT_SYMBOLIC_EXPR_H

#include "act/IR/ActOps.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
#include <optional>
#include <string>

namespace mlir::act {

struct SymExpr {
  enum class Kind { Constant, Param, Add, Mul };
  Kind kind;
  int64_t value = 0;                 // for constant
  unsigned paramIdx = 0;             // for param
  std::shared_ptr<SymExpr> lhs, rhs; // for add, mul

  SymExpr() : kind(Kind::Constant) {}

  static SymExpr constant(int64_t v) {
    SymExpr expr;
    expr.value = v;
    expr.kind = Kind::Constant;
    return expr;
  }

  static SymExpr param(unsigned idx) {
    SymExpr expr;
    expr.paramIdx = idx;
    expr.kind = Kind::Param;
    return expr;
  }

  SymExpr add(const SymExpr &other) const {
    if (kind == Kind::Constant && other.kind == Kind::Constant)
      return constant(value + other.value);
    SymExpr expr;
    expr.kind = Kind::Add;
    expr.lhs = std::make_shared<SymExpr>(*this);
    expr.rhs = std::make_shared<SymExpr>(other);
    return expr;
  }

  SymExpr operator+(const SymExpr &other) const { return add(other); }

  SymExpr mul(const SymExpr &other) const {
    if (kind == Kind::Constant && other.kind == Kind::Constant)
      return constant(value * other.value);
    if (kind == Kind::Constant && value == 1)
      return other;
    if (other.kind == Kind::Constant && other.value == 1)
      return *this;
    SymExpr expr;
    expr.kind = Kind::Mul;
    expr.lhs = std::make_shared<SymExpr>(*this);
    expr.rhs = std::make_shared<SymExpr>(other);
    return expr;
  }

  SymExpr operator*(const SymExpr &other) const { return mul(other); }

  int64_t evaluate(ArrayRef<int64_t> paramValues) const;
  void collectParams(DenseSet<unsigned> &params) const;

  bool isConstant() const { return kind == Kind::Constant; }
  bool isParam() const { return kind == Kind::Param; }

  std::optional<int64_t> getConstantValue() const;
  std::optional<unsigned> getParamIdx() const;
  std::string toString() const;
};

using SymShape = SmallVector<SymExpr>;

FailureOr<SymExpr> buildSymExpr(Value value);
FailureOr<SymExpr> buildSymExpr(OpFoldResult value);

FailureOr<SymShape> generateShapeExpr(Operation *accessOp,
                                      BufferTypeInterface bufferType);

std::string symShapeToString(const SymShape &shape);

} // namespace mlir::act

#endif // ACT_SUPPORT_SYMBOLIC_EXPR_H
