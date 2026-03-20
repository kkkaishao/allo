/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_PYTHON_IR_H
#define ALLO_PYTHON_IR_H

#include "mlir/IR/Builders.h"
#include "nanobind/nanobind.h"

namespace nb = nanobind;

class AlloOpBuilder : public mlir::OpBuilder {
public:
  using OpBuilder::OpBuilder;
  mlir::Location getLocation() const { return loc; }
  void setLocation(mlir::Location newLoc) { loc = newLoc; }
  void setUnknownLoc() { loc = getUnknownLoc(); }
  std::pair<OpBuilder::InsertPoint, mlir::Location>
  getInsertionPointAndLoc() const {
    return {saveInsertionPoint(), loc};
  }
  void setInsertionPointAndLoc(const OpBuilder::InsertPoint &ip,
                               mlir::Location newLoc) {
    restoreInsertionPoint(ip);
    loc = newLoc;
  }

private:
  // default init to unknown
  mlir::Location loc = getUnknownLoc();
};

template <typename Fn>
struct FunctionTraits : FunctionTraits<decltype(&Fn::operator())> {};

template <typename ClassType, typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (ClassType::*)(Args...) const> {
  using return_type = ReturnType;
  using args_tuple = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);
};

template <typename ReturnType, typename... Args>
struct FunctionTraits<ReturnType (*)(Args...)> {
  using return_type = ReturnType;
  using args_tuple = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);
};

template <typename ConcreteOp, typename Base = mlir::OpState>
using OpClass = nb::class_<ConcreteOp, Base>;

template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base> bindOp(nb::module_ &m, const char *pyName) {
  return nb::class_<ConcreteOp, Base>(m, pyName);
}

template <typename ConcreteOp, typename Class, typename BuilderFn,
          std::size_t... I, typename... NbArgs>
inline Class &bindConstructorImpl(Class &cls, BuilderFn &&builderFn,
                                  std::index_sequence<I...>,
                                  NbArgs &&...nbArgs) {
  using FnTraits = FunctionTraits<std::decay_t<BuilderFn>>;
  using ArgsTuple = typename FnTraits::args_tuple;
  using ReturnType = typename FnTraits::return_type;
  static_assert(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<ReturnType>>,
                     ConcreteOp>,
      "builder init lambda must return the concrete op type");
  static_assert(
      std::is_same_v<std::remove_cv_t<std::remove_reference_t<
                         std::tuple_element_t<0, ArgsTuple>>>,
                     AlloOpBuilder>,
      "builder init lambda must take AlloOpBuilder as its first argument");

  return cls.def(
      "__init__",
      [builderFn = std::forward<BuilderFn>(builderFn)](
          ConcreteOp &self, AlloOpBuilder &builder,
          std::tuple_element_t<I + 1, ArgsTuple>... args) {
        self = builderFn(builder, args...);
      },
      nb::arg("builder"), std::forward<NbArgs>(nbArgs)...);
}

template <typename ConcreteOp, typename Class, typename BuilderFn,
          typename... NbArgs>
inline Class &bindConstructor(Class &cls, BuilderFn &&builderFn,
                              NbArgs &&...nbArgs) {
  using FnTraits = FunctionTraits<std::decay_t<BuilderFn>>;
  static_assert(FnTraits::arity >= 1,
                "builder init lambda must take AlloOpBuilder");
  return bindConstructorImpl<ConcreteOp>(
      cls, std::forward<BuilderFn>(builderFn),
      std::make_index_sequence<FnTraits::arity - 1>{},
      std::forward<NbArgs>(nbArgs)...);
}

template <typename ConcreteOp, typename Base, typename BuilderFn,
          typename... NbArgs>
inline OpClass<ConcreteOp, Base> &
bindConstructor(OpClass<ConcreteOp, Base> &cls, BuilderFn &&builderFn,
                NbArgs &&...nbArgs) {
  using FnTraits = FunctionTraits<std::decay_t<BuilderFn>>;
  static_assert(FnTraits::arity >= 1,
                "builder init lambda must take AlloOpBuilder");
  return bindConstructorImpl<ConcreteOp>(
      cls, std::forward<BuilderFn>(builderFn),
      std::make_index_sequence<FnTraits::arity - 1>{},
      std::forward<NbArgs>(nbArgs)...);
}

template <typename Class, typename Getter>
inline Class &bindGetter(Class &cls, const char *name, Getter &&getter) {
  return cls.def(name, std::forward<Getter>(getter), nb::rv_policy::reference);
}

///===--------------------------------------------------------------------===//
/// Common op builder patterns
///===--------------------------------------------------------------------===//

// Bind a unary op that takes a single mlir::Value operand
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base> bindUnaryValueOp(nb::module_ &m,
                                                  const char *pyName,
                                                  const char *argName = "val") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &value) {
        return ConcreteOp::create(builder, builder.getLocation(), value);
      },
      nb::arg(argName));
  return cls;
}

// Bind a binary op that takes two mlir::Value operands
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindBinaryValueOp(nb::module_ &m, const char *pyName,
                  const char *lhsName = "lhs", const char *rhsName = "rhs") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs) {
        return ConcreteOp::create(builder, builder.getLocation(), lhs, rhs);
      },
      nb::arg(lhsName), nb::arg(rhsName));
  return cls;
}

// Bind an op that takes a mlir::Value operand and a mlir::ValueRange operand
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindValueRangeOp(nb::module_ &m, const char *pyName,
                 const char *valueName = "value",
                 const char *rangeName = "values") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &value,
         const std::vector<mlir::Value> &range) {
        return ConcreteOp::create(builder, builder.getLocation(), value, range);
      },
      nb::arg(valueName), nb::arg(rangeName));
  return cls;
}

// Bind an op that takes two mlir::Value operands and a mlir::ValueRange operand
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base> bindTwoValueRangeOp(
    nb::module_ &m, const char *pyName, const char *firstName = "first",
    const char *secondName = "second", const char *rangeName = "values") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &first, mlir::Value &second,
         const std::vector<mlir::Value> &range) {
        return ConcreteOp::create(builder, builder.getLocation(), first, second,
                                  range);
      },
      nb::arg(firstName), nb::arg(secondName), nb::arg(rangeName));
  return cls;
}

// Bind an op that takes a mlir::Value operand
// and a mlir::Value operand as init value
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindUnaryInitOp(nb::module_ &m, const char *pyName,
                const char *inputName = "input",
                const char *initName = "init") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &input, mlir::Value &init) {
        return ConcreteOp::create(builder, builder.getLocation(), input, init);
      },
      nb::arg(inputName), nb::arg(initName));
  return cls;
}

// Bind an op that takes two mlir::Value operands
// and a mlir::Value operand as init value
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindBinaryInputsInitOp(nb::module_ &m, const char *pyName,
                       const char *lhsName = "lhs", const char *rhsName = "rhs",
                       const char *initName = "init") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &lhs, mlir::Value &rhs,
         mlir::Value &init) {
        return ConcreteOp::create(builder, builder.getLocation(),
                                  std::initializer_list<mlir::Value>{lhs, rhs},
                                  init);
      },
      nb::arg(lhsName), nb::arg(rhsName), nb::arg(initName));
  return cls;
}

// Bind an op that takes a mlir::Value operand and a mlir::Type operand,
// where the value is the source and the type is the destination type (e.g. a
// cast op)
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindSourceToTypeOp(nb::module_ &m, const char *pyName,
                   const char *srcName = "src",
                   const char *dstTypeName = "dst_type") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Value &src, mlir::Type &dstType) {
        return ConcreteOp::create(builder, builder.getLocation(), dstType, src);
      },
      nb::arg(srcName), nb::arg(dstTypeName));
  return cls;
}

// Bind an op that takes a mlir::Type operand and a mlir::Value operand,
// where the type is the destination type and the value is the source (e.g. a
// cast op)
template <typename ConcreteOp, typename Base = mlir::OpState>
inline OpClass<ConcreteOp, Base>
bindTypeToSourceOp(nb::module_ &m, const char *pyName,
                   const char *dstTypeName = "dst_type",
                   const char *srcName = "src") {
  auto cls = bindOp<ConcreteOp, Base>(m, pyName);
  bindConstructor(
      cls,
      [](AlloOpBuilder &builder, mlir::Type &dstType, mlir::Value &src) {
        return ConcreteOp::create(builder, builder.getLocation(), dstType, src);
      },
      nb::arg(dstTypeName), nb::arg(srcName));
  return cls;
}

void bindIR(nb::module_ &m);
void bindMathOps(nb::module_ &m);
void bindArithOps(nb::module_ &m);
void bindSCFOps(nb::module_ &m);
void bindCFOps(nb::module_ &m);
void bindFuncOps(nb::module_ &m);
void bindAffineOps(nb::module_ &m);
void bindTensorOps(nb::module_ &m);
void bindMemRefOps(nb::module_ &m);
void bindLinalgOps(nb::module_ &m);
void bindTransform(nb::module_ &m);
void bindUtils(nb::module_ &m);
void bindUBOps(nb::module_ &m);
void bindAlloOps(nb::module_ &m);
void bindPasses(nb::module_ &m);

inline mlir::OpPrintingFlags getOpPrintingFlags(bool debug = false) {
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.enableDebugInfo(debug);
  printingFlags.printNameLocAsPrefix(true);
  printingFlags.printGenericOpForm(false);
  return printingFlags;
}

#endif // ALLO_PYTHON_IR_H
