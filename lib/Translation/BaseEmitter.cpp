#include "allo/Translation/BaseEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::allo;

SmallString<8> AlloBaseEmitter::addName(Value val, bool isPtr,
                                        const std::string &name) const {
  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  if (!name.empty()) {
    if (state.nameConflictCnt.count(name) > 0) {
      state.nameConflictCnt[name]++;
      valName += name + std::to_string(state.nameConflictCnt[name]);
    } else { // first time
      state.nameConflictCnt[name] = 0;
      valName += name;
    }
  } else {
    valName += StringRef("v" + std::to_string(state.nameTable.size()));
  }
  state.nameTable[val] = valName;
  return valName;
}

SmallString<8> AlloBaseEmitter::getName(Value val) {
  if (auto defOp = val.getDefiningOp()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      auto constAttr = constOp.getValue();

      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(constAttr)) {
        return {std::to_string(boolAttr.getValue())};
      }
      if (auto floatAttr = llvm::dyn_cast<FloatAttr>(constAttr)) {
        unsigned bitwidth =
            llvm::dyn_cast<FloatType>(floatAttr.getType()).getWidth();
        std::string prefix = (bitwidth == 32) ? "(float)" : "(double)";
        if (auto value = floatAttr.getValueAsDouble(); std::isfinite(value))
          return {prefix + std::to_string(value)};
        else if (value > 0)
          return {"INFINITY"};
        return {"-INFINITY"};
      }
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constAttr)) {
        auto value = intAttr.getInt();
        return {std::to_string(value)};
      }
    }
  }
  return state.nameTable.lookup(val);
}