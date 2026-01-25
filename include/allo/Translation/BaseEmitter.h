#ifndef ALLO_BASEEMITTER_H
#define ALLO_BASEEMITTER_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>

namespace mlir::allo {
struct AlloEmitterState {
  explicit AlloEmitterState(llvm::raw_ostream &os) : os(os) {}
  AlloEmitterState(const AlloEmitterState &) = delete;
  void operator=(const AlloEmitterState &) = delete;

  llvm::raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  llvm::DenseMap<mlir::Value, llvm::SmallString<8>> nameTable;
  std::unordered_map<std::string, int> nameConflictCnt;

  bool linearizePtrs = false;
  llvm::DenseSet<mlir::Value> topLevelFuncArgs;
};

struct AlloBaseEmitter {
  explicit AlloBaseEmitter(AlloEmitterState &state)
      : state(state), os(state.os) {}
  AlloBaseEmitter(const AlloBaseEmitter &) = delete;
  void operator=(const AlloBaseEmitter &) = delete;

  llvm::raw_ostream &indent() const { return os.indent(state.currentIndent); }

  void addIndent() const { state.currentIndent += 2; }

  void reduceIndent() const {
    if (state.currentIndent >= 2) {
      state.currentIndent -= 2;
    }
  }

  mlir::InFlightDiagnostic emitError(mlir::Operation *op,
                                     const llvm::Twine &message) const {
    state.encounteredError = true;
    return op->emitError(message);
  }

  AlloEmitterState &state;
  llvm::raw_ostream &os;

  /// Value name management methods.
  llvm::SmallString<8> addName(mlir::Value val, bool isPtr = false,
                               const std::string &name = "") const;

  llvm::SmallString<8> getName(mlir::Value val);
};
} // namespace mlir::allo

#endif // ALLO_BASEEMITTER_H
