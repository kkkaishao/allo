/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITTERSTATE_H
#define ALLO_TRANSLATION_EMITTERSTATE_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::allo {

struct EmitterState {
  llvm::raw_ostream &os;
  std::size_t currentIndent = 0;
  std::size_t indentSize = 2;
  DenseMap<Value, std::string> nameTable;
  unsigned indexWidth = 32;
  bool withLocation = false;
  bool failed = false;
  bool enabledApFloat = false;

  explicit EmitterState(llvm::raw_ostream &os) : os(os) {}

  void addIndent() { currentIndent += indentSize; }
  void reduceIndent() { currentIndent -= indentSize; }
  void setIndentSize(std::size_t size) { indentSize = size; }

  std::string addName(Value v) {
    assert(!nameTable.contains(v) &&
           "Value already has a name in the name table");
    std::string name = "v" + std::to_string(nameTable.size());
    nameTable[v] = name;
    return name;
  }

  std::string getOrAddName(Value v) {
    auto it = nameTable.find(v);
    if (it != nameTable.end())
      return it->second;
    return addName(v);
  }

  std::string getName(Value v) const {
    auto it = nameTable.find(v);
    assert(it != nameTable.end() && "Value not found in name table");
    return it->second;
  }

  bool hasName(Value v) const { return nameTable.contains(v); }
};
} // namespace mlir::allo

#endif // ALLO_TRANSLATION_EMITTERSTATE_H
