/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSLATION_EMITTERSTATE_H
#define ALLO_TRANSLATION_EMITTERSTATE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <optional>
#include <string>

namespace mlir::allo {

// Turn an arbitrary source name into a valid C++ identifier: characters that
// are not alphanumeric or '_' become '_', and a leading digit (or an empty
// result) is prefixed with '_'.
inline std::string sanitizeCppIdentifier(llvm::StringRef name) {
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    unsigned char uc = static_cast<unsigned char>(c);
    result.push_back(std::isalnum(uc) || c == '_' ? c : '_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

// C++ keywords and emitter-reserved identifiers that a source-derived name must
// never alias -- a local named ``int``/``default``/``allo_bitcast`` would break
// compilation. Such a name is uniquified with a numeric suffix instead.
inline bool isReservedCppName(llvm::StringRef name) {
  static const llvm::StringSet<> reserved = {"allo_bitcast",
                                             "alignas",
                                             "alignof",
                                             "and",
                                             "and_eq",
                                             "asm",
                                             "auto",
                                             "bitand",
                                             "bitor",
                                             "bool",
                                             "break",
                                             "case",
                                             "catch",
                                             "char",
                                             "class",
                                             "compl",
                                             "const",
                                             "constexpr",
                                             "const_cast",
                                             "continue",
                                             "decltype",
                                             "default",
                                             "delete",
                                             "do",
                                             "double",
                                             "dynamic_cast",
                                             "else",
                                             "enum",
                                             "explicit",
                                             "export",
                                             "extern",
                                             "false",
                                             "float",
                                             "for",
                                             "friend",
                                             "goto",
                                             "if",
                                             "inline",
                                             "int",
                                             "long",
                                             "mutable",
                                             "namespace",
                                             "new",
                                             "noexcept",
                                             "not",
                                             "not_eq",
                                             "nullptr",
                                             "operator",
                                             "or",
                                             "or_eq",
                                             "private",
                                             "protected",
                                             "public",
                                             "register",
                                             "reinterpret_cast",
                                             "return",
                                             "short",
                                             "signed",
                                             "sizeof",
                                             "static",
                                             "static_cast",
                                             "struct",
                                             "switch",
                                             "template",
                                             "this",
                                             "throw",
                                             "true",
                                             "try",
                                             "typedef",
                                             "typeid",
                                             "typename",
                                             "union",
                                             "unsigned",
                                             "using",
                                             "virtual",
                                             "void",
                                             "volatile",
                                             "while",
                                             "xor",
                                             "xor_eq"};
  return reserved.contains(name);
}

// The readable name carried by a value's NameLoc, if any. The value's own name
// is the outermost NameLoc (frontend attaches it as ``NameLoc(name, ...)``).
inline std::optional<std::string> nameFromLoc(Location loc) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return nameLoc.getName().str();
  return std::nullopt;
}

struct EmitterState {
  llvm::raw_ostream &os;
  std::size_t currentIndent = 0;
  std::size_t indentSize = 2;
  DenseMap<Value, std::string> nameTable;
  // The C++ signedness each named integer value was declared with. Sign-
  // sensitive uses consult this so they only cast when the wanted signedness
  // differs from the declared one; a missing entry means unsigned (the
  // default).
  DenseMap<Value, bool> signedness;
  // Names handed out in the current function scope, so source-derived names are
  // uniquified per function (see `beginValueScope`); `fallbackCounter` feeds
  // the synthetic `v<n>` names for values without a NameLoc and stays global.
  llvm::StringSet<> usedValueNames;
  unsigned fallbackCounter = 0;
  unsigned indexWidth = 32;
  bool withLocation = false;
  bool failed = false;
  bool enabledApFloat = false;
  std::string topName;

  explicit EmitterState(llvm::raw_ostream &os) : os(os) {}

  void addIndent() { currentIndent += indentSize; }
  void reduceIndent() { currentIndent -= indentSize; }
  void setIndentSize(std::size_t size) { indentSize = size; }

  std::string addName(Value v) {
    assert(!nameTable.contains(v) &&
           "Value already has a name in the name table");
    // Prefer the value's source name (from its NameLoc) for readability, else a
    // synthetic `v<n>`; then uniquify within the current function scope.
    std::string base;
    if (auto name = nameFromLoc(v.getLoc()))
      base = sanitizeCppIdentifier(*name);
    else
      base = "v" + std::to_string(fallbackCounter++);
    std::string unique = base;
    unsigned suffix = 0;
    while (usedValueNames.contains(unique) || isReservedCppName(unique))
      unique = base + "_" + std::to_string(++suffix);
    usedValueNames.insert(unique);
    nameTable[v] = unique;
    return unique;
  }

  // Start a fresh value-name scope for a function, so per-function locals reuse
  // clean source names. `seeded` are names already assigned to the function's
  // arguments (during the declaration pass) that body locals must not collide
  // with; pass the argument values so their existing names are reserved.
  template <typename ValueSeq> void beginValueScope(ValueSeq &&seeded) {
    usedValueNames.clear();
    for (Value v : seeded) {
      auto it = nameTable.find(v);
      if (it != nameTable.end())
        usedValueNames.insert(it->second);
    }
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

  void setSigned(Value v, bool isSigned) { signedness[v] = isSigned; }

  bool signednessOf(Value v) const {
    auto it = signedness.find(v);
    return it != signedness.end() && it->second;
  }
};
} // namespace mlir::allo

#endif // ALLO_TRANSLATION_EMITTERSTATE_H
