/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_SUPPORT_LOGGING_H
#define ALLO_SUPPORT_LOGGING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace mlir {
class Operation;
class Location;
} // namespace mlir

namespace mlir::allo::logging {

// Severity, ascending. Mapped onto spdlog levels in Logging.cpp.
enum class Level { Debug, Info, Warn, Error };

// Compiler stage printed in the second bracket. Extend as new stages log.
enum class Stage { Prep, Sched, Dcp, Emit };

namespace detail {
// Format `[LEVEL][STAGE] message[ (at where)]` and route to the backend. For
// `Level::Error` with a non-null `subject`, additionally emit an MLIR error
// diagnostic on it, so a fatal error both logs and propagates (fails the pass /
// surfaces to the caller) -- the logger augments MLIR error reporting, it does
// not replace it.
void emit(Level level, Stage stage, llvm::StringRef where,
          llvm::StringRef message, mlir::Operation *subject);
// Whether `level` currently passes the threshold (skip building dropped lines).
// `Level::Error` is never filtered.
bool enabled(Level level);
// Concise source anchor for an op / location (symbolic name + file:line:col).
std::string describe(mlir::Operation *op);
std::string describe(const mlir::Location &loc);
} // namespace detail

// RAII stream proxy: accumulate a message with `<<`, emit it on destruction.
// Create through the factories below; it relies on C++17 guaranteed copy
// elision, so it needs no move or copy constructor.
class Diagnostic {
public:
  Diagnostic(Level level, Stage stage, std::string where,
             mlir::Operation *subject)
      : level(level), stage(stage), active(detail::enabled(level)),
        subject(subject), where(std::move(where)), stream(message) {}
  ~Diagnostic() {
    if (active) {
      stream.flush();
      detail::emit(level, stage, where, message, subject);
    }
  }

  Diagnostic(const Diagnostic &) = delete;
  Diagnostic &operator=(const Diagnostic &) = delete;

  template <typename T> Diagnostic &operator<<(T &&value) {
    if (active)
      stream << std::forward<T>(value);
    return *this;
  }

private:
  Level level;
  Stage stage;
  bool active;
  mlir::Operation *subject;
  std::string where;
  std::string message;
  llvm::raw_string_ostream stream;
};

// Factories. The op/location overloads render a source anchor (a null op omits
// it); the convenience wrappers fix the level. Pass the subject op at an error
// site so the failure propagates, not just logs.
inline Diagnostic log(Level level, Stage stage) {
  return Diagnostic(level, stage, std::string(), nullptr);
}
inline Diagnostic log(Level level, Stage stage, mlir::Operation *subject) {
  return Diagnostic(level, stage, detail::describe(subject), subject);
}
inline Diagnostic log(Level level, Stage stage, const mlir::Location &loc) {
  return Diagnostic(level, stage, detail::describe(loc), nullptr);
}

inline Diagnostic debug(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Debug, stage, op);
}
inline Diagnostic info(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Info, stage, op);
}
inline Diagnostic warn(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Warn, stage, op);
}
inline Diagnostic error(Stage stage, mlir::Operation *op = nullptr) {
  return log(Level::Error, stage, op);
}

// Runtime configuration (the threshold is also seeded from the ALLO_LOG_LEVEL
// environment variable on first use).
void setLevel(Level level);
Level getLevel();
void setColor(bool enable);

} // namespace mlir::allo::logging

#endif // ALLO_SUPPORT_LOGGING_H
