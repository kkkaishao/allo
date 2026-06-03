// RUN: allo-opt %s -allow-unregistered-dialect | FileCheck %s

// The access-pattern producer ops land in Phase 1; for now exercise the
// !allo.pattern token type's parse/print via an unregistered op.

// CHECK-LABEL: func.func @pattern_roundtrip
func.func @pattern_roundtrip() {
  // CHECK: "test.produce"() : () -> !allo.pattern
  %0 = "test.produce"() : () -> !allo.pattern
  // CHECK: () -> !allo.desc<["offset":i8, "stride":i8]>
  %1 = "test.produce"() : () -> !allo.desc<["offset":i8, "stride":i8]>
  // CHECK: () -> !allo.state<i32>
  %2 = "test.produce"() : () -> !allo.state<i32>
  return
}
