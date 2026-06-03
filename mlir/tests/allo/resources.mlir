// RUN: allo-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: allo.state @mode enums ["os", "ws"] default "os" : !allo.state<i32>
allo.state @mode enums ["os", "ws"] default "os" : !allo.state<i32>

// CHECK-LABEL: func.func @rw_state
func.func @rw_state() {
  // CHECK: allo.state.write "ws" to @mode
  allo.state.write "ws" to @mode
  // CHECK: allo.state.read @mode : i32
  %0 = allo.state.read @mode : i32
  return
}

// -----

// CHECK: allo.desc @tma size(8) : !allo.desc<["offset":i8, "stride":i8]>
allo.desc @tma size(8) : !allo.desc<["offset":i8, "stride":i8]>

// CHECK-LABEL: func.func @rw_desc
func.func @rw_desc(%v: i8) {
  // CHECK: allo.desc.write %{{.*}} to "offset" in @tma : i8
  allo.desc.write %v to "offset" in @tma : i8
  // CHECK: allo.desc.read "stride" from @tma : i8
  %0 = allo.desc.read "stride" from @tma : i8
  return
}

// -----

allo.state @mode enums ["os", "ws"] : !allo.state<i32>
func.func @bad_state() {
  // expected-error @+1 {{value must be one of the enumerated states}}
  allo.state.write "bogus" to @mode
  return
}

// -----

// expected-error @+1 {{default state must be one of the enumerated states}}
allo.state @bad enums ["os", "ws"] default "xx" : !allo.state<i32>
