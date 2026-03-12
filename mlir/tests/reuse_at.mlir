// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @reuse_at_basic
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK:   %[[REUSE:.*]] = memref.alloc() {sym_name = "in_buf::reuse"} : memref<3xi32>
// CHECK:   affine.for %{{.*}} = 0 to 6 {
// CHECK:     affine.if
// CHECK:       affine.for %{{.*}} = 0 to 3 {
// CHECK:         affine.store %{{.*}}, %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:     } else {
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK:         %{{.*}} = affine.load %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:         affine.store %{{.*}}, %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:       }
// CHECK:       affine.store %{{.*}}, %[[REUSE]][%{{.*}}] : memref<3xi32>
// CHECK:     }
// CHECK:     %{{.*}} = affine.load %[[REUSE]][%{{.*}}] : memref<3xi32>
func.func @reuse_at_basic(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "in_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      %a0 = affine.load %in_buf[%y, %x] : memref<8x8xi32>
      %a1 = affine.load %in_buf[%y, %x + 1] : memref<8x8xi32>
      %a2 = affine.load %in_buf[%y, %x + 2] : memref<8x8xi32>
      %t0 = arith.addi %a0, %a1 : i32
      %t1 = arith.addi %t0, %a2 : i32
      affine.store %t1, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_affine_apply_chain
// CHECK: memref.alloc() {sym_name = "chain_buf::reuse"} : memref<3xi32>
// CHECK: affine.if
// CHECK: affine.load %{{.*}}[%{{.*}}] : memref<3xi32>
func.func @reuse_at_affine_apply_chain(%out: memref<8x6xi32>) {
  %chain_buf = memref.alloc() {sym_name = "chain_buf"} : memref<8x10xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      %x_shift = affine.apply affine_map<(d0) -> (d0 + 1)>(%x)
      %a0 = affine.load %chain_buf[%y, %x_shift] : memref<8x10xi32>
      %a1 = affine.load %chain_buf[%y, %x_shift + 1] : memref<8x10xi32>
      %a2 = affine.load %chain_buf[%y, %x_shift + 2] : memref<8x10xi32>
      %t0 = arith.addi %a0, %a1 : i32
      %t1 = arith.addi %t0, %a2 : i32
      affine.store %t1, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %chain_buf : memref<8x10xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_stationary_inner_dims
// CHECK: memref.alloc() {sym_name = "line_buf::reuse"} : memref<3x8x2xi32>
// CHECK: affine.if
// CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<3x8x2xi32>
func.func @reuse_at_stationary_inner_dims(%out: memref<6x8x2xi32>) {
  %line_buf = memref.alloc() {sym_name = "line_buf"} : memref<8x8x2xi32>
  affine.for %y = 0 to 6 {
    affine.for %x = 0 to 8 {
      affine.for %c = 0 to 2 {
        %a0 = affine.load %line_buf[%y, %x, %c] : memref<8x8x2xi32>
        %a1 = affine.load %line_buf[%y + 1, %x, %c] : memref<8x8x2xi32>
        %a2 = affine.load %line_buf[%y + 2, %x, %c] : memref<8x8x2xi32>
        %t0 = arith.addi %a0, %a1 : i32
        %t1 = arith.addi %t0, %a2 : i32
        affine.store %t1, %out[%y, %x, %c] : memref<6x8x2xi32>
      } {sym_name = "c"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %line_buf : memref<8x8x2xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "line_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_reduction_window
// CHECK: memref.alloc() {sym_name = "reduce_buf::reuse"} : memref<3xi32>
// CHECK: affine.if
// CHECK: affine.for %{{.*}} = 0 to 3 {
// CHECK:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<3xi32>
// CHECK: affine.load %{{.*}}[%{{.*}}] : memref<3xi32>
func.func @reuse_at_reduction_window(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "reduce_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      affine.for %r = 0 to 3 {
        %v = affine.load %in_buf[%y, %x + %r] : memref<8x8xi32>
        affine.store %v, %out[%y, %x] : memref<8x6xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_2d_window_axis_x
// CHECK: memref.alloc() {sym_name = "window_x_buf::reuse"} : memref<3x3xi32>
// CHECK: affine.if
// CHECK: affine.for %{{.*}} = 0 to 3 {
// CHECK:   affine.for %{{.*}} = 0 to 3 {
// CHECK:     affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<3x3xi32>
// CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<3x3xi32>
func.func @reuse_at_2d_window_axis_x(%out: memref<6x6xi32>) {
  %window_x_buf = memref.alloc() {sym_name = "window_x_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 6 {
    affine.for %x = 0 to 6 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.load %window_x_buf[%y + %ry, %x + %rx] : memref<8x8xi32>
          affine.store %v, %out[%y, %x] : memref<6x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %window_x_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "window_x_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @reuse_at_2d_window_axis_y
// CHECK: memref.alloc() {sym_name = "window_y_buf::reuse"} : memref<3x8xi32>
// CHECK: affine.if
// CHECK: affine.for %{{.*}} = 0 to 8 {
// CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<3x8xi32>
func.func @reuse_at_2d_window_axis_y(%out: memref<6x6xi32>) {
  %window_y_buf = memref.alloc() {sym_name = "window_y_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 6 {
    affine.for %x = 0 to 6 {
      affine.for %ry = 0 to 3 {
        affine.for %rx = 0 to 3 {
          %v = affine.load %window_y_buf[%y + %ry, %x + %rx] : memref<8x8xi32>
          affine.store %v, %out[%y, %x] : memref<6x6xi32>
        } {sym_name = "rx"}
      } {sym_name = "ry"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %window_y_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "window_y_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "y"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reuse_at_noncontiguous_window_rejected(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "noncontig_buf"} : memref<8x16xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      affine.for %r = 0 to 3 {
        // expected-note @+1 {{contiguous local footprint}}
        %v = affine.load %in_buf[%y, %x + %r * 2] : memref<8x16xi32> // expected-error {{bounded contiguous affine footprints}}
        affine.store %v, %out[%y, %x] : memref<8x6xi32>
      } {sym_name = "r"}
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x16xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "noncontig_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{analyze reuse state plan}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reuse_at_incompatible_sliding_dims(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "incompatible_buf"} : memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      %a0 = affine.load %in_buf[%y, %x] : memref<8x8xi32>
      // expected-note @+1 {{previous candidates}}
      %a1 = affine.load %in_buf[%x, %y] : memref<8x8xi32> // expected-error {{common sliding dimension}}
      %t0 = arith.addi %a0, %a1 : i32
      affine.store %t0, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "incompatible_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{analyze reuse state plan}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reuse_at_target_write_hazard(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "hazard_buf"} : memref<8x8xi32>
  %c0 = arith.constant 0 : i32
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 { // expected-error {{read-only}}
      %a0 = affine.load %in_buf[%y, %x] : memref<8x8xi32>
      %a1 = affine.load %in_buf[%y, %x + 1] : memref<8x8xi32>
      // expected-note @+1 {{write op}}
      affine.store %c0, %in_buf[%y, %x + 2] : memref<8x8xi32>
      %t0 = arith.addi %a0, %a1 : i32
      affine.store %t0, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "hazard_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "x"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{collect reuse candidate accesses}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @reuse_at_ignore_unrelated_store(%out: memref<8x6xi32>) {
  %in_buf = memref.alloc() {sym_name = "classify_buf"} : memref<8x8xi32>
  %scratch = memref.alloc() : memref<3xi32>
  %c0 = arith.constant 0 : i32
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 6 {
      affine.for %r = 0 to 3 {
        %v = affine.load %in_buf[%y, %x + %r] : memref<8x8xi32>
        affine.store %c0, %scratch[%r] : memref<3xi32>
      } {sym_name = "r"}
      affine.store %c0, %out[%y, %x] : memref<8x6xi32>
    } {sym_name = "x"}
  } {sym_name = "y"}
  memref.dealloc %scratch : memref<3xi32>
  memref.dealloc %in_buf : memref<8x8xi32>
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    %alloc = transform.structured.match ops{["memref.alloc"]} attributes {sym_name = "classify_buf"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"memref.alloc">
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.op<"memref.alloc"> -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "r"} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"affine.for">
    // expected-error @below {{reduction loop}}
    %reuse = transform.allo.reuse_at %target at %axis
      : (!transform.any_value, !transform.op<"affine.for">) -> !transform.any_op
    transform.yield
  }
}
