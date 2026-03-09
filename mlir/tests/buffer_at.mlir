// RUN: allo-opt %s -split-input-file -transform-interpreter -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @buffer_at_basic
func.func @buffer_at_basic() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    // CHECK: memref.alloc(){{.*}} : memref<8xi32>
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i, %j] : memref<8x8xi32>
    } {sym_name = "j"}
    // CHECK: affine.load
    // CHECK: affine.store
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      :(!transform.any_op) -> !transform.any_op
    transform.allo.buffer_at %target at %axis
      : !transform.any_value, !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @buffer_at_offset
func.func @buffer_at_offset() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i + 1, %j] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.allo.buffer_at %target at %axis
      : !transform.any_value, !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL func.func @buffer_at_multiple_access
func.func @buffer_at_multiple_access() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %0 = affine.load %tmp[%i, %j] : memref<8x8xi32>
      affine.store %0, %tmp[%j, %i] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.allo.buffer_at %target at %axis
      : !transform.any_value, !transform.any_op
    transform.yield
  }
}


// -----

// CHECK-LABEL: func.func @buffer_at_gemm_non_reduction
func.func @buffer_at_gemm_non_reduction(%a: memref<4x4xi32>,
                                        %b: memref<4x8xi32>) {
  %acc = memref.alloc() {sym_name = "acc"} : memref<4x8xi32>
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      affine.for %r = 0 to 4 {
        %lhs = affine.load %a[%i, %r] : memref<4x4xi32>
        %rhs = affine.load %b[%r, %j] : memref<4x8xi32>
        %old = affine.load %acc[%i, %j] : memref<4x8xi32>
        %mul = arith.muli %lhs, %rhs : i32
        %new = arith.addi %old, %mul : i32
        affine.store %new, %acc[%i, %j] : memref<4x8xi32>
      } {sym_name = "r"}
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "acc"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "i"} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.allo.buffer_at %target at %axis
      : !transform.any_value, !transform.any_op
    transform.yield
  }
}

// -----

func.func @buffer_at_gemm_reduction_rejected(%a: memref<4x4xi32>,
                                             %b: memref<4x8xi32>) {
  %acc = memref.alloc() {sym_name = "acc"} : memref<4x8xi32>
  affine.for %i = 0 to 4 {
    affine.for %r = 0 to 4 {
      affine.for %j = 0 to 8 {
        %lhs = affine.load %a[%i, %r] : memref<4x4xi32>
        %rhs = affine.load %b[%r, %j] : memref<4x8xi32>
        %old = affine.load %acc[%i, %j] : memref<4x8xi32>
        %mul = arith.muli %lhs, %rhs : i32
        %new = arith.addi %old, %mul : i32
        affine.store %new, %acc[%i, %j] : memref<4x8xi32>
      } {sym_name = "j"}
    } {sym_name = "r"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "acc"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "r"} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{cannot buffer at loop axis that contains reduction-like store}}
    transform.allo.buffer_at %target at %axis
      : !transform.any_value, !transform.any_op
    transform.yield
  }
}

// -----

func.func @buffer_at_innermost_rejected() {
  %tmp = memref.alloc() {sym_name = "tmp"} : memref<8x8xi32>
  %c1 = arith.constant 1 : i32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      affine.store %c1, %tmp[%i, %j] : memref<8x8xi32>
    } {sym_name = "j"}
  } {sym_name = "i"}
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %alloc = transform.structured.match attributes {sym_name = "tmp"} in %root
      : (!transform.any_op) -> !transform.any_op
    %target = transform.allo.match_value 0 of %alloc kind 2
      : !transform.any_op -> !transform.any_value
    %axis = transform.structured.match attributes {sym_name = "j"} in %root
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{cannot buffer at innermost loop axis}}
    transform.allo.buffer_at %target at %axis
      : !transform.any_value, !transform.any_op
    transform.yield
  }
}
