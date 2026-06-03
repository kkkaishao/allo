// RUN: allo-opt %s -lower-instructions | FileCheck %s

// Oracle lowering: buffers become memref.globals and each allo.emit is inlined
// into linalg/tensor reading from / writing back to those globals.

allo.buffer @X size(8) : !allo.vector<16xf32>
allo.buffer @Y size(8) : !allo.vector<16xf32>
allo.buffer @Z size(8) : !allo.vector<16xf32>

allo.define @vadd {
  src(@X, @Y) dst(@Z)
  addr(%i: index) {
    %0 = allo.patterns.strided basis(%i) counts(1) strides(1)
    %1 = allo.patterns.strided basis(%i) counts(1) strides(1)
    %2 = allo.patterns.strided basis(%i) counts(1) strides(1)
    allo.yield %0, %1, %2 : !allo.pattern, !allo.pattern, !allo.pattern
  }
  compute(%a: tensor<16xf32>, %b: tensor<16xf32>, %c: tensor<16xf32>) {
    %r = linalg.add ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
                    outs(%c : tensor<16xf32>) -> tensor<16xf32>
    allo.yield %r : tensor<16xf32>
  }
}

// CHECK-DAG: memref.global @X : memref<8x16xf32>
// CHECK-DAG: memref.global @Z : memref<8x16xf32>
// CHECK-LABEL: func.func @main
// CHECK: memref.get_global
// CHECK: bufferization.to_tensor
// CHECK: tensor.extract_slice
// CHECK: linalg.add
// CHECK: tensor.insert_slice
// CHECK: bufferization.materialize_in_destination
// CHECK-NOT: allo.emit
// CHECK-NOT: allo.define
func.func @main() {
  allo.emit @vadd addr(0) compute()
  return
}
