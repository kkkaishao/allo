// RUN: allo-opt %s -lower-instructions | FileCheck %s

// Control-flow safety: an allo.emit nested in scf.for must lower in place,
// with the inlined load/compute/store landing inside the loop body. This is
// the case the old (tensor-handle-threading) oracle could not handle.

allo.buffer @X size(8) : !allo.vector<16xf32>
allo.buffer @Z size(8) : !allo.vector<16xf32>

allo.define @copy {
  src(@X) dst(@Z)
  addr(%i: index) {
    %0 = allo.patterns.strided basis(%i) counts(1) strides(1)
    %1 = allo.patterns.strided basis(%i) counts(1) strides(1)
    allo.yield %0, %1 : !allo.pattern, !allo.pattern
  }
  compute(%a: tensor<16xf32>, %c: tensor<16xf32>) {
    allo.yield %a : tensor<16xf32>
  }
}

// CHECK-LABEL: func.func @main
// CHECK: scf.for
// CHECK:   memref.get_global @Z
// CHECK:   bufferization.to_tensor
// CHECK:   bufferization.materialize_in_destination
// CHECK: }
// CHECK-NOT: allo.emit
func.func @main() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c8 step %c1 {
    allo.emit @copy addr(%i) compute()
  }
  return
}
