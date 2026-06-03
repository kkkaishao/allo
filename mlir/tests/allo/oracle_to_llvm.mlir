// RUN: allo-opt %s -lower-instructions -allo-lower-to-llvm | FileCheck %s

// The oracle output lowers cleanly to the LLVM dialect via the reused pipeline.

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

// CHECK: llvm.func @main
func.func @main() {
  allo.emit @vadd addr(0) compute()
  return
}
