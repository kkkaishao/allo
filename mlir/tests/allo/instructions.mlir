// RUN: allo-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// A small ISA catalog: a parametric matmul instruction whose addr region maps a
// flat HBM region into 2-D operands, with a pure linalg.matmul compute region,
// invoked by an allo.emit inside an allo.sequence.

allo.buffer @devmem size(1) : !allo.hbm<4096xf32>

// CHECK: allo.define @matmul
// CHECK: allo.patterns.strided
// CHECK: allo.patterns.expand_shape
// CHECK: linalg.matmul
allo.define @matmul {
  src(@devmem, @devmem) dst(@devmem)
  addr(%a: index, %b: index, %c: index, %size: index) {
    %cnt = arith.muli %size, %size : index
    %sa = allo.patterns.strided basis(%a) counts(%cnt) strides(1)
    %ea = allo.patterns.expand_shape %sa [[0, 1]] output_shape [%size, %size]
    %sb = allo.patterns.strided basis(%b) counts(%cnt) strides(1)
    %eb = allo.patterns.expand_shape %sb [[0, 1]] output_shape [%size, %size]
    %sc = allo.patterns.strided basis(%c) counts(%cnt) strides(1)
    %ec = allo.patterns.expand_shape %sc [[0, 1]] output_shape [%size, %size]
    allo.yield %ea, %eb, %ec : !allo.pattern, !allo.pattern, !allo.pattern
  }
  compute(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>) {
    %0 = linalg.matmul ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
                       outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>
    allo.yield %0 : tensor<?x?xf32>
  }
}

// CHECK: allo.sequence @prog
// CHECK: allo.emit @matmul addr(0, 64, 128, 8) compute()
allo.sequence @prog {
  allo.emit @matmul addr(0, 64, 128, 8) compute()
}

// -----

allo.buffer @devmem size(1) : !allo.hbm<4096xf32>

allo.define @id {
  src(@devmem) dst(@devmem)
  addr(%a: index, %c: index, %n: index) {
    %0 = allo.patterns.strided basis(%a) counts(%n) strides(1)
    %1 = allo.patterns.strided basis(%c) counts(%n) strides(1)
    allo.yield %0, %1 : !allo.pattern, !allo.pattern
  }
  compute(%a: tensor<?xf32>, %c: tensor<?xf32>) {
    allo.yield %a : tensor<?xf32>
  }
}

allo.sequence @bad {
  // @id has 3 addr params; passing 2 must fail.
  // expected-error @+1 {{number of address parameters must match}}
  allo.emit @id addr(0, 4) compute()
}
