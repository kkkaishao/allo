// RUN: allo-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: allo.buffer @b size(8) : !allo.scalar<f32>
allo.buffer @b size(8) : !allo.scalar<f32>

// -----

// CHECK: allo.buffer @vec size(128) : !allo.vector<64xbf16>
allo.buffer @vec size(128) : !allo.vector<64xbf16>

// -----

// CHECK: allo.buffer @t size(4) : !allo.tile<4x4xf32>
allo.buffer @t size(4) : !allo.tile<4x4xf32>

// -----

// CHECK: allo.buffer @mem size(1) : !allo.hbm<8192xbf16>
allo.buffer @mem size(1) : !allo.hbm<8192xbf16>
