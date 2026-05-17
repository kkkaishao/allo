# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.backend import CPU
from allo.exp.lang.core import i32, range as allo_range
from allo.exp.lang.kernel import kernel


def test_dataflow_simulator_scalar_stream():
    @kernel
    def top(x: "i32[8]", out: "i32[8]"):
        fifo: "Stream[i32]"

        @kernel
        def producer(src: "i32[8]", stream: "Stream[i32]"):
            for i in allo_range(8):
                stream.put(src[i] + 1)

        @kernel
        def consumer(stream: "Stream[i32]", dst: "i32[8]"):
            for i in allo_range(8):
                dst[i] = stream.get() * 2

        producer(x, fifo)
        consumer(fifo, out)

    x = np.arange(8, dtype=np.int32)
    out = np.zeros((8,), dtype=np.int32)

    CPU(top).run(x, out)

    np.testing.assert_array_equal(out, (x + 1) * 2)
