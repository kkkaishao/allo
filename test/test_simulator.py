# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.lang.core import i32
from allo.exp.lang.kernel import kernel


def test_simulator_scalar_stream():
    @kernel
    def top(x: "i32[8]", out: "i32[8]"):
        fifo: "Stream[i32]"

        @kernel
        def producer(src: "i32[8]", stream: "Stream[i32]"):
            for i in range(8):
                stream.put(src[i] + 1)

        @kernel
        def consumer(stream: "Stream[i32]", dst: "i32[8]"):
            for i in range(8):
                dst[i] = stream.get() * 2

        producer(x, fifo)
        consumer(fifo, out)

    x = np.arange(8, dtype=np.int32)
    out = np.zeros((8,), dtype=np.int32)

    top(x, out)

    np.testing.assert_array_equal(out, (x + 1) * 2)


def test_simulator_block_stream():
    @kernel
    def top(out: "i32[2,2,2]"):
        fifo: "Stream[i32[2,2]]"

        @kernel
        def producer(stream: "Stream[i32[2,2]]"):
            buf: "i32[2,2]"
            buf[0, 0] = 1
            buf[0, 1] = 2
            buf[1, 0] = 3
            buf[1, 1] = 4
            stream.put(buf)
            buf[0, 0] = 10
            buf[0, 1] = 20
            buf[1, 0] = 30
            buf[1, 1] = 40
            stream.put(buf)

        @kernel
        def consumer(stream: "Stream[i32[2,2]]", dst: "i32[2,2,2]"):
            first = stream.get()
            second = stream.get()
            dst[0, 0, 0] = first[0, 0]
            dst[0, 0, 1] = first[0, 1]
            dst[0, 1, 0] = first[1, 0]
            dst[0, 1, 1] = first[1, 1]
            dst[1, 0, 0] = second[0, 0]
            dst[1, 0, 1] = second[0, 1]
            dst[1, 1, 0] = second[1, 0]
            dst[1, 1, 1] = second[1, 1]

        producer(fifo)
        consumer(fifo, out)

    out = np.zeros((2, 2, 2), dtype=np.int32)

    top(out)

    expected = np.array(
        [[[1, 2], [3, 4]], [[10, 20], [30, 40]]],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out, expected)
