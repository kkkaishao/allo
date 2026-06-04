# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from allo.exp.lang.core import APInt, i32, range as arange
from allo.exp.lang.kernel import kernel


def test_simulator_apint_buffers():
    i5 = APInt(5, signed=True)
    u5 = APInt(5, signed=False)

    @kernel
    def addsub(A: i5[8], B: u5[8], C: i5[8]):
        for i in arange(8, name="i"):
            C[i] = A[i] + B[i]

    A = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int8)
    B = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
    C = np.zeros(8, dtype=np.int8)
    addsub(A, B, C)
    expected = ((A.astype(np.int16) + B + 16) % 32 - 16).astype(np.int8)
    np.testing.assert_array_equal(C, expected)


def test_simulator_apint_scalar_return():
    i13 = APInt(13, signed=True)

    @kernel
    def acc(A: i13[6]) -> i13:
        s: i13 = 0
        for i in arange(6, name="i"):
            s = s + A[i]
        return s

    A = np.array([-4000, 4000, -100, 100, -1, 1], dtype=np.int16)
    expected = int((int(A.sum()) + 4096) % 8192 - 4096)
    assert int(acc(A)) == expected


def test_simulator_scalar_stream():
    @kernel
    def top(x: i32[8], out: i32[8]):
        fifo: Stream[i32]

        @kernel
        def producer(src: i32[8], stream: Stream[i32]):
            for i in range(8):
                stream.put(src[i] + 1)

        @kernel
        def consumer(stream: Stream[i32], dst: i32[8]):
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
    def top(out: i32[2, 2, 2]):
        fifo: Stream[i32[2, 2]]

        @kernel
        def producer(stream: Stream[i32[2, 2]]):
            buf: i32[2, 2]
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
        def consumer(stream: Stream[i32[2, 2]], dst: i32[2, 2, 2]):
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
