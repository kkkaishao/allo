# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-reconfigurable systolic GEMM arrays. A runtime ``inst`` bool selects
the dataflow per call -- ``True`` -> output-stationary (OS), ``False`` ->
weight-stationary (WS) -- broadcast across the PE array through instruction
FIFOs. Three memory schemes are covered: a plain array, a tiled array, and an
L2 daisy-chain with bit-packed operands.

These kernels do not compile under the old frontend (it emits invalid MLIR).

Functional checks run through Vitis csim, in both OS and WS modes; the CPU
dataflow simulator currently deadlocks on multi-PE arrays.
"""

import tempfile

import numpy as np
import pytest

import allo
from allo.lang.core import i16, i32, APInt, bool as allo_bool, Stream
from allo.lang.kernel import kernel
from allo.backend.vitis.core import is_vitis_available

requires_vitis = pytest.mark.skipif(
    not is_vitis_available(), reason="Vitis HLS toolchain not detected"
)
PART = "xcvu9p-flga2104-2-i"

U = 4
M, N, K = U, 4, U
P0, P1 = U + 2, U + 2  # plain + daisy PE grid
PW = U * 16  # daisy packed-word width (64)
upacked = APInt(PW, signed=False)

Rt, Ct = 2, 2
PT0, PT1 = Rt + 2, Ct + 2  # tiled PE grid


# ===========================================================================
# Plain OS/WS reconfigurable GEMM
# ===========================================================================


@kernel
def osws_gemm(A: i32[M, K], B: i32[K, N], inst: allo_bool, C: i32[M, N]):
    fifo_R: Stream[i32][P0, P1 - 1]
    fifo_C: Stream[i32][P0 - 1, P1]
    inst_broad: Stream[allo_bool][P1 - 1]
    inst_chain: Stream[allo_bool][P0 - 1, P1]

    @kernel(mapping=[P0, P1])
    def pe(
        A: i32[M, K],
        B: i32[K, N],
        inst: allo_bool,
        C: i32[M, N],
        fifo_R: Stream[i32][P0, P1 - 1],
        fifo_C: Stream[i32][P0 - 1, P1],
        inst_broad: Stream[allo_bool][P1 - 1],
        inst_chain: Stream[allo_bool][P0 - 1, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)

        # instruction decode & dispatch
        flowtag: allo_bool = inst
        if i == 0 and j == 0:
            inst_broad[j].put(flowtag)
            inst_chain[i, j].put(flowtag)
        else:
            if i == 0:
                flowtag = inst_broad[j - 1].get()
            else:
                flowtag = inst_chain[i - 1, j].get()
            if i == 0 and j != P1 - 1:
                inst_broad[j].put(flowtag)
            if i != P0 - 1:
                inst_chain[i, j].put(flowtag)

        # computation
        if (i == 0 or i == U + 1) and (j == 0 or j == U + 1):
            pass
        else:
            Tlength: i32 = U  # M == K here, so flowtag-independent
            Czero: i32 = 0
            if i == 0:
                for t in range(Tlength):
                    if flowtag:
                        fifo_C[i, j].put(B[t, j - 1])
                    else:
                        fifo_C[i, j].put(Czero)
            elif j == 0:
                for t in range(Tlength):
                    if flowtag:
                        fifo_R[i, j].put(A[i - 1, t])
                    else:
                        fifo_R[i, j].put(A[t, i - 1])
            elif i == U + 1:
                for t in range(Tlength):
                    if flowtag:
                        c_drain: i32 = fifo_C[i - 1, j].get()
                    else:
                        C[t, j - 1] = fifo_C[i - 1, j].get()
            elif j == U + 1:
                for t in range(Tlength):
                    r_drain: i32 = fifo_R[i, j - 1].get()
            else:
                local_S: i32 = 0 if flowtag else B[i - 1, j - 1]
                for t in range(Tlength):
                    s: i32 = local_S
                    r: i32 = fifo_R[i, j - 1].get()
                    c: i32 = fifo_C[i - 1, j].get()
                    weight: i32 = c if flowtag else s
                    psum: i32 = s if flowtag else c
                    accu: i32 = r * weight + psum
                    local_S = accu if flowtag else s
                    fifo_R[i, j].put(r)
                    fifo_C[i, j].put(c if flowtag else accu)
                if flowtag:
                    C[i - 1, j - 1] = local_S

    pe(A, B, inst, C, fifo_R, fifo_C, inst_broad, inst_chain)


# ===========================================================================
# Tiled OS/WS reconfigurable GEMM: a Rt x Ct array sweeps (M/Rt) x (N/Ct) tiles
# ===========================================================================


@kernel
def osws_gemm_tiled(A: i32[M, K], B: i32[K, N], inst: allo_bool, C: i32[M, N]):
    fifo_R: Stream[i32][PT0, PT1 - 1]
    fifo_C: Stream[i32][PT0 - 1, PT1]
    inst_broad: Stream[allo_bool][PT1 - 1]
    inst_chain: Stream[allo_bool][PT0 - 1, PT1]

    @kernel(mapping=[PT0, PT1])
    def pe(
        A: i32[M, K],
        B: i32[K, N],
        inst: allo_bool,
        C: i32[M, N],
        fifo_R: Stream[i32][PT0, PT1 - 1],
        fifo_C: Stream[i32][PT0 - 1, PT1],
        inst_broad: Stream[allo_bool][PT1 - 1],
        inst_chain: Stream[allo_bool][PT0 - 1, PT1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)

        # instruction decode & dispatch
        flowtag: allo_bool = inst
        if i == 0 and j == 0:
            inst_broad[j].put(flowtag)
            inst_chain[i, j].put(flowtag)
        else:
            if i == 0:
                flowtag = inst_broad[j - 1].get()
            else:
                flowtag = inst_chain[i - 1, j].get()
            if i == 0 and j != PT1 - 1:
                inst_broad[j].put(flowtag)
            if i != PT0 - 1:
                inst_chain[i, j].put(flowtag)

        # computation
        if (i == 0 or i == Rt + 1) and (j == 0 or j == Ct + 1):
            pass
        else:
            Tlength: i32 = K  # M == K here, so flowtag-independent
            Czero: i32 = 0
            for ri in range(M // Rt):
                for ci in range(N // Ct):
                    if i == 0:
                        for t in range(Tlength):
                            if flowtag:
                                fifo_C[i, j].put(B[t, ci * Ct + (j - 1)])
                            else:
                                fifo_C[i, j].put(Czero)
                    elif j == 0:
                        for t in range(Tlength):
                            if flowtag:
                                fifo_R[i, j].put(A[ri * Rt + (i - 1), t])
                            else:
                                fifo_R[i, j].put(A[t, ri * Rt + (i - 1)])
                    elif i == Rt + 1:
                        for t in range(Tlength):
                            if flowtag:
                                c_drain: i32 = fifo_C[i - 1, j].get()
                            else:
                                C[t, ci * Ct + (j - 1)] = (
                                    C[t, ci * Ct + (j - 1)] + fifo_C[i - 1, j].get()
                                )
                    elif j == Ct + 1:
                        for t in range(Tlength):
                            r_drain: i32 = fifo_R[i, j - 1].get()
                    else:
                        local_S: i32 = (
                            0 if flowtag else B[ri * Rt + (i - 1), ci * Ct + (j - 1)]
                        )
                        for t in range(Tlength):
                            s: i32 = local_S
                            r: i32 = fifo_R[i, j - 1].get()
                            c: i32 = fifo_C[i - 1, j].get()
                            weight: i32 = c if flowtag else s
                            psum: i32 = s if flowtag else c
                            accu: i32 = r * weight + psum
                            local_S = accu if flowtag else s
                            fifo_R[i, j].put(r)
                            fifo_C[i, j].put(c if flowtag else accu)
                        if flowtag:
                            C[ri * Rt + (i - 1), ci * Ct + (j - 1)] = local_S

    pe(A, B, inst, C, fifo_R, fifo_C, inst_broad, inst_chain)


# ===========================================================================
# Daisy-chain OS/WS reconfigurable GEMM: an L2 chain feeds bit-packed
# UInt(U*16) rows/columns into the array, packed/unpacked per PE via bit slicing
# ===========================================================================


@kernel
def osws_gemm_daisy(A: i16[M, K], B: i16[K, N], inst: allo_bool, C: i16[M, N]):
    L2_R: Stream[upacked][P0 - 1]
    L2_C: Stream[upacked][P1 - 1]
    L1_S: Stream[upacked][U + 1, N]
    L2_S_in: Stream[upacked][N]
    L2_S_out: Stream[upacked][N]
    fifo_R: Stream[i16][U, N]
    fifo_C: Stream[i16][U + 1, N]
    inst_broad: Stream[allo_bool][P1 - 1]
    inst_chain: Stream[allo_bool][P0 - 1, P1]

    @kernel(mapping=[P0, P1])
    def pe(
        A: i16[M, K],
        B: i16[K, N],
        inst: allo_bool,
        C: i16[M, N],
        L2_R: Stream[upacked][P0 - 1],
        L2_C: Stream[upacked][P1 - 1],
        L1_S: Stream[upacked][U + 1, N],
        L2_S_in: Stream[upacked][N],
        L2_S_out: Stream[upacked][N],
        fifo_R: Stream[i16][U, N],
        fifo_C: Stream[i16][U + 1, N],
        inst_broad: Stream[allo_bool][P1 - 1],
        inst_chain: Stream[allo_bool][P0 - 1, P1],
    ):
        i = allo.get_wid(0)
        j = allo.get_wid(1)
        Czero: i16 = 0

        # instruction decode & dispatch
        flowtag: allo_bool = inst
        if i == 0 and j == 0:
            inst_broad[j].put(flowtag)
            inst_chain[i, j].put(flowtag)
        else:
            if i == 0:
                flowtag = inst_broad[j - 1].get()
            else:
                flowtag = inst_chain[i - 1, j].get()
            if i == 0 and j != P1 - 1:
                inst_broad[j].put(flowtag)
            if i != P0 - 1:
                inst_chain[i, j].put(flowtag)

        # corner (0,0): pack A rows / B cols into the L2 chain
        if i == 0 and j == 0:
            if not flowtag:
                for n in range(N):
                    packed_S_in: upacked = 0
                    for k in range(U):
                        packed_S_in[k * 16 : k * 16 + 16] = B[k, n]
                    L2_S_in[0].put(packed_S_in)
            for u in range(U):
                packed_R: upacked = 0
                if flowtag:
                    for m in range(U):
                        packed_R[m * 16 : m * 16 + 16] = A[m, u]
                else:
                    for k in range(U):
                        packed_R[k * 16 : k * 16 + 16] = A[u, k]
                L2_R[1].put(packed_R)
                packed_C: upacked = 0
                if flowtag:
                    for n in range(N):
                        packed_C[n * 16 : n * 16 + 16] = B[u, n]
                else:
                    for n in range(N):
                        packed_C[n * 16 : n * 16 + 16] = Czero
                L2_C[1].put(packed_C)
        # corner (P0-1,P1-1): unpack results
        elif i == P0 - 1 and j == P1 - 1:
            for n in range(N):
                packed_S_out: upacked = L2_S_out[N - 1].get()
                for m in range(M):
                    C[m, n] = packed_S_out[m * 16 : m * 16 + 16]
        # other corners: idle
        elif (i == 0 or i == P0 - 1) and (j == 0 or j == P1 - 1):
            pass
        # west column: distribute packed rows, unpack one lane
        elif j == 0:
            for u in range(U):
                r: upacked = L2_R[i].get()
                fifo_R[i - 1, 0].put(r[16 * (i - 1) : 16 * (i - 1) + 16])
                if i < U:
                    L2_R[i + 1].put(r)
        # north row: distribute packed cols, forward stationary cache
        elif i == 0:
            if not flowtag:
                L1_S[0, j - 1].put(L2_S_in[j - 1].get())
                if j != P1 - 2:
                    for ind in range(N - j):
                        L2_S_in[j].put(L2_S_in[j - 1].get())
            for u in range(U):
                c: upacked = L2_C[j].get()
                fifo_C[0, j - 1].put(c[16 * (j - 1) : 16 * (j - 1) + 16])
                if j < N:
                    L2_C[j + 1].put(c)
        # south row: collect / forward the output cache chain
        elif i == P0 - 1:
            if flowtag:  # OS
                c_C: upacked = L1_S[i - 1, N - j].get()
                L2_S_out[j - 1].put(c_C)
                if j != 1:
                    for ind in range(j - 1):
                        L2_S_out[j - 1].put(L2_S_out[j - 2].get())
            else:  # WS
                if j != 1:
                    for ind in range(j - 1):
                        L2_S_out[j - 1].put(L2_S_out[j - 2].get())
                c_C2: upacked = 0
                for m in range(U):
                    c_C2[m * 16 : m * 16 + 16] = fifo_C[U, j - 1].get()
                L2_S_out[j - 1].put(c_C2)
        # east column: idle
        elif j == P1 - 1:
            pass
        # main PE body
        else:
            local_s: i16 = 0
            if not flowtag:
                packed_in: upacked = L1_S[i - 1, j - 1].get()
                local_s = packed_in[16 * (i - 1) : 16 * (i - 1) + 16]
                if i < U:
                    L1_S[i, j - 1].put(packed_in)
            for u in range(U):
                r2: i16 = fifo_R[i - 1, j - 1].get()
                c2: i16 = fifo_C[i - 1, j - 1].get()
                weight: i16 = c2 if flowtag else local_s
                psum: i16 = local_s if flowtag else c2
                accu: i16 = r2 * weight + psum
                if flowtag:
                    local_s = accu
                if j < N:
                    fifo_R[i - 1, j].put(r2)
                if i < U:
                    fifo_C[i, j - 1].put(c2 if flowtag else accu)
                if i == U:
                    if not flowtag:
                        fifo_C[i, j - 1].put(accu)
            # stationary cache-out (OS only)
            if flowtag:
                packed_out: upacked = 0
                if i != 1:
                    packed_out = L1_S[i - 1, j - 1].get()
                packed_c: upacked = 0
                for m in range(U):
                    if m == i - 1:
                        packed_c[m * 16 : m * 16 + 16] = local_s
                    else:
                        packed_c[m * 16 : m * 16 + 16] = packed_out[
                            m * 16 : m * 16 + 16
                        ]
                L1_S[i, j - 1].put(packed_c)

    pe(
        A,
        B,
        inst,
        C,
        L2_R,
        L2_C,
        L1_S,
        L2_S_in,
        L2_S_out,
        fifo_R,
        fifo_C,
        inst_broad,
        inst_chain,
    )


# ===========================================================================
# Tests
# ===========================================================================


def _check_both_modes(top, dtype, lo, hi):
    A = np.random.randint(lo, hi, (M, K)).astype(dtype)
    B = np.random.randint(lo, hi, (K, N)).astype(dtype)
    truth = (A.astype(np.int32) @ B.astype(np.int32)).astype(dtype)
    with tempfile.TemporaryDirectory() as proj:
        backend = top.schedule().export("vitis", project_path=proj)
        for flowtag in (False, True):  # weight-stationary, output-stationary
            C = np.zeros((M, N), dtype=dtype)
            backend(A, B, flowtag, C)
            np.testing.assert_allclose(C, truth, atol=1e-5)


def test_plain_codegen():
    assert "void osws_gemm_pe_1_1(" in osws_gemm.schedule().export("vitis").hls_code


def test_plain_cpu_sim():
    A = np.random.randint(-8, 8, (M, K)).astype(np.int32)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int32)
    truth = A @ B
    C = np.zeros((M, N), dtype=np.int32)
    osws_gemm(A, B, True, C)
    np.testing.assert_allclose(C, truth)


@requires_vitis
def test_plain_csim():
    _check_both_modes(osws_gemm, np.int32, -8, 8)


def test_tiled_codegen():
    code = osws_gemm_tiled.schedule().export("vitis").hls_code
    assert "void osws_gemm_tiled_pe_1_1(" in code


def test_tiled_cpu_sim():
    A = np.random.randint(-8, 8, (M, K)).astype(np.int32)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int32)
    truth = A @ B
    C = np.zeros((M, N), dtype=np.int32)
    osws_gemm_tiled(A, B, True, C)
    np.testing.assert_allclose(C, truth)
    C = np.zeros((M, N), dtype=np.int32)
    osws_gemm_tiled(A, B, False, C)
    np.testing.assert_allclose(C, truth)


@requires_vitis
def test_tiled_csim():
    _check_both_modes(osws_gemm_tiled, np.int32, -8, 8)


def test_daisy_codegen():
    code = osws_gemm_daisy.schedule().export("vitis").hls_code
    assert "void osws_gemm_daisy_pe_1_1(" in code


def test_daisy_cpu_sim():
    A = np.random.randint(0, 8, (M, K)).astype(np.int16)
    B = np.random.randint(0, 8, (K, N)).astype(np.int16)
    truth = A @ B
    C = np.zeros((M, N), dtype=np.int16)
    osws_gemm_daisy(A, B, True, C)
    np.testing.assert_allclose(C, truth)


@requires_vitis
def test_daisy_csim():
    _check_both_modes(osws_gemm_daisy, np.int16, 0, 8)
