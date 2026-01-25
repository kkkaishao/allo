import allo.experimental as allo
from allo.experimental import settings
import numpy as np
import ast
import inspect

settings.USE_TENSOR = False

Ty = allo.Array(allo.bf16, (16,64))
Ty2 = allo.Array(allo.int8, (16,64))

# @allo.kernel
# def vadd(a: Ty, b: Ty, c: Ty, N: allo.constexpr):
#     for i in range(N):
#         for j in range(N):
#             c[i, j] = a[i, j] + b[i, j]

int512 = allo.APInt(512)
@allo.kernel(mapping=[3,4])
def matmul(a: Ty2, b: Ty2, c: Ty, N: allo.index):
    # allo.channel("chan_group", Ty, 10, shape=(2, 4))
    # buf1, buf2 = allo.acquire_buffer("chan_group", (0,0), allo.Producer, size=2)
    a = allo.channel("chan_group", Ty, 10, shape=(2, 4))
    s = a[1, 2].to_stream(2)
    s.put(2)
    b1, b2 = a[0,2].acq_buf(2)
    allo.rel_buf(a[0, 2], b1, b2)
    z = allo.int16(43)
    z[2::5] = 42
    f = z[2::5]
    if f == 42:
        g = z[2::5]
    elif f == 43:
        g = z[2::5]
    else:
        g = z[2::5]
    


StreamTy = allo.Stream(int512, 3)
@allo.kernel
def stream(a: StreamTy):
    pass

if __name__ == "__main__":
    a = np.random.rand(64, 64).astype(np.int8)
    b = np.random.rand(64, 64).astype(np.int8)
    c = np.zeros_like(a)
    # tree = ast.parse(matmul.src)
    # print(ast.dump(tree, indent=4))
    mod = matmul(a, b, c, 16)
    print(mod.module)
    # mod = stream[(1,)](a)
    # print(mod.module)