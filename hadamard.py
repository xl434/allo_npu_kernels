import allo
import allo.dataflow as df
from allo.ir.types import float32
import numpy as np

KERNEL_LIB_PATH = "cc/"


Ty = float32
M = 768 * 4
linear_M = 768

@df.region()
def top():
    @df.kernel(mapping=[4])
    def core(A: Ty[linear_M], B: Ty[linear_M], C: Ty[linear_M]):
        C[:] = allo.mul(A, B)

# Build the kernel
mod = df.build(top, target="aie")

# Helper function to apply kernel row-by-row
def apply_rowwise_hadamard(mod, A_np, B_np, C_np):
    assert A_np.shape == B_np.shape == C_np.shape
    N = A_np.shape[0]
    M = A_np.shape[1]
    for i in range(N):
        for j in range(M // linear_M):
            mod(A_np[i, j * linear_M : (j + 1) * linear_M], 
            B_np[i, j * linear_M : (j + 1) * linear_M], 
            C_np[i, j * linear_M : (j + 1) * linear_M])

N = 64
A_np = np.random.rand(N, M).astype(np.float32)
B_np = np.random.rand(N, M).astype(np.float32)
C_np = np.zeros((N, M), dtype=np.float32)

apply_rowwise_hadamard(mod, A_np, B_np, C_np)

ref = A_np * B_np
np.testing.assert_allclose(C_np, ref, rtol=1e-5)
print("Row-wise Hadamard product matches NumPy result.")