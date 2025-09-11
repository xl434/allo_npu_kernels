import os, numpy as np, torch
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie import ExternalModule

# Shapes
SEQ, HEAD_DIM  = 64, 64
HEAD_DIM_HALF = HEAD_DIM // 2

# Layouts / dtypes
Ty = float32
VecLy = Layout("S0")
MatLy = Layout("S0S1")

# External kernels (AIE)
KERNEL_LIB_PATH = "../cc/"

OPS_IMPL = KERNEL_LIB_PATH + "rope_vec_ops.cc"   # the C++ file above
SIN_IMPL = KERNEL_LIB_PATH + "sine.cc"       # your LUT-based sin kernel
COS_IMPL = KERNEL_LIB_PATH + "cosine.cc"     # your LUT-based cos kernel

radians_ext = ExternalModule(
    top="rope_make_radians_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],   # positions, inv_timescale
    output_idx=[2],     # radians32
)

pack_ext = ExternalModule(
    top="pack32to64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0],      # radians32
    output_idx=[1],     # radians64
)

copyL_ext = ExternalModule(
    top="copy_left32_from64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0],      # in64
    output_idx=[1],     # out32
)
copyR_ext = ExternalModule(
    top="copy_right32_from64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0],      # in64
    output_idx=[1],     # out32
)

join_ext = ExternalModule(
    top="join32_to_64_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],   # left32, right32
    output_idx=[2],     # out64
)

mul32_ext = ExternalModule(
    top="mul32_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],   # A, B
    output_idx=[2],     # C
)
add32_ext = ExternalModule(
    top="add32_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],   # A, B
    output_idx=[2],     # C
)
sub32_ext = ExternalModule(
    top="sub32_float32",
    impl_path=OPS_IMPL,
    input_idx=[0, 1],   # A, B
    output_idx=[2],     # C
)

sin_ext = ExternalModule(
    top="sin_float32",
    impl_path=SIN_IMPL,
    input_idx=[0],      # in64
    output_idx=[1],     # out64
)

cos_ext = ExternalModule(
    top="cos_float32",
    impl_path=COS_IMPL,
    input_idx=[0],      # in64
    output_idx=[1],     # out64
)

# -------- Regions (mapping=[1,1]) --------
@df.region()
def radians_region():
    @df.kernel(mapping=[1, 1])
    def core(positions: Ty[SEQ] @ VecLy,
             inv_ts:    Ty[HEAD_DIM_HALF] @ VecLy,
             radians32: Ty[SEQ, HEAD_DIM_HALF] @ MatLy):
        radians_ext(positions, inv_ts, radians32)

@df.region()
def pack_region():
    @df.kernel(mapping=[1, 1])
    def core(radians32: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             radians64: Ty[SEQ, HEAD_DIM]      @ MatLy):
        pack_ext(radians32, radians64)

@df.region()
def sin_region():
    @df.kernel(mapping=[1, 2])
    def core(in64:  Ty[SEQ, HEAD_DIM] @ MatLy,
             out64: Ty[SEQ, HEAD_DIM] @ MatLy):
        sin_ext(in64, out64)

@df.region()
def cos_region():
    @df.kernel(mapping=[1, 2])
    def core(in64:  Ty[SEQ, HEAD_DIM] @ MatLy,
             out64: Ty[SEQ, HEAD_DIM] @ MatLy):
        cos_ext(in64, out64)

@df.region()
def copy_left_region():
    @df.kernel(mapping=[1, 1])
    def core(in64:  Ty[SEQ, HEAD_DIM]      @ MatLy,
             out32: Ty[SEQ, HEAD_DIM_HALF] @ MatLy):
        copyL_ext(in64, out32)

@df.region()
def copy_right_region():
    @df.kernel(mapping=[1, 1])
    def core(in64:  Ty[SEQ, HEAD_DIM]      @ MatLy,
             out32: Ty[SEQ, HEAD_DIM_HALF] @ MatLy):
        copyR_ext(in64, out32)

@df.region()
def join_region():
    @df.kernel(mapping=[1, 2])
    def core(left32:  Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             right32: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             out64:   Ty[SEQ, HEAD_DIM]      @ MatLy):
        join_ext(left32, right32, out64)

@df.region()
def mul32_region():
    @df.kernel(mapping=[1, 1])
    def core(A: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             B: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             C: Ty[SEQ, HEAD_DIM_HALF] @ MatLy):
        mul32_ext(A, B, C)

@df.region()
def add32_region():
    @df.kernel(mapping=[1, 1])
    def core(A: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             B: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             C: Ty[SEQ, HEAD_DIM_HALF] @ MatLy):
        add32_ext(A, B, C)

@df.region()
def sub32_region():
    @df.kernel(mapping=[1, 1])
    def core(A: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             B: Ty[SEQ, HEAD_DIM_HALF] @ MatLy,
             C: Ty[SEQ, HEAD_DIM_HALF] @ MatLy):
        sub32_ext(A, B, C)

# -------- Build all modules --------

radians_mod = df.build(radians_region, target="aie", project="rope/radians.prj")
pack_mod    = df.build(pack_region,    target="aie", project="rope/pack.prj")
sin_mod     = df.build(sin_region,     target="aie", project="rope/sin.prj")
cos_mod     = df.build(cos_region,     target="aie", project="rope/cos.prj")
copyL_mod   = df.build(copy_left_region,  target="aie", project="rope/copyL.prj")
copyR_mod   = df.build(copy_right_region, target="aie", project="rope/copyR.prj")
join_mod    = df.build(join_region,    target="aie", project="rope/join.prj")
mul32_mod   = df.build(mul32_region,   target="aie", project="rope/mul32.prj")
add32_mod   = df.build(add32_region,   target="aie", project="rope/add32.prj")
sub32_mod   = df.build(sub32_region,   target="aie", project="rope/sub32.prj")

# -------- Host helper: run the full RoPE on one 32x64 tile --------
def rope_apply_packed(
        packed: np.ndarray,
        heads: int,
        head_dim: int = 64,
        max_wavelength: float = 10_000.0,
        pos_offset: int = 0,
    ) -> np.ndarray:
        """
        Apply RoPE to a packed tensor of shape (SEQ, heads*head_dim).
        Returns array with the same shape.

        Requires your existing external kernels:
        radians_mod, pack_mod, sin_mod, cos_mod,
        copyL_mod, copyR_mod, mul32_mod, add32_mod, sub32_mod, join_mod
        """
        seq_len, total_dim = packed.shape
        assert total_dim == heads * head_dim, "packed width must be heads*head_dim"
        # Current AIE kernels are compiled for 32-row tiles and head_dim=64
        tile_rows = SEQ
        D = head_dim
        HALF = D // 2
        assert D == 64 and HALF * 2 == D, "this helper expects HEAD_DIM=64 kernels"

        out = np.empty_like(packed, dtype=np.float32)

        # inv_timescale[k] = max_wavelength^{-(2/D)*k}, k = 0..HALF-1
        k = np.arange(HALF, dtype=np.float32)
        inv_ts = (max_wavelength ** (-(2.0 / D) * k)).astype(np.float32)

        # process sequence in 32-row tiles
        for t0 in range(0, seq_len, tile_rows):
            rows = min(tile_rows, seq_len - t0)

            # positions for this tile (pad to 32 rows)
            pos32 = (pos_offset + np.arange(t0, t0 + rows, dtype=np.float32)).astype(np.float32)
            pos_pad = np.zeros(tile_rows, dtype=np.float32)
            pos_pad[:rows] = pos32

            # radians32 = pos_pad[:,None] * inv_ts[None,:]  (via external kernel)
            radians32 = np.zeros((tile_rows, HALF), dtype=np.float32)
            radians_mod(pos_pad, inv_ts, radians32)

            # pack to 32x64 and get LUT sin/cos
            radians64 = np.zeros((tile_rows, D), dtype=np.float32)
            pack_mod(radians32, radians64)

            sin64 = np.zeros((tile_rows, D), dtype=np.float32)
            cos64 = np.zeros((tile_rows, D), dtype=np.float32)
            sin_mod(radians64, sin64)
            cos_mod(radians64, cos64)

            # rotate each head using same sin/cos tile
            for h in range(heads):
                x_tile = np.zeros((tile_rows, D), dtype=np.float32)
                x_tile[:rows, :] = packed[t0:t0 + rows, h*D:(h+1)*D]

                # split x and (sin,cos) into halves, compute, then join
                xL = np.zeros((tile_rows, HALF), dtype=np.float32)
                xR = np.zeros((tile_rows, HALF), dtype=np.float32)
                s  = np.zeros((tile_rows, HALF), dtype=np.float32)
                c  = np.zeros((tile_rows, HALF), dtype=np.float32)

                copyL_mod(x_tile, xL)
                copyR_mod(x_tile, xR)
                copyL_mod(sin64, s)    # only first 32 cols needed
                copyL_mod(cos64, c)

                tmp1 = np.zeros_like(xL);  mul32_mod(xL, c, tmp1)   # xL*c
                tmp2 = np.zeros_like(xL);  mul32_mod(xR, s, tmp2)   # xR*s
                yL   = np.zeros_like(xL);  sub32_mod(tmp1, tmp2, yL)# yL = xL*c - xR*s

                tmp3 = np.zeros_like(xL);  mul32_mod(xR, c, tmp3)   # xR*c
                tmp4 = np.zeros_like(xL);  mul32_mod(xL, s, tmp4)   # xL*s
                yR   = np.zeros_like(xL);  add32_mod(tmp3, tmp4, yR)# yR = xR*c + xL*s

                y64 = np.zeros((tile_rows, D), dtype=np.float32)
                join_mod(yL, yR, y64)

                out[t0:t0 + rows, h*D:(h+1)*D] = y64[:rows, :]
        return out

if __name__ == "__main__":
    torch.manual_seed(0)

    SEQ, Q_H, D = 64, 15, 64
    q = torch.randn(SEQ, Q_H * D, dtype=torch.float32).numpy()

    # Allo/AIE path
    y_allo = rope_apply_packed(q, heads=Q_H, head_dim=D, max_wavelength=10_000.0, pos_offset=0)

    # Torch reference
    x4  = torch.from_numpy(q).view(SEQ, Q_H, D)
    k   = torch.arange(D // 2, dtype=torch.float32)
    inv = (10_000.0 ** (-(2.0 / D) * k)).view(1, 1, -1)       # [1,1,32]
    pos = torch.arange(SEQ, dtype=torch.float32).view(SEQ, 1, 1)
    rad = pos * inv                                           # [SEQ,1,32] (broadcast over heads)
    s   = torch.sin(rad); c = torch.cos(rad)
    xL  = x4[:, :, :D//2]; xR = x4[:, :, D//2:]
    y_ref = torch.cat([xL*c - xR*s, xR*c + xL*s], dim=-1).reshape(SEQ, Q_H * D).numpy()

    np.testing.assert_allclose(y_allo, y_ref, rtol=1e-3, atol=1e-4)
    print("✅ multi-head RoPE matches PyTorch (SEQ=64, Q_H=15, D=64)")