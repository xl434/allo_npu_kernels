# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model Architecture:
    (lm_expert): LlamaModel(
        (embed_tokens): None
        (layers): ModuleList(
            (0-15): LlamaDecoderLayer(
            (self_attn): LlamaAttention(
                (q_proj): Linear(in_features=720, out_features=960, bias=False)
                (k_proj): Linear(in_features=720, out_features=320, bias=False)
                (v_proj): Linear(in_features=720, out_features=320, bias=False)
                (o_proj): Linear(in_features=960, out_features=720, bias=False)
            )
            (mlp): LlamaMLP(
                (gate_proj): Linear(in_features=720, out_features=2048, bias=False)
                (up_proj): Linear(in_features=720, out_features=2048, bias=False)
                (down_proj): Linear(in_features=2048, out_features=720, bias=False)
                (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((720,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((720,), eps=1e-05)
            )
        )
        (norm): LlamaRMSNorm((720,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import MultiHeadAttention
import numpy as np
import allo
import allo.dataflow as df
from allo.ir.types import float32, bfloat16, int32
from allo.memory import Layout
from allo.backend.aie import ExternalModule
import time

torch.manual_seed(0)
np.random.seed(0)

# ===============================================================================
# Model Configuration
# ===============================================================================
USE_ALL_NPU_KERNELS = True  # if False, we will offload softmax and silu to cpu
KERNEL_LIB_PATH = "../cc/"
BATCH = 1  # fixme: don't care for now
SEQ = 64
EMBD = 768  # 64 * 12
Q_H = 15
KV_H = 5
HEAD_DIM = 64

FFN_HID = EMBD * 4

# assert SEQ == 64, "SEQ must be 64 (to use masked softmax external kernel)"
assert EMBD % 64 == 0, "EMBD must be a multiple of 64"
assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"
assert FFN_HID % EMBD == 0, "FFN_HID must be a multiple of FFN_HID size"

# ===============================================================================
# Torch Version
# ===============================================================================

class AttentionExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        q_proj = nn.Linear(EMBD, Q_H * HEAD_DIM, bias=False)
        k_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)
        v_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)
        o_proj = nn.Linear(Q_H * HEAD_DIM, EMBD, bias=False)

        self.attn = MultiHeadAttention(
            embed_dim=Q_H * HEAD_DIM,
            num_heads=Q_H,
            num_kv_heads=KV_H,
            head_dim=HEAD_DIM,
            q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, output_proj=o_proj,
            is_causal=True,
        )
        self.gate_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.ln_1 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.up_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.down_proj = nn.Linear(FFN_HID, EMBD, bias=False)
        self.silu = nn.SiLU()
        self.ln_2 = nn.RMSNorm(EMBD, elementwise_affine=True)

        # -------- RoPE cache --------
        self.max_wavelength = 10_000.0
        self.head_dim = HEAD_DIM
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
        d_half = self.head_dim // 2

        # timescale_k = max_wavelength^(2k/D), k=0..D/2-1   (float32 buffer)
        freq_exponents = (2.0 / self.head_dim) * torch.arange(d_half, dtype=torch.float32)
        timescale = self.max_wavelength ** freq_exponents
        self.register_buffer("rope_timescale", timescale, persistent=False)

        # optional: small positions cache (we can grow it on demand)
        self.register_buffer("pos_cache", torch.arange(0, 1, dtype=torch.float32), persistent=False)

    def _positions(self, L: int, device) -> torch.Tensor:
        # grow pos_cache if needed
        if self.pos_cache.numel() < L:
            self.pos_cache = torch.arange(L, dtype=torch.float32, device=device)
        return self.pos_cache[:L].unsqueeze(0)  # [1, L]

    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, H, D]  (even D)
        positions: [B, L]  (float32)
        uses cached self.rope_timescale (float32)
        """
        B, L, H, D = x.shape
        d_half = D // 2
        dtype = x.dtype

        # [d_half] -> ensure device match
        ts = self.rope_timescale.to(x.device)  # timescale
        # radians: [B, L, d_half]
        radians = positions.to(torch.float32)[..., None] / ts[None, None, :]
        # broadcast to heads: [B, L, 1, d_half]
        radians = radians[..., None, :]

        x = x.to(torch.float32)
        x1, x2 = x.split(d_half, dim=-1)
        s = torch.sin(radians)
        c = torch.cos(radians)

        out = torch.empty_like(x)
        out[..., :d_half] = x1 * c - x2 * s
        out[..., d_half:] = x2 * c + x1 * s
        return out.to(dtype)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)  # [B, L, EMBD]
        B, L, _ = x.shape
        D = self.head_dim

        # projections
        q = self.attn.q_proj(x).view(B, L, Q_H, D)
        k = self.attn.k_proj(x).view(B, L, KV_H, D)
        v = self.attn.v_proj(x).view(B, L, KV_H, D)

        # RoPE positions (shared across batch)
        pos = self._positions(L, x.device).expand(B, -1)  # [B, L]
        q = self.apply_rope(q, pos)
        k = self.apply_rope(k, pos)

        # grouped-KV map kv_idx[h] = floor(h*KV_H/Q_H)
        kv_map = torch.div(torch.arange(Q_H, device=x.device) * KV_H, Q_H, rounding_mode='floor')
        k_sel = k.index_select(dim=2, index=kv_map)   # [B, L, Q_H, D]
        v_sel = v.index_select(dim=2, index=kv_map)

        # attention
        q_h = q.transpose(1, 2)                        # [B, Q_H, L, D]
        k_h = k_sel.transpose(1, 2)                    # [B, Q_H, L, D]
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (D ** 0.5)  # [B, Q_H, L, L]
        scores.masked_fill_(torch.ones(L, L, device=x.device).triu(1).bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        v_h  = v_sel.transpose(1, 2)                   # [B, Q_H, L, D]
        ctx  = torch.matmul(attn, v_h).transpose(1, 2).contiguous().view(B, L, Q_H * D)

        x = self.attn.output_proj(ctx) + residual

        # MLP
        residual = x
        x = self.ln_2(x)
        act = self.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(act) + residual
        return x


# ===============================================================================
# Allo Version
# ===============================================================================
Ty = float32  # All tensors use float32
N = BATCH * SEQ  # 16   flattened (batch*seq)

# ----------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------
norm = ExternalModule(
    top="rms_norm",
    impl_path=KERNEL_LIB_PATH + "rms_norm.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0
norm_io_layout = Layout("S0R")
norm_arg_layout = Layout("R")

@df.region()
def rms_norm_kernel():
    @df.kernel(mapping=[NORM_P0])
    def core(A: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout, B: Ty[EMBD] @ norm_arg_layout, C: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout):
        norm(A, B, C)

# ----------------------------------------------------------------
# Linear
# ----------------------------------------------------------------
LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
linear_A_layout = Layout("S0R")
linear_B_layout = Layout("RS1")
linear_C_layout = Layout("S0S1")

@df.region()
def linear_matmul_kernel():
    @df.kernel(mapping=[4, 4])
    def gemm(
        A: Ty[LINEAR_M, LINEAR_K] @ linear_A_layout,
        B: Ty[LINEAR_K, LINEAR_N] @ linear_B_layout,
        C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        C[:, :] = allo.matmul(A, B)

@df.region()
def linear_accumulate_kernel():
    @df.kernel(mapping=[2, 4])
    def core(
        A: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        B: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
        C: Ty[LINEAR_M, LINEAR_N] @ linear_C_layout,
    ):
        C[:, :] = allo.add(A, B)

# ----------------------------------------------------------------
# Attention Score
# ----------------------------------------------------------------
attn_score = ExternalModule(
    top="transpose_matmul_with_scale",
    impl_path=KERNEL_LIB_PATH + "transpose_matmul_with_scale.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
ATTN_P0 = 2
ATTN_P1 = 2
ATTN_SCORE_M_TILE = ATTN_P0 * 32
ATTN_SCORE_N_TILE = ATTN_P1 * 32
ATTN_SCORE_LyA = Layout("S0R")
ATTN_SCORE_LyB = Layout("S1R")
ATTN_SCORE_LyC = Layout("S0S1")

@df.region()
def attn_score_kernel():
    @df.kernel(mapping=[ATTN_P0, ATTN_P1])
    def core(
        A: Ty[ATTN_SCORE_M_TILE, HEAD_DIM] @ ATTN_SCORE_LyA,
        B: Ty[ATTN_SCORE_N_TILE, HEAD_DIM] @ ATTN_SCORE_LyB,
        C: Ty[ATTN_SCORE_M_TILE, ATTN_SCORE_N_TILE] @ ATTN_SCORE_LyC,
    ):
        attn_score(A, B, C)

# ----------------------------------------------------------------
# Masked Softmax
# ----------------------------------------------------------------
masked_softmax = ExternalModule(
    top="masked_softmax_float32",
    impl_path=KERNEL_LIB_PATH + "masked_softmax.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
Tint = int32
SOFTMAX_P0 = 2
SOFTMAX_P1 = 3
# SEQ = 128
SOFTMAX_HEAD_TILE = SOFTMAX_P1
SOFTMAX_SEQ_TILE = SEQ // SOFTMAX_P0
SOFTMAX_Ly = Layout("S1S0")
SOFTMAX_ROW_Ly = Layout("S1")

@df.region()
def masked_softmax_kernel():
    @df.kernel(mapping=[SOFTMAX_P0, SOFTMAX_P1])
    def core(
        input_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
        row: Tint[SOFTMAX_P0] @ SOFTMAX_ROW_Ly,
        output_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
    ):
        masked_softmax(input_x, row, output_x)

# ----------------------------------------------------------------
# SiLU
# ----------------------------------------------------------------
silu = ExternalModule(
    top="silu_float32",
    impl_path=KERNEL_LIB_PATH + "silu.cc",
    input_idx=[0],
    output_idx=[1],
)
GELU_P0 = 4
GELU_P1 = 4
GELU_SEQ_TILE = 16
GELU_Ly = Layout("S0S1")

@df.region()
def silu_kernel():
    @df.kernel(mapping=[GELU_P0, GELU_P1])
    def core(
        input_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
        output_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
    ):
        silu(input_x, output_x)

# ----------------------------------------------------------------
# Hadamard
# ----------------------------------------------------------------
@df.region()
def hadamard_kernel():
    @df.kernel(mapping=[1])
    def core(A: Ty[EMBD], B: Ty[EMBD], C: Ty[EMBD]):
        C[:] = allo.mul(A, B)


# ----------------------------------------------------------------
# Rope
# ----------------------------------------------------------------
SEQ, HEAD_DIM  = 64, 64
HEAD_DIM_HALF = HEAD_DIM // 2

# Layouts / dtypes
Ty = float32
VecLy = Layout("S0")
MatLy = Layout("S0S1")

# External kernels (ROPE)
OPS_IMPL = KERNEL_LIB_PATH + "rope_vec_ops.cc"
SIN_IMPL = KERNEL_LIB_PATH + "sine.cc"
COS_IMPL = KERNEL_LIB_PATH + "cosine.cc"

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

# ##############################################################
# BUILD
# ##############################################################
masked_softmax_mod = df.build(
    masked_softmax_kernel, target="aie", project="llama/masked_softmax.prj"
)
rms_norm_mod = df.build(rms_norm_kernel, target="aie", project="llama/rms_norm.prj")
linear_matmul_mod = df.build(
    linear_matmul_kernel, target="aie", project="llama/linear_matmul.prj"
)
linear_accumulate_mod = df.build(
    linear_accumulate_kernel, target="aie", project="llama/linear_accumulate.prj"
)
attn_score_mod = df.build(
    attn_score_kernel, target="aie", project="llama/attn_score.prj"
)
silu_mod = df.build(silu_kernel, target="aie", project="llama/silu.prj")
hadamard_mod = df.build(
    hadamard_kernel, target="aie", project="llama/hadamard.prj"
)

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


# ##############################################################
# TOOL
# ##############################################################
def linear_projection(A, B, C, M, N, K):
    assert A.ndim == 2 and B.ndim == 2 and C.ndim == 2, \
        f"Inputs must be 2D: got A.ndim={A.ndim}, B.ndim={B.ndim}, C.ndim={C.ndim}."

    assert A.shape == (M, K), \
        f"A.shape {A.shape} does not match (M, K)=({M}, {K})."

    assert B.shape == (K, N), \
        f"B.shape {B.shape} does not match (K, N)=({K}, {N})."

    assert C.shape == (M, N), \
        f"C.shape {C.shape} does not match (M, N)=({M}, {N})."
    
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            C_tmp = np.zeros((LINEAR_M, LINEAR_N)).astype(np.float32)
            for k in range(K // LINEAR_K):
                tile_A = A[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                ]
                tile_B = B[
                    k * LINEAR_K : (k + 1) * LINEAR_K,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ]
                linear_matmul_mod(tile_A, tile_B, C_tmp)
                linear_accumulate_mod(
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                    C_tmp,
                    C[
                        i * LINEAR_M : (i + 1) * LINEAR_M,
                        j * LINEAR_N : (j + 1) * LINEAR_N,
                    ],
                )
                
def rmsnorm(input_x, weight, output_x):
    for i in range(SEQ // NORM_SEQ_TILE):
        tile_input = input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        rms_norm_mod(
            tile_input,
            weight,
            # bias,
            output_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
        )

def add_residual(residual, x, M, N):
    """
    reuse 'linear_accumulate_mod' for residual
    residual = residual + x
    """
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            linear_accumulate_mod(
                residual[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
                x[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
                residual[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
            )

def masked_softmax(attention_score, attention_weight):
    row_idx = np.array(list(range(0, SEQ, SOFTMAX_SEQ_TILE))).astype(np.int32)
    for i in range(Q_H // SOFTMAX_HEAD_TILE):
        masked_softmax_mod(
            attention_score[
                :, i * SOFTMAX_HEAD_TILE : (i + 1) * SOFTMAX_HEAD_TILE, :
            ],
            row_idx,
            attention_weight[
                :,
                i * (SOFTMAX_HEAD_TILE * SEQ) : (i + 1) * (SOFTMAX_HEAD_TILE * SEQ),
            ],
        )
def rowwise_hadamard(A, B, C):
    assert A.shape == B.shape == C.shape
    N = A.shape[0]
    M = A.shape[1]
    for i in range(N):
        for j in range(M // EMBD):
            hadamard_mod(A[i, j * EMBD : (j + 1) * EMBD], 
            B[i, j * EMBD : (j + 1) * EMBD], 
            C[i, j * EMBD : (j + 1) * EMBD])

def rope_apply_packed(
    packed: np.ndarray,
    heads: int,
    head_dim: int = 64,
    max_wavelength: float = 10_000.0,
    pos_offset: int = 0,
) -> np.ndarray:
    
    seq_len, total_dim = packed.shape
    assert total_dim == heads * head_dim, "packed width must be heads*head_dim"
    # Current AIE kernels are compiled for 32-row tiles and head_dim=64
    tile_rows = SEQ
    D = head_dim
    HALF = D // 2
    assert D == 64 and HALF * 2 == D

    out = np.empty_like(packed, dtype=np.float32)

    k = np.arange(HALF, dtype=np.float32)
    inv_ts = (max_wavelength ** (-(2.0 / D) * k)).astype(np.float32)

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

def llama_block_rope(x_fp32: np.ndarray, params: dict):
    # ##############################################################
    # FORWARD
    # ##############################################################
    x = x_fp32.astype(np.float32)
    residual = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=np.float32)
    rmsnorm(residual, params["W_norm_1"], x)

    query = np.zeros((SEQ, Q_H * HEAD_DIM)).astype(np.float32)     ## need rope
    key = np.zeros((SEQ, KV_H * HEAD_DIM)).astype(np.float32)      ## need rope
    value = np.zeros((SEQ, KV_H * HEAD_DIM)).astype(np.float32)
    linear_projection(x, params["Wq"], query, SEQ, Q_H * HEAD_DIM, EMBD)
    linear_projection(x, params["Wk"], key, SEQ, KV_H * HEAD_DIM, EMBD)
    linear_projection(x, params["Wv"], value, SEQ, KV_H * HEAD_DIM, EMBD)

    query = rope_apply_packed(query, heads=Q_H, head_dim=HEAD_DIM)
    key   = rope_apply_packed(key,   heads=KV_H, head_dim=HEAD_DIM)


    # attention score
    attention_score = np.empty((SEQ, Q_H, SEQ), dtype=np.float32)
    for i in range(SEQ // ATTN_SCORE_M_TILE):
        for j in range(SEQ // ATTN_SCORE_N_TILE):
            for k in range(Q_H):
                k_key_idx = int(k * KV_H // Q_H)
                attn_score_mod(
                    query[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k * HEAD_DIM : (k + 1) * HEAD_DIM,
                    ],
                    key[
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                        k_key_idx * HEAD_DIM : (k_key_idx + 1) * HEAD_DIM,
                    ],
                    attention_score[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k,
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                    ],
                )

    # safe softmax
    if USE_ALL_NPU_KERNELS:
        attn_weight = np.zeros((SEQ, Q_H * SEQ)).astype(np.float32)
        masked_softmax(attention_score, attn_weight)
    else:
        mask = torch.triu(torch.ones(SEQ, SEQ), 1).bool()
        mask = np.repeat(mask[:, np.newaxis, :], Q_H, axis=1)
        attention_score[mask == 1] = -np.inf
        tensor_atten_score = torch.from_numpy(attention_score)
        attn_weight = F.softmax(tensor_atten_score, dim=-1)
        attn_weight = attn_weight.numpy()

    # attention value
    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM)).astype(np.float32)
    for k in range(Q_H):
        kv_idx = int(k * KV_H // Q_H)
        linear_projection(
            (
                attn_weight[:, k * SEQ : (k + 1) * SEQ]
                if USE_ALL_NPU_KERNELS
                else attn_weight[:, k, :]
            ),
            value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM],
            attn_value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM],
            SEQ,
            HEAD_DIM,
            SEQ,
        )
    # output projection
    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(attn_value, params["Wo"], x, SEQ, EMBD, Q_H * HEAD_DIM)
    # add residual
    add_residual(residual, x, SEQ, EMBD)
    # norm
    rmsnorm(residual, params["W_norm_2"], x)
    # up projection
    up_proj_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
    linear_projection(x, params["W_up"], up_proj_x, SEQ, FFN_HID, EMBD)
    # gate projection
    gate_proj_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
    linear_projection(x, params["W_gate"], gate_proj_x, SEQ, FFN_HID, EMBD)

    if USE_ALL_NPU_KERNELS:
        activeated_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
        for i in range(SEQ // GELU_SEQ_TILE):
            silu_mod(
                gate_proj_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
                activeated_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
            )
        rowwise_hadamard(activeated_x, up_proj_x, activeated_x)
    else:
        tensor_gate_proj_x = torch.from_numpy(gate_proj_x)
        tensor_up_proj_x = torch.from_numpy(up_proj_x)
        silu_func = nn.SiLU()
        activeated_x = (silu_func(tensor_gate_proj_x) * tensor_up_proj_x).numpy()

    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    print("x", x.shape)
    print("W_gate", params["W_gate"].shape)
    print("gate_proj_x", gate_proj_x.shape)
    linear_projection(activeated_x, params["W_down"], x, SEQ, EMBD, FFN_HID)
    add_residual(residual, x, SEQ, EMBD)
    return residual

def llama_block_rope_cross(
    query: np.ndarray,   # [SEQ, Q_H * HEAD_DIM]  (RoPE applied upstream if desired)
    key:   np.ndarray,   # [SEQ, KV_H * HEAD_DIM]
    value: np.ndarray,   # [SEQ, KV_H * HEAD_DIM]
    x_fp32: np.ndarray,
    params: dict,
):
    # ##############################################################
    # FORWARD (cross-attn variant: Q/K/V are inputs)
    # ##############################################################
    x = x_fp32.astype(np.float32)
    residual = x.reshape(SEQ, EMBD)

    x = np.empty((SEQ, EMBD), dtype=np.float32)
    rmsnorm(residual, params["W_norm_1"], x)

    # attention score
    attention_score = np.empty((SEQ, Q_H, SEQ), dtype=np.float32)
    for i in range(SEQ // ATTN_SCORE_M_TILE):
        for j in range(SEQ // ATTN_SCORE_N_TILE):
            for k in range(Q_H):
                k_key_idx = int(k * KV_H // Q_H)
                attn_score_mod(
                    query[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k * HEAD_DIM : (k + 1) * HEAD_DIM,
                    ],
                    key[
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                        k_key_idx * HEAD_DIM : (k_key_idx + 1) * HEAD_DIM,
                    ],
                    attention_score[
                        i * ATTN_SCORE_M_TILE : (i + 1) * ATTN_SCORE_M_TILE,
                        k,
                        j * ATTN_SCORE_N_TILE : (j + 1) * ATTN_SCORE_N_TILE,
                    ],
                )

    # safe softmax
    if USE_ALL_NPU_KERNELS:
        attn_weight = np.zeros((SEQ, Q_H * SEQ)).astype(np.float32)
        masked_softmax(attention_score, attn_weight)
    else:
        mask = torch.triu(torch.ones(SEQ, SEQ), 1).bool()
        mask = np.repeat(mask[:, np.newaxis, :], Q_H, axis=1)
        attention_score[mask == 1] = -np.inf
        tensor_atten_score = torch.from_numpy(attention_score)
        attn_weight = F.softmax(tensor_atten_score, dim=-1)
        attn_weight = attn_weight.numpy()

    # attention value
    attn_value = np.zeros((SEQ, Q_H * HEAD_DIM)).astype(np.float32)
    for k in range(Q_H):
        kv_idx = int(k * KV_H // Q_H)
        linear_projection(
            (
                attn_weight[:, k * SEQ : (k + 1) * SEQ]
                if USE_ALL_NPU_KERNELS
                else attn_weight[:, k, :]
            ),
            value[:, kv_idx * HEAD_DIM : (kv_idx + 1) * HEAD_DIM],
            attn_value[:, k * HEAD_DIM : (k + 1) * HEAD_DIM],
            SEQ,
            HEAD_DIM,
            SEQ,
        )
    # output projection
    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    linear_projection(attn_value, params["Wo"], x, SEQ, EMBD, Q_H * HEAD_DIM)
    # add residual
    add_residual(residual, x, SEQ, EMBD)
    # norm
    rmsnorm(residual, params["W_norm_2"], x)
    # up projection
    up_proj_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
    linear_projection(x, params["W_up"], up_proj_x, SEQ, FFN_HID, EMBD)
    # gate projection
    gate_proj_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
    linear_projection(x, params["W_gate"], gate_proj_x, SEQ, FFN_HID, EMBD)

    if USE_ALL_NPU_KERNELS:
        activeated_x = np.zeros((SEQ, FFN_HID)).astype(np.float32)
        for i in range(SEQ // GELU_SEQ_TILE):
            silu_mod(
                gate_proj_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
                activeated_x[i * GELU_SEQ_TILE : (i + 1) * GELU_SEQ_TILE, :],
            )
        rowwise_hadamard(activeated_x, up_proj_x, activeated_x)
    else:
        tensor_gate_proj_x = torch.from_numpy(gate_proj_x)
        tensor_up_proj_x = torch.from_numpy(up_proj_x)
        silu_func = nn.SiLU()
        activeated_x = (silu_func(tensor_gate_proj_x) * tensor_up_proj_x).numpy()

    x = np.zeros((SEQ, EMBD)).astype(np.float32)
    print("x", x.shape)
    print("W_gate", params["W_gate"].shape)
    print("gate_proj_x", gate_proj_x.shape)
    linear_projection(activeated_x, params["W_down"], x, SEQ, EMBD, FFN_HID)
    add_residual(residual, x, SEQ, EMBD)

    return residual


def llama_q_from_emb_rope(text_seq_np: np.ndarray, params: dict):
    """
    Input:
      text_seq_np: [SEQ(=64), EMBD] float32 — text tokens only
    Output:
      Q_exp: [SEQ, Q_H*HEAD_DIM] with RoPE applied
      X_pre: [SEQ, EMBD]
    """
    assert text_seq_np.shape == (SEQ, EMBD)
    x = text_seq_np.astype(np.float32)

    # pre-norm
    X_pre = np.zeros((SEQ, EMBD), dtype=np.float32)
    rmsnorm(x, params["W_norm_1"], X_pre)

    # Q projection
    Q_exp = np.zeros((SEQ, Q_H * HEAD_DIM)).astype(np.float32)
    linear_projection(X_pre, params["Wq"], Q_exp, SEQ, Q_H * HEAD_DIM, EMBD)

    Q_exp = rope_apply_packed(Q_exp, heads=Q_H, head_dim=HEAD_DIM)

    return Q_exp, X_pre

def vlm_qkv_from_mm_seq(x_fp32: np.ndarray, params: dict):
    """
    Input:
      mm_seq_np: [SEQ, EMBD]
    Output:
      K_vlm: [SEQ, KV_H*HEAD_DIM]   (no RoPE; for expert cross-attn)
      V_vlm: [SEQ, KV_H*HEAD_DIM]   (no RoPE; for expert cross-attn)
      Y_vlm: [SEQ, EMBD]            (post 1-depth VLM self-attn output)
      X_pre: [SEQ, EMBD]            (pre-norm hidden used to produce K,V)
    """
    assert x_fp32.shape == (SEQ, EMBD)
    x = x_fp32.astype(np.float32)
    X_pre = x.reshape(SEQ, EMBD)
    x = np.empty((SEQ, EMBD), dtype=np.float32)
    rmsnorm(X_pre, params["W_norm_1"], x)

    key = np.zeros((SEQ, KV_H * HEAD_DIM)).astype(np.float32)
    value = np.zeros((SEQ, KV_H * HEAD_DIM)).astype(np.float32)
    linear_projection(x, params["Wk"], key, SEQ, KV_H * HEAD_DIM, EMBD)
    linear_projection(x, params["Wv"], value, SEQ, KV_H * HEAD_DIM, EMBD)

    # ----- full VLM layer output (self-attn with RoPE on Q,K inside) -----
    
    Y_vlm = llama_block_rope(X_pre, params)
    Y_vlm = x

    return key, value, Y_vlm, X_pre


import time

if __name__ == "__main__":
    ref_model = AttentionExpertBlock().eval()
    # reference weights (float32)
    p = {n: v.detach().numpy() for n, v in ref_model.named_parameters()}
    params_fp32 = {
        "Wq": p["attn.q_proj.weight"].T,
        "Wk": p["attn.k_proj.weight"].T,
        "Wv": p["attn.v_proj.weight"].T,
        "Wo": p["attn.output_proj.weight"].T,
        "W_gate": p["gate_proj.weight"].T,
        "W_up": p["up_proj.weight"].T,
        "W_down": p["down_proj.weight"].T,
        "W_norm_1": p["ln_1.weight"],
        "W_norm_2": p["ln_2.weight"],
    }

    params = {
        k: v.astype(np.float32) if isinstance(v, np.ndarray) else v
        for k, v in params_fp32.items()
    }

    # random input
    x_float = torch.randn(BATCH, SEQ, EMBD)

    # ---- timings ----
    t0 = time.time()
    sample = ref_model(x_float)[0, :, :]
    t1 = time.time()

    x_float = x_float[0, :, :]
    a0 = time.time()
    allo_out = llama_block_rope(x_float.numpy(), params)
    a1 = time.time()

    print(f"PyTorch forward time: {t1 - t0:.6f} s | Allo forward time: {a1 - a0:.6f} s")

    # ---- correctness check ----
    np.testing.assert_allclose(allo_out, sample.detach().numpy(), rtol=1e-1)
    print("Allo float32 block matches PyTorch float32 reference within tolerance ✔️")
