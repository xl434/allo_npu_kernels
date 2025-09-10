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

assert SEQ == 64, "SEQ must be 64 (to use masked softmax external kernel)"
assert EMBD % 64 == 0, "EMBD must be a multiple of 64"
assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"
assert FFN_HID % EMBD == 0, "FFN_HID must be a multiple of FFN_HID size"


# ===============================================================================
# Torch Version
# ===============================================================================

class AttentionExpertBlock(nn.Module):

    def __init__(self):
        super().__init__()
        q_proj = nn.Linear(EMBD, Q_H * HEAD_DIM, bias=False)   # 720 → 960
        k_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)  # 720 → 320
        v_proj = nn.Linear(EMBD, KV_H * HEAD_DIM, bias=False)  # 720 → 320
        o_proj = nn.Linear(Q_H * HEAD_DIM, EMBD, bias=False)   # 960 → 720

        self.attn = MultiHeadAttention(
            embed_dim=Q_H * HEAD_DIM,
            num_heads=Q_H,
            num_kv_heads=KV_H,
            head_dim=HEAD_DIM,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=o_proj,
            # pos_embeddings
            is_causal=True,  # causal attention for decoder
        )
        self.gate_proj = nn.Linear(EMBD, FFN_HID, bias=False) 
        self.ln_1 = nn.RMSNorm(EMBD, elementwise_affine=True)
        self.up_proj = nn.Linear(EMBD, FFN_HID, bias=False)
        self.down_proj = nn.Linear(FFN_HID, EMBD, bias=False)
        self.silu = nn.SiLU()
        self.ln_2 = nn.RMSNorm(EMBD, elementwise_affine=True)
        
        # self.attn.in_proj_bias.data.zero_()
        # self.attn.out_proj.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        activeated_x = self.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(activeated_x)
        return x


# ===============================================================================
# Allo Version
# ===============================================================================
Ty = float32  # All tensors use float32
N = BATCH * SEQ  # 16   flattened (batch*seq)

def run(x_fp32: np.ndarray, params: dict):

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

    # ##############################################################
    # BUILD
    # ##############################################################
    linear_matmul_mod = df.build(
        linear_matmul_kernel, target="aie", project="linear_matmul.prj"
    )
    linear_accumulate_mod = df.build(
        linear_accumulate_kernel, target="aie", project="linear_accumulate.prj"
    )
    silu_mod = df.build(silu_kernel, target="aie", project="silu.prj")
    hadamard_mod = df.build(
        hadamard_kernel, target="aie", project="hadamard.prj"
    )

    # ##############################################################
    # TOOL
    # ##############################################################

    def linear_projection(A, B, C, M, N, K):
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
    def rowwise_hadamard(A, B, C):
        assert A.shape == B.shape == C.shape
        N = A.shape[0]
        M = A.shape[1]
        for i in range(N):
            for j in range(M // EMBD):
                hadamard_mod(A[i, j * EMBD : (j + 1) * EMBD], 
                B[i, j * EMBD : (j + 1) * EMBD], 
                C[i, j * EMBD : (j + 1) * EMBD])

    # ##############################################################
    # FORWARD
    # ##############################################################
    x = x_fp32.astype(np.float32)

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
    return x


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
    # test
    sample = ref_model(x_float)[0, :, :]
    x_float = x_float[0,:,:]
    print(x_float.shape)
    allo_out = run(x_float.numpy(), params)
    np.testing.assert_allclose(allo_out, sample.detach().numpy(), rtol=1e-1)
    print("Allo float32 block matches PyTorch float32 reference within tolerance")