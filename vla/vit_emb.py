import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

# Allo
import allo
import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout


# -----------------------------
# Config & Embedding Class
# -----------------------------
@dataclass
class SiglipVisionConfig:
    num_channels: int = 3
    embed_dim: int = 768
    image_size: int = 128
    patch_size: int = 16


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, config.embed_dim)
        self.register_buffer(
            "position_ids", torch.arange(self.num_patches).expand(1, -1), persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        x = self.patch_embedding(pixel_values)          # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2).contiguous()   # [B, N, D]
        x = x + self.position_embedding(self.position_ids)
        return x

# -----------------------------
# Allo Linear Projection Kernel
# -----------------------------
LINEAR_M, LINEAR_N, LINEAR_K = 64, 64, 64
Ty = float32
A_Ly = Layout("S0R")
B_Ly = Layout("RS1")
C_Ly = Layout("S0S1")

@df.region()
def linear_matmul_kernel():
    @df.kernel(mapping=[4, 4])
    def gemm(
        A: Ty[LINEAR_M, LINEAR_K] @ A_Ly,
        B: Ty[LINEAR_K, LINEAR_N] @ B_Ly,
        C: Ty[LINEAR_M, LINEAR_N] @ C_Ly,
    ):
        C[:, :] = allo.matmul(A, B)
@df.region()
def linear_accumulate_kernel():
    @df.kernel(mapping=[2, 4])
    def core(
        A: Ty[LINEAR_M, LINEAR_N] @ C_Ly,
        B: Ty[LINEAR_M, LINEAR_N] @ C_Ly,
        C: Ty[LINEAR_M, LINEAR_N] @ C_Ly,
    ):
        C[:, :] = allo.add(A, B)

# -----------------------------
# Vision Embedding kernel: copy one [P,P] tile
# -----------------------------
cfg = SiglipVisionConfig()
C, H, W, P = cfg.num_channels, cfg.image_size, cfg.image_size, cfg.patch_size
OUT_H, OUT_W = H // P, W // P
N = OUT_H * OUT_W
K = C * P * P   # flattened patch length

@df.region()
def tile_copy_region():
    @df.kernel(mapping=[2, 2])
    def tile_copy(
        X_tile: float32[P, P] @ C_Ly,
        Y_tile: float32[P, P] @ C_Ly,
    ):
        Y_tile[:, :] = X_tile[:, :]


linear_matmul_mod = df.build(linear_matmul_kernel, target="aie-mlir", project="emb.proj")
linear_accumulate_mod = df.build(linear_accumulate_kernel, target="aie-mlir", project="emb.prj")
tile_copy_mod = df.build(tile_copy_region, target="aie-mlir", project="emb.prj")

# -----------------------------
# General helper: loop over patches and channels
# -----------------------------

def linear_projection(A, B, C, M, N, K):
    assert A.shape == (M, K) and B.shape == (K, N) and C.shape == (M, N)
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
def linear_accumulation(A, B, C, M, N):
    assert A.shape == (M, N) and B.shape == (M, N) and C.shape == (M, N)
    for i in range(M // LINEAR_M):
        for j in range(N // LINEAR_N):
            linear_accumulate_mod(
                A[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
                B[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ], 
                C[
                    i * LINEAR_M : (i + 1) * LINEAR_M,
                    j * LINEAR_N : (j + 1) * LINEAR_N,
                ],
            )

def allo_unfold_host(img_bchw: np.ndarray) -> np.ndarray:
    """
    img_bchw: [1, C, H, W]
    return: X_cols [N, K] patches
    """
    assert img_bchw.shape == (1, C, H, W)
    X_cols = np.empty((N, K), dtype=np.float32)

    n = 0
    for oh in range(OUT_H):
        for ow in range(OUT_W):
            out_patch = np.empty((C, P, P), dtype=np.float32)
            for c in range(C):
                X_tile = np.ascontiguousarray(
                    img_bchw[0, c, oh*P:oh*P+P, ow*P:ow*P+P].astype(np.float32)
                )
                Y_tile = np.empty((P, P), dtype=np.float32)
                tile_copy_mod(X_tile, Y_tile)
                out_patch[c] = Y_tile
            # flatten [C,P,P] into [K] row
            X_cols[n] = out_patch.reshape(-1)
            n += 1
    return X_cols

# -----------------------------
# siglip embeddings Function (A, B with Allo, C, D in torch)
# -----------------------------
def siglip_embeddings_allo(torch_model, img_bchw: np.ndarray) -> np.ndarray:
    """
    Allo path for SigLIP embeddings:
      1) Unfold image into patch matrix [N, K]
      2) GEMM with conv weights reshaped to [K, D]
      3) Add conv bias
      4) Add learned position embeddings
    Returns:
      out: np.ndarray of shape [1, N, D]
    """
    assert img_bchw.ndim == 4 and img_bchw.shape[0] == 1, "Expect [1, C, H, W]"

    # --- derive sizes from the torch model ---
    B = 1
    C = torch_model.patch_embedding.in_channels
    D = torch_model.patch_embedding.out_channels
    P = torch_model.patch_embedding.kernel_size[0]
    stride = torch_model.patch_embedding.stride[0]
    assert P == stride, "Connector assumes non-overlapping patches"
    H = W = int(np.sqrt(torch_model.num_patches) * P)
    N = torch_model.num_patches
    K = C * P * P

    # --- fetch weights ---
    with torch.no_grad():
        W_conv = torch_model.patch_embedding.weight.detach().cpu().numpy()         # [D, C, P, P]
        b_conv = torch_model.patch_embedding.bias.detach().cpu().numpy()           # [D]
        W_col  = W_conv.reshape(D, -1).T.copy()                                    # [K, D]
        pos_ref = torch_model.position_embedding(torch_model.position_ids)[0].cpu().numpy()  # [N, D]

    # --- Allo path ---
    # [1, C, H, W] -> [N, K]
    X_cols = allo_unfold_host(img_bchw.astype(np.float32))                         # [N, K]

    # [N, K] @ [K, D] -> [N, D]
    out_mm = np.zeros((N, D), dtype=np.float32)
    linear_projection(X_cols, W_col, out_mm, N, D, K)

    # + bias
    out_bias = np.zeros_like(out_mm)
    b_broadcast = np.broadcast_to(b_conv, (N, D))
    linear_accumulation(out_mm, b_broadcast, out_bias, N, D)

    # + position embedding
    out = np.zeros_like(out_bias)
    linear_accumulation(out_bias, pos_ref, out, N, D)

    # add batch dim
    return out[None, ...]  # [1, N, D]


import time

if __name__ == "__main__":
    # --- setup torch reference ---
    config = SiglipVisionConfig()
    model = SiglipVisionEmbeddings(config).eval()
    torch.manual_seed(0)
    np.random.seed(0)

    # random input
    img = torch.randn(
        1,
        config.num_channels,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )
    img_np = img.numpy()

    with torch.no_grad():
        torch_full = model(img)[0]  # [N, D]
        torch_out = torch_full[None, ...].cpu().numpy()

    allo_out = siglip_embeddings_allo(model, img_np)

    # --- compare ---
    max_abs = float(np.max(np.abs(allo_out - torch_out)))
    print(f"Max |diff| (Allo vs Torch): {max_abs:.3e}")
    np.testing.assert_allclose(allo_out, torch_out, rtol=1e-4, atol=2e-5)

    print(torch_out.shape)
    print("✅ Allo embeddings match Torch within tolerance")
