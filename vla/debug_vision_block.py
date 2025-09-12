# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
(vision_model): SmolVLMVisionTransformer(
            (embeddings): SmolVLMVisionEmbeddings(
              (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=valid)
              (position_embedding): Embedding(1024, 768)
            )
            (encoder): SmolVLMEncoder(
              (layers): ModuleList(
                (0-11): 12 x SmolVLMEncoderLayer(
                  (self_attn): SmolVLMVisionAttention(
                    (k_proj): Linear(in_features=768, out_features=768, bias=True)
                    (v_proj): Linear(in_features=768, out_features=768, bias=True)
                    (q_proj): Linear(in_features=768, out_features=768, bias=True)
                    (out_proj): Linear(in_features=768, out_features=768, bias=True)
                  )
                  (layer_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                  (mlp): SmolVLMVisionMLP(
                    (activation_fn): PytorchGELUTanh()
                    (fc1): Linear(in_features=768, out_features=3072, bias=True)
                    (fc2): Linear(in_features=3072, out_features=768, bias=True)
                  )
                  (layer_norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                )
              )
            )
            (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
USE_ALL_NPU_KERNELS = True  # if False, we will offload softmax and gelu to cpu
KERNEL_LIB_PATH = "../cc/"
BATCH = 1  # fixme: don't care for now
SEQ = 32
EMBD = 768  # 64 * 12
N_HEAD = 12
HEAD_DIM = EMBD // N_HEAD
FFN_HID = EMBD * 4

# assert SEQ == 64, "SEQ must be 64 (to use masked softmax external kernel)"
assert EMBD % 64 == 0, "EMBD must be a multiple of 64"
assert HEAD_DIM % 64 == 0, "HEAD_DIM must be a multiple of 64"


# ===============================================================================
# Torch Version
# ===============================================================================
class MiniVit(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBD, N_HEAD, batch_first=True)
        self.ln_1 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.ffn_up = nn.Linear(EMBD, FFN_HID, bias=False)
        self.ffn_down = nn.Linear(FFN_HID, EMBD, bias=False)
        self.gelu = nn.GELU()
        self.ln_2 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.attn.in_proj_bias.data.zero_()
        self.attn.out_proj.bias.data.zero_()


    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            need_weights=False,
            # attn_mask=torch.triu(torch.ones(SEQ, SEQ), 1).bool(),
        )
        x = attn_out + residual
        residual = x
        x = self.ln_2(x)
        activeated_x = self.gelu(self.ffn_up(x))
        x = self.ffn_down(activeated_x)
        x = residual + x
        return x


# ========= Debug helpers =========
from collections import OrderedDict

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def _snr_db(ref, err):
    # 20*log10(||ref|| / ||err||)
    ref_n = np.linalg.norm(ref.ravel())
    err_n = np.linalg.norm(err.ravel())
    if err_n == 0:
        return float('inf')
    if ref_n == 0:
        return -float('inf')
    return 20.0 * np.log10(ref_n / err_n)

def _relative_l2(a, b):
    denom = np.linalg.norm(b.ravel())
    num = np.linalg.norm((a - b).ravel())
    return num / (denom + 1e-12)

def _mismatch_rate(a, b, atol=1e-5, rtol=1e-3):
    diff = np.abs(a - b)
    thresh = atol + rtol * np.abs(b)
    return float(np.mean(diff > thresh))

def _metrics(a, b, name):
    a = _to_numpy(a).astype(np.float32)
    b = _to_numpy(b).astype(np.float32)
    err = a - b
    return {
        "name": name,
        "shape": tuple(a.shape),
        "SNR[dB]": _snr_db(b, err),
        "max_abs_err": float(np.max(np.abs(err))),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "rel_L2": _relative_l2(a, b),
        "mismatch_rate": _mismatch_rate(a, b),
    }

def _print_table(rows):
    # compact aligned table
    headers = ["step","shape","SNR[dB]","max_abs_err","mean_abs_err","rel_L2","mismatch_rate"]
    colw = [max(len(h), max(len(str(r[k])) for r in rows)) for k,h in enumerate(headers)]
    def fmt_row(r):
        return "  ".join([
            str(r["name"]).ljust(colw[0]),
            str(r["shape"]).ljust(colw[1]),
            f'{r["SNR[dB]"]:.2f}'.rjust(colw[2]),
            f'{r["max_abs_err"]:.4e}'.rjust(colw[3]),
            f'{r["mean_abs_err"]:.4e}'.rjust(colw[4]),
            f'{r["rel_L2"]:.4e}'.rjust(colw[5]),
            f'{r["mismatch_rate"]:.4f}'.rjust(colw[6]),
        ])
    header_line = "  ".join([headers[i].ljust(colw[i]) for i in range(len(headers))])
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(fmt_row(r))



# ===============================================================================
# Allo Version
# ===============================================================================


Ty = float32  # All tensors use float32
N = BATCH * SEQ  # 16   flattened (batch*seq)
# ----------------------------------------------------------------
# LayerNorm
# ----------------------------------------------------------------
norm = ExternalModule(
    top="layer_norm",
    impl_path=KERNEL_LIB_PATH + "layer_norm.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
NORM_P0 = 4
NORM_SEQ_TILE = 16
NORM_TILE = NORM_SEQ_TILE // NORM_P0
norm_io_layout = Layout("S0R")
norm_arg_layout = Layout("R")

@df.region()
def layer_norm_kernel():
    pipe = df.array(
        df.pipe(dtype=Ty, shape=(NORM_TILE, EMBD), depth=1), shape=(NORM_P0,)
    )

    @df.kernel(mapping=[NORM_P0])
    def norm_no_bias(
        input_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
        weight: Ty[EMBD] @ norm_arg_layout,
    ):
        pi = df.get_pid()
        tmp: Ty[NORM_TILE, EMBD] = 0
        norm(input_x, weight, tmp)
        pipe[pi].put(tmp)

    @df.kernel(mapping=[NORM_P0])
    def norm_add_bias(
        bias: Ty[EMBD] @ norm_arg_layout,
        output_x: Ty[NORM_SEQ_TILE, EMBD] @ norm_io_layout,
    ):
        pi = df.get_pid()
        data = pipe[pi].get()
        output_x[:, :] = allo.add(data, bias)

# ----------------------------------------------------------------
# Linear
# ----------------------------------------------------------------
LINEAR_M, LINEAR_N, LINEAR_K = 32, 64, 32
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
ATTN_P0 = 1
ATTN_P1 = 1
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
softmax = ExternalModule(
    top="softmax_float32",
    impl_path=KERNEL_LIB_PATH + "softmax.cc",
    input_idx=[0, 1],
    output_idx=[2],
)
Tint = int32
SOFTMAX_P0 = 1
SOFTMAX_P1 = 4
SOFTMAX_HEAD_TILE = SOFTMAX_P1
SOFTMAX_SEQ_TILE = SEQ // SOFTMAX_P0
SOFTMAX_Ly = Layout("S1S0")
SOFTMAX_ROW_Ly = Layout("S1")

@df.region()
def softmax_kernel():
    @df.kernel(mapping=[SOFTMAX_P0, SOFTMAX_P1])
    def core(
        input_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
        row: Tint[SOFTMAX_P0] @ SOFTMAX_ROW_Ly,
        output_x: Ty[SEQ, SEQ * SOFTMAX_HEAD_TILE] @ SOFTMAX_Ly,
    ):
        softmax(input_x, row, output_x)

# ----------------------------------------------------------------
# Gelu
# ----------------------------------------------------------------
gelu = ExternalModule(
    top="gelu_float32",
    impl_path=KERNEL_LIB_PATH + "gelu.cc",
    input_idx=[0],
    output_idx=[1],
)
GELU_P0 = 4
GELU_P1 = 4
GELU_SEQ_TILE = 16
GELU_Ly = Layout("S0S1")

@df.region()
def gelu_kernel():
    @df.kernel(mapping=[GELU_P0, GELU_P1])
    def core(
        input_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
        output_x: Ty[GELU_SEQ_TILE, FFN_HID] @ GELU_Ly,
    ):
        gelu(input_x, output_x)

# ##############################################################
# BUILD
# ##############################################################
softmax_mod = df.build(
    softmax_kernel, target="aie", project="softmax.prj"
)
layer_norm_mod = df.build(layer_norm_kernel, target="aie", project="norm.prj")
linear_matmul_mod = df.build(
    linear_matmul_kernel, target="aie", project="linear_matmul.prj"
)
linear_accumulate_mod = df.build(
    linear_accumulate_kernel, target="aie", project="linear_accumulate.prj"
)
attn_score_mod = df.build(
    attn_score_kernel, target="aie", project="attn_score.prj"
)
gelu_mod = df.build(gelu_kernel, target="aie", project="gelu.prj")

# ##############################################################
# TOOL
# ##############################################################
def layernorm(input_x, weight, bias, output_x):
    for i in range(SEQ // NORM_SEQ_TILE):
        tile_input = input_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :]
        layer_norm_mod(
            tile_input,
            weight,
            bias,
            output_x[i * NORM_SEQ_TILE : (i + 1) * NORM_SEQ_TILE, :],
        )

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

def softmax(attention_score, attention_weight):
    row_idx = np.array(list(range(0, SEQ, SOFTMAX_SEQ_TILE))).astype(np.int32)
    for i in range(N_HEAD // SOFTMAX_HEAD_TILE):
        softmax_mod(
            attention_score[
                :, i * SOFTMAX_HEAD_TILE : (i + 1) * SOFTMAX_HEAD_TILE, :
            ],
            row_idx,
            attention_weight[
                :,
                i * (SOFTMAX_HEAD_TILE * SEQ) : (i + 1) * (SOFTMAX_HEAD_TILE * SEQ),
            ],
        )



def vision_block(x_fp32: np.ndarray, params: dict, debug: bool = False):
    taps = OrderedDict() if debug else None

    x = x_fp32.astype(np.float32)
    residual = x.reshape(SEQ, EMBD)

    x_ln1 = np.empty((SEQ, EMBD), dtype=np.float32)
    layernorm(residual, params["W_norm_1"], params["b_norm_1"], x_ln1)
    if debug: taps["ln1_out"] = x_ln1.copy()

    # qkv
    query = np.zeros((SEQ, EMBD), dtype=np.float32)
    key   = np.zeros((SEQ, EMBD), dtype=np.float32)
    value = np.zeros((SEQ, EMBD), dtype=np.float32)
    linear_projection(x_ln1, params["Wq"], query, SEQ, EMBD, EMBD)
    linear_projection(x_ln1, params["Wk"], key,   SEQ, EMBD, EMBD)
    linear_projection(x_ln1, params["Wv"], value, SEQ, EMBD, EMBD)
    if debug:
        taps["Q"] = query.copy()
        taps["K"] = key.copy()
        taps["V"] = value.copy()

    # attention scores: [SEQ, H, SEQ]
    attention_score = np.empty((SEQ, N_HEAD, SEQ), dtype=np.float32)
    for i in range(SEQ // ATTN_SCORE_M_TILE):
        for j in range(SEQ // ATTN_SCORE_N_TILE):
            for k in range(N_HEAD):
                attn_score_mod(
                    query[i*ATTN_SCORE_M_TILE:(i+1)*ATTN_SCORE_M_TILE, k*HEAD_DIM:(k+1)*HEAD_DIM],
                    key  [j*ATTN_SCORE_N_TILE:(j+1)*ATTN_SCORE_N_TILE, k*HEAD_DIM:(k+1)*HEAD_DIM],
                    attention_score[i*ATTN_SCORE_M_TILE:(i+1)*ATTN_SCORE_M_TILE, k, j*ATTN_SCORE_N_TILE:(j+1)*ATTN_SCORE_N_TILE],
                )
    if debug: taps["attn_scores"] = attention_score.copy()

    # softmax
    if USE_ALL_NPU_KERNELS:
        attn_weight = np.zeros((SEQ, N_HEAD * SEQ), dtype=np.float32)
        softmax(attention_score, attn_weight)  # produces [SEQ, H*SEQ]
        attn_weight_reshaped = attn_weight.reshape(SEQ, N_HEAD, SEQ)
    else:
        tensor_atten_score = torch.from_numpy(attention_score)
        attn_weight_reshaped = F.softmax(tensor_atten_score, dim=-1).numpy()
    if debug: taps["attn_weights"] = attn_weight_reshaped.copy()

    # attention value: concat heads -> [SEQ, EMBD]
    attn_value = np.zeros((SEQ, EMBD), dtype=np.float32)
    for k in range(N_HEAD):
        A = (attn_weight_reshaped[:, k, :]).astype(np.float32)
        Vh = value[:, k * HEAD_DIM:(k + 1) * HEAD_DIM].astype(np.float32)
        linear_projection(A, Vh, attn_value[:, k * HEAD_DIM:(k + 1) * HEAD_DIM], SEQ, HEAD_DIM, SEQ)
    if debug: taps["attn_value"] = attn_value.copy()

    # output projection
    out_proj = np.zeros((SEQ, EMBD), dtype=np.float32)
    linear_projection(attn_value, params["Wo"], out_proj, SEQ, EMBD, EMBD)
    if debug: taps["attn_out_proj"] = out_proj.copy()

    # residual 1
    res1 = residual.copy()
    add_residual(res1, out_proj, SEQ, EMBD)
    if debug: taps["residual1"] = res1.copy()

    # norm2
    x_ln2 = np.zeros((SEQ, EMBD), dtype=np.float32)
    layernorm(res1, params["W_norm_2"], params["b_norm_2"], x_ln2)
    if debug: taps["ln2_out"] = x_ln2.copy()

    # FFN up -> GELU -> down
    ffn_up_x = np.zeros((SEQ, FFN_HID), dtype=np.float32)
    linear_projection(x_ln2, params["W_up"], ffn_up_x, SEQ, FFN_HID, EMBD)
    if debug: taps["ffn_up"] = ffn_up_x.copy()

    if USE_ALL_NPU_KERNELS:
        activated_x = np.zeros((SEQ, FFN_HID), dtype=np.float32)
        for i in range(SEQ // GELU_SEQ_TILE):
            gelu_mod(ffn_up_x[i*GELU_SEQ_TILE:(i+1)*GELU_SEQ_TILE, :],
                     activated_x[i*GELU_SEQ_TILE:(i+1)*GELU_SEQ_TILE, :])
    else:
        activated_x = nn.GELU()(torch.from_numpy(ffn_up_x)).numpy()
    if debug: taps["gelu"] = activated_x.copy()

    ffn_down_x = np.zeros((SEQ, EMBD), dtype=np.float32)
    linear_projection(activated_x, params["W_down"], ffn_down_x, SEQ, EMBD, FFN_HID)
    if debug: taps["ffn_down"] = ffn_down_x.copy()

    # residual 2 (final)
    res2 = res1.copy()
    add_residual(res2, ffn_down_x, SEQ, EMBD)
    if debug: taps["output"] = res2.copy()

    return (res2, taps) if debug else res2


def torch_stepwise_reference(x: torch.Tensor, ref_model: MiniVit):
    # x: [SEQ, EMBD] (unbatched)
    taps = OrderedDict()

    # LN1
    x_ln1 = F.layer_norm(x, (EMBD,), ref_model.ln_1.weight, ref_model.ln_1.bias, eps=1e-6)
    taps["ln1_out"] = x_ln1

    # Project Q,K,V (biases are zero per your init)
    Wqkv = ref_model.attn.in_proj_weight    # [3*E, E]
    bqkv = ref_model.attn.in_proj_bias      # zeros
    Wq, Wk, Wv = torch.split(Wqkv, EMBD, dim=0)
    bq, bk, bv = torch.split(bqkv, EMBD, dim=0)
    Q = x_ln1 @ Wq.T + bq
    K = x_ln1 @ Wk.T + bk
    V = x_ln1 @ Wv.T + bv
    taps["Q"] = Q; taps["K"] = K; taps["V"] = V

    # Reshape to heads
    Qh = Q.view(SEQ, N_HEAD, HEAD_DIM)
    Kh = K.view(SEQ, N_HEAD, HEAD_DIM)
    Vh = V.view(SEQ, N_HEAD, HEAD_DIM)

    # Attention scores: [SEQ, H, SEQ]
    scale = 1.0 / (HEAD_DIM ** 0.5)
    # scores[t, h, s] = dot(Qh[t,h], Kh[s,h]) * scale
    scores = torch.einsum("thd,shd->ths", Qh, Kh) * scale
    taps["attn_scores"] = scores

    # Softmax over keys axis
    attn_weights = torch.softmax(scores, dim=-1)  # [SEQ, H, SEQ]
    taps["attn_weights"] = attn_weights

    # Weighted value per head then concat
    attn_value = torch.einsum("ths,shd->thd", attn_weights, Vh).reshape(SEQ, EMBD)
    taps["attn_value"] = attn_value

    # Output projection
    out_proj = attn_value @ ref_model.attn.out_proj.weight.T + ref_model.attn.out_proj.bias
    taps["attn_out_proj"] = out_proj

    # Residual 1
    res1 = x + out_proj
    taps["residual1"] = res1

    # LN2
    x_ln2 = F.layer_norm(res1, (EMBD,), ref_model.ln_2.weight, ref_model.ln_2.bias, eps=1e-6)
    taps["ln2_out"] = x_ln2

    # FFN
    ffn_up = x_ln2 @ ref_model.ffn_up.weight.T  # bias=False
    taps["ffn_up"] = ffn_up
    gelu_out = F.gelu(ffn_up)
    taps["gelu"] = gelu_out
    ffn_down = gelu_out @ ref_model.ffn_down.weight.T  # bias=False
    taps["ffn_down"] = ffn_down

    # Residual 2 (final)
    out = res1 + ffn_down
    taps["output"] = out
    return taps

def compare_allo_vs_torch(allo_taps: OrderedDict, torch_taps: OrderedDict, per_head_attention: bool = True):
    rows = []
    ordered_keys = [
        "ln1_out","Q","K","V","attn_scores","attn_weights",
        "attn_value","attn_out_proj","residual1",
        "ln2_out","ffn_up","gelu","ffn_down","output"
    ]
    for k in ordered_keys:
        if k in allo_taps and k in torch_taps:
            a = allo_taps[k]
            t = _to_numpy(torch_taps[k])
            # unify shapes for attn_weights (Allo stored [SEQ,H,SEQ])
            if k == "attn_weights" and a.ndim == 2:  # if you kept flattened [SEQ, H*SEQ]
                a = a.reshape(SEQ, N_HEAD, SEQ)
            rows.append(_metrics(a, t, k))
    _print_table(rows)

    # Optional: per-head diagnostics
    if per_head_attention and "attn_weights" in allo_taps:
        print("\nPer-head attention weight metrics (top 5 worst by rel_L2):")
        a = allo_taps["attn_weights"]
        if a.ndim == 2:
            a = a.reshape(SEQ, N_HEAD, SEQ)
        t = _to_numpy(torch_taps["attn_weights"])
        head_rows = []
        for h in range(N_HEAD):
            head_rows.append(_metrics(a[:, h, :], t[:, h, :], f"attn_w_head{h}"))
        head_rows.sort(key=lambda r: r["rel_L2"], reverse=True)
        _print_table(head_rows[:5])

        if "attn_value" in allo_taps:
            print("\nPer-head attention VALUE metrics (top 5 worst by rel_L2):")
            av = allo_taps["attn_value"].reshape(SEQ, N_HEAD, HEAD_DIM)
            tv = _to_numpy(torch_taps["attn_value"]).reshape(SEQ, N_HEAD, HEAD_DIM)
            head_rows = []
            for h in range(N_HEAD):
                head_rows.append(_metrics(av[:, h, :], tv[:, h, :], f"attn_val_head{h}"))
            head_rows.sort(key=lambda r: r["rel_L2"], reverse=True)
            _print_table(head_rows[:5])


if __name__ == "__main__":
    ref_model = MiniVit().eval()
    # reference weights (float32)
    p = {n: v.detach().numpy() for n, v in ref_model.named_parameters()}
    params_fp32 = {
        "Wq": p["attn.in_proj_weight"][:EMBD, :].T,
        "Wk": p["attn.in_proj_weight"][EMBD : 2 * EMBD, :].T,
        "Wv": p["attn.in_proj_weight"][2 * EMBD :, :].T,
        "Wo": p["attn.out_proj.weight"].T,
        "W_up": p["ffn_up.weight"].T,
        "W_down": p["ffn_down.weight"].T,
        "W_norm_1": p["ln_1.weight"],
        "b_norm_1": p["ln_1.bias"],
        "W_norm_2": p["ln_2.weight"],
        "b_norm_2": p["ln_2.bias"],
    }
    params = {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in params_fp32.items()}

    x_float = torch.randn(SEQ, EMBD)

    # Timings
    t0 = time.time()
    sample = ref_model(x_float)
    t1 = time.time()
    print(f"PyTorch forward time: {t1 - t0:.6f} s")

    a0 = time.time()
    allo_out, allo_taps = vision_block(x_float.numpy(), params, debug=True)
    a1 = time.time()
    print(f"Allo forward time:    {a1 - a0:.6f} s")

    # Stepwise PyTorch taps
    torch_taps = torch_stepwise_reference(x_float, ref_model)

    # Step-by-step precision report
    compare_allo_vs_torch(allo_taps, torch_taps, per_head_attention=True)

    # Final check (still keep your assert)
    np.testing.assert_allclose(allo_out, sample.detach().numpy(), rtol=1e-1)
    print("Allo float32 block matches PyTorch float32 reference within tolerance ✔️")