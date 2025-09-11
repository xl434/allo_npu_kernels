# vla_depth1.py
import time
import numpy as np
import torch

from vision_block import (
    SEQ as SEQ_V, EMBD as EMBD_V,  # SEQ_V==64, EMBD_V==768
    vision_block,
)

from llama_block import (    # <-- make this by copying llama rope file, set SEQ=128, USE_ALL_NPU_KERNELS=False
    SEQ as SEQ_MM, EMBD as EMBD_MM, Q_H, KV_H, HEAD_DIM,
    llama_block_rope as vlm_block_rope,   # full VLM self-attn layer (with RoPE on Q,K)
    vlm_qkv_from_mm_seq,                  # the helper you just added (K,V from pre-norm; Y_vlm full output)
)

from llama_block import (           # <-- your working expert with SEQ=64
    SEQ as SEQ_T, EMBD as EMBD_T,
    llama_q_from_emb_rope,          # pre-norm + Q (RoPE) for text
    expert_cross_attn_from_qkv,     # cross attn Qexp x (Kvlm,Vvlm) + MLP
)

assert SEQ_V == 64 and SEQ_T == 64, "expect 64 vision tokens + 64 text tokens."
assert EMBD_V == EMBD_T == EMBD_MM == 768
assert SEQ_MM == SEQ_V + SEQ_T == 128, "VLM sequence is 128 (vision + text)."

def main():
    rng = np.random.default_rng(0)


    def rand_mat(m, n): return rng.standard_normal((m, n), dtype=np.float32)
    def rand_vec(n):    return rng.standard_normal((n,), dtype=np.float32)

    # Vision params (ViT-style)
    params_vit = dict(
        Wq=rand_mat(EMBD_V, EMBD_V), Wk=rand_mat(EMBD_V, EMBD_V), Wv=rand_mat(EMBD_V, EMBD_V),
        Wo=rand_mat(EMBD_V, EMBD_V),
        W_up=rand_mat(EMBD_V, 4*EMBD_V), W_down=rand_mat(4*EMBD_V, EMBD_V),
        W_norm_1=rand_vec(EMBD_V), b_norm_1=rand_vec(EMBD_V),
        W_norm_2=rand_vec(EMBD_V), b_norm_2=rand_vec(EMBD_V),
    )

    # VLM (SmolLM) layer params
    params_vlm = dict(
        Wq=rand_mat(EMBD_MM, Q_H*HEAD_DIM),
        Wk=rand_mat(EMBD_MM, KV_H*HEAD_DIM),
        Wv=rand_mat(EMBD_MM, KV_H*HEAD_DIM),
        Wo=rand_mat(Q_H*HEAD_DIM, EMBD_MM),
        W_up=rand_mat(EMBD_MM, 4*EMBD_MM),
        W_gate=rand_mat(EMBD_MM, 4*EMBD_MM),
        W_down=rand_mat(4*EMBD_MM, EMBD_MM),
        W_norm_1=rand_vec(EMBD_MM),
        W_norm_2=rand_vec(EMBD_MM),
    )

    # Expert layer params (text-only, SEQ_T=64)
    params_exp = dict(
        Wq=rand_mat(EMBD_T, Q_H*HEAD_DIM),
        Wk=rand_mat(EMBD_T, KV_H*HEAD_DIM),
        Wv=rand_mat(EMBD_T, KV_H*HEAD_DIM),
        Wo=rand_mat(Q_H*HEAD_DIM, EMBD_T),
        W_up=rand_mat(EMBD_T, 4*EMBD_T),
        W_gate=rand_mat(EMBD_T, 4*EMBD_T),
        W_down=rand_mat(4*EMBD_T, EMBD_T),
        W_norm_1=rand_vec(EMBD_T),
        W_norm_2=rand_vec(EMBD_T),
    )

    # -----------------------------
    # 1) Inputs (already-embedded)
    # -----------------------------
    # vision_emb: [64, 768], text_emb: [64, 768]
    vision_emb = rng.standard_normal((SEQ_V, EMBD_V), dtype=np.float32)
    text_emb   = rng.standard_normal((SEQ_T, EMBD_T), dtype=np.float32)

    # -----------------------------
    # 2) Vision encoder (1 depth)
    #    get Y_v (for concat).
    # -----------------------------
    t0 = time.perf_counter()
    Y_v = vision_block(vision_emb, params_vit)
    t1 = time.perf_counter()

    # -----------------------------
    # 3) Multimodal concat for VLM layer (64 vision + 64 text = 128)
    # -----------------------------
    mm_seq = np.concatenate([Y_v, text_emb], axis=0)  # [128, 768]
    assert mm_seq.shape == (SEQ_MM, EMBD_MM)

    # -----------------------------
    # 4) VLM (SmolLM) layer (1 depth)
    #    - K_vlm, V_vlm from pre-norm (no RoPE) for expert cross-attn
    #    - Y_vlm: full layer output (self-attn w/ RoPE + MLP)
    # -----------------------------
    t2 = time.perf_counter()
    K_vlm, V_vlm, Y_vlm, X_pre_vlm = vlm_qkv_from_mm_seq(mm_seq, params_vlm)
    t3 = time.perf_counter()

    # -----------------------------
    # 5) Expert layer (text-only, 1 depth)
    # -----------------------------
    residual_exp = text_emb.copy()  # [64, 768]

    t4 = time.perf_counter()
    Q_exp, X_pre_exp = llama_q_from_emb_rope(text_emb, params_exp)             # [64, 960], [64, 768]
    Y_exp = expert_cross_attn_from_qkv(Q_exp, K_vlm, V_vlm, residual_exp, params_exp, causal=True)  # [64, 768]
    t5 = time.perf_counter()

    # -----------------------------
    # 6) Report shapes + timings
    # -----------------------------
    print("== Shapes ==")
    print("vision_emb :", vision_emb.shape)
    print("Y_v        :", Y_v.shape)
    print("mm_seq     :", mm_seq.shape)
    print("K_vlm      :", K_vlm.shape, "  V_vlm:", V_vlm.shape, "  Y_vlm:", Y_vlm.shape)
    print("Q_exp      :", Q_exp.shape,  "  Y_exp:", Y_exp.shape)

    print("\n== Timings ==")
    print(f"Vision enc (1L)      : {(t1 - t0)*1e3:.2f} ms")
    print(f"VLM layer (1L)       : {(t3 - t2)*1e3:.2f} ms")
    print(f"Expert x-attn (1L)   : {(t5 - t4)*1e3:.2f} ms")

if __name__ == "__main__":
    main()
