import time
import numpy as np
import torch

"""
SmolVLA

[Paper]()

Designed by Hugging Face.
┌──────────────────────────────┐
│                 actions      │
│                    ▲         │
│ ┌─────────┐      ┌─|────┐    │
│ |         │────► │      │    │
│ |         │ kv   │      │    │
│ |         │────► │Action│    │
│ |   VLM   │cache │Expert│    |
│ │         │────► |      │    │
│ │         │      │      │    │
│ └▲──▲───▲─┘      └───▲──┘    |
│  │  |   |            │       |
│  |  |   |          noise     │
│  │  │ state                  │
│  │ language tokens           │
│  image(s)                    │
└──────────────────────────────┘
"""


from llama_block_rope import (
    SEQ as SEQ_MM, EMBD as EMBD_MM, Q_H, KV_H, HEAD_DIM,
    vlm_qkv_from_mm_seq,
    llama_q_from_emb_rope,          # pre-norm + Q (RoPE) for text
    llama_block_rope_cross, 
)

from vision_block import (
    SEQ as SEQ_V, EMBD as EMBD_V,  # SEQ_V==64, EMBD_V==768
    vision_block,
)

SEQ_T = 32
EMBD_T = EMBD_V

assert SEQ_V == 64 and SEQ_T == 32, SEQ_MM == 64
assert EMBD_V == EMBD_T == EMBD_MM == 768
# assert SEQ_MM == SEQ_V + SEQ_T == 128, "VLM sequence is 128 (vision + text)."

def token_reduce(x):  # x:[B,S,D]
    S, D = x.shape
    assert S % 2 == 0
    return 0.5 * (x[0::2, :] + x[1::2, :])

def main():
    rng = np.random.default_rng(0)


    def rand_mat(m, n): return rng.standard_normal((m, n)).astype(np.float32)
    def rand_vec(n):    return rng.standard_normal((n,)).astype(np.float32)

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
    # vision_emb: [64, 768], text_emb: [32, 768]
    image_token = rng.standard_normal((SEQ_V, EMBD_V), dtype=np.float32)
    text_emb   = rng.standard_normal((SEQ_T, EMBD_T), dtype=np.float32)

    # -----------------------------
    # 2) Vision encoder (1 depth)
    #    get Y_v (for concat).
    # -----------------------------
    t0 = time.perf_counter()
    vision_emb = vision_block(image_token, params_vit)
    # vision_emb = image_token
    vision_emb = token_reduce(vision_emb) # [64, 768] --> [32, 768]
    t1 = time.perf_counter()

    # -----------------------------
    # 3) Multimodal concat for VLM layer (32 vision + 32 text = 64)
    # -----------------------------
    mm_seq = np.concatenate([vision_emb, text_emb], axis=0)  # [64, 768]
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
    action_noise   = rng.standard_normal((SEQ_MM, EMBD_T), dtype=np.float32)
    residual_exp = action_noise.copy()  # [64, 768]

    t4 = time.perf_counter()
    Q_exp, X_pre_exp = llama_q_from_emb_rope(action_noise, params_exp)             # [64, 960], [64, 768]
    Y_exp = llama_block_rope_cross(Q_exp, K_vlm, V_vlm, residual_exp, params_exp)  # [64, 768]
    t5 = time.perf_counter()

    # -----------------------------
    # 6) Report shapes + timings
    # -----------------------------
    print("== Shapes ==")
    print("img_token :", image_token.shape)
    print("vision_emb :", vision_emb.shape)
    print("mm_seq     :", mm_seq.shape)
    print("K_vlm      :", K_vlm.shape, "  V_vlm:", V_vlm.shape, "  Y_vlm:", Y_vlm.shape)
    print("Q_exp      :", Q_exp.shape,  "  Y_exp:", Y_exp.shape)

    print("\n== Timings ==")
    print(f"Vision enc (1L)      : {t1 - t0:.3f} s")
    print(f"VLM layer (1L)       : {t3 - t2:.3f} s")
    print(f"Expert x-attn (1L)   : {t5 - t4:.3f} s")

if __name__ == "__main__":
    main()
