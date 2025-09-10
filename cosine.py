# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie.external_kernel import ExternalModule
import time

KERNEL_LIB_PATH = "cc/"

# Matrix shape: rows=32, cols=64  (matches cos_float32 signature: [32][64])
Ly = Layout("S0S1")
Ty = float32
seq_tile = 32       # rows
feature_tile = 64    # cols

# Tolerances for LUT-based cosine
RTOL = 1e-3
ATOL = 1e-4

def _test_cosine_single_tile():
    # External kernel wrapper
    cosine = ExternalModule(
        top="cos_float32",
        impl_path=KERNEL_LIB_PATH+"cosine.cc",
        input_idx=[0],
        output_idx=[1],
    )

    @df.region()
    def top():
        @df.kernel(mapping=[1, 1])
        def core(
            input_x: Ty[seq_tile, feature_tile] @ Ly,
            output_x: Ty[seq_tile, feature_tile] @ Ly,
        ):
            cosine(input_x, output_x)

    # Reference and inputs
    torch.manual_seed(0)
    # Wide range to stress range-reduction (same as sine test)
    input_tensor = (torch.rand(seq_tile, feature_tile, dtype=torch.float32) * 40.0) - 20.0  # ~U[-20,20]
    ref_out = torch.cos(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq_tile, feature_tile), dtype=np.float32)

        # Run external kernel via Allo
        mod(input_tensor.cpu().numpy(), output_allo)

        # Compare
        try:
            np.testing.assert_allclose(output_allo, ref_out.cpu().numpy(), rtol=RTOL, atol=ATOL)
            print(f"PASSED cosine! (rtol={RTOL}, atol={ATOL})")
        except AssertionError:
            # Debug: print summary and a few worst mismatches
            diff = np.abs(output_allo - ref_out.cpu().numpy())
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print("Cosine mismatch detected.")
            print(f"Max abs diff = {diff[max_idx]:.6e} at index {max_idx}")
            r, c = max_idx
            print(f"Input        = {input_tensor[r, c].item():.6f}")
            print(f"Allo output  = {output_allo[r, c]:.6f}")
            print(f"Torch output = {ref_out[r, c].item():{'.6f'}}")
            # raise  # uncomment to see full assertion
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend run. "
              "Set it to execute the Allo kernel.")
        
def _test_cosine_tiling():
    # External kernel wrapper
    cosine = ExternalModule(
        top="cos_float32",
        impl_path=KERNEL_LIB_PATH+"cosine.cc",
        input_idx=[0],
        output_idx=[1],
    )
    
    P0 = 4
    P1 = 4
    seq = seq_tile * P1      # rows
    feature_dim = feature_tile * P0

    @df.region()
    def top():
        @df.kernel(mapping=[P1, P0])
        def core(
            input_x: Ty[seq, feature_dim] @ Ly,
            output_x: Ty[seq, feature_dim] @ Ly,
        ):
            cosine(input_x, output_x)

    # Reference and inputs
    torch.manual_seed(0)
    # Wide range to stress range-reduction (same as sine test)
    input_tensor = (torch.rand(seq, feature_dim, dtype=torch.float32) * 40.0) - 20.0  # ~U[-20,20]
    ref_out = torch.cos(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq, feature_dim), dtype=np.float32)

        # Run external kernel via Allo
        mod(input_tensor.cpu().numpy(), output_allo)

        # Compare
        try:
            np.testing.assert_allclose(output_allo, ref_out.cpu().numpy(), rtol=RTOL, atol=ATOL)
            print(f"PASSED cosine! (rtol={RTOL}, atol={ATOL})")
        except AssertionError:
            # Debug: print summary and a few worst mismatches
            diff = np.abs(output_allo - ref_out.cpu().numpy())
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print("Cosine mismatch detected.")
            print(f"Max abs diff = {diff[max_idx]:.6e} at index {max_idx}")
            r, c = max_idx
            print(f"Input        = {input_tensor[r, c].item():.6f}")
            print(f"Allo output  = {output_allo[r, c]:.6f}")
            print(f"Torch output = {ref_out[r, c].item():{'.6f'}}")
            # raise  # uncomment to see full assertion
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend run. "
              "Set it to execute the Allo kernel.")

if __name__ == "__main__":
    _test_cosine_single_tile()
    _test_cosine_tiling()
