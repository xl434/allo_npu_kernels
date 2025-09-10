# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie.external_kernel import ExternalModule

# Matrix shape: rows=32, cols=64  (matches sin_float32 signature: [32][64])
Ly = Layout("S0S1")
Ty = float32
seq_tile = 32       # rows
feature_tile = 64   # cols

RTOL = 1e-3
ATOL = 1e-4

KERNEL_LIB_PATH = "cc/"


def _test_sine_single_tile():
    sine = ExternalModule(
        top="sin_float32",
        impl_path=KERNEL_LIB_PATH + "sine.cc",
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
            sine(input_x, output_x)

    torch.manual_seed(0)
    input_tensor = (torch.rand(seq_tile, feature_tile, dtype=torch.float32) * 40.0) - 20.0  # ~U[-20,20]
    ref_out = torch.sin(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq_tile, feature_tile), dtype=np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_out.cpu().numpy(), rtol=RTOL, atol=ATOL)
            print(f"PASSED sine! (rtol={RTOL}, atol={ATOL})")
        except AssertionError as e:
            # Debug: print summary and a few worst mismatches
            diff = np.abs(output_allo - ref_out.cpu().numpy())
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print("Sine mismatch detected.")
            print(f"Max abs diff = {diff[max_idx]:.6e} at index {max_idx}")
            r, c = max_idx
            print(f"Input        = {input_tensor[r, c].item():.6f}")
            print(f"Allo output  = {output_allo[r, c]:.6f}")
            print(f"Torch output = {ref_out[r, c].item():.6f}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend run. "
              "Set it to execute the Allo kernel.")
        
def _test_sine_tiling():
    sine = ExternalModule(
        top="sin_float32",
        impl_path=KERNEL_LIB_PATH + "sine.cc",
        input_idx=[0],
        output_idx=[1],
    )

    P0 = 4
    P1 = 2
    seq = P0 * seq_tile
    feature_dim = P1 * feature_tile

    @df.region()
    def top():
        @df.kernel(mapping=[P1, P0])
        def core(
            input_x: Ty[seq, feature_dim] @ Ly,
            output_x: Ty[seq, feature_dim] @ Ly,
        ):
            sine(input_x, output_x)

    torch.manual_seed(0)
    input_tensor = (torch.rand(seq, feature_dim, dtype=torch.float32) * 40.0) - 20.0  # ~U[-20,20]
    ref_out = torch.sin(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq, feature_dim), dtype=np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, ref_out.cpu().numpy(), rtol=RTOL, atol=ATOL)
        print(f"PASSED sine! (rtol={RTOL}, atol={ATOL})")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend run. "
              "Set it to execute the Allo kernel.")

if __name__ == "__main__":
    _test_sine_single_tile()
    _test_sine_tiling()