# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import torch.nn as nn
import numpy as np
import allo.dataflow as df
from allo.memory import Layout
from allo.ir.types import float32
from allo.backend.aie.external_kernel import ExternalModule


Ly = Layout("S0S1")
Ty = float32

feature_tile = 768
seq_tile = 4

KERNEL_LIB_PATH = "cc/"


def _test_silu_single_tile():
    silu = ExternalModule(
        top="silu_float32",
        impl_path=KERNEL_LIB_PATH + "silu.cc",  # Make sure this path is correct
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
            silu(input_x, output_x)

    # Reference PyTorch SiLU
    silu_model = nn.SiLU().cpu()
    input_tensor = torch.randn(seq_tile, feature_tile, dtype=torch.float32)
    output = silu_model(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq_tile, feature_tile), dtype=np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output.numpy(), rtol=1e-2)
        print("PASSED SiLU!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

def _test_silu_tiling():
    silu = ExternalModule(
        top="silu_float32",
        impl_path=KERNEL_LIB_PATH + "silu.cc",  # Make sure this path is correct
        input_idx=[0],
        output_idx=[1],
    )

    P0 = 4
    P1 = 4
    seq = seq_tile * P0
    feature_dim = feature_tile * P1

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def core(
            input_x: Ty[seq, feature_dim] @ Ly,
            output_x: Ty[seq, feature_dim] @ Ly,
        ):
            silu(input_x, output_x)

    # Reference PyTorch SiLU
    silu_model = nn.SiLU().cpu()
    input_tensor = torch.randn(seq, feature_dim, dtype=torch.float32)
    output = silu_model(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq, feature_dim), dtype=np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        np.testing.assert_allclose(output_allo, output.numpy(), rtol=1e-2)
        print("PASSED SiLU!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    _test_silu_single_tile()
    _test_silu_tiling()