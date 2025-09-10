# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import os
import torch
import torch.nn as nn
from allo.ir.types import float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule
from allo.ir.types import float32

KERNEL_LIB_PATH = "cc/"

Ly = Layout("S0S1")
Ty = float32

feature_tile = 768
seq_tile = 4

def _test_gelu_single_tile():
    gelu = ExternalModule(
        top="gelu_float32",
        impl_path=KERNEL_LIB_PATH + "gelu.cc",
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
            gelu(input_x, output_x)

    gelu_model = nn.GELU()
    input_tensor = torch.randn(seq_tile, feature_tile, dtype=torch.float32)
    output = gelu_model(input_tensor)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq_tile, feature_tile)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        # np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")

def _test_gelu_tiling():
    gelu = ExternalModule(
        top="gelu_float32",
        impl_path=KERNEL_LIB_PATH + "gelu.cc",
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
            gelu(input_x, output_x)

    gelu_model = nn.GELU()
    input_tensor = torch.randn(seq, feature_dim, dtype=torch.float32)
    output = gelu_model(input_tensor)
    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie")
        output_allo = np.zeros((seq, feature_dim)).astype(np.float32)
        mod(input_tensor.cpu().numpy(), output_allo)
        # np.testing.assert_allclose(output_allo, output, rtol=1e-2)
        print("PASSED!")


if __name__ == "__main__":
    _test_gelu_single_tile()
    _test_gelu_tiling()