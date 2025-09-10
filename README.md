# allo_npu_kernels

.
├── cc/ # C++ kernels
│ ├── <kernel>.cc # Low-level compute kernels in C++
│ └── ...
│
├── kernels/ # Allo kernel calls
│ ├── single_tile/ # Kernels running on a single tile
│ ├── tiling/ # Tiled kernel implementations
│ └── ...
│
├── vla/ # Model components
│ ├── llama_block/ # LLaMA transformer blocks
│ ├── vision_block/ # Vision encoder layer
│ └── ...
│
├── gpt2/ # Model components
│ ├── gpt2_block/ # Transformer layers 
│ └── ...
│
└── README.md