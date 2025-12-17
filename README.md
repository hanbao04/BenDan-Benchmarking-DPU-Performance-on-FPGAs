# BenDan - Benchmarking DPU Performance on FPGAs

## Project Structure

This repository is organized into **six parts**:

- **`conv2d_bm/`**  
  Benchmarking standard 2D convolutional layers.

- **`depthwise_conv2d_bm/`**  
  Benchmarking depthwise 2D convolutional layers.

- **`gemm_bm/`**  
  Benchmarking standard 2D convolutional layers with $1 \times 1$ kernels.

- **`gemm_linear_bm/`**  
  Benchmarking linear (fully connected) layers.

- **`arch_pp_cp/`**  
  Benchmarking under different architecture configurations (PP/CP variations).

- **`shared/`**  
  Contains abstract classes and utilities for configuration and shared benchmarking logic.

---

## `_bm` Folder Details

Each folder with the `_bm` suffix has the **same structure**:

---

## `_bm` Folder Details

Each folder with the `_bm` suffix has the **same structure**:
```plaintext
_bm/
├── arch.json           # Architecture fingerprint and DPU configuration
├── cpp/                # C++ kernels for benchmarking
├── _compiler.sh        # Compilation script for C++/bitstream if needed
├── _config.py          # Benchmark layer configuration (runnable for summary)
├── _inspector.py       # Inspector utilities for analyzing models/kernels
├── _model.py           # Model construction for benchmarking
├── _quantizer.py       # Quantization utilities for layer generation
└── _run.sh             # One-command script to run the entire benchmarking pipeline
```


### Notes:
- `arch.json`  
  Stores architecture fingerprint and configuration for the DPU.

- `cpp/`  
  Contains C++ kernels used in benchmarking.

- `_config.py`  
  Configures the layer generation for benchmarking. This file is executable, and running it will print a summary of the layers that will be generated.

- `_run.sh`  
  Runs the entire benchmarking pipeline for the target layer.

You can **run `_run.sh` to execute the full benchmarking workflow end-to-end**.

---

## Adapting to Other DPU Versions

To benchmark on different DPU versions, you typically need to:

1. **Modify the fingerprint in `arch.json`** to match the target DPU version.
2. If architecture configuration names for CP/PP differ, adjust `_arch_map` in `shared/base_config.py` accordingly to align with your specific environment.

---

## Key Features

✅ Modular benchmarking for Conv2D, DepthwiseConv2D, GEMM, Linear layers  
✅ Easily configurable layer shapes, batch sizes, strides, and quantization  
✅ One-command benchmarking (`_run.sh`) for reproducibility  
✅ Supports architecture-aware benchmarking for FPGA DPUs under different configurations

## Getting Started

1️⃣ Clone this repository:
```bash
git clone https://gitlab.com/etrovub/embedded-systems/publications/bendan.git
cd BenDan
```
2️⃣ Choose a benchmarking folder, modify _config.py as needed for your layer shape and DPU constraints.

3️⃣ Run:
```bash
bash _run.sh
```
In each _bm folder to generate the corresponding layers for benchmarking.

## Supported Environment

This framework is developed and tested under the following environment:

- **`Vivado 2022.2`**  
- **`Petalinux 2022.2`**  
- **`Vitis Ai 3.0`**  
- **`DPUCZDX8G module (Xilinx DPU v4.1)`** 
- **`Xilinx ZCU104 UltraScale+ MPSoC`** 
