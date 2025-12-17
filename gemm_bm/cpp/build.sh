#!/bin/bash

set -e
set -o pipefail

EXE_NAME="gemm_vart"

if [[ -f "$EXE_NAME" ]]; then
    echo "Removing old version"
    rm $EXE_NAME
    echo "[1] Old version removed..."
fi

echo "[2] Compiling new version..."
g++ src/$EXE_NAME.cpp -I/usr/include/opencv4 -std=c++17 -lglog -lvart-dpu-runner -lvart-runner -lxir -lunilog -lvart-mem-manager -lvart-util -o $EXE_NAME
echo "[3] Compilation success!"