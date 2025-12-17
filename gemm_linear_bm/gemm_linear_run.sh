#!/bin/bash

set -e
set -o pipefail 

function print_info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}
function print_ok() {
  echo -e "\033[1;32m[DONE]\033[0m $1"
}
function print_err() {
  echo -e "\033[41;97m[ERROR]\033[0m $1"
  exit 1
}

WEIGHT_PATH="weight_output/"
INSPECT_PATH="inspector_output/"
QUANT_PATH="quantizer_output/"
COMPILE_PATH="compile_output/"
ZCU_PATH="../zcu_apps/"
COMPILER_SCRIPT="./gemm_linear_compiler.sh"

for path in "$WEIGHT_PATH" "$INSPECT_PATH" "$QUANT_PATH" "$COMPILE_PATH"; do
  print_info "Cleaning $path"
  rm -rf "$path" || print_err "Failed to remove $path"
  mkdir -p "$path" || print_err "Failed to create $path"
done

print_info "Generating models..."
python gemm_linear_model.py || print_err "gemm_linear_model.py failed"
print_ok "All model generated."

print_info "Running model inspecting..."
python gemm_linear_inspector.py || print_err "gemm_linear_inspector.py failed"
print_ok "Model inspecting finished."

print_info "Running model quantizing..."
python gemm_linear_quantizer.py || print_err "gemm_linear_quantizer.py failed"
rm -rf $INSPECT_PATH
rm -rf $WEIGHT_PATH
echo "Success: memory freed..."
mkdir -p $INSPECT_PATH
mkdir -p $WEIGHT_PATH
print_ok "Model quantizing finished."

if [[ -f "$COMPILER_SCRIPT" ]]; then
  print_info "Compling models..."
  bash "$COMPILER_SCRIPT" || print_err "Compiler.sh failed"
  print_ok "Compiling finished."
else
  print_err "Compiler script not found: $COMPILER_SCRIPT"
fi

rm -rf $QUANT_PATH
echo "Success: memory freed..."
mkdir -p $QUANT_PATH

# print_info "Copying compile output to $ZCU_PATH..."
# cp -r "$COMPILE_PATH" "$ZCU_PATH" || print_err "Failed to copy compile_output to $ZCU_PATH"
# print_ok "All steps completed successfully."