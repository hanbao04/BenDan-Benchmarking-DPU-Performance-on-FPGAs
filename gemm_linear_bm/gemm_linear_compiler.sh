#!/bin/bash

set -e
set -o pipefail 

# Set DPU architecture json path
ARCH_JSON="./arch.json"         
OUTPUT_ROOT="./compile_output"           
INPUT_ROOT="./quantizer_output"         

if [ ! -f "$ARCH_JSON" ]; then
    print_error "Architecture JSON file not found at $ARCH_JSON"
fi

function print_error() {
    echo -e "\033[41;97m[ERROR]\033[0m $1"
    exit 1
}

for model_dir in ${INPUT_ROOT}/gemm_*; do
    if [ -d "$model_dir" ]; then
        xmodel_path="${model_dir}/GEMMModel_int.xmodel"

        if [ -f "$xmodel_path" ]; then
            model_name=$(basename "$model_dir")
            echo "Compiling: $model_name"

            output_path="${OUTPUT_ROOT}/${model_name}"
            mkdir -p "$output_path"

            # Try compiling, catch failure manually
            if ! vai_c_xir -x "$xmodel_path" -a "$ARCH_JSON" -o "$output_path" -n "$model_name"; then
                print_error "Compilation failed for $model_name"
            fi

            echo "Compiled $model_name → $output_path"
        else
            print_error "No .xmodel found in $model_dir"
        fi
    fi
done

echo -e "\033[1;32mAll models compiled successfully.\033[0m"