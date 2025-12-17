#!/bin/bash

OUTPUT_DIR="./gemm_linear_trace_summary.txt"
EXE_NAME="gemm_linear_vart"
rm -rf $OUTPUT_DIR

echo "Old summary removed"

trap "echo -e '\nInterrupted by user. Exiting...'; exit 1" SIGINT

for xmodel_name in /home/root/gemm_linear/compile_output/*/*.xmodel; do
        if [ -f "$xmodel_name" ]; then
		model_name=$(basename $(dirname "$xmodel_name"))
                echo -e "\e[44;97m--------------------------- $model_name ---------------------------\e[0m"
		echo "--------------------------------- $model_name ------------------------------------" >> $OUTPUT_DIR
		# echo "" >> $OUTPUT_DIR
		echo "done: model: $model_name done"
                vaitrace --txt ./$EXE_NAME "$xmodel_name" >> $OUTPUT_DIR 2>&1
                # ./gemm_vart "$xmodel_name"
		sync; sleep 1
        else
                echo "no xmodel found!"
        fi
done

echo "Success: ALL FINISH RUNNING"
