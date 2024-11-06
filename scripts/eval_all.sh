#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Define the LLM and retriever models to evaluate here
LLM_MODELS=("gpt" "llama3" "qwen2")
RETRIEVER_MODELS=("facebook/contriever-msmarco" "ToolBench/ToolBench_IR_bert_based_uncased")

# Define the paths to the attacked results, B-Tools, and data here
ATTACKED_RESULTS_DIR="./attack_results"
B_TOOLs=("tool1" "tool2" "tool3")
DATA_PATH_TRAIN="./data/g1_train_a.json"
DATA_PATH_EVAL="./data/g1_eval_a.json"

# Define the assets to evaluate here
assets=("g1_train_a" "g1_train_b" "g1_train_c")

for llm in "${LLM_MODELS[@]}"; do
    for retriever in "${RETRIEVER_MODELS[@]}"; do
        for i in "${!assets[@]}"; do
            asset="${assets[i]}"
            b_tool="${B_TOOLs[i]}"
            retriever_name=$(echo $retriever | sed 's/\//_/g')

            result_files=("${ATTACKED_RESULTS_DIR}/${asset}/attack_results_${retriever_name}.json")

            for result_file in "${result_files[@]}"; do
                echo "Evaluating LLM: $llm, Retriever: $retriever, Result File: $result_file, B-Tool: $b_tool..."

                python evaluate.py \
                    --llm_model="$llm" \
                    --retriever_model="$retriever" \
                    --attacked_results="$result_file" \
                    --data_path_train="$DATA_PATH_TRAIN" \
                    --data_path_eval="$DATA_PATH_EVAL" \
                    --b_tool="$b_tool"
            done
        done
    done
done

echo "All evaluations completed!"