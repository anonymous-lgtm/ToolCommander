#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Define the paths to the assets, corpus, and tool root directory here
ASSETS_PATH_BASE="./data"
CORPUS_PATH="./data/toolbench/corpus.tsv"
TOOL_ROOT_DIR="./data/toolbench/tools"
RESULTS_DIR="./attack_results"

# Define the retrievers to attack here
retrievers=("facebook/contriever-msmarco" "ToolBench/ToolBench_IR_bert_based_uncased")

# Define the assets to attack with here
assets=("g1_train_a.json" "g1_train_b.json" "g1_train_c.json")

mkdir -p $RESULTS_DIR

for asset in "${assets[@]}"; do
    asset_name="${asset%.json}"
    asset_results_dir="${RESULTS_DIR}/${asset_name}"
    mkdir -p "$asset_results_dir"

    for retriever in "${retrievers[@]}"; do
        echo "Attacking with target retriever: $retriever using asset: $asset..."
        retriever_name=$(echo $retriever | sed 's/\//_/g')
        result_file="${asset_results_dir}/attack_results_${retriever_name}.json"

        # Execute the attack
        python attack.py attack \
            --model_retriever="$retriever" \
            --assets_path="${ASSETS_PATH_BASE}/${asset}" \
            --corpus_tsv_path="$CORPUS_PATH" \
            --tool_root_dir="$TOOL_ROOT_DIR" \
            --attack_result_save_path="$result_file" \
            --n_samples=256 \
            --topk=32 \
            --epochs=64 \
            --tag_length=64
    done
done


echo "All attacks completed!"
