#!/bin/bash

IMG_DIR=~/Documents/ai/chatbotai/docs/

for img in "${IMG_DIR}"*.jpg; do
    base_name=$(basename "$img" .jpg)
    output_file="${IMG_DIR}${base_name}.txt"

    /Users/lev/Documents/ai/llama.cpp/build/bin/llama-mtmd-cli \
        -m /Users/lev/Documents/ai/llama.cpp/models/ggml-model-f16.gguf \
        --mmproj /Users/lev/Documents/ai/llama.cpp/models/mmproj-model-f16.gguf \
        --temp 0.1 \
        -p "Describe the image in detail. Be specific about graphs, such as bar plots." \
        --image "$img" > "$output_file"
done