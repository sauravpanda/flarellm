#!/bin/bash
# Download a Q4_K_M benchmark model: Llama-3.2-1B-Instruct Q4_K_M (~770MB)
#
# Usage: ./scripts/download_q4k_model.sh

set -e

MODEL_DIR="${MODEL_DIR:-models}"
MODEL_NAME="llama-3.2-1b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Model already exists at $MODEL_DIR/$MODEL_NAME"
    exit 0
fi

echo "Downloading Llama-3.2-1B-Instruct Q4_K_M (~770MB)..."
curl -L "$MODEL_URL" -o "$MODEL_DIR/$MODEL_NAME" --progress-bar

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    SIZE=$(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1)
    echo "Downloaded $MODEL_DIR/$MODEL_NAME ($SIZE)"
else
    echo "Download failed" >&2
    exit 1
fi
