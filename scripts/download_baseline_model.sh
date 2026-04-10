#!/bin/bash
# Download the baseline benchmark model: SmolLM2-135M-Instruct Q8_0 (~138MB)
#
# Usage: ./scripts/download_baseline_model.sh

set -e

MODEL_DIR="${MODEL_DIR:-models}"
MODEL_NAME="smollm2-135m-instruct-q8_0.gguf"
MODEL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Model already exists at $MODEL_DIR/$MODEL_NAME"
    exit 0
fi

echo "Downloading SmolLM2-135M-Instruct Q8_0 (~138MB)..."
curl -L "$MODEL_URL" -o "$MODEL_DIR/$MODEL_NAME" --progress-bar

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    SIZE=$(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1)
    echo "Downloaded $MODEL_DIR/$MODEL_NAME ($SIZE)"
else
    echo "Download failed" >&2
    exit 1
fi
