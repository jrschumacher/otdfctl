#!/bin/bash

# Download lightweight test model for CI and development testing
set -e

# Default to tiny model for CI, allow override for manual testing
MODEL_TYPE="${1:-ci}"

if [ "$MODEL_TYPE" = "ci" ]; then
    MODEL_URL="https://huggingface.co/mradermacher/Tiny-LLM-GGUF/resolve/main/Tiny-LLM.Q4_K_S.gguf"
    MODEL_PATH="models/tiny-test-model.gguf"
    EXPECTED_SIZE=11500000  # ~11.9MB (allow for slight variation)
    MODEL_NAME="Tiny-LLM (CI)"
elif [ "$MODEL_TYPE" = "dev" ]; then
    MODEL_URL="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
    MODEL_PATH="models/dev-test-model.gguf"
    EXPECTED_SIZE=700000000  # ~700MB
    MODEL_NAME="TinyLlama-1.1B (Development)"
else
    echo "❌ Usage: $0 [ci|dev]"
    echo "   ci  - Download tiny 11MB model for CI testing"
    echo "   dev - Download 700MB model for development testing"
    exit 1
fi

echo "🤖 OpenTDF Test Model Downloader"
echo "================================"
echo "📋 Model Type: $MODEL_NAME"
echo ""

# Create models directory
mkdir -p models

# Check if model already exists and is complete
if [ -f "$MODEL_PATH" ]; then
    CURRENT_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    if [ "$CURRENT_SIZE" -ge "$EXPECTED_SIZE" ]; then
        echo "✅ $MODEL_NAME already exists and appears complete ($CURRENT_SIZE bytes)"
        echo "📍 Model location: $MODEL_PATH"
        exit 0
    else
        echo "⚠️  Existing model incomplete ($CURRENT_SIZE bytes), re-downloading..."
        rm -f "$MODEL_PATH"
    fi
fi

echo "📥 Downloading $MODEL_NAME..."
echo "🔗 URL: $MODEL_URL"
echo "📍 Target: $MODEL_PATH"
echo ""

# Download with progress bar and resume support
if command -v curl >/dev/null 2>&1; then
    echo "Using curl for download..."
    curl -L -C - --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    echo "Using wget for download..."
    wget -c -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "❌ Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Validate download
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Download failed: Model file not found"
    exit 1
fi

FINAL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
if [ "$FINAL_SIZE" -lt "$EXPECTED_SIZE" ]; then
    echo "❌ Download incomplete: Expected ~$EXPECTED_SIZE bytes, got $FINAL_SIZE bytes"
    rm -f "$MODEL_PATH"
    exit 1
fi

echo ""
echo "✅ Model downloaded successfully!"
echo "📊 Size: $FINAL_SIZE bytes (~$(($FINAL_SIZE / 1000000))MB)"
echo "📍 Location: $MODEL_PATH"
echo ""
echo "🧪 Test the model:"
echo "   ./otdfctl llm eval --query 'What is OpenTDF?' --model $MODEL_PATH --verbose"
echo ""
if [ "$MODEL_TYPE" = "ci" ]; then
    echo "💡 Model Info:"
    echo "   • Name: Tiny-LLM (CI optimized)"
    echo "   • Size: ~11.9MB"
    echo "   • Quantization: Q4_K_S"
    echo "   • License: Apache 2.0"
    echo "   • Use Case: Fast CI testing and validation"
    echo ""
    echo "📝 Note: For development testing with better quality:"
    echo "   ./scripts/download-test-model.sh dev"
else
    echo "💡 Model Info:"
    echo "   • Name: TinyLlama-1.1B-Chat-v1.0"
    echo "   • Parameters: 1.1 billion"
    echo "   • Quantization: Q4_K_M"
    echo "   • License: Apache 2.0"
    echo "   • Use Case: Development testing and evaluation"
fi