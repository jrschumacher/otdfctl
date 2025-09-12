#!/bin/bash

# Test evaluation with development model (higher quality)
set -e

echo "🔧 Development Model Testing"
echo "============================"
echo ""

# Download development model if needed
echo "📥 Ensuring development model is available..."
./scripts/download-test-model.sh dev

echo ""
echo "🧪 Running development evaluation tests..."
echo ""

# Build otdfctl
echo "🔨 Building otdfctl..."
go build -o otdfctl .

echo ""
echo "📊 Single Query Test (Development Model):"
./otdfctl llm eval \
  --query "What is OpenTDF and how does it work?" \
  --model models/dev-test-model.gguf \
  --output detailed \
  --verbose

echo ""
echo "🤖 Agent Routing Test (Development Model):"
./otdfctl llm eval \
  --query "How do I create an attribute namespace?" \
  --model models/dev-test-model.gguf \
  --agent-mode=true \
  --output detailed

echo ""
echo "📋 Quality Regression Suite (Development Model):"
./otdfctl llm eval \
  --test-suite evals/quality-regression-tests.json \
  --model models/dev-test-model.gguf \
  --output detailed

echo ""
echo "✅ Development model testing complete!"
echo ""
echo "💡 Model comparison:"
echo "   • CI Model (tiny): ~11MB, basic functionality validation"  
echo "   • Dev Model (this): ~700MB, high-quality evaluation"
echo ""
echo "🔄 To test CI model: ./test-eval.sh"