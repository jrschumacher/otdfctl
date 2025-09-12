#!/bin/bash

# Chat script using local RAG storage

echo "🤖 Starting OpenTDF Chat with Local Vector RAG..."
echo "📁 Using RAG index: ./rag-data/vector_index.json"

# Check if local index exists
if [ ! -f "./rag-data/vector_index.json" ]; then
    echo "❌ No local vector index found at ./rag-data/vector_index.json"
    echo "💡 Run ./ingest_docs.sh first to create the index"
    exit 1
fi

# Show index stats
DOC_COUNT=$(jq '.documents | length' ./rag-data/vector_index.json)
echo "📊 Knowledge base: $DOC_COUNT documents loaded"
echo ""

# Start chat with local index
./otdfctl llm chat llama3.2:1b --rag --index-path ./rag-data/vector_index.json --embedding-model nomic-embed-text