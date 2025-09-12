#!/bin/bash

# Script to ingest all documentation files with vector embeddings

echo "🚀 Starting OpenTDF documentation ingestion with vector embeddings..."
echo "📁 Using local RAG storage: ./rag-data/vector_index.json"

# Find all markdown files in docs-main
find ./docs-main -name "*.md" -o -name "*.mdx" | while read -r file; do
    echo "📄 Ingesting: $file"
    ./otdfctl llm ingest "$file" --embeddings --embedding-model nomic-embed-text --index-path ./rag-data/vector_index.json
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully ingested: $file"
    else
        echo "❌ Failed to ingest: $file"
    fi
    echo "---"
done

echo "🎉 Documentation ingestion complete!"
echo "📊 Vector index saved at: ./rag-data/vector_index.json"
echo "📊 Total documents: $(jq '.documents | length' ./rag-data/vector_index.json)"