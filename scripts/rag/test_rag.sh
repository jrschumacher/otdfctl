#!/bin/bash

# Test script to verify RAG is working
echo "Testing OpenTDF Vector RAG System..."

# Check if we have any chat models
echo "Available models:"
ollama list

echo ""
echo "Testing vector RAG with a simple question..."

# Test with a simple question
(echo "What is OpenTDF?"; sleep 2; echo "exit") | ./otdfctl llm chat llama3.2:3b --rag