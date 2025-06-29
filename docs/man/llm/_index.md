---
title: llm
command:
  name: llm
  usage: llm
  description: Interact with local LLM models for OpenTDF subject matter expertise
---

# llm

The `llm` command provides offline access to local LLM models fine-tuned with OpenTDF context.
This allows users to chat with an AI assistant that has deep knowledge about OpenTDF concepts,
policies, and troubleshooting without requiring internet connectivity.

## Usage

```shell
otdfctl llm chat <model-path>
```

## Features

- **Offline Operation**: No internet required once model is loaded
- **OpenTDF Context**: Pre-loaded with OpenTDF documentation and best practices  
- **Streaming Responses**: Real-time token generation for better user experience
- **Conversation History**: Maintains context across chat interactions
- **Memory Efficient**: Model stays loaded in memory for fast subsequent responses

## Requirements

- Compatible with Ollama model formats
- Model file must be accessible on local filesystem
- Sufficient RAM for model inference (varies by model size)

## Examples

Start a chat session with a local model:
```shell
otdfctl llm chat /path/to/model.gguf
```

## Commands

- [chat](chat.md) - Start interactive chat session with LLM model