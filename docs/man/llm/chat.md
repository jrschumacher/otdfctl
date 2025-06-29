---
title: llm chat
command:
  name: chat
  usage: chat <model-path>
  description: Start interactive chat session with local LLM model
---

# llm chat

Start an interactive chat session with a local LLM model that has been fine-tuned or prompted 
with OpenTDF context. The chat session runs the model inference in a background goroutine 
for optimal performance.

## Usage

```shell
otdfctl llm chat <model-path> [flags]
```

## Arguments

- `model-path` - Path to the local LLM model file (required)

## Flags

- `--stream` - Enable streaming responses for real-time output (default: true)
- `--context-size` - Maximum context window size for the model (default: 4096)  
- `--temperature` - Sampling temperature from 0.0-1.0, higher values are more creative (default: 0.7)
- `--system-prompt` - Override the default OpenTDF system prompt with custom context

## Interactive Commands

Once in the chat session, these commands are available:

- `exit` or `quit` - Exit the chat session
- `clear` - Clear conversation history  
- `/stream` - Toggle streaming mode on/off
- `/help` - Show available commands

## Examples

Start a basic chat session:
```shell
otdfctl llm chat /models/openai-assistant.gguf
```

Use custom model parameters:
```shell
otdfctl llm chat /models/llama2.gguf --temperature 0.3 --context-size 8192
```

Override the system prompt:
```shell
otdfctl llm chat /models/custom.gguf --system-prompt "You are a security expert focused on data protection."
```

## Model Requirements

- Model must be in a format compatible with Ollama's inference engine
- Recommended formats: GGUF, GGML
- Ensure sufficient system RAM for model size (4GB+ recommended for 7B models)

## Performance Notes

- Initial model load may take 30-60 seconds depending on model size
- Subsequent interactions are much faster as model stays in memory
- Use `--stream=false` for batch-style responses if preferred
- Larger context sizes use more memory but allow longer conversations