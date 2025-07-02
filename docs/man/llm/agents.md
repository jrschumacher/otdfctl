---
title: llm agents
command:
  name: agents
  usage: agents [model-path]
  description: Start interactive agent-based chat with specialized OpenTDF assistants
---

# otdfctl llm agents

Start an interactive agent-based chat session with specialized OpenTDF assistants.

The agent chat provides a simple menu-driven interface where you can select and interact with specialized AI agents that have expertise in different areas of OpenTDF:

- **Policy Agent**: Expert in ABAC design, attribute management, and policy creation
- **Development Agent**: Specialist in SDK integration and application development  
- **Deployment Agent**: Focused on installation, configuration, and troubleshooting
- **General Agent**: Provides explanations of OpenTDF concepts and architecture

## Features

- 🤖 **Smart Agent Selection**: Menu-driven agent selection with clear descriptions
- 🔄 **Easy Agent Switching**: Return to menu anytime with Ctrl+C
- 💡 **Agent Suggestions**: System suggests better agents based on your queries
- 📚 **RAG Support**: Enable retrieval-augmented generation for enhanced responses
- ⌨️ **Simple Controls**: Clean interface with intuitive keyboard shortcuts
- 🚀 **Fast and Reliable**: No complex UI dependencies, just pure functionality

## Usage

```bash
# Start agent chat with default settings
otdfctl llm agents [model-path]

# Start with specific agent directory
otdfctl llm agents --agent-dir ./my-agents model.gguf

# Enable RAG support
otdfctl llm agents --rag --index-path ~/.otdfctl/rag_index.json

# Custom context size and temperature
otdfctl llm agents --context-size 8192 --temperature 0.8
```

## Interface Controls

### Agent Selection Menu
- **Number (1-4)**: Select an agent to chat with
- **q/quit/exit**: Exit the application

### Chat Interface  
- **Enter**: Send message
- **Ctrl+C**: Return to agent selection menu
- **Ctrl+D**: Exit the application
- **/help**: Show available commands
- **/switch**: Return to agent selection menu
- **/quit**: Exit the application

## Agent Directory Structure

Agents are defined in markdown files with YAML frontmatter:

```markdown
---
agent_name: My Custom Agent
description: A specialized agent for my use case
tags: ["custom", "specialized"]
capabilities: ["domain expertise", "specific knowledge"]
---

You are a specialized OpenTDF agent with expertise in...

## Your Knowledge Areas
- Specific domain knowledge
- Best practices
- Common patterns

## Communication Style
- Clear and helpful responses
- Practical examples
- Step-by-step guidance
```

## Context Management

The interface displays real-time context usage similar to Claude Code:

- 🟢 **Green (0-50%)**: Plenty of context available
- 🟡 **Yellow (50-80%)**: Moderate context usage
- 🔴 **Red (80%+)**: Context limit approaching, auto-summarization recommended

## Agent Selection Logic

The system automatically selects agents based on:

1. **Keyword Matching**: Query content matches agent specialization
2. **Context Analysis**: Previous conversation topics
3. **Usage Patterns**: Recently used and effective agents
4. **Explicit Requests**: Direct agent mentions in queries

## Examples

### Policy Questions
*Automatically routed to Policy Agent*
```
Q: How do I create attributes for data classification?
A: [Policy Agent] I'll help you design an attribute structure...
```

### Development Questions  
*Automatically routed to Development Agent*
```
Q: How do I encrypt files using the Java SDK?
A: [Development Agent] Here's how to integrate TDF encryption...
```

### General Questions
*Routed to General Agent*
```
Q: What is OpenTDF and how does it work?
A: [General Agent] OpenTDF is a data-centric security framework...
```

## Configuration

Configure default settings in `otdfctl.yaml`:

```yaml
llm:
  agent_directory: "./agents"
  default_model_path: "path/to/model.gguf"
  context_size: 4096
  temperature: 0.7
```

## Troubleshooting

- **No agents found**: Check the agent directory path and file format
- **Agent selection issues**: Use `otdfctl llm agents validate` to check agent files
- **Context overflow**: Use `Ctrl+L` to clear history or enable auto-summarization
- **Performance issues**: Reduce context size or use a smaller model

## See Also

- `otdfctl llm agents list` - List available agents
- `otdfctl llm agents validate` - Validate agent configuration
- `otdfctl llm chat` - Traditional single-agent chat
- `otdfctl llm ingest` - Prepare RAG knowledge base