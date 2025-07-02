---
title: llm agents list
command:
  name: list
  usage: list
  description: List all available agents in the configured agent directory
---

# otdfctl llm agents list

List all available agents in the configured agent directory.

This command discovers and displays all agents that can be loaded for the agent chat interface. It shows agent metadata, capabilities, and validation status.

## Usage

```bash
# List agents in default directory
otdfctl llm agents list

# List agents in specific directory
otdfctl llm agents list --agent-dir ./my-agents

# Output as JSON
otdfctl llm agents list --json
```

## Output Format

### Styled Output (Default)
```
📁 Agent Directory: pkg/llm/agents
🤖 Available Agents (4):

  1. Policy Agent (policy)
     📝 OpenTDF policy specialist expert in ABAC design
     📍 pkg/llm/agents/policy_agent.md
     🏷️  Tags: [policy, ABAC, attributes]
     ⚡ Capabilities: [policy design, attribute management]

  2. Development Agent (development)
     📝 SDK development specialist for multiple languages
     📍 pkg/llm/agents/development_agent.md
     🏷️  Tags: [development, SDK, programming]
     ⚡ Capabilities: [Java SDK, Go SDK, JavaScript SDK]
```

### JSON Output
```json
{
  "agent_directory": "pkg/llm/agents",
  "total_agents": 4,
  "agents": [
    {
      "name": "Policy Agent",
      "type": "policy",
      "description": "OpenTDF policy specialist expert in ABAC design",
      "file_path": "pkg/llm/agents/policy_agent.md",
      "tags": ["policy", "ABAC", "attributes"],
      "capabilities": ["policy design", "attribute management"]
    }
  ]
}
```

## Agent Information

For each agent, the command displays:

- **Name**: The agent's display name
- **Type**: Agent category (policy, development, deployment, general)
- **Description**: Brief description of the agent's purpose
- **File Path**: Location of the agent definition file
- **Tags**: Keywords for categorization and discovery
- **Capabilities**: Specific skills and knowledge areas

## Agent Types

- **policy**: Specialists in ABAC design, policy creation, and access control
- **development**: Experts in SDK integration and application development
- **deployment**: Focused on installation, configuration, and operations
- **general**: Provide explanations of OpenTDF concepts and architecture

## Creating New Agents

To create a new agent, add a markdown file to the agent directory:

```markdown
---
agent_name: Custom Agent
description: Your custom agent description
tags: ["custom", "domain-specific"]
capabilities: ["specialized knowledge", "specific tasks"]
---

You are a custom OpenTDF agent with expertise in...
```

## See Also

- `otdfctl llm agents` - Start agent chat
- `otdfctl llm agents validate` - Validate agent configuration