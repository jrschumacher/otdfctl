---
title: llm agents validate
command:
  name: validate
  usage: validate
  description: Validate agent configuration files and report any issues
---

# otdfctl llm agents validate

Validate agent configuration files and report any issues.

This command checks all agent files in the specified directory for proper formatting, required fields, and structural validity. It provides detailed feedback on loading success and failure reasons.

## Usage

```bash
# Validate agents in default directory
otdfctl llm agents validate

# Validate agents in specific directory
otdfctl llm agents validate --agent-dir ./my-agents

# Output detailed results as JSON
otdfctl llm agents validate --json
```

## Validation Checks

The validation process verifies:

1. **File Format**: Proper markdown with YAML frontmatter
2. **Required Fields**: `agent_name` and `description` are present
3. **YAML Syntax**: Valid YAML structure in frontmatter
4. **Content Structure**: Readable system prompt content
5. **File Access**: Proper file permissions and encoding

## Output Format

### Styled Output (Default)
```
🔍 Validating agents in: pkg/llm/agents

📊 Validation Results:
   📁 Files processed: 4
   ✅ Successfully loaded: 3 agents
   ❌ Failed to load: 1 agents
   ⏱️  Total time: 45ms

✅ Successfully Loaded Agents:
  1. Policy Agent (policy)
     📍 pkg/llm/agents/policy_agent.md (loaded in 12ms)

  2. Development Agent (development)
     📍 pkg/llm/agents/development_agent.md (loaded in 8ms)

❌ Failed to Load:
  1. pkg/llm/agents/broken_agent.md
     ❌ Error: failed to parse YAML frontmatter: yaml: line 3: could not find expected ':'
     ⏱️  Time: 2ms

💡 Tips for fixing agent files:
   - Ensure YAML frontmatter is properly formatted
   - Required fields: agent_name, description
   - Check file permissions and encoding
   - Validate YAML syntax with yamllint
```

### JSON Output
```json
{
  "agent_directory": "pkg/llm/agents",
  "total_files": 4,
  "loaded_agents": 3,
  "failed_agents": 1,
  "validation_time": "45.234ms",
  "loaded_details": [
    {
      "agent": {
        "metadata": {
          "name": "Policy Agent",
          "description": "OpenTDF policy specialist"
        },
        "type": "policy"
      },
      "file_path": "pkg/llm/agents/policy_agent.md",
      "load_time": "12.1ms"
    }
  ],
  "failed_details": [
    {
      "file_path": "pkg/llm/agents/broken_agent.md",
      "error": "failed to parse YAML frontmatter: yaml: line 3: could not find expected ':'",
      "load_time": "2.3ms"
    }
  ]
}
```

## Common Issues and Solutions

### Invalid YAML Frontmatter
```
Error: failed to parse YAML frontmatter
```
**Solution**: Check YAML syntax, ensure proper indentation and colons

### Missing Required Fields
```
Error: agent_name is required in frontmatter
```
**Solution**: Add required fields to frontmatter:
```yaml
---
agent_name: My Agent
description: Agent description
---
```

### File Access Issues
```
Error: permission denied
```
**Solution**: Check file permissions and ensure files are readable

### Empty Content
```
Error: processed content is empty
```
**Solution**: Add content after the frontmatter section

## Agent File Template

Use this template for creating valid agent files:

```markdown
---
agent_name: My Agent Name
description: Brief description of the agent's purpose and expertise
tags: ["tag1", "tag2"]
capabilities: ["capability1", "capability2"]
version: "1.0"
---

You are a specialized OpenTDF agent with expertise in [domain].

## Core Knowledge
- Domain-specific knowledge
- Best practices
- Common patterns

## Communication Style
- Clear and helpful responses
- Practical examples
- Step-by-step guidance

## When to Use
- Specific use cases
- Problem types
- Question categories
```

## Performance Information

The validation command reports timing information:

- **File Processing Time**: Time to read and parse each file
- **Total Validation Time**: Complete validation duration
- **Load Success Rate**: Percentage of successfully loaded agents

## Exit Codes

- `0`: All agents validated successfully
- `1`: Some agents failed validation
- `2`: Command execution error

## See Also

- `otdfctl llm agents list` - List available agents
- `otdfctl llm agents` - Start agent chat
- YAML validation tools: `yamllint`, `yq`