---
title: llm eval
command:
  name: eval
  usage: eval [flags]
  description: Evaluate LLM responses for quality and consistency using OpenTDF-specific metrics
---

# llm eval

Evaluate LLM responses for quality and consistency using OpenTDF-specific metrics.

## Synopsis

```shell
otdfctl llm eval [flags]
```

## Description

The `eval` command provides comprehensive evaluation capabilities for LLM responses, enabling quality assessment, regression testing, and automated CI validation. It supports both single query evaluation and batch test suite processing with customizable scoring functions.

## Options

### Evaluation Mode
- `--query string` - Single query to evaluate
- `--test-suite string` - Path to JSON test suite file
- `--agent-mode` - Enable agent-based routing for queries

### Model Configuration  
- `--model string` - Path to GGUF model file (required)
- `--config string` - Path to model configuration file

### Output Options
- `--output string` - Output format: json, detailed, summary (default "summary")
- `--verbose` - Enable verbose logging and detailed metrics
- `--baseline string` - Path to baseline results for comparison

### Quality Thresholds
- `--min-score float` - Minimum acceptable quality score (default 5.0)
- `--timeout duration` - Maximum evaluation time per query (default 30s)

## Examples

### Single Query Evaluation
```shell
# Basic query evaluation
otdfctl llm eval --query "What is OpenTDF?" --model models/test-model.gguf

# Verbose evaluation with JSON output
otdfctl llm eval \
  --query "How do I create an attribute namespace?" \
  --model models/test-model.gguf \
  --output json \
  --verbose

# Agent-based routing
otdfctl llm eval \
  --query "Deploy OpenTDF with Docker" \
  --model models/test-model.gguf \
  --agent-mode \
  --output detailed
```

### Test Suite Evaluation
```shell
# Run full regression test suite
otdfctl llm eval \
  --test-suite evals/quality-regression-tests.json \
  --model models/test-model.gguf \
  --output detailed

# Compare against baseline
otdfctl llm eval \
  --test-suite evals/quality-regression-tests.json \
  --model models/test-model.gguf \
  --baseline previous-results.json \
  --output json
```

### CI/CD Integration
```shell
# Fast CI validation with tiny model
otdfctl llm eval \
  --test-suite evals/quality-regression-tests.json \
  --model models/tiny-test-model.gguf \
  --min-score 2.0 \
  --output json

# Quality gate validation
RESULT=$(otdfctl llm eval --test-suite evals/quality-regression-tests.json --model models/test-model.gguf --output json)
SCORE=$(echo "$RESULT" | jq -r '.summary.average_score')
if (( $(echo "$SCORE < 5.0" | bc -l) )); then
  echo "Quality threshold not met: $SCORE"
  exit 1
fi
```

## Test Suite Format

Test suites are JSON files with the following structure:

```json
{
  "name": "Quality Regression Tests",
  "description": "Comprehensive evaluation of OpenTDF prompt quality",
  "test_cases": [
    {
      "id": "policy_001",
      "query": "How do I create an attribute namespace?",
      "category": "policy",
      "expected_agent": "policy_agent",
      "min_length": 200,
      "must_include": ["namespace", "attribute"],
      "expected_keywords": ["otdfctl", "policy", "create"]
    }
  ]
}
```

## Quality Metrics

The evaluation system assesses responses using multiple metrics:

### Core Metrics (0-10 scale)
- **Helpfulness**: Practical value and relevance to the query
- **Completeness**: Thoroughness and comprehensive coverage
- **Actionability**: Specific examples, commands, and implementation guidance

### Content Analysis
- **Length Analysis**: Word and character counts
- **Code Detection**: Command examples and code blocks
- **Link Analysis**: Reference and documentation links
- **Keyword Matching**: Domain-specific terminology

### OpenTDF-Specific Scoring
- **Domain Bonus**: +0.5 for mentioning "otdfctl"
- **Platform Bonus**: +0.3 for mentioning "opentdf"
- **Command Bonus**: +0.2 for including CLI commands
- **Example Bonus**: +0.4 for providing code examples

## Output Formats

### Summary (Default)
```
Evaluation Results:
Query: "What is OpenTDF?"
Score: 7.8/10
Agent: general_agent (confidence: 0.95)
Response Length: 342 words
```

### JSON
```json
{
  "query": "What is OpenTDF?",
  "score": 7.8,
  "agent_used": "general_agent",
  "metrics": {
    "helpfulness": 8.2,
    "completeness": 7.5,
    "actionability": 7.7
  }
}
```

### Detailed
Comprehensive output including full response text, metric breakdowns, agent reasoning, and performance statistics.

## Exit Codes

- `0` - Evaluation completed successfully
- `1` - Quality threshold not met
- `2` - Model or configuration error
- `3` - Test suite validation failed

## See Also

- [llm chat](../chat) - Interactive LLM chat interface
- [llm ingest](../ingest) - RAG document ingestion
- [llm agents](../agents) - Agent management