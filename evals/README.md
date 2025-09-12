# OpenTDF LLM Evaluation Framework

This directory contains OpenTDF-specific test cases and evaluation configurations that use the generic evaluation framework from **chanollama** to assess and improve the quality of OpenTDF's LLM-powered assistant responses.

## 🏗️ **Architecture**

The evaluation system uses a clean separation of concerns:
- **chanollama/evaluation.go**: Generic LLM evaluation utilities and framework
- **cmd/llm_eval.go**: OpenTDF-specific CLI integration and domain scoring
- **evals/**: OpenTDF test cases, quality standards, and baselines

See [EVALUATION_ARCHITECTURE.md](../EVALUATION_ARCHITECTURE.md) for detailed architectural documentation.

## 🎯 **Overview**

The evaluation system helps detect quality regressions, optimize prompts, and ensure consistent high-quality responses across all agent specializations.

## 🚀 **Quick Start**

### Single Query Evaluation (CI-Friendly)
```bash
# Evaluate a single query with JSON output for CI
otdfctl llm eval --query "What is OpenTDF and how does it work?" --output json

# Evaluate with agent mode disabled for comparison
otdfctl llm eval --query "How do I create an attribute?" --agent-mode=false --output json

# Verbose evaluation with detailed metrics
otdfctl llm eval --query "Show me Java code to encrypt a file" --verbose
```

### Batch Evaluation
```bash
# Run the quality regression test suite
otdfctl llm eval --test-suite evals/quality-regression-tests.json --output detailed

# Compare with baseline (for regression detection)
otdfctl llm eval --test-suite evals/quality-regression-tests.json --baseline previous-results.json
```

## 📊 **Quality Metrics**

The evaluation system measures:

### **Response Quality (0-10 scale)**
- **Helpfulness**: Practical value and actionability
- **Completeness**: Thoroughness and depth of coverage  
- **Actionability**: Concrete steps, code examples, commands
- **Length Appropriateness**: Sufficient detail without verbosity

### **Agent Performance**
- **Routing Accuracy**: Correct agent selection for queries
- **Response Time**: Performance benchmarks
- **Consistency**: Cross-agent alignment on shared concepts

### **Content Analysis**
- Word count and sentence structure
- Code examples and command inclusion
- Link and reference quality
- Security and best practice coverage

## 🔍 **Quality Standards**

### **Minimum Quality Thresholds**
- **General concepts**: 200+ words with examples and context
- **Policy guidance**: 300+ words with commands and security considerations  
- **Development help**: 400+ words with complete code examples
- **Deployment guidance**: 500+ words with full procedures and configs

### **Required Elements**
- ✅ Direct answer to the core question
- ✅ Contextual background and explanation
- ✅ Practical examples (code, commands, configs)
- ✅ Security and best practice considerations
- ✅ Next steps or additional resources

## 🏗️ **Test Suite Structure**

### **Quality Regression Tests** (`quality-regression-tests.json`)
Core test cases covering:
- **Concept explanations** (OpenTDF fundamentals)
- **Policy design** (ABAC, attributes, mappings)
- **Development integration** (SDK usage, code examples)
- **Deployment procedures** (Docker, Kubernetes, configuration)
- **Agent routing** (correct specialization selection)

### **Test Case Format**
```json
{
  "id": "policy_001",
  "query": "How do I create an attribute namespace?", 
  "expected_agent": "policy",
  "category": "policy_design",
  "min_length": 250,
  "must_include": ["namespace", "otdfctl", "create"],
  "expected_keywords": ["policy", "attribute", "command", "example"]
}
```

## 🔧 **CI Integration**

### **GitHub Actions Example**
```yaml
- name: Evaluate LLM Quality
  run: |
    otdfctl llm eval --test-suite evals/quality-regression-tests.json --output json > eval-results.json
    
- name: Check Quality Thresholds
  run: |
    # Fail if average score < 6.0 or pass rate < 80%
    python scripts/check-quality-gates.py eval-results.json
```

### **Quality Gates**
- **Average Score**: ≥ 6.0/10
- **Pass Rate**: ≥ 80% of tests
- **Agent Accuracy**: ≥ 90% correct routing
- **Response Time**: ≤ 10 seconds average

## 📈 **Quality Improvement Process**

### **1. Baseline Establishment**
```bash
# Create baseline with current prompts
otdfctl llm eval --test-suite evals/quality-regression-tests.json --output json > baseline.json
```

### **2. Prompt Optimization**
- Update agent system prompts in `/pkg/llm/agents/`
- Focus on comprehensive response guidelines
- Emphasize practical examples and actionable guidance

### **3. Regression Testing**  
```bash
# Test improvements against baseline
otdfctl llm eval --test-suite evals/quality-regression-tests.json --baseline baseline.json
```

### **4. Continuous Monitoring**
- Run evaluations on every PR
- Track quality trends over time
- Alert on significant regressions

## 🎭 **Agent-Specific Quality Focus**

### **General Agent**
- Comprehensive concept explanations
- Real-world use cases and scenarios
- Architecture and design guidance
- Integration with other agents' domains

### **Policy Agent** 
- Security-first approach with threat considerations
- Complete command examples with otdfctl
- Enterprise governance and compliance guidance
- ABAC design patterns and best practices

### **Development Agent**
- Complete, working code examples with imports
- Production-ready error handling and security
- Multi-language SDK coverage
- Testing and deployment guidance

### **Deployment Agent**
- Production-ready deployment procedures
- Complete configuration templates
- Security hardening and compliance
- Operational excellence and monitoring

## 🚨 **Quality Regression Investigation**

If quality scores drop:

1. **Compare agent routing accuracy** - wrong agents give poor responses
2. **Check response length distribution** - too short indicates prompt issues
3. **Analyze content quality metrics** - missing examples, commands, context
4. **Review recent prompt changes** - correlation with score drops
5. **Test with agent mode disabled** - compare base engine quality

## 📋 **Development Workflow**

### **Before Prompt Changes**
```bash
# Establish baseline
otdfctl llm eval --test-suite evals/quality-regression-tests.json --output json > before.json
```

### **After Prompt Changes**
```bash
# Test improvements
otdfctl llm eval --test-suite evals/quality-regression-tests.json --output json > after.json

# Compare results
python scripts/compare-evaluations.py before.json after.json
```

### **Quality Validation Checklist**
- [ ] Average score ≥ 6.0/10
- [ ] No responses < 100 words for complex topics
- [ ] Agent routing accuracy ≥ 90%
- [ ] All responses include practical examples
- [ ] Security considerations addressed
- [ ] Next steps or resources provided

This evaluation framework ensures that OpenTDF's LLM assistant maintains consistently high quality while enabling rapid iteration and improvement of the user experience.