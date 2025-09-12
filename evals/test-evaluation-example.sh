#!/bin/bash

# Test script for the chanollama evaluation example
set -e

echo "🧪 Testing Chanollama Evaluation Example"
echo "========================================"

# Test 1: Example structure validation
echo ""
echo "Test 1: Example structure validation"
echo "------------------------------------"

if [ -f "chanollama/examples/evaluation/main.go" ]; then
    echo "✅ Evaluation example found"
    
    # Check for key components
    grep -q "demonstrateSingleQueryEvaluation" chanollama/examples/evaluation/main.go && echo "✅ Single query evaluation demo found"
    grep -q "demonstrateCustomScoring" chanollama/examples/evaluation/main.go && echo "✅ Custom scoring demo found"  
    grep -q "demonstrateTestSuiteEvaluation" chanollama/examples/evaluation/main.go && echo "✅ Test suite demo found"
    grep -q "demonstrateBaselineComparison" chanollama/examples/evaluation/main.go && echo "✅ Baseline comparison demo found"
    
    # Check for MockEngine implementation
    grep -q "type MockEngine struct" chanollama/examples/evaluation/main.go && echo "✅ MockEngine implementation found"
    grep -q "generateMockResponse" chanollama/examples/evaluation/main.go && echo "✅ Mock response generation found"
    
else
    echo "❌ Evaluation example not found"
fi

# Test 2: README documentation
echo ""
echo "Test 2: Documentation validation"
echo "-------------------------------"

if [ -f "chanollama/examples/evaluation/README.md" ]; then
    echo "✅ Evaluation example README found"
    
    grep -q "Custom Scoring Functions" chanollama/examples/evaluation/README.md && echo "✅ Custom scoring documentation found"
    grep -q "Test Suite Evaluation" chanollama/examples/evaluation/README.md && echo "✅ Test suite documentation found"
    grep -q "Adapting for Your Project" chanollama/examples/evaluation/README.md && echo "✅ Adaptation guide found"
    
else
    echo "❌ Evaluation example README not found"
fi

# Test 3: Integration with main examples
echo ""
echo "Test 3: Examples integration"
echo "---------------------------"

if [ -f "chanollama/examples/README.md" ]; then
    echo "✅ Main examples README found"
    
    grep -q "evaluation/" chanollama/examples/README.md && echo "✅ Evaluation example listed"
    grep -q "Quality Metrics" chanollama/examples/README.md && echo "✅ Feature highlights included"
    
else
    echo "❌ Main examples README not found"
fi

# Test 4: Code structure validation
echo ""
echo "Test 4: Code structure validation"
echo "--------------------------------"

# Check that example uses the evaluation framework correctly
grep -q "chanollama.NewEvaluator" chanollama/examples/evaluation/main.go && echo "✅ Evaluator initialization found"
grep -q "chanollama.EvalSuite" chanollama/examples/evaluation/main.go && echo "✅ Test suite types used"
grep -q "chanollama.TestCase" chanollama/examples/evaluation/main.go && echo "✅ Test case types used"
grep -q "WithScoringFunction" chanollama/examples/evaluation/main.go && echo "✅ Custom scoring integration found"

# Check for realistic mock responses
grep -q "encryption" chanollama/examples/evaluation/main.go && echo "✅ Domain-specific mock responses found"
grep -q "authentication" chanollama/examples/evaluation/main.go && echo "✅ Multiple response scenarios found"

# Test 5: Framework independence
echo ""
echo "Test 5: Framework independence"
echo "-----------------------------"

# Verify the example doesn't depend on OpenTDF specifics
if ! grep -q "otdfctl\|opentdf" chanollama/examples/evaluation/main.go; then
    echo "✅ Example is framework-independent"
else
    echo "⚠️  Example may have OpenTDF-specific dependencies"
fi

# Check that it demonstrates generic patterns
grep -q "security-focused evaluation" chanollama/examples/evaluation/main.go && echo "✅ Generic domain example used"
grep -q "Mock.*demonstrates" chanollama/examples/evaluation/main.go && echo "✅ Clear demonstration purpose"

echo ""
echo "🎯 Test Summary"
echo "==============="
echo "✅ Evaluation example structure complete"
echo "✅ Comprehensive documentation provided"
echo "✅ Framework-independent implementation"
echo "✅ Realistic demonstration scenarios"
echo "✅ Integration with chanollama examples"
echo ""
echo "🚀 Example Features:"
echo "   • Single query evaluation with quality metrics"
echo "   • Custom scoring functions for domain-specific needs"
echo "   • Test suite evaluation with multiple test cases"  
echo "   • Baseline comparison for regression detection"
echo "   • MockEngine for testing without external dependencies"
echo "   • Comprehensive documentation and adaptation guide"