package evaluation

import (
	"strings"

	"github.com/jrschumacher/chanollama"
)

// OpenTDFScoringFunction provides OpenTDF-specific quality scoring
func OpenTDFScoringFunction(response string, metrics chanollama.EvalMetrics, testCase *chanollama.TestCase) float64 {
	// Start with default scoring
	score := chanollama.DefaultScoringFunction(response, metrics, testCase)
	
	// OpenTDF-specific bonuses and penalties
	responseText := strings.ToLower(response)
	
	// Bonus for OpenTDF-specific content
	if strings.Contains(responseText, "otdfctl") { 
		score += 0.5 
	}
	if strings.Contains(responseText, "opentdf") { 
		score += 0.3 
	}
	if strings.Contains(responseText, "tdf") || strings.Contains(responseText, "nanotdf") { 
		score += 0.3 
	}
	
	// Bonus for practical elements
	if strings.Contains(responseText, "policy") && strings.Contains(responseText, "attribute") { 
		score += 0.5 
	}
	if strings.Contains(responseText, "encrypt") && strings.Contains(responseText, "decrypt") { 
		score += 0.4 
	}
	if strings.Contains(responseText, "docker") || strings.Contains(responseText, "kubernetes") { 
		score += 0.3 
	}
	
	// Bonus for security considerations
	if strings.Contains(responseText, "security") { 
		score += 0.3 
	}
	if strings.Contains(responseText, "best practice") { 
		score += 0.4 
	}
	if strings.Contains(responseText, "compliance") { 
		score += 0.3 
	}
	
	// Penalty for missing OpenTDF context in relevant topics
	if testCase != nil {
		category := testCase.Category
		if category == "policy_design" && !strings.Contains(responseText, "abac") { 
			score -= 0.5 
		}
		if category == "code_example" && metrics.CodeExampleCount == 0 { 
			score -= 1.0 
		}
		if category == "deployment" && !strings.Contains(responseText, "config") { 
			score -= 0.5 
		}
	}
	
	return clampScore(score)
}

// clampScore ensures score stays within valid range
func clampScore(score float64) float64 {
	if score > 10.0 {
		return 10.0
	}
	if score < 0.0 {
		return 0.0
	}
	return score
}

// CustomPolicyScoringFunction provides specialized scoring for policy-related queries
func CustomPolicyScoringFunction(response string, metrics chanollama.EvalMetrics, testCase *chanollama.TestCase) float64 {
	score := OpenTDFScoringFunction(response, metrics, testCase)
	responseText := strings.ToLower(response)
	
	// Additional policy-specific bonuses
	if strings.Contains(responseText, "abac") || strings.Contains(responseText, "attribute-based") {
		score += 0.4
	}
	if strings.Contains(responseText, "namespace") && strings.Contains(responseText, "create") {
		score += 0.3
	}
	if strings.Contains(responseText, "kas") || strings.Contains(responseText, "key access service") {
		score += 0.3
	}
	
	return clampScore(score)
}

// CustomDevelopmentScoringFunction provides specialized scoring for development queries
func CustomDevelopmentScoringFunction(response string, metrics chanollama.EvalMetrics, testCase *chanollama.TestCase) float64 {
	score := OpenTDFScoringFunction(response, metrics, testCase)
	responseText := strings.ToLower(response)
	
	// Development-specific bonuses
	if metrics.CodeExampleCount > 0 {
		score += 0.5 * float64(metrics.CodeExampleCount) // Bonus per code example
	}
	if strings.Contains(responseText, "sdk") {
		score += 0.4
	}
	if strings.Contains(responseText, "java") || strings.Contains(responseText, "python") || 
	   strings.Contains(responseText, "javascript") || strings.Contains(responseText, "go") {
		score += 0.3
	}
	if strings.Contains(responseText, "maven") || strings.Contains(responseText, "npm") || 
	   strings.Contains(responseText, "pip") || strings.Contains(responseText, "go mod") {
		score += 0.2
	}
	
	return clampScore(score)
}

// CustomDeploymentScoringFunction provides specialized scoring for deployment queries
func CustomDeploymentScoringFunction(response string, metrics chanollama.EvalMetrics, testCase *chanollama.TestCase) float64 {
	score := OpenTDFScoringFunction(response, metrics, testCase)
	responseText := strings.ToLower(response)
	
	// Deployment-specific bonuses
	if strings.Contains(responseText, "helm") || strings.Contains(responseText, "chart") {
		score += 0.4
	}
	if strings.Contains(responseText, "environment") && strings.Contains(responseText, "variable") {
		score += 0.3
	}
	if strings.Contains(responseText, "scaling") || strings.Contains(responseText, "load") {
		score += 0.3
	}
	if strings.Contains(responseText, "monitoring") || strings.Contains(responseText, "logging") {
		score += 0.2
	}
	
	return clampScore(score)
}

// GetScoringFunction returns the appropriate scoring function based on category
func GetScoringFunction(category string) func(string, chanollama.EvalMetrics, *chanollama.TestCase) float64 {
	switch category {
	case "policy", "policy_design":
		return CustomPolicyScoringFunction
	case "development", "code_example":
		return CustomDevelopmentScoringFunction
	case "deployment", "ops":
		return CustomDeploymentScoringFunction
	default:
		return OpenTDFScoringFunction
	}
}

// QualityThresholds defines quality thresholds for different contexts
type QualityThresholds struct {
	Production  float64 `json:"production"`  // Threshold for production readiness
	Development float64 `json:"development"` // Threshold for development use
	CI          float64 `json:"ci"`          // Threshold for CI validation
}

// DefaultQualityThresholds returns standard quality thresholds for OpenTDF
func DefaultQualityThresholds() QualityThresholds {
	return QualityThresholds{
		Production:  7.0, // High quality required for production
		Development: 5.0, // Moderate quality for development
		CI:          2.0, // Basic validation for CI (with tiny models)
	}
}

// EvaluateQualityLevel determines quality level based on score and context
func EvaluateQualityLevel(score float64, context string) (string, string, bool) {
	thresholds := DefaultQualityThresholds()
	
	var threshold float64
	switch context {
	case "production", "prod":
		threshold = thresholds.Production
	case "development", "dev":
		threshold = thresholds.Development
	case "ci", "continuous-integration":
		threshold = thresholds.CI
	default:
		threshold = thresholds.Development // Default to development
	}
	
	passes := score >= threshold
	
	var level, description string
	switch {
	case score >= 8.0:
		level = "excellent"
		description = "✅ Excellent - Ready for production use"
	case score >= 6.0:
		level = "good"
		description = "👍 Good - Minor improvements recommended"
	case score >= 4.0:
		level = "fair"
		description = "⚠️  Fair - Significant improvements needed"
	default:
		level = "poor"
		description = "❌ Poor - Major issues require attention"
	}
	
	return level, description, passes
}