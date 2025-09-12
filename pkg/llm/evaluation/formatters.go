package evaluation

import (
	"encoding/json"
	"strings"

	"github.com/jrschumacher/chanollama"
	"github.com/opentdf/otdfctl/pkg/cli"
)

// OutputFormatter handles different output formats for OpenTDF evaluation results
type OutputFormatter struct {
	format string
	title  string
}

// NewOutputFormatter creates a new formatter for the specified format
func NewOutputFormatter(format string) *OutputFormatter {
	return &OutputFormatter{
		format: format,
		title:  "OpenTDF LLM Evaluation",
	}
}

// NewOutputFormatterWithTitle creates a formatter with a custom title
func NewOutputFormatterWithTitle(format, title string) *OutputFormatter {
	return &OutputFormatter{
		format: format,
		title:  title,
	}
}

// FormatSingleQueryResult formats a single query evaluation result
func (of *OutputFormatter) FormatSingleQueryResult(c *cli.Cli, result chanollama.EvalResult) {
	switch of.format {
	case "json":
		of.formatSingleQueryJSON(c, result)
	case "detailed":
		of.formatSingleQueryDetailed(c, result)
	default: // summary
		of.formatSingleQuerySummary(c, result)
	}
}

// FormatEvaluationSummary formats an evaluation summary
func (of *OutputFormatter) FormatEvaluationSummary(c *cli.Cli, summary chanollama.EvalSummary) {
	switch of.format {
	case "json":
		of.formatSummaryJSON(c, summary)
	case "detailed":
		of.formatSummaryDetailed(c, summary)
	default: // summary
		of.formatSummaryStandard(c, summary)
	}
}

// FormatDetailedResults formats detailed test results
func (of *OutputFormatter) FormatDetailedResults(c *cli.Cli, results []chanollama.EvalResult) {
	if of.format != "detailed" {
		return // Only show detailed results in detailed mode
	}
	
	formatter := chanollama.NewDetailedFormatter(of.title)
	output := formatter.FormatDetailedResults(results)
	c.Print(output)
}

// formatSingleQueryJSON formats single query result as JSON
func (of *OutputFormatter) formatSingleQueryJSON(c *cli.Cli, result chanollama.EvalResult) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		c.Printf("Error formatting JSON: %v\n", err)
		return
	}
	c.Print(string(data))
}

// formatSingleQueryDetailed formats single query result with detailed information
func (of *OutputFormatter) formatSingleQueryDetailed(c *cli.Cli, result chanollama.EvalResult) {
	formatter := chanollama.NewDetailedFormatter(of.title)
	output := formatter.FormatSingleResult(result)
	c.Print(output)
	
	// Add OpenTDF-specific quality assessment
	_, description, _ := EvaluateQualityLevel(result.QualityScore, "development")
	c.Printf("\n🎯 OpenTDF Quality Assessment:\n%s\n", description)
	
	// Show response content for detailed view
	if len(result.Response) > 0 {
		c.Printf("\n📝 Response Content:\n")
		c.Printf("%s\n", strings.Repeat("-", 50))
		c.Printf("%s\n", result.Response)
	}
}

// formatSingleQuerySummary formats single query result as summary
func (of *OutputFormatter) formatSingleQuerySummary(c *cli.Cli, result chanollama.EvalResult) {
	formatter := chanollama.NewSummaryFormatter()
	output := formatter.FormatSingleResult(result)
	c.Print(output)
}

// formatSummaryJSON formats evaluation summary as JSON
func (of *OutputFormatter) formatSummaryJSON(c *cli.Cli, summary chanollama.EvalSummary) {
	data, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		c.Printf("Error formatting JSON: %v\n", err)
		return
	}
	c.Print(string(data))
}

// formatSummaryDetailed formats evaluation summary with comprehensive details
func (of *OutputFormatter) formatSummaryDetailed(c *cli.Cli, summary chanollama.EvalSummary) {
	formatter := chanollama.NewDetailedFormatter(of.title)
	output := formatter.FormatSummary(summary)
	c.Print(output)
}

// formatSummaryStandard formats evaluation summary in standard format
func (of *OutputFormatter) formatSummaryStandard(c *cli.Cli, summary chanollama.EvalSummary) {
	formatter := chanollama.NewSummaryFormatter()
	output := formatter.FormatSummary(summary)
	c.Print(output)
}

// PrintOpenTDFSummary shows OpenTDF-specific evaluation summary
func PrintOpenTDFSummary(c *cli.Cli, summary chanollama.EvalSummary) {
	c.Printf("\n📊 OpenTDF LLM Evaluation Summary\n")
	c.Printf("═══════════════════════════════════\n")
	c.Printf("Tests: %d total, %d passed, %d failed\n", 
		summary.TotalTests, summary.PassedTests, summary.FailedTests)
	c.Printf("Average Quality Score: %.2f/10\n", summary.AverageScore)
	c.Printf("Average Response Length: %.0f characters\n", summary.AverageLength)
	c.Printf("Average Response Time: %v\n", summary.AverageTime)
	
	if summary.AgentAccuracy > 0 {
		c.Printf("Agent Routing Accuracy: %.1f%%\n", summary.AgentAccuracy)
	}
	
	passRate := float64(summary.PassedTests) / float64(summary.TotalTests) * 100
	c.Printf("Pass Rate: %.1f%%\n", passRate)
	
	// OpenTDF-specific quality assessment
	_, description, _ := EvaluateQualityLevel(summary.AverageScore, "development")
	c.Printf("\n🎯 OpenTDF Quality Assessment:\n%s\n", description)
}

// PrintProgressResults shows progress during batch evaluation
func PrintProgressResults(c *cli.Cli, results []chanollama.EvalResult) {
	formatter := chanollama.NewDetailedFormatter("")
	output := formatter.FormatProgress(results)
	c.Print(output)
}

// PrintSingleQueryResult shows detailed results for a single query
func PrintSingleQueryResult(c *cli.Cli, result chanollama.EvalResult) {
	formatter := chanollama.NewDetailedFormatter("")
	output := formatter.FormatSingleResult(result)
	c.Print(output)
}

// FormatBaselineComparison formats baseline comparison results
func FormatBaselineComparison(c *cli.Cli, comparison chanollama.BaselineComparison, format string) {
	switch format {
	case "json":
		data, err := json.MarshalIndent(comparison, "", "  ")
		if err != nil {
			c.Printf("Error formatting baseline comparison JSON: %v\n", err)
			return
		}
		c.Print(string(data))
	case "detailed":
		c.Printf("\n📊 Baseline Comparison Results\n")
		c.Printf("════════════════════════════════\n")
		c.Printf("Baseline File: %s\n", comparison.BaselineFile)
		c.Printf("Comparison Time: %s\n", comparison.ComparisonTime.Format("2006-01-02 15:04:05"))
		c.Printf("\n📈 Overall Change: %s (%.2f%% change)\n", 
			strings.Title(comparison.OverallChange.Direction), 
			comparison.OverallChange.Percentage)
		
		c.Printf("\n🎯 Score Comparison:\n")
		c.Printf("  Baseline: %.2f/10\n", comparison.ScoreChange.BaselineAverage)
		c.Printf("  Current:  %.2f/10\n", comparison.ScoreChange.CurrentAverage)
		c.Printf("  Change:   %+.2f (%s)\n", 
			comparison.ScoreChange.Change.Magnitude,
			comparison.ScoreChange.Significance)
		
		c.Printf("\n⏱️  Performance Comparison:\n")
		c.Printf("  Baseline Time: %v\n", comparison.PerformanceChange.ResponseTime.BaselineAverage)
		c.Printf("  Current Time:  %v\n", comparison.PerformanceChange.ResponseTime.CurrentAverage)
		
		c.Printf("\n📋 Test Summary:\n")
		c.Printf("  Total Tests: %d\n", comparison.Summary.TotalTests)
		c.Printf("  Improved:    %d\n", comparison.Summary.ImprovedTests)
		c.Printf("  Degraded:    %d\n", comparison.Summary.DegradedTests)
		c.Printf("  Stable:      %d\n", comparison.Summary.StableTests)
		c.Printf("  Risk Level:  %s\n", strings.Title(comparison.Summary.RegressionRisk))
		
		c.Printf("\n💡 Recommendation:\n")
		c.Printf("   %s\n", comparison.Summary.Recommendation)
	default: // summary
		c.Printf("📊 Baseline Comparison: %s change of %.2f%% (%s significance)\n",
			comparison.OverallChange.Direction,
			comparison.OverallChange.Percentage,
			comparison.ScoreChange.Significance)
		c.Printf("💡 %s\n", comparison.Summary.Recommendation)
	}
}