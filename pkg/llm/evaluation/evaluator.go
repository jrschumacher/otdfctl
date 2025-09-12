package evaluation

import (
	"fmt"
	"strings"

	"github.com/jrschumacher/chanollama"
	"github.com/opentdf/otdfctl/pkg/llm"
)

// OpenTDFEvaluator wraps chanollama.Evaluator with OpenTDF-specific functionality
type OpenTDFEvaluator struct {
	evaluator   *chanollama.Evaluator
	engine      chanollama.Engine
	agentMode   bool
	model       string
	suiteLoader *chanollama.SuiteLoader
}

// EvaluatorConfig contains configuration for OpenTDF evaluator
type EvaluatorConfig struct {
	Model       string
	AgentMode   bool
	Verbose     bool
	Temperature float64
	Context     string // "production", "development", "ci"
}

// DefaultEvaluatorConfig returns default configuration for OpenTDF evaluation
func DefaultEvaluatorConfig(model string) EvaluatorConfig {
	return EvaluatorConfig{
		Model:       model,
		AgentMode:   false,
		Verbose:     false,
		Temperature: 0.1, // Lower temperature for consistent evaluation
		Context:     "development",
	}
}

// NewOpenTDFEvaluator creates an evaluator with OpenTDF-specific configuration
func NewOpenTDFEvaluator(config EvaluatorConfig, pathResolver *llm.ModelPathResolver) (*OpenTDFEvaluator, error) {
	var resolvedPath string
	var err error
	
	// Check if model looks like an Ollama model name (no file path indicators)
	if !strings.Contains(config.Model, "/") && !strings.HasSuffix(config.Model, ".gguf") && !strings.HasSuffix(config.Model, ".bin") {
		// Use model name directly for Ollama API mode
		resolvedPath = config.Model
	} else {
		// Use file path resolution for local models
		resolvedPath, err = pathResolver.ResolveChatModelPath(config.Model)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve model path: %w", err)
		}
	}
	
	// Create engine configuration optimized for evaluation
	engineConfig := chanollama.DefaultConfig(resolvedPath)
	engineConfig.Verbose = config.Verbose
	engineConfig.Temperature = config.Temperature
	
	var engine chanollama.Engine
	
	if config.AgentMode {
		// Create agent-aware engine for OpenTDF
		agentEngine, err := chanollama.NewSimpleAgentEngine(engineConfig)
		if err != nil {
			return nil, err
		}
		engine = chanollama.Engine(agentEngine)
	} else {
		// Create regular engine
		regularEngine, err := chanollama.NewEngine(engineConfig)
		if err != nil {
			return nil, err
		}
		engine = regularEngine
	}
	
	// Create evaluator with OpenTDF-specific scoring
	evaluator := chanollama.NewEvaluator(engine, config.AgentMode).
		WithScoringFunction(OpenTDFScoringFunction).
		WithSystemInfo(chanollama.SystemInfo{
			Version:   "v0.1.0", // TODO: Get from build info
			Model:     resolvedPath,
			AgentMode: config.AgentMode,
			EvalMode:  "opentdf-comprehensive",
		})
	
	// Create suite loader with OpenTDF-specific paths
	suiteLoader := chanollama.NewSuiteLoaderWithPaths([]string{
		"evals",           // OpenTDF evals directory
		"test_suites",     // Alternative location
		"tests",           // Common tests directory
		".",               // Current directory
	})
	
	return &OpenTDFEvaluator{
		evaluator:   evaluator,
		engine:      engine,
		agentMode:   config.AgentMode,
		model:       resolvedPath,
		suiteLoader: suiteLoader,
	}, nil
}

// EvaluateQuery evaluates a single query with OpenTDF-specific scoring
func (oe *OpenTDFEvaluator) EvaluateQuery(query string) (chanollama.EvalResult, error) {
	return oe.evaluator.EvaluateQuery(query)
}

// EvaluateSuite evaluates a test suite with OpenTDF-specific configuration
func (oe *OpenTDFEvaluator) EvaluateSuite(suiteFile string) (chanollama.EvalSummary, error) {
	// Load suite using OpenTDF-aware loader
	suite, err := oe.LoadOpenTDFTestSuite(suiteFile)
	if err != nil {
		return chanollama.EvalSummary{}, err
	}
	
	return oe.evaluator.EvaluateSuite(suite)
}

// LoadOpenTDFTestSuite loads test suite with OpenTDF-specific defaults
func (oe *OpenTDFEvaluator) LoadOpenTDFTestSuite(filePath string) (chanollama.EvalSuite, error) {
	return oe.suiteLoader.LoadSuite(filePath)
}

// EvaluateWithCategoryScoring evaluates using category-specific scoring functions
func (oe *OpenTDFEvaluator) EvaluateWithCategoryScoring(query, category string) (chanollama.EvalResult, error) {
	// Create a new evaluator with category-specific scoring
	categoryScoringFunc := GetScoringFunction(category)
	
	categoryEvaluator := chanollama.NewEvaluator(oe.engine, oe.agentMode).
		WithScoringFunction(categoryScoringFunc)
	
	return categoryEvaluator.EvaluateQuery(query)
}

// CompareWithBaseline compares results with OpenTDF baseline expectations
func (oe *OpenTDFEvaluator) CompareWithBaseline(current chanollama.EvalSummary, baselineFile string) (chanollama.BaselineComparison, error) {
	manager := chanollama.NewBaselineManager()
	return manager.CompareWithBaseline(current, baselineFile)
}

// GetQualityAssessment provides OpenTDF-specific quality assessment
func (oe *OpenTDFEvaluator) GetQualityAssessment(score float64, context string) (string, string, bool) {
	return EvaluateQualityLevel(score, context)
}

// Close closes the underlying engine
func (oe *OpenTDFEvaluator) Close() error {
	if oe.engine != nil {
		return oe.engine.Close()
	}
	return nil
}

// GetModel returns the model path being used
func (oe *OpenTDFEvaluator) GetModel() string {
	return oe.model
}

// IsAgentMode returns whether agent mode is enabled
func (oe *OpenTDFEvaluator) IsAgentMode() bool {
	return oe.agentMode
}

// GetSuiteLoader returns the suite loader for advanced operations
func (oe *OpenTDFEvaluator) GetSuiteLoader() *chanollama.SuiteLoader {
	return oe.suiteLoader
}

// AddSuiteSearchPath adds a custom search path for test suites
func (oe *OpenTDFEvaluator) AddSuiteSearchPath(path string) {
	oe.suiteLoader.AddSearchPath(path)
}

// ListAvailableTestSuites lists all available test suites in search paths
func (oe *OpenTDFEvaluator) ListAvailableTestSuites() ([]string, error) {
	return oe.suiteLoader.ListAvailableSuites()
}

// ValidateTestSuite validates a test suite file without loading all test cases
func (oe *OpenTDFEvaluator) ValidateTestSuite(filePath string) error {
	_, err := oe.suiteLoader.LoadSuite(filePath)
	return err
}

// GetTestSuiteInfo returns basic information about a test suite
func (oe *OpenTDFEvaluator) GetTestSuiteInfo(filePath string) (chanollama.SuiteInfo, error) {
	return oe.suiteLoader.GetSuiteInfo(filePath)
}