package llm

import (
	"context"
	"sync"
	"time"
)

// OperationType defines the type of operation to perform on the engine
type OperationType int

const (
	OpLoadModel OperationType = iota
	OpInference              // Send query and get response
	OpStream                 // Send query and get streaming response
	OpGetMetrics
	OpUnloadModel
	OpShutdown
	OpCancel
	OpConfigurePrompt
	OpConfigureRAG
	OpInspectModel
	OpHealthCheck
	OpConfigureGeneration
	OpTokenCount // Count tokens in text
	OpWarmup     // Pre-warm model
)

// String returns the string representation of the operation type
func (ot OperationType) String() string {
	switch ot {
	case OpLoadModel:
		return "LoadModel"
	case OpInference:
		return "Inference"
	case OpStream:
		return "Stream"
	case OpGetMetrics:
		return "GetMetrics"
	case OpUnloadModel:
		return "UnloadModel"
	case OpShutdown:
		return "Shutdown"
	case OpCancel:
		return "Cancel"
	case OpConfigurePrompt:
		return "ConfigurePrompt"
	case OpConfigureRAG:
		return "ConfigureRAG"
	case OpInspectModel:
		return "InspectModel"
	case OpHealthCheck:
		return "HealthCheck"
	case OpConfigureGeneration:
		return "ConfigureGeneration"
	case OpTokenCount:
		return "TokenCount"
	case OpWarmup:
		return "Warmup"
	default:
		return "Unknown"
	}
}

// EngineRequest represents a request to the LLM engine
type EngineRequest struct {
	ID       string
	Type     OperationType
	Payload  interface{}
	Response chan<- EngineResponse
	Context  context.Context
}

// EngineResponse represents a response from the LLM engine
type EngineResponse struct {
	RequestID string
	Success   bool
	Data      interface{}
	Error     error
	Metadata  map[string]interface{}
}

// StreamingResponse represents a streaming token response
type StreamingResponse struct {
	RequestID string
	Token     string
	Done      bool
	Error     error
	Metadata  map[string]interface{}
}

// Operation payload types

// LoadModelPayload contains parameters for loading a model
type LoadModelPayload struct {
	ModelPath        string
	SystemPrompt     string
	ContextSize      int
	EnableRAG        bool
	RAGIndexPath     string
	GenerationParams GenerationParams
}

// InferencePayload contains parameters for inference operations
type InferencePayload struct {
	Query       string
	MaxTokens   int
	Temperature float64
	IncludeRAG  bool
}

// StreamingPayload contains parameters for streaming operations
type StreamingPayload struct {
	Query       string
	MaxTokens   int
	Temperature float64
	IncludeRAG  bool
	ChunkSize   int // Tokens per streaming chunk
}

// ConfigurePromptPayload contains parameters for updating system prompt
type ConfigurePromptPayload struct {
	SystemPrompt string
}

// ConfigureRAGPayload contains parameters for RAG configuration
type ConfigureRAGPayload struct {
	Enabled   bool
	IndexPath string
}

// TokenCountPayload contains text to count tokens for
type TokenCountPayload struct {
	Text string
}

// CancelPayload contains the request ID to cancel
type CancelPayload struct {
	RequestID string
}

// GenerationParams contains parameters for text generation
type GenerationParams struct {
	Temperature    float64
	TopK           int
	TopP           float64
	RepeatPenalty  float64
	MaxTokens      int
	MinP           float64
	RepeatLastN    int
	PenalizeNl     bool
	Seed           int
}

// DefaultGenerationParams returns sensible defaults for generation
func DefaultGenerationParams() GenerationParams {
	return GenerationParams{
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		MaxTokens:     512,
		MinP:          0.1,
		RepeatLastN:   64,
		PenalizeNl:    true,
		Seed:          0,
	}
}

// EngineMetrics contains performance and resource metrics
type EngineMetrics struct {
	// Performance metrics
	TotalRequests       int64         `json:"total_requests"`
	SuccessfulRequests  int64         `json:"successful_requests"`
	FailedRequests      int64         `json:"failed_requests"`
	AverageResponseTime time.Duration `json:"average_response_time"`

	// Resource metrics
	ModelLoaded     bool   `json:"model_loaded"`
	ModelPath       string `json:"model_path"`
	MemoryUsage     int64  `json:"memory_usage_bytes"`
	TokensGenerated int64  `json:"tokens_generated"`

	// Current state
	ActiveRequests int       `json:"active_requests"`
	QueuedRequests int       `json:"queued_requests"`
	LastActivity   time.Time `json:"last_activity"`
	Uptime         time.Duration `json:"uptime"`
	StartTime      time.Time `json:"start_time"`

	// Error tracking
	LastError  string         `json:"last_error,omitempty"`
	ErrorCount map[string]int `json:"error_count"`

	// Thread safety
	mu sync.RWMutex `json:"-"`
}

// NewEngineMetrics creates a new metrics instance
func NewEngineMetrics() *EngineMetrics {
	return &EngineMetrics{
		StartTime:  time.Now(),
		ErrorCount: make(map[string]int),
	}
}

// IncrementRequests safely increments the total request counter
func (m *EngineMetrics) IncrementRequests() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TotalRequests++
	m.LastActivity = time.Now()
}

// IncrementSuccessful safely increments the successful request counter
func (m *EngineMetrics) IncrementSuccessful() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.SuccessfulRequests++
}

// IncrementFailed safely increments the failed request counter and tracks error
func (m *EngineMetrics) IncrementFailed(errorType string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.FailedRequests++
	m.ErrorCount[errorType]++
	m.LastError = errorType
}

// SetActiveRequests safely sets the number of active requests
func (m *EngineMetrics) SetActiveRequests(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ActiveRequests = count
}

// SetQueuedRequests safely sets the number of queued requests
func (m *EngineMetrics) SetQueuedRequests(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.QueuedRequests = count
}

// UpdateResponseTime safely updates the average response time
func (m *EngineMetrics) UpdateResponseTime(duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Simple moving average - could be made more sophisticated
	if m.AverageResponseTime == 0 {
		m.AverageResponseTime = duration
	} else {
		m.AverageResponseTime = (m.AverageResponseTime + duration) / 2
	}
}

// SetModelInfo safely sets model-related information
func (m *EngineMetrics) SetModelInfo(loaded bool, path string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ModelLoaded = loaded
	m.ModelPath = path
}

// AddTokensGenerated safely adds to the tokens generated counter
func (m *EngineMetrics) AddTokensGenerated(count int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TokensGenerated += count
}

// SetMemoryUsage safely sets the current memory usage
func (m *EngineMetrics) SetMemoryUsage(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.MemoryUsage = bytes
}

// GetSnapshot returns a thread-safe snapshot of current metrics
func (m *EngineMetrics) GetSnapshot() EngineMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Calculate uptime
	uptime := time.Since(m.StartTime)

	// Create a copy with current uptime
	snapshot := *m
	snapshot.Uptime = uptime
	
	// Copy error count map
	snapshot.ErrorCount = make(map[string]int)
	for k, v := range m.ErrorCount {
		snapshot.ErrorCount[k] = v
	}

	return snapshot
}

// EngineConfig contains configuration for the LLM engine
type EngineConfig struct {
	RequestChannelSize   int           `json:"request_channel_size"`
	StreamingChannelSize int           `json:"streaming_channel_size"`
	OperationTimeout     time.Duration `json:"operation_timeout"`
	MaxConcurrentOps     int           `json:"max_concurrent_ops"`
	MetricsInterval      time.Duration `json:"metrics_interval"`
	ShutdownTimeout      time.Duration `json:"shutdown_timeout"`
}

// DefaultEngineConfig returns sensible defaults for engine configuration
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{
		RequestChannelSize:   100,
		StreamingChannelSize: 1000,
		OperationTimeout:     30 * time.Second,
		MaxConcurrentOps:     10,
		MetricsInterval:      5 * time.Second,
		ShutdownTimeout:      10 * time.Second,
	}
}

// ModelInfo contains information about the loaded model
type ModelInfo struct {
	Path         string            `json:"path"`
	ContextSize  int               `json:"context_size"`
	VocabSize    int               `json:"vocab_size,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	SystemPrompt string            `json:"system_prompt"`
	RAGEnabled   bool              `json:"rag_enabled"`
	RAGIndexPath string            `json:"rag_index_path,omitempty"`
}

// HealthStatus represents the health status of the engine
type HealthStatus struct {
	Status      string        `json:"status"` // "healthy", "degraded", "unhealthy"
	Timestamp   time.Time     `json:"timestamp"`
	Uptime      time.Duration `json:"uptime"`
	ModelLoaded bool          `json:"model_loaded"`
	LastError   string        `json:"last_error,omitempty"`
	Version     string        `json:"version,omitempty"`
}