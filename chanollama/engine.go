package chanollama

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/google/uuid"
)

// OperationType defines the type of operation to perform on the engine
type OperationType int

const (
	OpInference OperationType = iota // Send query and get response
	OpStream                         // Send query and get streaming response
	OpGetMetrics
	OpHealthCheck
	OpConfigureGeneration
	OpTokenCount // Count tokens in text
	OpShutdown
)

// String returns the string representation of the operation type
func (ot OperationType) String() string {
	switch ot {
	case OpInference:
		return "Inference"
	case OpStream:
		return "Stream"
	case OpGetMetrics:
		return "GetMetrics"
	case OpHealthCheck:
		return "HealthCheck"
	case OpConfigureGeneration:
		return "ConfigureGeneration"
	case OpTokenCount:
		return "TokenCount"
	case OpShutdown:
		return "Shutdown"
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

// InferencePayload contains parameters for inference operations
type InferencePayload struct {
	Messages    []Message
	MaxTokens   int
	Temperature float64
}

// StreamingPayload contains parameters for streaming operations
type StreamingPayload struct {
	Messages    []Message
	MaxTokens   int
	Temperature float64
	ChunkSize   int // Tokens per streaming chunk
}

// TokenCountPayload contains text to count tokens for
type TokenCountPayload struct {
	Text string
}

// GenerationParams contains parameters for text generation
type GenerationParams struct {
	Temperature   float64
	TopK          int
	TopP          float64
	RepeatPenalty float64
	MaxTokens     int
	Seed          int
}

// DefaultGenerationParams returns sensible defaults for generation
func DefaultGenerationParams() GenerationParams {
	return GenerationParams{
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		MaxTokens:     512,
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
	ModelName       string `json:"model_name"`
	TokensGenerated int64  `json:"tokens_generated"`

	// Current state
	ActiveRequests int           `json:"active_requests"`
	QueuedRequests int           `json:"queued_requests"`
	LastActivity   time.Time     `json:"last_activity"`
	Uptime         time.Duration `json:"uptime"`
	StartTime      time.Time     `json:"start_time"`

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
func (m *EngineMetrics) SetModelInfo(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ModelName = name
}

// AddTokensGenerated safely adds to the tokens generated counter
func (m *EngineMetrics) AddTokensGenerated(count int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TokensGenerated += count
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

// HealthStatus represents the health status of the engine
type HealthStatus struct {
	Status    string        `json:"status"` // "healthy", "degraded", "unhealthy"
	Timestamp time.Time     `json:"timestamp"`
	Uptime    time.Duration `json:"uptime"`
	LastError string        `json:"last_error,omitempty"`
	Version   string        `json:"version,omitempty"`
}

// AdvancedEngine provides a message-passing based engine for complex LLM operations
type AdvancedEngine struct {
	// Communication channels
	requests  chan EngineRequest
	streaming chan StreamingResponse
	shutdown  chan struct{}

	// Core client
	client *Client

	// Generation parameters
	genParams GenerationParams

	// Metrics and monitoring
	metrics *EngineMetrics

	// Cancellation tracking
	activeOps map[string]context.CancelFunc
	opsLock   sync.RWMutex

	// Configuration
	config *EngineConfig

	// Engine state
	running   bool
	stateLock sync.RWMutex
}

// NewAdvancedEngine creates a new advanced LLM engine instance
func NewAdvancedEngine(config Config) *AdvancedEngine {
	engineConfig := DefaultEngineConfig()

	return &AdvancedEngine{
		requests:  make(chan EngineRequest, engineConfig.RequestChannelSize),
		streaming: make(chan StreamingResponse, engineConfig.StreamingChannelSize),
		shutdown:  make(chan struct{}),
		client:    NewWithConfig(config),
		metrics:   NewEngineMetrics(),
		activeOps: make(map[string]context.CancelFunc),
		config:    &engineConfig,
		genParams: DefaultGenerationParams(),
	}
}

// NewAdvancedEngineWithConfig creates a new advanced engine with custom configuration
func NewAdvancedEngineWithConfig(config Config, engineConfig EngineConfig) *AdvancedEngine {
	return &AdvancedEngine{
		requests:  make(chan EngineRequest, engineConfig.RequestChannelSize),
		streaming: make(chan StreamingResponse, engineConfig.StreamingChannelSize),
		shutdown:  make(chan struct{}),
		client:    NewWithConfig(config),
		metrics:   NewEngineMetrics(),
		activeOps: make(map[string]context.CancelFunc),
		config:    &engineConfig,
		genParams: DefaultGenerationParams(),
	}
}

// Start initializes the engine and begins processing requests
func (e *AdvancedEngine) Start(ctx context.Context) error {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	if e.running {
		return fmt.Errorf("engine already running")
	}

	log.Printf("Starting Advanced LLM Engine...")

	e.running = true

	// Start the main engine loop
	go e.run(ctx)

	// Start metrics collection
	go e.collectMetrics(ctx)

	// Set model info
	e.metrics.SetModelInfo(e.client.config.Model)

	log.Printf("Advanced LLM Engine started successfully")
	return nil
}

// Stop gracefully shuts down the engine
func (e *AdvancedEngine) Stop() error {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	if !e.running {
		return nil
	}

	log.Printf("Stopping Advanced LLM Engine...")

	// Signal shutdown
	close(e.shutdown)

	// Cancel all active operations
	e.opsLock.Lock()
	for id, cancel := range e.activeOps {
		log.Printf("Cancelling active operation: %s", id)
		cancel()
	}
	e.activeOps = make(map[string]context.CancelFunc)
	e.opsLock.Unlock()

	e.running = false
	log.Printf("Advanced LLM Engine stopped")
	return nil
}

// SubmitRequest submits a request to the engine
func (e *AdvancedEngine) SubmitRequest(req EngineRequest) error {
	e.stateLock.RLock()
	defer e.stateLock.RUnlock()

	if !e.running {
		return fmt.Errorf("engine not running")
	}

	select {
	case e.requests <- req:
		e.metrics.IncrementRequests()
		e.metrics.SetQueuedRequests(len(e.requests))
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("request queue full, engine may be overloaded")
	}
}

// GetStreamingChannel returns the streaming response channel
func (e *AdvancedEngine) GetStreamingChannel() <-chan StreamingResponse {
	return e.streaming
}

// run is the main engine loop
func (e *AdvancedEngine) run(ctx context.Context) {
	log.Printf("Advanced Engine main loop started")

	for {
		select {
		case req := <-e.requests:
			e.metrics.SetQueuedRequests(len(e.requests))
			go e.handleRequest(req)

		case <-ctx.Done():
			log.Printf("Advanced Engine context cancelled")
			return

		case <-e.shutdown:
			log.Printf("Advanced Engine shutdown signal received")
			return
		}
	}
}

// handleRequest processes a single request
func (e *AdvancedEngine) handleRequest(req EngineRequest) {
	startTime := time.Now()

	// Track operation for cancellation
	e.trackOperation(req.ID, req.Context)
	defer e.untrackOperation(req.ID)

	log.Printf("Handling request %s: %s", req.ID, req.Type.String())

	var response EngineResponse

	// Set current active requests count
	e.opsLock.RLock()
	activeCount := len(e.activeOps)
	e.opsLock.RUnlock()
	e.metrics.SetActiveRequests(activeCount)

	// Handle the request based on type
	switch req.Type {
	case OpInference:
		response = e.handleInference(req)
	case OpStream:
		response = e.handleStreaming(req)
	case OpGetMetrics:
		response = e.handleGetMetrics(req)
	case OpHealthCheck:
		response = e.handleHealthCheck(req)
	case OpConfigureGeneration:
		response = e.handleConfigureGeneration(req)
	case OpTokenCount:
		response = e.handleTokenCount(req)
	case OpShutdown:
		response = e.handleShutdown(req)
	default:
		response = EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("unknown operation type: %s", req.Type.String()),
		}
	}

	// Update metrics
	duration := time.Since(startTime)
	e.metrics.UpdateResponseTime(duration)

	if response.Success {
		e.metrics.IncrementSuccessful()
	} else {
		errorType := "unknown"
		if response.Error != nil {
			errorType = response.Error.Error()
		}
		e.metrics.IncrementFailed(errorType)
	}

	// Send response back (non-blocking)
	select {
	case req.Response <- response:
		log.Printf("Response sent for request %s (success: %v)", req.ID, response.Success)
	case <-req.Context.Done():
		log.Printf("Request %s was cancelled before response could be sent", req.ID)
	case <-time.After(5 * time.Second):
		log.Printf("Warning: Response channel blocked for request %s", req.ID)
	}
}

// Operation handlers

func (e *AdvancedEngine) handleInference(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(InferencePayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for Inference"),
		}
	}

	log.Printf("Processing inference request: %s", req.ID)

	// Check for cancellation
	select {
	case <-req.Context.Done():
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     req.Context.Err(),
		}
	default:
	}

	// Perform inference using the HTTP client
	response, err := e.client.Chat(payload.Messages)
	if err != nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     err,
		}
	}

	// Update token count
	tokenCount := int64(response.Tokens)
	if tokenCount == 0 {
		// Estimate if not provided
		tokenCount = int64(len(response.Content) / 4)
	}
	e.metrics.AddTokensGenerated(tokenCount)

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      response.Content,
		Metadata: map[string]interface{}{
			"tokens_generated": tokenCount,
			"model":           e.client.config.Model,
		},
	}
}

func (e *AdvancedEngine) handleStreaming(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(StreamingPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for Stream"),
		}
	}

	log.Printf("Processing streaming request: %s", req.ID)

	// Start streaming inference in goroutine
	go e.performStreamingInference(req.Context, req.ID, payload.Messages)

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Streaming started",
		Metadata: map[string]interface{}{
			"model": e.client.config.Model,
		},
	}
}

func (e *AdvancedEngine) handleGetMetrics(req EngineRequest) EngineResponse {
	snapshot := e.metrics.GetSnapshot()

	// Add current memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      snapshot,
	}
}

func (e *AdvancedEngine) handleHealthCheck(req EngineRequest) EngineResponse {
	status := "healthy"

	if !e.client.IsHealthy() {
		status = "degraded"
	}

	if e.metrics.FailedRequests > e.metrics.SuccessfulRequests {
		status = "unhealthy"
	}

	health := HealthStatus{
		Status:    status,
		Timestamp: time.Now(),
		Uptime:    time.Since(e.metrics.StartTime),
		LastError: e.metrics.LastError,
		Version:   "1.0.0",
	}

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      health,
	}
}

func (e *AdvancedEngine) handleConfigureGeneration(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(GenerationParams)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for ConfigureGeneration"),
		}
	}

	e.genParams = payload

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Generation parameters updated",
	}
}

func (e *AdvancedEngine) handleTokenCount(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(TokenCountPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for TokenCount"),
		}
	}

	// Estimate token count using our utility
	estimator := NewTokenEstimator()
	tokenCount := estimator.EstimateTokens(payload.Text)

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      tokenCount,
	}
}

func (e *AdvancedEngine) handleShutdown(req EngineRequest) EngineResponse {
	log.Printf("Shutdown requested")

	// Signal shutdown
	go func() {
		time.Sleep(100 * time.Millisecond) // Give time for response to be sent
		close(e.shutdown)
	}()

	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Shutdown initiated",
	}
}

// Helper methods

func (e *AdvancedEngine) trackOperation(id string, ctx context.Context) {
	e.opsLock.Lock()
	defer e.opsLock.Unlock()

	opCtx, cancel := context.WithCancel(ctx)
	e.activeOps[id] = cancel

	// Monitor for cancellation
	go func() {
		<-opCtx.Done()
		e.untrackOperation(id)
	}()
}

func (e *AdvancedEngine) untrackOperation(id string) {
	e.opsLock.Lock()
	defer e.opsLock.Unlock()

	if cancel, exists := e.activeOps[id]; exists {
		cancel()
		delete(e.activeOps, id)
	}
}

func (e *AdvancedEngine) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(e.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Metrics are automatically updated during requests

		case <-ctx.Done():
			return
		case <-e.shutdown:
			return
		}
	}
}

// performStreamingInference performs streaming inference using the HTTP client
func (e *AdvancedEngine) performStreamingInference(ctx context.Context, requestID string, messages []Message) {
	defer func() {
		// Send final streaming response
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Done:      true,
		}
	}()

	var tokenCount int64

	// Use the client's streaming functionality
	response := e.client.ChatStream(messages, func(token string) {
		tokenCount++

		// Stream the token
		select {
		case e.streaming <- StreamingResponse{
			RequestID: requestID,
			Token:     token,
			Done:      false,
		}:
		case <-ctx.Done():
			return
		case <-time.After(1 * time.Second):
			log.Printf("Warning: Streaming channel blocked for request %s", requestID)
		}
	})

	// Handle any final error
	if response.Error != nil {
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Error:     response.Error,
			Done:      true,
		}
		return
	}

	// Update metrics
	e.metrics.AddTokensGenerated(tokenCount)
}

// generateRequestID generates a unique request ID
func generateRequestID() string {
	return uuid.New().String()
}

// Chat provides a simple synchronous chat interface
func (e *AdvancedEngine) Chat(messages []Message) (*Response, error) {
	requestID := generateRequestID()
	responseChan := make(chan EngineResponse, 1)

	request := EngineRequest{
		ID:      requestID,
		Type:    OpInference,
		Payload: InferencePayload{Messages: messages},
		Response: responseChan,
		Context: context.Background(),
	}

	err := e.SubmitRequest(request)
	if err != nil {
		return nil, err
	}

	// Wait for response
	select {
	case response := <-responseChan:
		if !response.Success {
			return nil, response.Error
		}

		content, ok := response.Data.(string)
		if !ok {
			return nil, fmt.Errorf("unexpected response type")
		}

		tokens := 0
		if tokenCount, exists := response.Metadata["tokens_generated"]; exists {
			if tc, ok := tokenCount.(int64); ok {
				tokens = int(tc)
			}
		}

		return &Response{
			Content: content,
			Tokens:  tokens,
		}, nil

	case <-time.After(e.config.OperationTimeout):
		return nil, fmt.Errorf("operation timeout")
	}
}

// ChatStream provides a simple streaming chat interface  
func (e *AdvancedEngine) ChatStream(messages []Message, callback StreamCallback) *Response {
	requestID := generateRequestID()
	responseChan := make(chan EngineResponse, 1)

	request := EngineRequest{
		ID:      requestID,
		Type:    OpStream,
		Payload: StreamingPayload{Messages: messages},
		Response: responseChan,
		Context: context.Background(),
	}

	err := e.SubmitRequest(request)
	if err != nil {
		return &Response{Error: err}
	}

	// Wait for streaming to start
	select {
	case response := <-responseChan:
		if !response.Success {
			return &Response{Error: response.Error}
		}
	case <-time.After(5 * time.Second):
		return &Response{Error: fmt.Errorf("streaming startup timeout")}
	}

	// Process streaming tokens
	go func() {
		for streamResponse := range e.streaming {
			if streamResponse.RequestID == requestID {
				if streamResponse.Error != nil {
					return
				}
				if streamResponse.Done {
					return
				}
				if callback != nil {
					callback(streamResponse.Token)
				}
			}
		}
	}()

	return &Response{Content: "Streaming started"}
}

// IsHealthy checks if the engine is healthy
func (e *AdvancedEngine) IsHealthy() bool {
	return e.client.IsHealthy()
}

// Close shuts down the engine
func (e *AdvancedEngine) Close() error {
	return e.Stop()
}