package llm

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/ollama/ollama/llama"
	"github.com/opentdf/otdfctl/pkg/config"
)

// LLMEngineV2 is the new decoupled engine that handles all LLM operations via message passing
type LLMEngineV2 struct {
	// Communication channels
	requests  chan EngineRequest
	streaming chan StreamingResponse
	shutdown  chan struct{}

	// Engine state
	model        *llama.Model
	context      *llama.Context
	systemPrompt string
	ragStore     *SimpleRAGStore
	ragEnabled   bool

	// Generation parameters
	genParams GenerationParams

	// Metrics and monitoring
	metrics *EngineMetrics

	// Cancellation tracking
	activeOps map[string]context.CancelFunc
	opsLock   sync.RWMutex

	// Configuration
	config *EngineConfig

	// Resource management
	resourceManager *ResourceManager

	// Engine state
	running   bool
	stateLock sync.RWMutex
}

// NewLLMEngineV2 creates a new LLM engine instance
func NewLLMEngineV2(cfg *config.Config) *LLMEngineV2 {
	engineConfig := DefaultEngineConfig()
	
	return &LLMEngineV2{
		requests:        make(chan EngineRequest, engineConfig.RequestChannelSize),
		streaming:       make(chan StreamingResponse, engineConfig.StreamingChannelSize),
		shutdown:        make(chan struct{}),
		metrics:         NewEngineMetrics(),
		activeOps:       make(map[string]context.CancelFunc),
		config:          &engineConfig,
		genParams:       DefaultGenerationParams(),
		resourceManager: NewResourceManagerFromConfig(cfg),
	}
}

// NewLLMEngineV2WithConfig creates a new LLM engine with custom configuration
func NewLLMEngineV2WithConfig(engineConfig EngineConfig, cfg *config.Config) *LLMEngineV2 {
	return &LLMEngineV2{
		requests:        make(chan EngineRequest, engineConfig.RequestChannelSize),
		streaming:       make(chan StreamingResponse, engineConfig.StreamingChannelSize),
		shutdown:        make(chan struct{}),
		metrics:         NewEngineMetrics(),
		activeOps:       make(map[string]context.CancelFunc),
		config:          &engineConfig,
		genParams:       DefaultGenerationParams(),
		resourceManager: NewResourceManagerFromConfig(cfg),
	}
}

// Start initializes the engine and begins processing requests
func (e *LLMEngineV2) Start(ctx context.Context) error {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	if e.running {
		return fmt.Errorf("engine already running")
	}

	log.Printf("Starting LLM Engine V2...")
	
	// Initialize llama backend
	llama.BackendInit()
	
	e.running = true
	
	// Start the main engine loop
	go e.run(ctx)
	
	// Start metrics collection
	go e.collectMetrics(ctx)
	
	log.Printf("LLM Engine V2 started successfully")
	return nil
}

// Stop gracefully shuts down the engine
func (e *LLMEngineV2) Stop() error {
	e.stateLock.Lock()
	defer e.stateLock.Unlock()

	if !e.running {
		return nil
	}

	log.Printf("Stopping LLM Engine V2...")
	
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
	
	// Wait for shutdown with timeout
	shutdownTimer := time.NewTimer(e.config.ShutdownTimeout)
	defer shutdownTimer.Stop()
	
	// Clean up resources
	e.cleanup()
	
	e.running = false
	log.Printf("LLM Engine V2 stopped")
	return nil
}

// SubmitRequest submits a request to the engine
func (e *LLMEngineV2) SubmitRequest(req EngineRequest) error {
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
func (e *LLMEngineV2) GetStreamingChannel() <-chan StreamingResponse {
	return e.streaming
}

// run is the main engine loop
func (e *LLMEngineV2) run(ctx context.Context) {
	defer e.cleanup()
	
	log.Printf("Engine V2 main loop started")
	
	for {
		select {
		case req := <-e.requests:
			e.metrics.SetQueuedRequests(len(e.requests))
			go e.handleRequest(req)
			
		case <-ctx.Done():
			log.Printf("Engine V2 context cancelled")
			return
			
		case <-e.shutdown:
			log.Printf("Engine V2 shutdown signal received")
			return
		}
	}
}

// handleRequest processes a single request
func (e *LLMEngineV2) handleRequest(req EngineRequest) {
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
	case OpLoadModel:
		response = e.handleLoadModel(req)
	case OpInference:
		response = e.handleInference(req)
	case OpStream:
		response = e.handleStreaming(req)
	case OpGetMetrics:
		response = e.handleGetMetrics(req)
	case OpUnloadModel:
		response = e.handleUnloadModel(req)
	case OpShutdown:
		response = e.handleShutdown(req)
	case OpConfigurePrompt:
		response = e.handleConfigurePrompt(req)
	case OpConfigureRAG:
		response = e.handleConfigureRAG(req)
	case OpInspectModel:
		response = e.handleInspectModel(req)
	case OpHealthCheck:
		response = e.handleHealthCheck(req)
	case OpConfigureGeneration:
		response = e.handleConfigureGeneration(req)
	case OpTokenCount:
		response = e.handleTokenCount(req)
	case OpWarmup:
		response = e.handleWarmup(req)
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

func (e *LLMEngineV2) handleLoadModel(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(LoadModelPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for LoadModel"),
		}
	}
	
	log.Printf("Loading model: %s", payload.ModelPath)
	
	// Unload existing model if loaded
	e.unloadModelInternal()
	
	// Set up model parameters
	modelParams := llama.ModelParams{
		NumGpuLayers: -1, // Use all available GPU layers
		UseMmap:      true,
		VocabOnly:    false,
	}
	
	// Load model
	model, err := llama.LoadModelFromFile(payload.ModelPath, modelParams)
	if err != nil {
		log.Printf("Model loading failed: %v", err)
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("failed to load model: %v", err),
		}
	}
	
	// Create context
	contextSize := payload.ContextSize
	if contextSize <= 0 {
		contextSize = 4096
	}
	
	contextParams := llama.NewContextParams(
		contextSize, // numCtx
		512,         // batchSize
		1,           // numSeqMax
		4,           // threads
		false,       // flashAttention
		"",          // kvCacheType
	)
	
	context, err := llama.NewContextWithModel(model, contextParams)
	if err != nil {
		llama.FreeModel(model)
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("failed to create context: %v", err),
		}
	}
	
	// Set engine state
	e.model = model
	e.context = context
	e.systemPrompt = payload.SystemPrompt
	e.genParams = payload.GenerationParams
	
	// Configure RAG if requested
	if payload.EnableRAG && payload.RAGIndexPath != "" {
		ragStore := NewSimpleRAGStore(payload.RAGIndexPath)
		if err := ragStore.LoadIndex(); err != nil {
			log.Printf("Warning: Failed to load RAG index: %v", err)
		} else {
			e.ragStore = ragStore
			e.ragEnabled = true
			log.Printf("RAG enabled with %d documents", ragStore.GetDocumentCount())
		}
	}
	
	// Update metrics
	e.metrics.SetModelInfo(true, payload.ModelPath)
	
	log.Printf("Model loaded successfully: %s", payload.ModelPath)
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Model loaded successfully",
		Metadata: map[string]interface{}{
			"model_path":    payload.ModelPath,
			"context_size":  contextSize,
			"rag_enabled":   e.ragEnabled,
		},
	}
}

func (e *LLMEngineV2) handleInference(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(InferencePayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for Inference"),
		}
	}
	
	if e.model == nil || e.context == nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("no model loaded"),
		}
	}
	
	log.Printf("Processing inference request: %s", req.ID)
	
	// Build messages
	messages := []ChatMessage{
		{Role: "system", Content: e.systemPrompt},
		{Role: "user", Content: payload.Query},
	}
	
	// Build prompt with optional RAG context
	prompt, err := e.buildPromptWithRAG(messages, payload.Query, payload.IncludeRAG)
	if err != nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("failed to build prompt: %v", err),
		}
	}
	
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
	
	// Perform inference
	response, err := e.performInference(req.Context, prompt, payload.MaxTokens)
	if err != nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     err,
		}
	}
	
	// Update token count
	tokenCount := int64(len(response) / 4) // Rough approximation
	e.metrics.AddTokensGenerated(tokenCount)
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      response,
		Metadata: map[string]interface{}{
			"tokens_generated": tokenCount,
			"rag_used":        payload.IncludeRAG && e.ragEnabled,
		},
	}
}

func (e *LLMEngineV2) handleStreaming(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(StreamingPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for Stream"),
		}
	}
	
	if e.model == nil || e.context == nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("no model loaded"),
		}
	}
	
	log.Printf("Processing streaming request: %s", req.ID)
	
	// Build messages
	messages := []ChatMessage{
		{Role: "system", Content: e.systemPrompt},
		{Role: "user", Content: payload.Query},
	}
	
	// Build prompt with optional RAG context
	prompt, err := e.buildPromptWithRAG(messages, payload.Query, payload.IncludeRAG)
	if err != nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("failed to build prompt: %v", err),
		}
	}
	
	// Start streaming inference in goroutine
	go e.performStreamingInference(req.Context, req.ID, prompt, payload.MaxTokens)
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Streaming started",
		Metadata: map[string]interface{}{
			"rag_used": payload.IncludeRAG && e.ragEnabled,
		},
	}
}

func (e *LLMEngineV2) handleGetMetrics(req EngineRequest) EngineResponse {
	snapshot := e.metrics.GetSnapshot()
	
	// Add current memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	snapshot.SetMemoryUsage(int64(m.Alloc))
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      snapshot,
	}
}

func (e *LLMEngineV2) handleUnloadModel(req EngineRequest) EngineResponse {
	log.Printf("Unloading model")
	
	e.unloadModelInternal()
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Model unloaded successfully",
	}
}

func (e *LLMEngineV2) handleShutdown(req EngineRequest) EngineResponse {
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

func (e *LLMEngineV2) handleConfigurePrompt(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(ConfigurePromptPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for ConfigurePrompt"),
		}
	}
	
	e.systemPrompt = payload.SystemPrompt
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "System prompt updated",
	}
}

func (e *LLMEngineV2) handleConfigureRAG(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(ConfigureRAGPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for ConfigureRAG"),
		}
	}
	
	if payload.Enabled && payload.IndexPath != "" {
		ragStore := NewSimpleRAGStore(payload.IndexPath)
		if err := ragStore.LoadIndex(); err != nil {
			return EngineResponse{
				RequestID: req.ID,
				Success:   false,
				Error:     fmt.Errorf("failed to load RAG index: %v", err),
			}
		}
		e.ragStore = ragStore
		e.ragEnabled = true
	} else {
		e.ragEnabled = false
		e.ragStore = nil
	}
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      fmt.Sprintf("RAG %s", map[bool]string{true: "enabled", false: "disabled"}[e.ragEnabled]),
	}
}

func (e *LLMEngineV2) handleInspectModel(req EngineRequest) EngineResponse {
	if e.model == nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("no model loaded"),
		}
	}
	
	info := ModelInfo{
		Path:         e.metrics.ModelPath,
		SystemPrompt: e.systemPrompt,
		RAGEnabled:   e.ragEnabled,
	}
	
	if e.ragStore != nil {
		info.RAGIndexPath = "configured" // Don't expose full path
	}
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      info,
	}
}

func (e *LLMEngineV2) handleHealthCheck(req EngineRequest) EngineResponse {
	status := "healthy"
	
	if e.model == nil {
		status = "degraded"
	}
	
	if e.metrics.FailedRequests > e.metrics.SuccessfulRequests {
		status = "unhealthy"
	}
	
	health := HealthStatus{
		Status:      status,
		Timestamp:   time.Now(),
		Uptime:      time.Since(e.metrics.StartTime),
		ModelLoaded: e.model != nil,
		LastError:   e.metrics.LastError,
		Version:     "2.0.0", // V2 engine
	}
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      health,
	}
}

func (e *LLMEngineV2) handleConfigureGeneration(req EngineRequest) EngineResponse {
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

func (e *LLMEngineV2) handleTokenCount(req EngineRequest) EngineResponse {
	payload, ok := req.Payload.(TokenCountPayload)
	if !ok {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("invalid payload type for TokenCount"),
		}
	}
	
	if e.model == nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("no model loaded"),
		}
	}
	
	// Tokenize the text
	tokens, err := e.model.Tokenize(payload.Text, true, true)
	if err != nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("tokenization failed: %v", err),
		}
	}
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      len(tokens),
	}
}

func (e *LLMEngineV2) handleWarmup(req EngineRequest) EngineResponse {
	if e.model == nil || e.context == nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("no model loaded"),
		}
	}
	
	// Perform a small inference to warm up the model
	warmupPrompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
	
	_, err := e.performInference(req.Context, warmupPrompt, 10)
	if err != nil {
		return EngineResponse{
			RequestID: req.ID,
			Success:   false,
			Error:     fmt.Errorf("warmup failed: %v", err),
		}
	}
	
	return EngineResponse{
		RequestID: req.ID,
		Success:   true,
		Data:      "Model warmed up successfully",
	}
}

// Helper methods

func (e *LLMEngineV2) trackOperation(id string, ctx context.Context) {
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

func (e *LLMEngineV2) untrackOperation(id string) {
	e.opsLock.Lock()
	defer e.opsLock.Unlock()
	
	if cancel, exists := e.activeOps[id]; exists {
		cancel()
		delete(e.activeOps, id)
	}
}

func (e *LLMEngineV2) unloadModelInternal() {
	if e.model != nil {
		llama.FreeModel(e.model)
		e.model = nil
	}
	e.context = nil
	e.metrics.SetModelInfo(false, "")
}

func (e *LLMEngineV2) cleanup() {
	log.Printf("Cleaning up engine V2 resources")
	
	e.unloadModelInternal()
	
	// Cancel all active operations
	e.opsLock.Lock()
	for id, cancel := range e.activeOps {
		log.Printf("Cancelling operation during cleanup: %s", id)
		cancel()
	}
	e.activeOps = make(map[string]context.CancelFunc)
	e.opsLock.Unlock()
}

func (e *LLMEngineV2) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(e.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Update memory usage
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			e.metrics.SetMemoryUsage(int64(m.Alloc))
			
		case <-ctx.Done():
			return
		case <-e.shutdown:
			return
		}
	}
}

// buildPromptWithRAG builds prompt with optional RAG context
func (e *LLMEngineV2) buildPromptWithRAG(messages []ChatMessage, userQuery string, includeRAG bool) (string, error) {
	var systemMessage string
	var conversationMessages []ChatMessage
	
	// Separate system message from conversation
	for _, msg := range messages {
		if msg.Role == "system" {
			systemMessage = msg.Content
		} else {
			conversationMessages = append(conversationMessages, msg)
		}
	}
	
	// Add RAG context if enabled
	if includeRAG && e.ragEnabled && userQuery != "" && e.ragStore != nil {
		results, err := e.ragStore.Search(userQuery, 2) // Top 2 results
		if err != nil {
			log.Printf("Warning: RAG search failed: %v", err)
		} else if len(results) > 0 {
			ragContext := BuildSimpleRAGContext(userQuery, results, 800)
			if ragContext.NumDocuments > 0 {
				enhancedSystem := fmt.Sprintf("%s\n\n%s\n\nBased on the above documentation, please provide accurate and helpful responses about OpenTDF.",
					systemMessage, ragContext.ContextText)
				systemMessage = enhancedSystem
				log.Printf("RAG: Retrieved %d relevant documents", ragContext.NumDocuments)
			}
		}
	}
	
	return e.buildPrompt(systemMessage, conversationMessages), nil
}

// buildPrompt creates the final prompt string
func (e *LLMEngineV2) buildPrompt(systemMessage string, messages []ChatMessage) string {
	var prompt strings.Builder
	
	// Add system message
	if systemMessage != "" {
		prompt.WriteString(fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n", systemMessage))
	}
	
	// Add conversation messages
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			prompt.WriteString(fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n", msg.Content))
		case "assistant":
			prompt.WriteString(fmt.Sprintf("<|im_start|>assistant\n%s<|im_end|>\n", msg.Content))
		}
	}
	
	// Add assistant prompt to start generation
	prompt.WriteString("<|im_start|>assistant\n")
	
	return prompt.String()
}

// performInference does actual model inference
func (e *LLMEngineV2) performInference(ctx context.Context, prompt string, maxTokens int) (string, error) {
	// Tokenize the prompt
	tokens, err := e.model.Tokenize(prompt, true, true)
	if err != nil {
		return "", fmt.Errorf("tokenization failed: %v", err)
	}
	
	// Limit batch size
	maxBatchSize := 512
	if len(tokens) > maxBatchSize {
		tokens = tokens[:maxBatchSize]
	}
	
	// Create batch
	batch, err := llama.NewBatch(len(tokens), 1, 0)
	if err != nil {
		return "", fmt.Errorf("batch creation failed: %v", err)
	}
	defer batch.Free()
	
	// Add tokens to batch
	for i, token := range tokens {
		batch.Add(token, nil, i, i == len(tokens)-1, 0)
	}
	
	// Process the batch
	err = e.context.Decode(batch)
	if err != nil {
		return "", fmt.Errorf("context decode failed: %v", err)
	}
	
	// Set up sampling
	samplingParams := llama.SamplingParams{
		TopK:           e.genParams.TopK,
		TopP:           float32(e.genParams.TopP),
		MinP:           float32(e.genParams.MinP),
		Temp:           float32(e.genParams.Temperature),
		RepeatLastN:    e.genParams.RepeatLastN,
		PenaltyRepeat:  float32(e.genParams.RepeatPenalty),
		PenalizeNl:     e.genParams.PenalizeNl,
		Seed:           uint32(e.genParams.Seed),
	}
	
	sampler, err := llama.NewSamplingContext(e.model, samplingParams)
	if err != nil {
		return "", fmt.Errorf("sampling context creation failed: %v", err)
	}
	
	var response strings.Builder
	if maxTokens <= 0 {
		maxTokens = e.genParams.MaxTokens
	}
	
	// Generate tokens
	for i := 0; i < maxTokens; i++ {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return response.String(), ctx.Err()
		default:
		}
		
		// Sample next token
		token := sampler.Sample(e.context, batch.NumTokens()-1)
		
		// Check for end of generation
		if e.model.TokenIsEog(token) {
			break
		}
		
		// Convert token to text
		piece := e.model.TokenToPiece(token)
		response.WriteString(piece)
		
		// Accept the token
		sampler.Accept(token, true)
		
		// Prepare for next iteration
		batch.Clear()
		batch.Add(token, nil, len(tokens)+i, true, 0)
		
		err = e.context.Decode(batch)
		if err != nil {
			log.Printf("Decode failed during generation: %v", err)
			break
		}
	}
	
	return strings.TrimSpace(response.String()), nil
}

// performStreamingInference does streaming inference
func (e *LLMEngineV2) performStreamingInference(ctx context.Context, requestID string, prompt string, maxTokens int) {
	defer func() {
		// Send final streaming response
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Done:      true,
		}
	}()
	
	// Similar to performInference but with streaming
	tokens, err := e.model.Tokenize(prompt, true, true)
	if err != nil {
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Error:     fmt.Errorf("tokenization failed: %v", err),
			Done:      true,
		}
		return
	}
	
	maxBatchSize := 512
	if len(tokens) > maxBatchSize {
		tokens = tokens[:maxBatchSize]
	}
	
	batch, err := llama.NewBatch(len(tokens), 1, 0)
	if err != nil {
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Error:     fmt.Errorf("batch creation failed: %v", err),
			Done:      true,
		}
		return
	}
	defer batch.Free()
	
	for i, token := range tokens {
		batch.Add(token, nil, i, i == len(tokens)-1, 0)
	}
	
	err = e.context.Decode(batch)
	if err != nil {
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Error:     fmt.Errorf("context decode failed: %v", err),
			Done:      true,
		}
		return
	}
	
	samplingParams := llama.SamplingParams{
		TopK:           e.genParams.TopK,
		TopP:           float32(e.genParams.TopP),
		MinP:           float32(e.genParams.MinP),
		Temp:           float32(e.genParams.Temperature),
		RepeatLastN:    e.genParams.RepeatLastN,
		PenaltyRepeat:  float32(e.genParams.RepeatPenalty),
		PenalizeNl:     e.genParams.PenalizeNl,
		Seed:           uint32(e.genParams.Seed),
	}
	
	sampler, err := llama.NewSamplingContext(e.model, samplingParams)
	if err != nil {
		e.streaming <- StreamingResponse{
			RequestID: requestID,
			Error:     fmt.Errorf("sampling context creation failed: %v", err),
			Done:      true,
		}
		return
	}
	
	if maxTokens <= 0 {
		maxTokens = e.genParams.MaxTokens
	}
	
	var tokenCount int64
	
	for i := 0; i < maxTokens; i++ {
		// Check for cancellation
		select {
		case <-ctx.Done():
			e.streaming <- StreamingResponse{
				RequestID: requestID,
				Error:     ctx.Err(),
				Done:      true,
			}
			return
		default:
		}
		
		token := sampler.Sample(e.context, batch.NumTokens()-1)
		
		if e.model.TokenIsEog(token) {
			break
		}
		
		piece := e.model.TokenToPiece(token)
		tokenCount++
		
		// Stream the token
		select {
		case e.streaming <- StreamingResponse{
			RequestID: requestID,
			Token:     piece,
			Done:      false,
		}:
		case <-ctx.Done():
			return
		case <-time.After(1 * time.Second):
			log.Printf("Warning: Streaming channel blocked for request %s", requestID)
		}
		
		sampler.Accept(token, true)
		
		batch.Clear()
		batch.Add(token, nil, len(tokens)+i, true, 0)
		
		err = e.context.Decode(batch)
		if err != nil {
			log.Printf("Decode failed during streaming generation: %v", err)
			break
		}
	}
	
	// Update metrics
	e.metrics.AddTokensGenerated(tokenCount)
}

// generateRequestID generates a unique request ID
func generateRequestID() string {
	return uuid.New().String()
}