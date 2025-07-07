package llm

import (
	"context"
	"fmt"
	"time"
)

// EngineClient provides a high-level interface to the LLM engine
type EngineClient struct {
	engine    *LLMEngineV2
	responses chan EngineResponse
	streaming chan StreamingResponse
	timeout   time.Duration
}

// NewEngineClient creates a new engine client
func NewEngineClient(engine *LLMEngineV2) *EngineClient {
	return &EngineClient{
		engine:    engine,
		responses: make(chan EngineResponse, 10),
		streaming: make(chan StreamingResponse, 100),
		timeout:   30 * time.Second,
	}
}

// NewEngineClientWithTimeout creates a new engine client with custom timeout
func NewEngineClientWithTimeout(engine *LLMEngineV2, timeout time.Duration) *EngineClient {
	return &EngineClient{
		engine:    engine,
		responses: make(chan EngineResponse, 10),
		streaming: make(chan StreamingResponse, 100),
		timeout:   timeout,
	}
}

// LoadModel loads a model with the specified configuration
func (c *EngineClient) LoadModel(ctx context.Context, payload LoadModelPayload) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpLoadModel,
		Payload:  payload,
		Response: c.responses,
		Context:  ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// SendQuery sends a query and returns the complete response
func (c *EngineClient) SendQuery(ctx context.Context, query string) (string, error) {
	return c.SendQueryWithOptions(ctx, InferencePayload{
		Query:       query,
		MaxTokens:   0,    // Use default
		Temperature: 0,    // Use default
		IncludeRAG:  true, // Enable RAG by default
	})
}

// SendQueryWithOptions sends a query with specific options
func (c *EngineClient) SendQueryWithOptions(ctx context.Context, payload InferencePayload) (string, error) {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpInference,
		Payload:  payload,
		Response: c.responses,
		Context:  ctx,
	}
	
	resp, err := c.sendRequestAndWaitForResponse(req)
	if err != nil {
		return "", err
	}
	
	if result, ok := resp.Data.(string); ok {
		return result, nil
	}
	return "", fmt.Errorf("unexpected response type: %T", resp.Data)
}

// StreamQuery sends a query and streams the response through a callback
func (c *EngineClient) StreamQuery(ctx context.Context, query string, callback func(string)) error {
	return c.StreamQueryWithOptions(ctx, StreamingPayload{
		Query:       query,
		MaxTokens:   0,    // Use default
		Temperature: 0,    // Use default
		IncludeRAG:  true, // Enable RAG by default
		ChunkSize:   1,    // Token by token
	}, callback)
}

// StreamQueryWithOptions sends a streaming query with specific options
func (c *EngineClient) StreamQueryWithOptions(ctx context.Context, payload StreamingPayload, callback func(string)) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpStream,
		Payload:  payload,
		Response: c.responses,
		Context:  ctx,
	}
	
	// Send request
	if err := c.sendRequest(req); err != nil {
		return err
	}
	
	// Wait for initial response
	select {
	case resp := <-c.responses:
		if !resp.Success {
			return resp.Error
		}
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(c.timeout):
		return fmt.Errorf("timeout waiting for streaming to start")
	}
	
	// Handle streaming responses
	return c.handleStreamingResponse(ctx, req.ID, callback)
}

// GetMetrics retrieves current engine metrics
func (c *EngineClient) GetMetrics(ctx context.Context) (*EngineMetrics, error) {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpGetMetrics,
		Payload:  nil,
		Response: c.responses,
		Context:  ctx,
	}
	
	resp, err := c.sendRequestAndWaitForResponse(req)
	if err != nil {
		return nil, err
	}
	
	if metrics, ok := resp.Data.(EngineMetrics); ok {
		return &metrics, nil
	}
	return nil, fmt.Errorf("unexpected response type: %T", resp.Data)
}

// UnloadModel unloads the currently loaded model
func (c *EngineClient) UnloadModel(ctx context.Context) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpUnloadModel,
		Payload:  nil,
		Response: c.responses,
		Context:  ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// Shutdown gracefully shuts down the engine
func (c *EngineClient) Shutdown(ctx context.Context) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpShutdown,
		Payload:  nil,
		Response: c.responses,
		Context:  ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// ConfigurePrompt updates the system prompt
func (c *EngineClient) ConfigurePrompt(ctx context.Context, systemPrompt string) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpConfigurePrompt,
		Payload:  ConfigurePromptPayload{SystemPrompt: systemPrompt},
		Response: c.responses,
		Context:  ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// ConfigureRAG configures RAG settings
func (c *EngineClient) ConfigureRAG(ctx context.Context, enabled bool, indexPath string) error {
	req := EngineRequest{
		ID:      generateRequestID(),
		Type:    OpConfigureRAG,
		Payload: ConfigureRAGPayload{Enabled: enabled, IndexPath: indexPath},
		Response: c.responses,
		Context: ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// InspectModel returns information about the loaded model
func (c *EngineClient) InspectModel(ctx context.Context) (*ModelInfo, error) {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpInspectModel,
		Payload:  nil,
		Response: c.responses,
		Context:  ctx,
	}
	
	resp, err := c.sendRequestAndWaitForResponse(req)
	if err != nil {
		return nil, err
	}
	
	if info, ok := resp.Data.(ModelInfo); ok {
		return &info, nil
	}
	return nil, fmt.Errorf("unexpected response type: %T", resp.Data)
}

// HealthCheck checks the health status of the engine
func (c *EngineClient) HealthCheck(ctx context.Context) (*HealthStatus, error) {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpHealthCheck,
		Payload:  nil,
		Response: c.responses,
		Context:  ctx,
	}
	
	resp, err := c.sendRequestAndWaitForResponse(req)
	if err != nil {
		return nil, err
	}
	
	if health, ok := resp.Data.(HealthStatus); ok {
		return &health, nil
	}
	return nil, fmt.Errorf("unexpected response type: %T", resp.Data)
}

// ConfigureGeneration updates generation parameters
func (c *EngineClient) ConfigureGeneration(ctx context.Context, params GenerationParams) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpConfigureGeneration,
		Payload:  params,
		Response: c.responses,
		Context:  ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// CountTokens counts tokens in the given text
func (c *EngineClient) CountTokens(ctx context.Context, text string) (int, error) {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpTokenCount,
		Payload:  TokenCountPayload{Text: text},
		Response: c.responses,
		Context:  ctx,
	}
	
	resp, err := c.sendRequestAndWaitForResponse(req)
	if err != nil {
		return 0, err
	}
	
	if count, ok := resp.Data.(int); ok {
		return count, nil
	}
	return 0, fmt.Errorf("unexpected response type: %T", resp.Data)
}

// WarmupModel performs a small inference to warm up the model
func (c *EngineClient) WarmupModel(ctx context.Context) error {
	req := EngineRequest{
		ID:       generateRequestID(),
		Type:     OpWarmup,
		Payload:  nil,
		Response: c.responses,
		Context:  ctx,
	}
	
	return c.sendRequestAndWait(req)
}

// Helper methods

// sendRequest sends a request to the engine (non-blocking)
func (c *EngineClient) sendRequest(req EngineRequest) error {
	return c.engine.SubmitRequest(req)
}

// sendRequestAndWait sends a request and waits for any response
func (c *EngineClient) sendRequestAndWait(req EngineRequest) error {
	if err := c.sendRequest(req); err != nil {
		return err
	}
	
	select {
	case resp := <-c.responses:
		if !resp.Success {
			return resp.Error
		}
		return nil
	case <-req.Context.Done():
		return req.Context.Err()
	case <-time.After(c.timeout):
		return fmt.Errorf("timeout waiting for response")
	}
}

// sendRequestAndWaitForResponse sends a request and waits for a specific response
func (c *EngineClient) sendRequestAndWaitForResponse(req EngineRequest) (EngineResponse, error) {
	if err := c.sendRequest(req); err != nil {
		return EngineResponse{}, err
	}
	
	select {
	case resp := <-c.responses:
		return resp, nil
	case <-req.Context.Done():
		return EngineResponse{}, req.Context.Err()
	case <-time.After(c.timeout):
		return EngineResponse{}, fmt.Errorf("timeout waiting for response")
	}
}

// handleStreamingResponse handles streaming responses from the engine
func (c *EngineClient) handleStreamingResponse(ctx context.Context, requestID string, callback func(string)) error {
	streamingChan := c.engine.GetStreamingChannel()
	
	for {
		select {
		case stream := <-streamingChan:
			// Check if this streaming response is for our request
			if stream.RequestID != requestID {
				continue // Not our response, keep waiting
			}
			
			if stream.Error != nil {
				return stream.Error
			}
			
			if stream.Done {
				return nil // Streaming complete
			}
			
			// Call the callback with the token
			if callback != nil {
				callback(stream.Token)
			}
			
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(c.timeout):
			return fmt.Errorf("timeout waiting for streaming response")
		}
	}
}

// SetTimeout updates the client timeout
func (c *EngineClient) SetTimeout(timeout time.Duration) {
	c.timeout = timeout
}

// GetTimeout returns the current timeout
func (c *EngineClient) GetTimeout() time.Duration {
	return c.timeout
}

// Close cleans up the client resources
func (c *EngineClient) Close() {
	close(c.responses)
	close(c.streaming)
}