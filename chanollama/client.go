package chanollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	defaultBaseURL = "http://localhost:11434"
	chatEndpoint   = "/api/chat"
)

// Client implements the Engine interface for Ollama HTTP API
type Client struct {
	config     Config
	httpClient *http.Client
	baseURL    string
}

// New creates a new chanollama client with default configuration
func New(model string) *Client {
	return NewWithConfig(DefaultConfig(model))
}

// NewWithConfig creates a new chanollama client with custom configuration
func NewWithConfig(config Config) *Client {
	return &Client{
		config:  config,
		baseURL: defaultBaseURL,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Chat performs a non-streaming chat request
func (c *Client) Chat(messages []Message) (*Response, error) {
	req := c.buildRequest(messages, false)
	
	ctx, cancel := context.WithTimeout(context.Background(), c.config.Timeout)
	defer cancel()
	
	resp, err := c.makeRequest(ctx, req)
	if err != nil {
		return &Response{Error: err}, err
	}
	
	var ollamaResp ollamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		resp.Body.Close()
		return &Response{Error: err}, err
	}
	resp.Body.Close()
	
	return &Response{
		Content: ollamaResp.Message.Content,
		Tokens:  c.estimateTokens(ollamaResp.Message.Content),
	}, nil
}

// ChatStream performs a streaming chat request
func (c *Client) ChatStream(messages []Message, callback StreamCallback) *Response {
	req := c.buildRequest(messages, true)
	
	ctx, cancel := context.WithTimeout(context.Background(), c.config.Timeout)
	defer cancel()
	
	resp, err := c.makeRequest(ctx, req)
	if err != nil {
		return &Response{Error: err}
	}
	defer resp.Body.Close()
	
	var fullContent strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		var streamResp ollamaStreamResponse
		if err := json.Unmarshal([]byte(line), &streamResp); err != nil {
			continue // Skip malformed lines
		}
		
		content := streamResp.Message.Content
		if content != "" {
			fullContent.WriteString(content)
			if callback != nil {
				callback(content)
			}
		}
		
		if streamResp.Done {
			break
		}
	}
	
	if err := scanner.Err(); err != nil {
		return &Response{Error: err}
	}
	
	finalContent := fullContent.String()
	return &Response{
		Content: finalContent,
		Tokens:  c.estimateTokens(finalContent),
	}
}

// IsHealthy checks if the Ollama service is available
func (c *Client) IsHealthy() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL, nil)
	if err != nil {
		return false
	}
	
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK
}

// Close cleans up resources (nothing to do for HTTP client)
func (c *Client) Close() error {
	return nil
}

// buildRequest creates an Ollama API request
func (c *Client) buildRequest(messages []Message, stream bool) ollamaRequest {
	options := map[string]interface{}{
		"temperature": c.config.Temperature,
	}
	
	if c.config.MaxTokens > 0 {
		options["num_predict"] = c.config.MaxTokens
	}
	
	return ollamaRequest{
		Model:    c.config.Model,
		Messages: messages,
		Stream:   stream,
		Options:  options,
	}
}

// makeRequest executes an HTTP request to Ollama
func (c *Client) makeRequest(ctx context.Context, req ollamaRequest) (*http.Response, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+chatEndpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("ollama API error (status %d): %s", resp.StatusCode, string(body))
	}
	
	return resp, nil
}

// estimateTokens provides a rough token count estimate
func (c *Client) estimateTokens(content string) int {
	// Rough estimation: 1 token ≈ 4 characters
	return len(content) / 4
}