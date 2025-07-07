package chanollama

import (
	"context"
	"fmt"
	"time"
)

// StreamingEngine wraps a Client to provide channel-based streaming
type StreamingEngine struct {
	client *Client
}

// NewStreamingEngine creates a new streaming engine
func NewStreamingEngine(model string) *StreamingEngine {
	return &StreamingEngine{
		client: New(model),
	}
}

// NewStreamingEngineWithConfig creates a streaming engine with custom config
func NewStreamingEngineWithConfig(config Config) *StreamingEngine {
	return &StreamingEngine{
		client: NewWithConfig(config),
	}
}

// ChatStreamChan performs streaming chat with channel-based token delivery
func (s *StreamingEngine) ChatStreamChan(messages []Message) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 100)
	errorCh := make(chan error, 1)
	
	go func() {
		defer close(tokenCh)
		defer close(errorCh)
		
		response := s.client.ChatStream(messages, func(token string) {
			select {
			case tokenCh <- token:
				// Token sent successfully
			default:
				// Channel full, skip token (non-blocking)
			}
		})
		
		if response.Error != nil {
			errorCh <- response.Error
		}
	}()
	
	return tokenCh, errorCh
}

// ChatStreamChanWithTimeout performs streaming with timeout
func (s *StreamingEngine) ChatStreamChanWithTimeout(messages []Message, timeout time.Duration) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 100)
	errorCh := make(chan error, 1)
	
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	
	go func() {
		defer close(tokenCh)
		defer close(errorCh)
		defer cancel()
		
		done := make(chan struct{})
		var streamErr error
		
		// Start streaming in another goroutine
		go func() {
			defer close(done)
			response := s.client.ChatStream(messages, func(token string) {
				select {
				case tokenCh <- token:
					// Token sent successfully
				case <-ctx.Done():
					// Context cancelled, stop streaming
					return
				default:
					// Channel full, skip token (non-blocking)
				}
			})
			
			if response.Error != nil {
				streamErr = response.Error
			}
		}()
		
		// Wait for completion or timeout
		select {
		case <-done:
			if streamErr != nil {
				errorCh <- streamErr
			}
		case <-ctx.Done():
			errorCh <- fmt.Errorf("streaming timeout after %v", timeout)
		}
	}()
	
	return tokenCh, errorCh
}

// Chat delegates to the underlying client
func (s *StreamingEngine) Chat(messages []Message) (*Response, error) {
	return s.client.Chat(messages)
}

// ChatStream delegates to the underlying client
func (s *StreamingEngine) ChatStream(messages []Message, callback StreamCallback) *Response {
	return s.client.ChatStream(messages, callback)
}

// IsHealthy delegates to the underlying client
func (s *StreamingEngine) IsHealthy() bool {
	return s.client.IsHealthy()
}

// Close delegates to the underlying client
func (s *StreamingEngine) Close() error {
	return s.client.Close()
}

// StreamCollector collects streaming tokens into a complete response
type StreamCollector struct {
	tokens []string
	err    error
}

// NewStreamCollector creates a new stream collector
func NewStreamCollector() *StreamCollector {
	return &StreamCollector{
		tokens: make([]string, 0),
	}
}

// Collect reads from token and error channels and accumulates the response
func (sc *StreamCollector) Collect(tokenCh <-chan string, errorCh <-chan error) (*Response, error) {
	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				// Token channel closed, check for errors
				select {
				case err := <-errorCh:
					if err != nil {
						return &Response{Error: err}, err
					}
				default:
					// No error
				}
				
				// Return collected response
				content := ""
				for _, t := range sc.tokens {
					content += t
				}
				
				return &Response{
					Content: content,
					Tokens:  len(content) / 4, // Rough estimate
				}, nil
			}
			
			sc.tokens = append(sc.tokens, token)
			
		case err := <-errorCh:
			if err != nil {
				return &Response{Error: err}, err
			}
		}
	}
}

// TokenBatcher batches streaming tokens for efficient processing
type TokenBatcher struct {
	batchSize int
	interval  time.Duration
}

// NewTokenBatcher creates a new token batcher
func NewTokenBatcher(batchSize int, interval time.Duration) *TokenBatcher {
	return &TokenBatcher{
		batchSize: batchSize,
		interval:  interval,
	}
}

// BatchTokens converts a token stream into batched chunks
func (tb *TokenBatcher) BatchTokens(tokenCh <-chan string) <-chan []string {
	batchCh := make(chan []string, 10)
	
	go func() {
		defer close(batchCh)
		
		var batch []string
		ticker := time.NewTicker(tb.interval)
		defer ticker.Stop()
		
		for {
			select {
			case token, ok := <-tokenCh:
				if !ok {
					// Send final batch if any tokens remain
					if len(batch) > 0 {
						batchCh <- batch
					}
					return
				}
				
				batch = append(batch, token)
				
				// Send batch if it reaches the size limit
				if len(batch) >= tb.batchSize {
					batchCh <- batch
					batch = make([]string, 0)
				}
				
			case <-ticker.C:
				// Send batch on interval if any tokens exist
				if len(batch) > 0 {
					batchCh <- batch
					batch = make([]string, 0)
				}
			}
		}
	}()
	
	return batchCh
}

// ConcurrentStreamer manages multiple concurrent streaming requests
type ConcurrentStreamer struct {
	engine   *StreamingEngine
	maxConns int
	semaphore chan struct{}
}

// NewConcurrentStreamer creates a new concurrent streamer
func NewConcurrentStreamer(model string, maxConnections int) *ConcurrentStreamer {
	return &ConcurrentStreamer{
		engine:    NewStreamingEngine(model),
		maxConns:  maxConnections,
		semaphore: make(chan struct{}, maxConnections),
	}
}

// Stream performs a streaming request with concurrency control
func (cs *ConcurrentStreamer) Stream(messages []Message) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 100)
	errorCh := make(chan error, 1)
	
	go func() {
		defer close(tokenCh)
		defer close(errorCh)
		
		// Acquire semaphore
		cs.semaphore <- struct{}{}
		defer func() { <-cs.semaphore }()
		
		// Perform streaming
		tokens, errors := cs.engine.ChatStreamChan(messages)
		
		// Forward tokens and errors
		for {
			select {
			case token, ok := <-tokens:
				if !ok {
					return
				}
				select {
				case tokenCh <- token:
				default:
					// Non-blocking send
				}
			case err, ok := <-errors:
				if ok && err != nil {
					errorCh <- err
					return
				}
			}
		}
	}()
	
	return tokenCh, errorCh
}

// Close cleans up the concurrent streamer
func (cs *ConcurrentStreamer) Close() error {
	return cs.engine.Close()
}