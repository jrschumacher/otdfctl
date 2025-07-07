package chanollama

import (
	"fmt"
	"time"
)

// Message represents a chat message
type Message struct {
	Role      string            `json:"role"`      // "system", "user", "assistant"
	Content   string            `json:"content"`
	Timestamp time.Time         `json:"timestamp,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// NewMessage creates a new message with timestamp
func NewMessage(role, content string) Message {
	return Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Metadata:  make(map[string]string),
	}
}

// NewUserMessage creates a user message
func NewUserMessage(content string) Message {
	return NewMessage("user", content)
}

// NewAssistantMessage creates an assistant message
func NewAssistantMessage(content string) Message {
	return NewMessage("assistant", content)
}

// NewSystemMessage creates a system message
func NewSystemMessage(content string) Message {
	return NewMessage("system", content)
}

// WithMetadata adds metadata to the message
func (m Message) WithMetadata(key, value string) Message {
	if m.Metadata == nil {
		m.Metadata = make(map[string]string)
	}
	m.Metadata[key] = value
	return m
}

// GetMetadata retrieves metadata value
func (m Message) GetMetadata(key string) (string, bool) {
	if m.Metadata == nil {
		return "", false
	}
	value, exists := m.Metadata[key]
	return value, exists
}

// IsRole checks if message has the specified role
func (m Message) IsRole(role string) bool {
	return m.Role == role
}

// IsUser checks if message is from user
func (m Message) IsUser() bool {
	return m.Role == "user"
}

// IsAssistant checks if message is from assistant
func (m Message) IsAssistant() bool {
	return m.Role == "assistant"
}

// IsSystem checks if message is a system message
func (m Message) IsSystem() bool {
	return m.Role == "system"
}

// Age returns how long ago the message was created
func (m Message) Age() time.Duration {
	if m.Timestamp.IsZero() {
		return 0
	}
	return time.Since(m.Timestamp)
}

// String returns a string representation of the message
func (m Message) String() string {
	return fmt.Sprintf("[%s] %s", m.Role, m.Content)
}

// Response represents a complete response from the model
type Response struct {
	Content      string            `json:"content"`
	Error        error             `json:"error,omitempty"`
	Tokens       int               `json:"tokens,omitempty"`
	Duration     time.Duration     `json:"duration,omitempty"`
	Model        string            `json:"model,omitempty"`
	Timestamp    time.Time         `json:"timestamp,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	FinishReason string            `json:"finish_reason,omitempty"` // "stop", "length", "error"
}

// NewResponse creates a new response with timestamp
func NewResponse(content string) *Response {
	return &Response{
		Content:   content,
		Timestamp: time.Now(),
		Metadata:  make(map[string]string),
	}
}

// WithError sets an error on the response
func (r *Response) WithError(err error) *Response {
	r.Error = err
	if err != nil {
		r.FinishReason = "error"
	}
	return r
}

// WithTokens sets token count on the response
func (r *Response) WithTokens(tokens int) *Response {
	r.Tokens = tokens
	return r
}

// WithDuration sets processing duration on the response
func (r *Response) WithDuration(duration time.Duration) *Response {
	r.Duration = duration
	return r
}

// WithModel sets the model name on the response
func (r *Response) WithModel(model string) *Response {
	r.Model = model
	return r
}

// WithFinishReason sets the finish reason on the response
func (r *Response) WithFinishReason(reason string) *Response {
	r.FinishReason = reason
	return r
}

// WithMetadata adds metadata to the response
func (r *Response) WithMetadata(key, value string) *Response {
	if r.Metadata == nil {
		r.Metadata = make(map[string]string)
	}
	r.Metadata[key] = value
	return r
}

// IsSuccess returns true if the response was successful
func (r *Response) IsSuccess() bool {
	return r.Error == nil
}

// IsError returns true if the response contains an error
func (r *Response) IsError() bool {
	return r.Error != nil
}

// ToMessage converts the response to an assistant message
func (r *Response) ToMessage() Message {
	msg := NewAssistantMessage(r.Content)
	if r.Model != "" {
		msg = msg.WithMetadata("model", r.Model)
	}
	if r.Duration > 0 {
		msg = msg.WithMetadata("duration", r.Duration.String())
	}
	if r.Tokens > 0 {
		msg = msg.WithMetadata("tokens", fmt.Sprintf("%d", r.Tokens))
	}
	return msg
}

// Config holds configuration for the engine
type Config struct {
	Model       string        `json:"model"`       // Required: e.g. "llama2:7b"
	Timeout     time.Duration `json:"timeout"`     // Default: 30s
	MaxTokens   int          `json:"max_tokens"`  // Default: 2048
	Temperature float64      `json:"temperature"` // Default: 0.7
}

// DefaultConfig returns a sensible default configuration
func DefaultConfig(model string) Config {
	return Config{
		Model:       model,
		Timeout:     30 * time.Second,
		MaxTokens:   2048,
		Temperature: 0.7,
	}
}

// StreamCallback is called for each token during streaming
type StreamCallback func(token string)

// Engine interface defines the core functionality
type Engine interface {
	Chat(messages []Message) (*Response, error)
	ChatStream(messages []Message, callback StreamCallback) *Response
	IsHealthy() bool
	Close() error
}

// Conversation helper for managing chat history
type Conversation struct {
	Messages    []Message `json:"messages"`
	MaxMessages int       `json:"max_messages"` // Auto-trim when exceeded
	TokenLimit  int       `json:"token_limit"`  // Auto-summarize when exceeded
}

// NewConversation creates a new conversation with sensible defaults
func NewConversation() *Conversation {
	return &Conversation{
		Messages:    make([]Message, 0),
		MaxMessages: 20,   // Keep last 20 messages
		TokenLimit:  4000, // Auto-summarize at 4k tokens
	}
}

// AddUserMessage adds a user message to the conversation
func (c *Conversation) AddUserMessage(content string) {
	c.addMessage(NewUserMessage(content))
}

// AddAssistantMessage adds an assistant message to the conversation
func (c *Conversation) AddAssistantMessage(content string) {
	c.addMessage(NewAssistantMessage(content))
}

// AddSystemMessage adds a system message to the conversation
func (c *Conversation) AddSystemMessage(content string) {
	c.addMessage(NewSystemMessage(content))
}

// AddMessage adds a pre-created message to the conversation
func (c *Conversation) AddMessage(msg Message) {
	c.addMessage(msg)
}

// AddResponse adds a response as an assistant message
func (c *Conversation) AddResponse(response *Response) {
	if response != nil && response.IsSuccess() {
		c.addMessage(response.ToMessage())
	}
}

// GetLastMessage returns the last message in the conversation
func (c *Conversation) GetLastMessage() (Message, bool) {
	if len(c.Messages) == 0 {
		return Message{}, false
	}
	return c.Messages[len(c.Messages)-1], true
}

// GetLastUserMessage returns the last user message in the conversation
func (c *Conversation) GetLastUserMessage() (Message, bool) {
	for i := len(c.Messages) - 1; i >= 0; i-- {
		if c.Messages[i].IsUser() {
			return c.Messages[i], true
		}
	}
	return Message{}, false
}

// GetSystemMessages returns all system messages
func (c *Conversation) GetSystemMessages() []Message {
	var systemMsgs []Message
	for _, msg := range c.Messages {
		if msg.IsSystem() {
			systemMsgs = append(systemMsgs, msg)
		}
	}
	return systemMsgs
}

// FilterMessagesByRole returns messages with the specified role
func (c *Conversation) FilterMessagesByRole(role string) []Message {
	var filtered []Message
	for _, msg := range c.Messages {
		if msg.IsRole(role) {
			filtered = append(filtered, msg)
		}
	}
	return filtered
}

// GetMessageCount returns the total number of messages
func (c *Conversation) GetMessageCount() int {
	return len(c.Messages)
}

// GetMessagesByAge returns messages newer than the specified duration
func (c *Conversation) GetMessagesByAge(maxAge time.Duration) []Message {
	var recent []Message
	for _, msg := range c.Messages {
		if msg.Age() <= maxAge {
			recent = append(recent, msg)
		}
	}
	return recent
}

// Clear removes all messages from the conversation
func (c *Conversation) Clear() {
	c.Messages = make([]Message, 0)
}

// ClearExceptSystem removes all non-system messages
func (c *Conversation) ClearExceptSystem() {
	systemMsgs := c.GetSystemMessages()
	c.Messages = systemMsgs
}

// addMessage adds a message and handles trimming
func (c *Conversation) addMessage(msg Message) {
	c.Messages = append(c.Messages, msg)
	
	// Trim if we exceed max messages (keep system messages)
	if len(c.Messages) > c.MaxMessages {
		var systemMsgs []Message
		var otherMsgs []Message
		
		for _, m := range c.Messages {
			if m.Role == "system" {
				systemMsgs = append(systemMsgs, m)
			} else {
				otherMsgs = append(otherMsgs, m)
			}
		}
		
		// Keep system messages + recent other messages
		keepCount := c.MaxMessages - len(systemMsgs)
		if keepCount > 0 && len(otherMsgs) > keepCount {
			otherMsgs = otherMsgs[len(otherMsgs)-keepCount:]
		}
		
		c.Messages = append(systemMsgs, otherMsgs...)
	}
}

// EstimateTokens provides a rough token count estimate
func (c *Conversation) EstimateTokens() int {
	total := 0
	for _, msg := range c.Messages {
		// Rough estimation: 1 token ≈ 4 characters
		total += len(msg.Content) / 4
	}
	return total
}

// ContextSummarizer interface for summarizing conversation history
type ContextSummarizer interface {
	SummarizeHistory(messages []Message) (string, error)
}

// Internal Ollama API types
type ollamaRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaResponse struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}

type ollamaStreamResponse struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done bool `json:"done"`
}