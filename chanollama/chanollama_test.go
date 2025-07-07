package chanollama

import (
	"testing"
	"time"
)

func TestNewClient(t *testing.T) {
	client := NewClient()
	if client == nil {
		t.Fatal("NewClient returned nil")
	}
	
	if client.config.BaseURL != "http://localhost:11434" {
		t.Errorf("Expected base URL http://localhost:11434, got %s", client.config.BaseURL)
	}
	
	if client.config.Timeout != 30*time.Second {
		t.Errorf("Expected timeout 30s, got %v", client.config.Timeout)
	}
}

func TestNewClientWithConfig(t *testing.T) {
	config := &Config{
		BaseURL: "http://custom:8080",
		Timeout: 60 * time.Second,
		Headers: map[string]string{
			"Authorization": "Bearer token",
		},
	}
	
	client := NewClientWithConfig(config)
	if client == nil {
		t.Fatal("NewClientWithConfig returned nil")
	}
	
	if client.config.BaseURL != config.BaseURL {
		t.Errorf("Expected base URL %s, got %s", config.BaseURL, client.config.BaseURL)
	}
	
	if client.config.Timeout != config.Timeout {
		t.Errorf("Expected timeout %v, got %v", config.Timeout, client.config.Timeout)
	}
}

func TestNewEngine(t *testing.T) {
	client := NewClient()
	config := &EngineConfig{
		Model:       "llama2",
		Temperature: 0.8,
		MaxTokens:   500,
	}
	
	engine := NewEngine(client, config)
	if engine == nil {
		t.Fatal("NewEngine returned nil")
	}
	
	if engine.config.Model != config.Model {
		t.Errorf("Expected model %s, got %s", config.Model, engine.config.Model)
	}
	
	if engine.config.Temperature != config.Temperature {
		t.Errorf("Expected temperature %f, got %f", config.Temperature, engine.config.Temperature)
	}
}

func TestEngineSystemPrompt(t *testing.T) {
	client := NewClient()
	config := &EngineConfig{
		Model:        "llama2",
		SystemPrompt: "You are a helpful assistant.",
	}
	
	engine := NewEngine(client, config)
	history := engine.GetHistory()
	
	if len(history) != 1 {
		t.Errorf("Expected 1 message in history, got %d", len(history))
	}
	
	if history[0].Role != "system" {
		t.Errorf("Expected first message to be system, got %s", history[0].Role)
	}
	
	if history[0].Content != config.SystemPrompt {
		t.Errorf("Expected system prompt %s, got %s", config.SystemPrompt, history[0].Content)
	}
}

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()
	
	if config.BaseURL != "http://localhost:11434" {
		t.Errorf("Expected default base URL http://localhost:11434, got %s", config.BaseURL)
	}
	
	if config.Timeout != 30*time.Second {
		t.Errorf("Expected default timeout 30s, got %v", config.Timeout)
	}
	
	if config.Headers == nil {
		t.Error("Expected headers to be initialized")
	}
}

func TestDefaultEngineConfig(t *testing.T) {
	config := DefaultEngineConfig()
	
	if config.Model != "llama2" {
		t.Errorf("Expected default model llama2, got %s", config.Model)
	}
	
	if config.Temperature != 0.7 {
		t.Errorf("Expected default temperature 0.7, got %f", config.Temperature)
	}
	
	if config.MaxTokens != 1000 {
		t.Errorf("Expected default max tokens 1000, got %d", config.MaxTokens)
	}
}