package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/jrschumacher/chanollama"
)

func main() {
	// Create a client with custom configuration
	config := &chanollama.Config{
		BaseURL: "http://localhost:11434",
		Timeout: 30 * time.Second,
		Headers: map[string]string{
			"X-Custom-Header": "example-value",
		},
	}

	client := chanollama.NewClientWithConfig(config)

	// Create an engine with advanced configuration
	engine := chanollama.NewEngine(client, &chanollama.EngineConfig{
		Model:       "llama2",
		Temperature: 0.7,
		MaxTokens:   1000,
		SystemPrompt: "You are a helpful assistant specialized in technical documentation.",
	})

	ctx := context.Background()

	// Example of using the engine with streaming
	stream, err := engine.ChatStream(ctx, "Explain the concept of microservices architecture.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Streaming response from technical assistant:")
	for response := range stream {
		if response.Error != nil {
			log.Printf("Error: %v", response.Error)
			break
		}
		fmt.Print(response.Content)
	}
	fmt.Println()

	// Example of getting conversation history
	history := engine.GetHistory()
	fmt.Printf("\nConversation has %d messages\n", len(history))
}