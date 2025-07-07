package main

import (
	"context"
	"fmt"
	"log"

	"github.com/jrschumacher/chanollama"
)

func main() {
	// Create a new client with default configuration
	client := chanollama.NewClient()

	// Make a simple chat request
	response, err := client.Chat(context.Background(), &chanollama.ChatRequest{
		Model: "llama2",
		Messages: []chanollama.Message{
			{Role: "user", Content: "What is the capital of France?"},
		},
	})

	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response.Message.Content)
}