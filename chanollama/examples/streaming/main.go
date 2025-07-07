package main

import (
	"context"
	"fmt"
	"log"

	"github.com/jrschumacher/chanollama"
)

func main() {
	// Create a new client with streaming enabled
	client := chanollama.NewClient()

	// Make a streaming chat request
	stream, err := client.ChatStream(context.Background(), &chanollama.ChatRequest{
		Model: "llama2",
		Messages: []chanollama.Message{
			{Role: "user", Content: "Tell me a short story about a robot."},
		},
		Stream: true,
	})

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Streaming response:")
	for response := range stream {
		if response.Error != nil {
			log.Printf("Error: %v", response.Error)
			break
		}
		fmt.Print(response.Message.Content)
	}
	fmt.Println()
}