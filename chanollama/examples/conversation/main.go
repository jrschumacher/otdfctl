package main

import (
	"context"
	"fmt"
	"log"

	"github.com/jrschumacher/chanollama"
)

func main() {
	// Create a new client
	client := chanollama.NewClient()

	// Create a conversation engine for managing chat history
	engine := chanollama.NewEngine(client, &chanollama.EngineConfig{
		Model: "llama2",
	})

	// Start a conversation
	ctx := context.Background()
	
	// First message
	response1, err := engine.Chat(ctx, "Hello, I'm planning a trip to Japan.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Assistant: %s\n\n", response1)

	// Follow-up message (engine remembers context)
	response2, err := engine.Chat(ctx, "What's the best season to visit?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Assistant: %s\n\n", response2)

	// Another follow-up
	response3, err := engine.Chat(ctx, "What about food recommendations?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Assistant: %s\n", response3)
}