package main

import (
	"fmt"
	"log"

	"github.com/jrschumacher/chanollama"
)

func main() {
	fmt.Println("🚀 Real-time Streaming Example")
	fmt.Println("==============================")

	// Create a streaming engine
	engine := chanollama.NewStreamingEngine("llama2:7b")
	defer engine.Close()

	// Check if Ollama is available
	if !engine.IsHealthy() {
		log.Fatal("❌ Ollama service is not available at localhost:11434")
	}

	fmt.Println("✅ Connected to Ollama")

	// Example 1: Callback-based streaming
	fmt.Println("\n🔄 Callback-based streaming:")
	fmt.Print("Assistant: ")

	messages := []chanollama.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Tell me a short joke"},
	}

	response := engine.ChatStream(messages, func(token string) {
		fmt.Print(token) // Real-time token display
	})

	if response.Error != nil {
		log.Printf("\n❌ Error: %v", response.Error)
	} else {
		fmt.Printf("\n\n📊 Total tokens: %d\n", response.Tokens)
	}

	// Example 2: Channel-based streaming
	fmt.Println("\n🔄 Channel-based streaming:")
	fmt.Print("Assistant: ")

	messages2 := []chanollama.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}

	tokenCh, errorCh := engine.ChatStreamChan(messages2)

	var fullResponse string
	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				// Streaming completed
				goto done
			}
			fmt.Print(token)
			fullResponse += token

		case err := <-errorCh:
			if err != nil {
				log.Printf("\n❌ Error: %v", err)
				return
			}
		}
	}

done:
	fmt.Printf("\n\n📊 Full response length: %d characters\n", len(fullResponse))

	// Example 3: Token batching
	fmt.Println("\n🔄 Token batching example:")
	fmt.Print("Assistant: ")

	messages3 := []chanollama.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Count from 1 to 10"},
	}

	tokenCh3, _ := engine.ChatStreamChan(messages3)

	// Create a batcher that groups tokens every 50ms or 3 tokens
	batcher := chanollama.NewTokenBatcher(3, 50) // Batch every 3 tokens or 50ms
	batchCh := batcher.BatchTokens(tokenCh3)

	for batch := range batchCh {
		// Process tokens in batches
		for _, token := range batch {
			fmt.Print(token)
		}
		fmt.Print("📦") // Visual indicator of batch boundaries
	}

	fmt.Println("\n\n✅ Streaming examples completed!")
}