package main

import (
	"fmt"
	"log"

	"github.com/jrschumacher/chanollama"
)

func main() {
	fmt.Println("🔧 System Prompts Example")
	fmt.Println("=========================")

	// Create client
	client := chanollama.New("llama2:7b")
	defer client.Close()

	if !client.IsHealthy() {
		log.Fatal("❌ Ollama service is not available")
	}

	// Example 1: Coding Assistant
	fmt.Println("\n👨‍💻 Coding Assistant:")
	fmt.Println("Question: How do I handle errors in Go?")

	codingMessages := []chanollama.Message{
		{
			Role: "system", 
			Content: `You are a helpful coding assistant specialized in Go programming. 
			Provide clear, concise answers with code examples. Focus on best practices and idiomatic Go code.`,
		},
		{Role: "user", Content: "How do I handle errors in Go?"},
	}

	response, err := client.Chat(codingMessages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
	} else {
		fmt.Printf("💡 Answer: %s\n", response.Content)
	}

	// Example 2: Documentation Expert  
	fmt.Println("\n📚 Documentation Expert:")
	fmt.Println("Question: What is OpenTDF?")

	docMessages := []chanollama.Message{
		{
			Role: "system",
			Content: `You are a technical documentation expert specializing in OpenTDF.
			Provide accurate, comprehensive explanations about OpenTDF concepts, architecture, and usage.
			Always include practical examples when possible.`,
		},
		{Role: "user", Content: "What is OpenTDF and how does it work?"},
	}

	response, err = client.Chat(docMessages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
	} else {
		fmt.Printf("📖 Answer: %s\n", response.Content)
	}

	// Example 3: Creative Writer
	fmt.Println("\n✍️ Creative Writer:")
	fmt.Println("Request: Write a haiku about programming")

	creativeMessages := []chanollama.Message{
		{
			Role: "system",
			Content: `You are a creative writer who specializes in crafting beautiful, meaningful poetry and prose.
			Your writing is concise but evocative, capturing deep emotions and ideas in few words.`,
		},
		{Role: "user", Content: "Write a haiku about the beauty of programming"},
	}

	response, err = client.Chat(creativeMessages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
	} else {
		fmt.Printf("🎨 Haiku:\n%s\n", response.Content)
	}

	// Example 4: Debugging Assistant
	fmt.Println("\n🐛 Debugging Assistant:")
	fmt.Println("Problem: My Go program panics with 'index out of range'")

	debugMessages := []chanollama.Message{
		{
			Role: "system",
			Content: `You are an expert debugging assistant. When users describe errors or bugs:
			1. Identify the likely cause
			2. Provide specific steps to fix the issue  
			3. Suggest prevention strategies
			4. Give code examples when helpful`,
		},
		{Role: "user", Content: "My Go program panics with 'index out of range [3] with length 2'. How do I fix this?"},
	}

	response, err = client.Chat(debugMessages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
	} else {
		fmt.Printf("🔧 Debug help: %s\n", response.Content)
	}

	// Example 5: Streaming with specialized prompt
	fmt.Println("\n🔄 Streaming Architecture Consultant:")
	fmt.Print("Consultant: ")

	archMessages := []chanollama.Message{
		{
			Role: "system",
			Content: `You are a senior software architect with 15+ years of experience.
			You provide strategic guidance on system design, scalability, and best practices.
			Your responses are thoughtful, detailed, and focus on long-term maintainability.`,
		},
		{Role: "user", Content: "How should I design a microservices architecture for a chat application?"},
	}

	engine := chanollama.NewStreamingEngine("llama2:7b")
	engine.ChatStream(archMessages, func(token string) {
		fmt.Print(token)
	})

	fmt.Println("\n\n✅ System prompts example completed!")
}