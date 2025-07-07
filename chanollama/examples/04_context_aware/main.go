package main

import (
	"fmt"
	"log"

	"github.com/jrschumacher/chanollama"
)

func main() {
	fmt.Println("🧠 Context-Aware Conversation Example")
	fmt.Println("=====================================")

	// Create client
	client := chanollama.New("llama2:7b")
	defer client.Close()

	if !client.IsHealthy() {
		log.Fatal("❌ Ollama service is not available")
	}

	// Create a conversation with history management
	conversation := chanollama.NewConversation()
	
	// Set system prompt for the conversation
	conversation.AddSystemMessage(`You are a helpful programming tutor. 
	Remember what we've discussed and build upon previous topics in our conversation.`)

	fmt.Println("🎓 Programming Tutor Session")
	fmt.Println("Let's have a conversation that builds upon itself...\n")

	// Turn 1: Ask about variables
	fmt.Println("👤 Student: What are variables in programming?")
	conversation.AddUserMessage("What are variables in programming?")

	response, err := client.Chat(conversation.Messages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
		return
	}

	fmt.Printf("🎓 Tutor: %s\n\n", response.Content)
	conversation.AddAssistantMessage(response.Content)

	// Turn 2: Follow up about types (context-aware)
	fmt.Println("👤 Student: What about different types of variables?")
	conversation.AddUserMessage("What about different types of variables?")

	response, err = client.Chat(conversation.Messages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
		return
	}

	fmt.Printf("🎓 Tutor: %s\n\n", response.Content)
	conversation.AddAssistantMessage(response.Content)

	// Turn 3: Ask for a code example (should remember context)
	fmt.Println("👤 Student: Can you show me an example in Go?")
	conversation.AddUserMessage("Can you show me an example in Go?")

	response, err = client.Chat(conversation.Messages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
		return
	}

	fmt.Printf("🎓 Tutor: %s\n\n", response.Content)
	conversation.AddAssistantMessage(response.Content)

	// Turn 4: Reference previous example (context continuity)
	fmt.Println("👤 Student: How would I modify that example to use pointers?")
	conversation.AddUserMessage("How would I modify that example to use pointers?")

	response, err = client.Chat(conversation.Messages)
	if err != nil {
		log.Printf("❌ Error: %v", err)
		return
	}

	fmt.Printf("🎓 Tutor: %s\n\n", response.Content)
	conversation.AddAssistantMessage(response.Content)

	// Show conversation statistics
	fmt.Printf("📊 Conversation Statistics:\n")
	fmt.Printf("   Messages: %d\n", len(conversation.Messages))
	fmt.Printf("   Estimated tokens: %d\n", conversation.EstimateTokens())
	fmt.Printf("   Max messages limit: %d\n", conversation.MaxMessages)
	fmt.Printf("   Token limit: %d\n", conversation.TokenLimit)

	// Example with streaming and context
	fmt.Println("\n🔄 Streaming with context:")
	fmt.Print("🎓 Tutor: ")

	conversation.AddUserMessage("Can you explain the difference between the examples you just showed me?")

	engine := chanollama.NewStreamingEngine("llama2:7b")
	response = engine.ChatStream(conversation.Messages, func(token string) {
		fmt.Print(token)
	})

	if response.Error != nil {
		log.Printf("\n❌ Error: %v", response.Error)
	} else {
		conversation.AddAssistantMessage(response.Content)
	}

	fmt.Printf("\n\n📈 Final conversation stats:\n")
	fmt.Printf("   Total messages: %d\n", len(conversation.Messages))
	fmt.Printf("   Total estimated tokens: %d\n", conversation.EstimateTokens())

	// Demonstrate automatic trimming by adding many messages
	fmt.Println("\n🔄 Testing automatic message trimming...")
	
	for i := 0; i < 25; i++ {
		conversation.AddUserMessage(fmt.Sprintf("Test message %d", i))
		conversation.AddAssistantMessage(fmt.Sprintf("Response %d", i))
	}

	fmt.Printf("📊 After adding 50 test messages:\n")
	fmt.Printf("   Messages kept: %d (should be ≤ %d)\n", len(conversation.Messages), conversation.MaxMessages)
	fmt.Printf("   System messages preserved: ")
	
	systemCount := 0
	for _, msg := range conversation.Messages {
		if msg.Role == "system" {
			systemCount++
		}
	}
	fmt.Printf("%d\n", systemCount)

	fmt.Println("\n✅ Context-aware conversation example completed!")
}