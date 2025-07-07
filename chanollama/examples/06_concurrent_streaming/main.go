package main

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/jrschumacher/chanollama"
)

func main() {
	fmt.Println("⚡ Concurrent Streaming Example")
	fmt.Println("===============================")

	// Create a concurrent streamer with max 3 connections
	streamer := chanollama.NewConcurrentStreamer("llama2:7b", 3)
	defer streamer.Close()

	// Example 1: Multiple concurrent conversations
	fmt.Println("🔄 Starting 5 concurrent conversations...")

	questions := []string{
		"What is Go programming language?",
		"Explain Docker containers",
		"What is machine learning?",
		"How does HTTP work?",
		"What is database indexing?",
	}

	var wg sync.WaitGroup
	
	for i, question := range questions {
		wg.Add(1)
		
		go func(id int, q string) {
			defer wg.Done()
			
			fmt.Printf("🚀 Starting conversation %d: %s\n", id+1, q)
			
			messages := []chanollama.Message{
				{Role: "system", Content: "You are a helpful technical assistant. Give concise, accurate answers."},
				{Role: "user", Content: q},
			}
			
			tokenCh, errorCh := streamer.Stream(messages)
			
			var response string
			fmt.Printf("💬 Conversation %d: ", id+1)
			
			for {
				select {
				case token, ok := <-tokenCh:
					if !ok {
						goto done
					}
					fmt.Print(token)
					response += token
					
				case err := <-errorCh:
					if err != nil {
						log.Printf("\n❌ Conversation %d error: %v", id+1, err)
						return
					}
				}
			}
			
		done:
			fmt.Printf("\n✅ Conversation %d completed (%d chars)\n\n", id+1, len(response))
			
		}(i, question)
		
		// Stagger the starts slightly
		time.Sleep(100 * time.Millisecond)
	}

	// Wait for all conversations to complete
	wg.Wait()
	fmt.Println("🎉 All concurrent conversations completed!")

	// Example 2: Chat room simulation
	fmt.Println("\n💬 Chat Room Simulation")
	fmt.Println("========================")

	chatRoom := NewChatRoom("llama2:7b")
	defer chatRoom.Close()

	// Simulate multiple users asking questions simultaneously
	users := []User{
		{"Alice", "What's the weather like?"},
		{"Bob", "Tell me a joke"},
		{"Charlie", "Explain quantum computing"},
		{"Diana", "How do I bake a cake?"},
	}

	fmt.Println("👥 Users joining chat room...")

	var chatWg sync.WaitGroup
	for _, user := range users {
		chatWg.Add(1)
		go func(u User) {
			defer chatWg.Done()
			chatRoom.HandleUser(u)
		}(user)
	}

	chatWg.Wait()
	fmt.Println("\n🏁 Chat room session ended!")

	// Example 3: Load testing
	fmt.Println("\n📈 Load Testing")
	fmt.Println("===============")

	fmt.Println("🔥 Sending 10 rapid requests...")
	
	start := time.Now()
	var loadWg sync.WaitGroup
	
	for i := 0; i < 10; i++ {
		loadWg.Add(1)
		
		go func(requestID int) {
			defer loadWg.Done()
			
			messages := []chanollama.Message{
				{Role: "user", Content: fmt.Sprintf("Count to %d", requestID+1)},
			}
			
			tokenCh, errorCh := streamer.Stream(messages)
			
			tokenCount := 0
			for {
				select {
				case _, ok := <-tokenCh:
					if !ok {
						goto loadDone
					}
					tokenCount++
					
				case err := <-errorCh:
					if err != nil {
						log.Printf("❌ Request %d error: %v", requestID, err)
						return
					}
				}
			}
			
		loadDone:
			fmt.Printf("✅ Request %d: %d tokens\n", requestID, tokenCount)
			
		}(i)
	}

	loadWg.Wait()
	duration := time.Since(start)
	fmt.Printf("⏱️ Load test completed in %v\n", duration)

	fmt.Println("\n✅ Concurrent streaming examples completed!")
}

// User represents a chat room user
type User struct {
	Name     string
	Question string
}

// ChatRoom simulates a multi-user chat environment
type ChatRoom struct {
	streamer *chanollama.ConcurrentStreamer
	mutex    sync.Mutex
}

// NewChatRoom creates a new chat room
func NewChatRoom(model string) *ChatRoom {
	return &ChatRoom{
		streamer: chanollama.NewConcurrentStreamer(model, 5), // Max 5 concurrent users
	}
}

// HandleUser processes a user's message in the chat room
func (cr *ChatRoom) HandleUser(user User) {
	cr.mutex.Lock()
	fmt.Printf("👤 %s joined: %s\n", user.Name, user.Question)
	cr.mutex.Unlock()

	messages := []chanollama.Message{
		{Role: "system", Content: "You are a friendly chat assistant. Keep responses brief and engaging."},
		{Role: "user", Content: user.Question},
	}

	tokenCh, errorCh := cr.streamer.Stream(messages)

	cr.mutex.Lock()
	fmt.Printf("🤖 Assistant to %s: ", user.Name)
	cr.mutex.Unlock()

	var response string
	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				goto chatDone
			}
			
			// In a real chat room, you'd broadcast to all users
			cr.mutex.Lock()
			fmt.Print(token)
			cr.mutex.Unlock()
			
			response += token

		case err := <-errorCh:
			if err != nil {
				cr.mutex.Lock()
				log.Printf("\n❌ Error for %s: %v", user.Name, err)
				cr.mutex.Unlock()
				return
			}
		}
	}

chatDone:
	cr.mutex.Lock()
	fmt.Printf("\n✅ %s's message handled (%d chars)\n", user.Name, len(response))
	cr.mutex.Unlock()
}

// Close cleans up the chat room
func (cr *ChatRoom) Close() {
	cr.streamer.Close()
}