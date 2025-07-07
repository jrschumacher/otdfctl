# 🚀 chanollama

**Channel-based streaming for Ollama models in Go**

`chanollama` provides a clean, efficient way to interact with local Ollama models using Go channels for real-time token streaming. Perfect for building chatbots, AI assistants, and applications requiring responsive LLM interactions.

## ✨ Features

- **🚀 Simple Setup** - Just `chanollama.New("llama2:7b")` and you're streaming
- **📡 Real-time Streaming** - Channel-based, non-blocking token delivery  
- **🧠 Context Management** - Built-in conversation history and trimming
- **🔧 System Prompts** - Easy specialization of AI behavior
- **⚡ Performance** - Concurrent conversations, efficient token handling
- **🛡️ Production Ready** - Error handling, timeouts, goroutine safety
- **🔌 Zero Dependencies** - Pure Go HTTP client for Ollama

## 📦 Installation

```bash
go get github.com/jrschumacher/chanollama
```

## 🚀 Quick Start

```go
package main

import (
    "fmt"
    "github.com/jrschumacher/chanollama"
)

func main() {
    // Create client (assumes Ollama running on localhost:11434)
    client := chanollama.New("llama2:7b")
    defer client.Close()

    // Simple chat
    messages := []chanollama.Message{
        {Role: "user", Content: "Hello!"},
    }

    response, err := client.Chat(messages)
    if err != nil {
        panic(err)
    }
    
    fmt.Println("Response:", response.Content)
    
    // Real-time streaming
    client.ChatStream(messages, func(token string) {
        fmt.Print(token) // Tokens appear in real-time
    })
}
```

## 📖 Examples

### 🔄 Channel-Based Streaming

```go
engine := chanollama.NewStreamingEngine("llama2:7b")

// Get channels for tokens and errors
tokenCh, errorCh := engine.ChatStreamChan(messages)

for {
    select {
    case token, ok := <-tokenCh:
        if !ok {
            return // Streaming complete
        }
        fmt.Print(token)
        
    case err := <-errorCh:
        if err != nil {
            log.Printf("Error: %v", err)
            return
        }
    }
}
```

### 🧠 Context-Aware Conversations

```go
// Automatic conversation management
conversation := chanollama.NewConversation()
conversation.AddSystemMessage("You are a helpful coding assistant.")

// Messages are automatically managed and trimmed
conversation.AddUserMessage("What is Go?")
response, _ := client.Chat(conversation.Messages)
conversation.AddAssistantMessage(response.Content)

conversation.AddUserMessage("Show me an example")
response, _ = client.Chat(conversation.Messages) // Includes full context
```

### 🔧 System Prompts

```go
messages := []chanollama.Message{
    {
        Role: "system", 
        Content: "You are a helpful coding assistant specialized in Go programming.",
    },
    {Role: "user", Content: "How do I handle errors in Go?"},
}

response, _ := client.Chat(messages)
```

### ⚡ Concurrent Streaming

```go
// Handle multiple conversations simultaneously
streamer := chanollama.NewConcurrentStreamer("llama2:7b", 5) // Max 5 concurrent

var wg sync.WaitGroup
for _, question := range questions {
    wg.Add(1)
    go func(q string) {
        defer wg.Done()
        
        messages := []chanollama.Message{{Role: "user", Content: q}}
        tokenCh, errorCh := streamer.Stream(messages)
        
        // Process streaming response...
    }(question)
}
wg.Wait()
```

### 📦 Token Batching

```go
// Batch tokens for efficient processing
batcher := chanollama.NewTokenBatcher(5, 50*time.Millisecond)
tokenCh, _ := engine.ChatStreamChan(messages)
batchCh := batcher.BatchTokens(tokenCh)

for batch := range batchCh {
    // Process 5 tokens at a time or every 50ms
    for _, token := range batch {
        fmt.Print(token)
    }
}
```

## 🔧 Configuration

```go
config := chanollama.Config{
    Model:       "llama2:7b",
    Timeout:     30 * time.Second,
    MaxTokens:   2048,
    Temperature: 0.7,
}

client := chanollama.NewWithConfig(config)
```

## 📋 Requirements

- **Ollama** running locally on `localhost:11434`
- **Go 1.21+**
- A downloaded model (e.g., `ollama pull llama2:7b`)

## 🎯 Use Cases

### 🤖 Chatbots
```go
conversation := chanollama.NewConversation()
conversation.AddSystemMessage("You are a friendly customer service bot.")

for userInput := range userInputChannel {
    conversation.AddUserMessage(userInput)
    response, _ := client.Chat(conversation.Messages)
    conversation.AddAssistantMessage(response.Content)
    
    // Send response back to user
    sendToUser(response.Content)
}
```

### 🔍 AI-Powered Search
```go
messages := []chanollama.Message{
    {Role: "system", Content: "You are a search assistant. Provide concise, relevant answers."},
    {Role: "user", Content: searchQuery},
}

client.ChatStream(messages, func(token string) {
    // Stream results in real-time
    updateSearchResults(token)
})
```

### 💻 Code Assistant
```go
codeAssistant := []chanollama.Message{
    {Role: "system", Content: "You are an expert programmer. Provide clean, well-documented code."},
    {Role: "user", Content: "Write a REST API handler in Go"},
}

response, _ := client.Chat(codeAssistant)
```

## 🔍 API Reference

### Core Types

```go
type Message struct {
    Role    string // "system", "user", "assistant"
    Content string
}

type Response struct {
    Content string
    Error   error
    Tokens  int
}

type StreamCallback func(token string)
```

### Client Methods

```go
// Basic client
client := chanollama.New("model-name")
response, err := client.Chat(messages)
response := client.ChatStream(messages, callback)
healthy := client.IsHealthy()
client.Close()

// Streaming engine
engine := chanollama.NewStreamingEngine("model-name")
tokenCh, errorCh := engine.ChatStreamChan(messages)
tokenCh, errorCh := engine.ChatStreamChanWithTimeout(messages, timeout)
```

### Conversation Management

```go
conv := chanollama.NewConversation()
conv.AddUserMessage("content")
conv.AddAssistantMessage("content")
conv.AddSystemMessage("content")
tokens := conv.EstimateTokens()
```

## 🛠️ Advanced Features

### Custom Timeout
```go
tokenCh, errorCh := engine.ChatStreamChanWithTimeout(messages, 10*time.Second)
```

### Stream Collection
```go
collector := chanollama.NewStreamCollector()
response, err := collector.Collect(tokenCh, errorCh)
```

### Concurrent Control
```go
streamer := chanollama.NewConcurrentStreamer("llama2:7b", maxConnections)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for the excellent local LLM platform
- The Go community for inspiration on clean API design

---

**Made with ❤️ for the Go and AI community**