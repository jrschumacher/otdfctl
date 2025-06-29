package cmd

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/opentdf/otdfctl/pkg/cli"
	"github.com/opentdf/otdfctl/pkg/llm"
	"github.com/opentdf/otdfctl/pkg/man"
	"github.com/spf13/cobra"
)

var llmCmd = man.Docs.GetCommand("llm")

var llmChatCmd = man.Docs.GetCommand("llm/chat", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	if len(args) == 0 {
		c.ExitWithError("Model path is required", nil)
	}
	
	modelPath := args[0]
	
	// Get flag values
	stream := c.Flags.GetOptionalBool("stream")
	contextSize := int(c.Flags.GetOptionalInt32("context-size"))
	temperatureFlag, _ := cmd.Flags().GetFloat64("temperature")
	temperature := temperatureFlag
	systemPrompt := c.Flags.GetOptionalString("system-prompt")
	enableRAG := c.Flags.GetOptionalBool("rag")
	indexPath := c.Flags.GetOptionalString("index-path")
	
	// Initialize simple chat engine to avoid goroutine issues
	simpleEngine := llm.NewSimpleChatEngine(modelPath)
	
	// Set defaults for RAG if enabled
	if enableRAG {
		if indexPath == "" {
			homeDir, _ := os.UserHomeDir()
			indexPath = filepath.Join(homeDir, ".otdfctl", "simple_rag_index.json")
		}
		
		c.Printf("🔧 Initializing Simple RAG support...\n")
		
		// Load simple RAG store
		simpleStore := llm.NewSimpleRAGStore(indexPath)
		if err := simpleStore.LoadIndex(); err != nil {
			c.ExitWithError("Failed to load simple RAG index", err)
		}
		
		if simpleStore.GetDocumentCount() == 0 {
			c.Printf("⚠️  Warning: No documents found in simple RAG index. Run 'otdfctl llm ingest-simple' first.\n")
		} else {
			// Enable simple RAG on the chat engine
			simpleEngine.EnableSimpleRAG(simpleStore)
			c.Printf("✅ Simple RAG enabled with %d documents\n", simpleStore.GetDocumentCount())
		}
	}
	
	// Start the engine
	if err := simpleEngine.Start(); err != nil {
		c.ExitWithError("Failed to start simple chat engine", err)
	}
	defer simpleEngine.Stop()
	
	// Check if JSON output is requested
	if jsonFlag, _ := cmd.Flags().GetBool("json"); jsonFlag {
		session := map[string]interface{}{
			"model_path": modelPath,
			"config": map[string]interface{}{
				"stream":       stream,
				"context_size": contextSize,
				"temperature":  temperature,
			},
			"rag_enabled": enableRAG,
			"status":      "initialized",
		}
		c.ExitWithJSON(session)
		return
	}
	
	// Start interactive chat session
	if err := startSimpleInteractiveChat(c, simpleEngine, systemPrompt, stream); err != nil {
		c.ExitWithError("Failed to start chat session", err)
	}
}))

func init() {
	// TODO: Fix flag documentation parsing and use proper doc-driven flags
	// For POC, hardcode flags temporarily
	llmChatCmd.Flags().Bool("stream", true, "Enable streaming responses")
	llmChatCmd.Flags().Int32("context-size", 4096, "Maximum context window size")
	llmChatCmd.Flags().Float64("temperature", 0.7, "Sampling temperature (0.0-1.0)")
	llmChatCmd.Flags().String("system-prompt", "", "Custom system prompt")
	llmChatCmd.Flags().Bool("rag", false, "Enable RAG (Retrieval-Augmented Generation)")
	llmChatCmd.Flags().String("index-path", "", "Path to RAG vector index (default: ~/.otdfctl/rag_index.json)")
	llmChatCmd.Flags().String("embedding-model", "", "Path to embedding model for RAG (default: same as chat model)")
	llmChatCmd.Flags().Bool("json", false, "Output in JSON format")
	
	// Add chat command to llm parent
	llmCmd.AddCommand(&llmChatCmd.Command)
	
	// Add llm command to root
	RootCmd.AddCommand(&llmCmd.Command)
}

// startSimpleInteractiveChat handles the interactive chat session with the simple engine
func startSimpleInteractiveChat(c *cli.Cli, engine *llm.SimpleChatEngine, systemPrompt string, stream bool) error {
	// Initialize conversation with system message
	messages := []llm.ChatMessage{}
	if systemPrompt != "" {
		messages = append(messages, llm.ChatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	} else {
		messages = append(messages, llm.ChatMessage{
			Role:    "system",
			Content: getDefaultSystemPrompt(),
		})
	}
	
	c.Printf("🤖 OpenTDF LLM Chat started! Type 'exit' to quit, 'clear' to clear history.\n")
	c.Printf("   Use '/stream' to toggle streaming mode, '/help' for commands.\n")
	c.Printf("   Simple engine mode (no complex goroutines)\n\n")
	
	scanner := bufio.NewScanner(os.Stdin)
	
	for {
		c.Printf("> ")
		
		if !scanner.Scan() {
			break
		}
		
		input := strings.TrimSpace(scanner.Text())
		
		if input == "" {
			continue
		}
		
		// Handle commands
		switch input {
		case "exit", "quit":
			c.Println("Goodbye! 👋")
			return nil
		case "clear":
			messages = messages[:1] // Keep system message
			c.Println("Chat history cleared.")
			continue
		case "/stream":
			stream = !stream
			c.Printf("Streaming mode: %v\n", stream)
			continue
		case "/help":
			printHelp(c)
			continue
		}
		
		// Add user message
		messages = append(messages, llm.ChatMessage{
			Role:    "user",
			Content: input,
		})
		
		// Get response
		c.Printf("🤖 ")
		
		start := time.Now()
		var fullResponse strings.Builder
		
		if stream {
			// Use streaming inference
			response := engine.ChatStream(messages, func(token string) {
				c.Printf("%s", token)
				os.Stdout.Sync() // Force flush for real-time streaming
				fullResponse.WriteString(token)
			})
			
			if response.Error != nil {
				c.Printf("\nError: %v\n", response.Error)
				continue
			}
			
			c.Printf("\n\n⏱️  Response time: %v\n", time.Since(start))
		} else {
			// Use non-streaming inference
			response := engine.Chat(messages)
			
			if response.Error != nil {
				c.Printf("\nError: %v\n", response.Error)
				continue
			}
			
			c.Printf("%s\n\n⏱️  Response time: %v\n", response.Content, time.Since(start))
			fullResponse.WriteString(response.Content)
		}
		
		// Add assistant response to history
		if fullResponse.Len() > 0 {
			messages = append(messages, llm.ChatMessage{
				Role:    "assistant",
				Content: fullResponse.String(),
			})
		}
	}
	
	return nil
}

// getDefaultSystemPrompt returns the default OpenTDF-focused system prompt
func getDefaultSystemPrompt() string {
	return `You are an OpenTDF subject matter expert assistant. You have deep knowledge about:

- OpenTDF (Trusted Data Format) architecture and concepts
- Policy management including attributes, namespaces, values, and subject mappings  
- TDF encryption/decryption workflows and best practices
- Key Access Service (KAS) configuration and operations
- otdfctl CLI tool usage and troubleshooting
- OpenTDF Platform deployment and administration
- Data security and access control patterns

You help users understand OpenTDF concepts, debug issues, write policies, and implement secure data workflows. Provide practical, actionable guidance with code examples when relevant.`
}

// printHelp displays available commands
func printHelp(c *cli.Cli) {
	c.Println("\nAvailable commands:")
	c.Println("  exit, quit  - Exit the chat")
	c.Println("  clear       - Clear chat history")
	c.Println("  /stream     - Toggle streaming mode")
	c.Println("  /help       - Show this help")
}