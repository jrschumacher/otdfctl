package llm

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/opentdf/otdfctl/pkg/config"
)

// ChatSession represents a chat session for JSON output
type ChatSession struct {
	ModelPath   string        `json:"model_path"`
	Config      ChatConfig    `json:"config"`
	Messages    []ChatMessage `json:"messages"`
	SessionInfo SessionInfo   `json:"session_info"`
}

type ChatConfig struct {
	Stream      bool    `json:"stream"`
	ContextSize int     `json:"context_size"`
	Temperature float64 `json:"temperature"`
}

type SessionInfo struct {
	Started   string `json:"started"`
	Responses int    `json:"responses"`
}

// PrintFunc is a function that prints output (to allow dependency injection)
type PrintFunc func(string, ...interface{})

// ExitWithJSONFunc handles JSON output and exits
type ExitWithJSONFunc func(interface{})

// Handler provides LLM chat functionality 
type Handler struct {
	config           *config.Config
	engine           *ChatEngine
	printFunc        PrintFunc
	printlnFunc      PrintFunc
	exitWithJSONFunc ExitWithJSONFunc
	isJSONMode       bool
}

// NewHandler creates a new LLM handler instance
func NewHandler(cfg *config.Config, printFunc, printlnFunc PrintFunc, exitWithJSONFunc ExitWithJSONFunc, isJSONMode bool) *Handler {
	return &Handler{
		config:           cfg,
		printFunc:        printFunc,
		printlnFunc:      printlnFunc,
		exitWithJSONFunc: exitWithJSONFunc,
		isJSONMode:       isJSONMode,
	}
}

// Close gracefully shuts down the LLM handler
func (h *Handler) Close() {
	if h.engine != nil {
		h.engine.Stop()
	}
}

// StartChat initializes and starts an interactive chat session
func (h *Handler) StartChat(modelPath string, stream bool, contextSize int, temperature float64, systemPrompt string) error {
	// Use config defaults if values not provided via flags
	if modelPath == "" && h.config != nil {
		modelPath = h.config.LLM.DefaultModelPath
	}
	if !stream && h.config != nil {
		stream = h.config.LLM.Stream
	}
	if contextSize == 0 && h.config != nil {
		contextSize = h.config.LLM.ContextSize
	}
	if temperature == 0.0 && h.config != nil {
		temperature = h.config.LLM.Temperature
	}
	if systemPrompt == "" && h.config != nil {
		systemPrompt = h.config.LLM.SystemPrompt
	}
	
	if modelPath == "" {
		return fmt.Errorf("model path is required (set via argument or config file)")
	}
	
	h.engine = NewChatEngine(modelPath)
	
	if err := h.engine.Start(); err != nil {
		return fmt.Errorf("failed to start chat engine: %w", err)
	}
	
	defer h.engine.Stop()
	
	// Initialize conversation with system message
	messages := []ChatMessage{}
	if systemPrompt != "" {
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	} else {
		messages = append(messages, ChatMessage{
			Role:    "system", 
			Content: h.getDefaultSystemPrompt(),
		})
	}
	
	// Check if JSON output is requested
	if h.isJSONMode {
		return h.startJSONSession(modelPath, stream, contextSize, temperature, messages)
	}
	
	h.printFunc("🤖 OpenTDF LLM Chat started! Type 'exit' to quit, 'clear' to clear history.\n")
	h.printFunc("   Use '/stream' to toggle streaming mode, '/help' for commands.\n")
	h.printFunc("   Model: %s\n\n", modelPath)
	
	scanner := bufio.NewScanner(os.Stdin)
	
	for {
		h.printFunc("> ")
		
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
			h.printlnFunc("Goodbye! 👋")
			return nil
		case "clear":
			messages = messages[:1] // Keep system message
			h.printlnFunc("Chat history cleared.")
			continue
		case "/stream":
			stream = !stream
			h.printFunc("Streaming mode: %v\n", stream)
			continue
		case "/help":
			h.printHelp()
			continue
		}
		
		// Add user message
		messages = append(messages, ChatMessage{
			Role:    "user",
			Content: input,
		})
		
		// Get response
		h.printFunc("🤖 ")
		
		start := time.Now()
		responseChan := h.engine.Chat(messages, stream)
		
		var assistantResponse strings.Builder
		
		for response := range responseChan {
			if response.Error != nil {
				h.printFunc("\nError: %v\n", response.Error)
				break
			}
			
			if stream && !response.Done {
				h.printFunc(response.Message.Content)
			}
			
			assistantResponse.WriteString(response.Message.Content)
			
			if response.Done {
				if !stream {
					h.printFunc(assistantResponse.String())
				}
				h.printFunc("\n\n⏱️  Response time: %v\n", time.Since(start))
				break
			}
		}
		
		// Add assistant response to history  
		if assistantResponse.Len() > 0 {
			messages = append(messages, ChatMessage{
				Role:    "assistant",
				Content: assistantResponse.String(),
			})
		}
	}
	
	return nil
}

// getDefaultSystemPrompt returns the default OpenTDF-focused system prompt
func (h *Handler) getDefaultSystemPrompt() string {
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

// startJSONSession handles JSON output mode for non-interactive use
func (h *Handler) startJSONSession(modelPath string, stream bool, contextSize int, temperature float64, messages []ChatMessage) error {
	session := ChatSession{
		ModelPath: modelPath,
		Config: ChatConfig{
			Stream:      stream,
			ContextSize: contextSize,
			Temperature: temperature,
		},
		Messages: messages,
		SessionInfo: SessionInfo{
			Started:   time.Now().Format(time.RFC3339),
			Responses: 0,
		},
	}
	
	// For JSON mode, output the session configuration and exit
	// Interactive mode is not suitable for JSON output
	h.exitWithJSONFunc(session)
	return nil
}

// printHelp displays available commands
func (h *Handler) printHelp() {
	h.printlnFunc("\nAvailable commands:")
	h.printlnFunc("  exit, quit  - Exit the chat")
	h.printlnFunc("  clear       - Clear chat history")
	h.printlnFunc("  /stream     - Toggle streaming mode")
	h.printlnFunc("  /help       - Show this help")
}