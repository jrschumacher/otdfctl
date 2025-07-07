package llm

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/opentdf/otdfctl/pkg/config"
)

// AgentChatCommand handles the agent-based chat command
type AgentChatCommand struct {
	config       *config.Config
	agentManager *AgentManager
	chatEngine   *SimpleChatEngine
}

// NewAgentChatCommand creates a new agent chat command
func NewAgentChatCommand(cfg *config.Config) *AgentChatCommand {
	return &AgentChatCommand{
		config: cfg,
	}
}

// RunAgentChat starts the agent-based TUI chat interface
func (acc *AgentChatCommand) RunAgentChat(ctx context.Context, modelPath string, agentDirectory string) error {
	// Initialize chat engine
	if modelPath == "" {
		pathResolver := NewModelPathResolver(acc.config)
		resolvedPath, err := pathResolver.ResolveChatModelPath("")
		if err != nil {
			return fmt.Errorf("model path required: %w", err)
		}
		modelPath = resolvedPath
	}
	
	// Create chat engine
	acc.chatEngine = NewSimpleChatEngineWithConfig(modelPath, acc.config)
	if err := acc.chatEngine.Start(); err != nil {
		return fmt.Errorf("failed to start chat engine: %w", err)
	}
	defer acc.chatEngine.Stop()
	
	// Set default agent directory if not provided
	if agentDirectory == "" {
		// Default to pkg/llm/agents relative to current directory
		agentDirectory = filepath.Join("pkg", "llm", "agents")
	}
	
	// Initialize agent manager
	acc.agentManager = NewAgentManager(agentDirectory, acc.config)
	if err := acc.agentManager.Initialize(ctx); err != nil {
		return fmt.Errorf("failed to initialize agent manager: %w", err)
	}
	
	// Check if we have agents loaded
	agents := acc.agentManager.ListAgents()
	if len(agents) == 0 {
		log.Printf("Warning: No agents found in directory: %s", agentDirectory)
		log.Printf("The chat will work with a default system prompt")
	} else {
		log.Printf("Loaded %d agents from %s", len(agents), agentDirectory)
	}
	
	// Create simple chat interface
	chatInterface := NewSimpleChatInterface(acc.agentManager, acc.chatEngine, acc.config)
	
	// Run the simple interface
	err := chatInterface.Run()
	if err != nil {
		return fmt.Errorf("chat interface error: %w", err)
	}
	
	return nil
}

// RunAgentChatV2 starts the agent-based chat interface using the new V2 engine architecture
func (acc *AgentChatCommand) RunAgentChatV2(ctx context.Context, modelPath string, agentDirectory string) error {
	// Resolve model path
	if modelPath == "" {
		pathResolver := NewModelPathResolver(acc.config)
		resolvedPath, err := pathResolver.ResolveChatModelPath("")
		if err != nil {
			return fmt.Errorf("model path required: %w", err)
		}
		modelPath = resolvedPath
	}
	
	// Create and start the V2 engine
	engine := NewLLMEngineV2(acc.config)
	if err := engine.Start(ctx); err != nil {
		return fmt.Errorf("failed to start LLM engine V2: %w", err)
	}
	defer engine.Stop()
	
	// Create engine client
	engineClient := NewEngineClient(engine)
	defer engineClient.Close()
	
	// Load the model
	loadPayload := LoadModelPayload{
		ModelPath:        modelPath,
		SystemPrompt:     "You are a helpful assistant.", // Default prompt, will be overridden per agent
		ContextSize:      4096,
		EnableRAG:        false, // Can be enabled per request
		GenerationParams: DefaultGenerationParams(),
	}
	
	loadCtx, loadCancel := context.WithTimeout(ctx, 30*time.Second)
	err := engineClient.LoadModel(loadCtx, loadPayload)
	loadCancel()
	
	if err != nil {
		log.Printf("Warning: Failed to load model: %v", err)
		log.Printf("Continuing anyway - engine will handle model loading errors gracefully")
	} else {
		log.Printf("Model loaded successfully: %s", modelPath)
	}
	
	// Set default agent directory if not provided
	if agentDirectory == "" {
		agentDirectory = filepath.Join("pkg", "llm", "agents")
	}
	
	// Initialize agent manager
	acc.agentManager = NewAgentManager(agentDirectory, acc.config)
	if err := acc.agentManager.Initialize(ctx); err != nil {
		return fmt.Errorf("failed to initialize agent manager: %w", err)
	}
	
	// Check if we have agents loaded
	agents := acc.agentManager.ListAgents()
	if len(agents) == 0 {
		log.Printf("Warning: No agents found in directory: %s", agentDirectory)
	} else {
		log.Printf("Loaded %d agents from %s", len(agents), agentDirectory)
		for _, agent := range agents {
			log.Printf("  - %s (%s): %s", agent.Metadata.Name, agent.Type, agent.Metadata.Description)
		}
	}
	
	// Display startup information
	log.Printf("🚀 Starting OpenTDF Agent Chat V2...")
	log.Printf("📁 Agent Directory: %s", agentDirectory)
	log.Printf("🤖 Model: %s", modelPath)
	log.Printf("🔧 Engine: V2 (Decoupled Architecture)")
	
	// Create V2 chat interface
	chatInterface := NewSimpleChatInterfaceV2(acc.agentManager, engineClient, acc.config)
	
	// Run the V2 interface
	err = chatInterface.Run()
	if err != nil {
		return fmt.Errorf("chat interface V2 error: %w", err)
	}
	
	log.Printf("👋 Thanks for using OpenTDF Agent Chat V2!")
	return nil
}

// RunAgentChatWithOptions provides more configuration options
type AgentChatOptions struct {
	ModelPath       string
	AgentDirectory  string
	EnableRAG       bool
	RAGIndexPath    string
	SystemPrompt    string
	ContextSize     int
	Temperature     float64
	MaxTokens       int
}

// RunAgentChatWithOptions starts the agent chat with detailed configuration
func (acc *AgentChatCommand) RunAgentChatWithOptions(ctx context.Context, opts AgentChatOptions) error {
	// Resolve model path
	modelPath := opts.ModelPath
	if modelPath == "" {
		pathResolver := NewModelPathResolver(acc.config)
		resolvedPath, err := pathResolver.ResolveChatModelPath("")
		if err != nil {
			return fmt.Errorf("model path required: %w", err)
		}
		modelPath = resolvedPath
	}
	
	// Create chat engine with configuration
	acc.chatEngine = NewSimpleChatEngineWithConfig(modelPath, acc.config)
	
	// Configure RAG if enabled
	if opts.EnableRAG && opts.RAGIndexPath != "" {
		log.Printf("Initializing RAG support with index: %s", opts.RAGIndexPath)
		ragStore := NewSimpleRAGStore(opts.RAGIndexPath)
		if err := ragStore.LoadIndex(); err != nil {
			log.Printf("Warning: Failed to load RAG index: %v", err)
		} else {
			acc.chatEngine.EnableSimpleRAG(ragStore)
			log.Printf("RAG enabled with %d documents", ragStore.GetDocumentCount())
		}
	}
	
	// Start chat engine
	if err := acc.chatEngine.Start(); err != nil {
		return fmt.Errorf("failed to start chat engine: %w", err)
	}
	defer acc.chatEngine.Stop()
	
	// Set default agent directory
	agentDirectory := opts.AgentDirectory
	if agentDirectory == "" {
		agentDirectory = filepath.Join("pkg", "llm", "agents")
	}
	
	// Initialize agent manager
	acc.agentManager = NewAgentManager(agentDirectory, acc.config)
	if err := acc.agentManager.Initialize(ctx); err != nil {
		return fmt.Errorf("failed to initialize agent manager: %w", err)
	}
	
	// Validate agent loading
	agents := acc.agentManager.ListAgents()
	if len(agents) == 0 {
		log.Printf("Warning: No agents found in directory: %s", agentDirectory)
	} else {
		log.Printf("Successfully loaded %d agents:", len(agents))
		for _, agent := range agents {
			log.Printf("  - %s (%s): %s", agent.Metadata.Name, agent.Type, agent.Metadata.Description)
		}
	}
	
	// Display startup information
	log.Printf("🚀 Starting OpenTDF Agent Chat...")
	log.Printf("📁 Agent Directory: %s", agentDirectory)
	log.Printf("🤖 Model: %s", modelPath)
	if opts.EnableRAG {
		log.Printf("📚 RAG: Enabled")
	}
	
	// Create enhanced Bubbletea TUI interface
	tuiModel := NewAgentChatModel(acc.agentManager, acc.chatEngine)
	
	// Run the enhanced TUI
	err := RunAgentChatTUI(tuiModel)
	if err != nil {
		return fmt.Errorf("TUI interface error: %w", err)
	}
	
	log.Printf("👋 Thanks for using OpenTDF Agent Chat!")
	return nil
}

// ValidateAgentDirectory checks if the agent directory exists and contains agents
func (acc *AgentChatCommand) ValidateAgentDirectory(agentDirectory string) error {
	loader := NewFileSystemAgentLoader()
	
	// Try to load agents
	agents, err := loader.LoadFromDirectory(agentDirectory)
	if err != nil {
		return fmt.Errorf("failed to load agents from %s: %w", agentDirectory, err)
	}
	
	if len(agents) == 0 {
		return fmt.Errorf("no valid agents found in directory: %s", agentDirectory)
	}
	
	return nil
}

// ListAvailableAgents shows agents that would be loaded
func (acc *AgentChatCommand) ListAvailableAgents(agentDirectory string) ([]AgentInfo, error) {
	loader := NewFileSystemAgentLoader()
	
	agents, err := loader.LoadFromDirectory(agentDirectory)
	if err != nil {
		return nil, fmt.Errorf("failed to load agents: %w", err)
	}
	
	agentInfos := make([]AgentInfo, len(agents))
	for i, agent := range agents {
		agentInfos[i] = AgentInfo{
			Name:        agent.Metadata.Name,
			Type:        string(agent.Type),
			Description: agent.Metadata.Description,
			FilePath:    agent.Metadata.FilePath,
			Tags:        agent.Metadata.Tags,
			Capabilities: agent.Metadata.Capabilities,
		}
	}
	
	return agentInfos, nil
}

// AgentInfo provides agent information for display
type AgentInfo struct {
	Name         string   `json:"name"`
	Type         string   `json:"type"`
	Description  string   `json:"description"`
	FilePath     string   `json:"file_path"`
	Tags         []string `json:"tags,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
}

// GetAgentManagerStats returns statistics about the agent system
func (acc *AgentChatCommand) GetAgentManagerStats() map[string]interface{} {
	if acc.agentManager == nil {
		return map[string]interface{}{
			"error": "agent manager not initialized",
		}
	}
	
	return acc.agentManager.GetAgentStats()
}

// RefreshAgents reloads agents from the directory
func (acc *AgentChatCommand) RefreshAgents(ctx context.Context) error {
	if acc.agentManager == nil {
		return fmt.Errorf("agent manager not initialized")
	}
	
	return acc.agentManager.RefreshAgents(ctx)
}