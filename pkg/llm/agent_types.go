package llm

import (
	"context"
	"time"
)

// AgentMetadata contains metadata about an agent loaded from frontmatter
type AgentMetadata struct {
	Name        string `yaml:"agent_name"`
	Description string `yaml:"description"`
	
	// Additional metadata
	FilePath    string    `yaml:"-"` // Path to the agent file
	LastModified time.Time `yaml:"-"` // When the agent was last modified
	Version     string    `yaml:"version,omitempty"`
	Tags        []string  `yaml:"tags,omitempty"`
	
	// Agent capabilities
	Capabilities []string `yaml:"capabilities,omitempty"`
	SubAgents    []string `yaml:"sub_agents,omitempty"`
}

// AgentType represents different categories of agents
type AgentType string

const (
	AgentTypeGeneral    AgentType = "general"
	AgentTypePolicy     AgentType = "policy"
	AgentTypeDevelopment AgentType = "development"
	AgentTypeDeployment AgentType = "deployment"
)

// Agent represents a specialized AI agent with specific knowledge and capabilities
type Agent struct {
	Metadata      AgentMetadata
	SystemPrompt  string
	Type          AgentType
	
	// Runtime properties
	IsActive      bool
	LastUsed      time.Time
	UsageCount    int64
	
	// Context management
	ContextWindow int
	MaxTokens     int
}

// AgentRegistry manages available agents and their lifecycle
type AgentRegistry interface {
	// Agent discovery and loading
	LoadAgents(ctx context.Context) error
	RefreshAgents(ctx context.Context) error
	
	// Agent retrieval
	GetAgent(name string) (*Agent, error)
	GetAgentByType(agentType AgentType) (*Agent, error)
	ListAgents() []*Agent
	ListAgentsByType(agentType AgentType) []*Agent
	
	// Agent management
	RegisterAgent(agent *Agent) error
	UnregisterAgent(name string) error
	
	// Agent matching
	FindBestAgent(query string, context map[string]interface{}) (*Agent, error)
	SuggestAgent(userInput string) (*Agent, error)
}

// AgentLoader handles loading agents from various sources
type AgentLoader interface {
	LoadFromDirectory(dirPath string) ([]*Agent, error)
	LoadFromFile(filePath string) (*Agent, error)
	ValidateAgent(agent *Agent) error
	WatchForChanges(ctx context.Context, callback func(*Agent)) error
}

// AgentContext represents the context in which an agent operates
type AgentContext struct {
	UserQuery     string
	ConversationHistory []ChatMessage
	SessionMetadata     map[string]interface{}
	
	// OpenTDF specific context
	CurrentProfile  string
	PolicyContext   map[string]interface{}
	ResourceContext map[string]interface{}
	
	// Agent selection context
	PreviousAgent   *Agent
	SuggestedAgents []*Agent
	UserPreferences map[string]interface{}
}

// AgentSelectionStrategy defines how agents are selected for queries
type AgentSelectionStrategy interface {
	SelectAgent(ctx *AgentContext, availableAgents []*Agent) (*Agent, float64, error)
	CalculateRelevanceScore(agent *Agent, context *AgentContext) float64
	ShouldSwitchAgent(currentAgent *Agent, context *AgentContext) (bool, *Agent, error)
}

// AgentManager orchestrates agent lifecycle and selection
type AgentManager struct {
	registry           AgentRegistry
	loader            AgentLoader
	selectionStrategy AgentSelectionStrategy
	resourceManager   *ResourceManager
	
	// Configuration
	agentDirectory    string
	autoRefresh       bool
	refreshInterval   time.Duration
	
	// Runtime state
	currentAgent      *Agent
	sessionContext    *AgentContext
	agentSwitchCount  int
}

// AgentSwitchEvent represents when the system switches between agents
type AgentSwitchEvent struct {
	Timestamp     time.Time
	FromAgent     *Agent
	ToAgent       *Agent
	Reason        string
	UserQuery     string
	Confidence    float64
	WasAutomatic  bool
}

// AgentResponse represents a response from an agent with metadata
type AgentResponse struct {
	Content         string
	Agent          *Agent
	Confidence     float64
	ProcessingTime time.Duration
	TokensUsed     int
	
	// Agent suggestions
	SuggestedAgents   []*Agent
	SuggestSwitch     bool
	SwitchReason      string
	
	// Context updates
	UpdatedContext    map[string]interface{}
	LearningInsights  []string
}

// AgentEvent represents various agent lifecycle events
type AgentEvent struct {
	Type      string
	Agent     *Agent
	Timestamp time.Time
	Data      map[string]interface{}
}

// AgentEventHandler handles agent events
type AgentEventHandler interface {
	OnAgentLoaded(agent *Agent)
	OnAgentUnloaded(agent *Agent)
	OnAgentSelected(agent *Agent, context *AgentContext)
	OnAgentSwitched(from *Agent, to *Agent, reason string)
	OnAgentError(agent *Agent, err error)
}