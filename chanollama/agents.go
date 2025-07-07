package chanollama

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// AgentMetadata contains metadata about an agent loaded from frontmatter
type AgentMetadata struct {
	Name        string `yaml:"agent_name"`
	Description string `yaml:"description"`
	
	// Additional metadata
	FilePath     string    `yaml:"-"` // Path to the agent file
	LastModified time.Time `yaml:"-"` // When the agent was last modified
	Version      string    `yaml:"version,omitempty"`
	Tags         []string  `yaml:"tags,omitempty"`
	
	// Agent capabilities
	Capabilities []string `yaml:"capabilities,omitempty"`
	SubAgents    []string `yaml:"sub_agents,omitempty"`
}

// AgentType represents different categories of agents
type AgentType string

const (
	AgentTypeGeneral     AgentType = "general"
	AgentTypePolicy      AgentType = "policy"
	AgentTypeDevelopment AgentType = "development"
	AgentTypeDeployment  AgentType = "deployment"
	AgentTypeCustom      AgentType = "custom"
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

// NewAgent creates a new agent with default values
func NewAgent(name, description, systemPrompt string) *Agent {
	return &Agent{
		Metadata: AgentMetadata{
			Name:        name,
			Description: description,
			Version:     "1.0.0",
			Tags:        []string{},
			Capabilities: []string{},
			SubAgents:   []string{},
		},
		SystemPrompt:  systemPrompt,
		Type:          AgentTypeGeneral,
		IsActive:      false,
		LastUsed:      time.Now(),
		UsageCount:    0,
		ContextWindow: 4096,
		MaxTokens:     512,
	}
}

// WithType sets the agent type
func (a *Agent) WithType(agentType AgentType) *Agent {
	a.Type = agentType
	return a
}

// WithCapabilities sets the agent capabilities
func (a *Agent) WithCapabilities(capabilities []string) *Agent {
	a.Metadata.Capabilities = capabilities
	return a
}

// WithTags sets the agent tags
func (a *Agent) WithTags(tags []string) *Agent {
	a.Metadata.Tags = tags
	return a
}

// WithContextWindow sets the context window size
func (a *Agent) WithContextWindow(size int) *Agent {
	a.ContextWindow = size
	return a
}

// WithMaxTokens sets the maximum tokens for responses
func (a *Agent) WithMaxTokens(maxTokens int) *Agent {
	a.MaxTokens = maxTokens
	return a
}

// Activate marks the agent as active
func (a *Agent) Activate() {
	a.IsActive = true
	a.LastUsed = time.Now()
}

// Deactivate marks the agent as inactive
func (a *Agent) Deactivate() {
	a.IsActive = false
}

// Use increments the usage counter and updates last used time
func (a *Agent) Use() {
	a.UsageCount++
	a.LastUsed = time.Now()
}

// GetID returns a unique identifier for the agent
func (a *Agent) GetID() string {
	return fmt.Sprintf("%s-%s", a.Type, a.Metadata.Name)
}

// HasCapability checks if the agent has a specific capability
func (a *Agent) HasCapability(capability string) bool {
	for _, cap := range a.Metadata.Capabilities {
		if cap == capability {
			return true
		}
	}
	return false
}

// HasTag checks if the agent has a specific tag
func (a *Agent) HasTag(tag string) bool {
	for _, t := range a.Metadata.Tags {
		if t == tag {
			return true
		}
	}
	return false
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
	UserQuery           string
	ConversationHistory []Message
	SessionMetadata     map[string]interface{}
	
	// Agent selection context
	PreviousAgent   *Agent
	SuggestedAgents []*Agent
	UserPreferences map[string]interface{}
}

// NewAgentContext creates a new agent context
func NewAgentContext(userQuery string) *AgentContext {
	return &AgentContext{
		UserQuery:           userQuery,
		ConversationHistory: []Message{},
		SessionMetadata:     make(map[string]interface{}),
		SuggestedAgents:     []*Agent{},
		UserPreferences:     make(map[string]interface{}),
	}
}

// AddMessage adds a message to the conversation history
func (ac *AgentContext) AddMessage(msg Message) {
	ac.ConversationHistory = append(ac.ConversationHistory, msg)
}

// SetMetadata sets session metadata
func (ac *AgentContext) SetMetadata(key string, value interface{}) {
	ac.SessionMetadata[key] = value
}

// GetMetadata gets session metadata
func (ac *AgentContext) GetMetadata(key string) (interface{}, bool) {
	value, exists := ac.SessionMetadata[key]
	return value, exists
}

// AgentSelectionStrategy defines how agents are selected for queries
type AgentSelectionStrategy interface {
	SelectAgent(ctx *AgentContext, availableAgents []*Agent) (*Agent, float64, error)
	CalculateRelevanceScore(agent *Agent, context *AgentContext) float64
	ShouldSwitchAgent(currentAgent *Agent, context *AgentContext) (bool, *Agent, error)
}

// SimpleAgentRegistry provides a basic in-memory agent registry
type SimpleAgentRegistry struct {
	agents map[string]*Agent
	mutex  sync.RWMutex
}

// NewSimpleAgentRegistry creates a new simple agent registry
func NewSimpleAgentRegistry() *SimpleAgentRegistry {
	return &SimpleAgentRegistry{
		agents: make(map[string]*Agent),
	}
}

// LoadAgents loads agents (no-op for simple registry)
func (r *SimpleAgentRegistry) LoadAgents(ctx context.Context) error {
	// No-op for simple registry
	return nil
}

// RefreshAgents refreshes agent list (no-op for simple registry)
func (r *SimpleAgentRegistry) RefreshAgents(ctx context.Context) error {
	// No-op for simple registry
	return nil
}

// GetAgent retrieves an agent by name
func (r *SimpleAgentRegistry) GetAgent(name string) (*Agent, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	for _, agent := range r.agents {
		if agent.Metadata.Name == name {
			return agent, nil
		}
	}
	
	return nil, fmt.Errorf("agent not found: %s", name)
}

// GetAgentByType retrieves the first agent of a specific type
func (r *SimpleAgentRegistry) GetAgentByType(agentType AgentType) (*Agent, error) {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	for _, agent := range r.agents {
		if agent.Type == agentType {
			return agent, nil
		}
	}
	
	return nil, fmt.Errorf("no agent found for type: %s", agentType)
}

// ListAgents returns all registered agents
func (r *SimpleAgentRegistry) ListAgents() []*Agent {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	agents := make([]*Agent, 0, len(r.agents))
	for _, agent := range r.agents {
		agents = append(agents, agent)
	}
	
	return agents
}

// ListAgentsByType returns agents of a specific type
func (r *SimpleAgentRegistry) ListAgentsByType(agentType AgentType) []*Agent {
	r.mutex.RLock()
	defer r.mutex.RUnlock()
	
	agents := []*Agent{}
	for _, agent := range r.agents {
		if agent.Type == agentType {
			agents = append(agents, agent)
		}
	}
	
	return agents
}

// RegisterAgent registers a new agent
func (r *SimpleAgentRegistry) RegisterAgent(agent *Agent) error {
	if agent == nil {
		return fmt.Errorf("agent cannot be nil")
	}
	
	if agent.Metadata.Name == "" {
		return fmt.Errorf("agent name cannot be empty")
	}
	
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	agentID := agent.GetID()
	r.agents[agentID] = agent
	
	return nil
}

// UnregisterAgent removes an agent by name
func (r *SimpleAgentRegistry) UnregisterAgent(name string) error {
	r.mutex.Lock()
	defer r.mutex.Unlock()
	
	for id, agent := range r.agents {
		if agent.Metadata.Name == name {
			delete(r.agents, id)
			return nil
		}
	}
	
	return fmt.Errorf("agent not found: %s", name)
}

// FindBestAgent finds the best agent for a query
func (r *SimpleAgentRegistry) FindBestAgent(query string, context map[string]interface{}) (*Agent, error) {
	agents := r.ListAgents()
	if len(agents) == 0 {
		return nil, fmt.Errorf("no agents registered")
	}
	
	// Simple strategy: return the first active agent, or first agent if none active
	for _, agent := range agents {
		if agent.IsActive {
			return agent, nil
		}
	}
	
	return agents[0], nil
}

// SuggestAgent suggests an agent based on user input
func (r *SimpleAgentRegistry) SuggestAgent(userInput string) (*Agent, error) {
	// Simple implementation - could be enhanced with ML/keyword matching
	return r.FindBestAgent(userInput, nil)
}

// SimpleSelectionStrategy provides a basic agent selection strategy
type SimpleSelectionStrategy struct{}

// NewSimpleSelectionStrategy creates a new simple selection strategy
func NewSimpleSelectionStrategy() *SimpleSelectionStrategy {
	return &SimpleSelectionStrategy{}
}

// SelectAgent selects the best agent for the context
func (s *SimpleSelectionStrategy) SelectAgent(ctx *AgentContext, availableAgents []*Agent) (*Agent, float64, error) {
	if len(availableAgents) == 0 {
		return nil, 0, fmt.Errorf("no agents available")
	}
	
	bestAgent := availableAgents[0]
	bestScore := s.CalculateRelevanceScore(bestAgent, ctx)
	
	for _, agent := range availableAgents[1:] {
		score := s.CalculateRelevanceScore(agent, ctx)
		if score > bestScore {
			bestAgent = agent
			bestScore = score
		}
	}
	
	return bestAgent, bestScore, nil
}

// CalculateRelevanceScore calculates how relevant an agent is to the context
func (s *SimpleSelectionStrategy) CalculateRelevanceScore(agent *Agent, context *AgentContext) float64 {
	score := 0.0
	
	// Base score for active agents
	if agent.IsActive {
		score += 0.3
	}
	
	// Score based on usage frequency
	if agent.UsageCount > 0 {
		score += 0.2
	}
	
	// Score based on recent usage
	if time.Since(agent.LastUsed) < 24*time.Hour {
		score += 0.1
	}
	
	// Score based on agent type preference (simple keyword matching)
	query := strings.ToLower(context.UserQuery)
	switch agent.Type {
	case AgentTypePolicy:
		if strings.Contains(query, "policy") || strings.Contains(query, "permission") {
			score += 0.4
		}
	case AgentTypeDevelopment:
		if strings.Contains(query, "code") || strings.Contains(query, "develop") || strings.Contains(query, "api") {
			score += 0.4
		}
	case AgentTypeDeployment:
		if strings.Contains(query, "deploy") || strings.Contains(query, "install") || strings.Contains(query, "setup") {
			score += 0.4
		}
	case AgentTypeGeneral:
		score += 0.2 // General agents are always somewhat relevant
	}
	
	return score
}

// ShouldSwitchAgent determines if the current agent should be switched
func (s *SimpleSelectionStrategy) ShouldSwitchAgent(currentAgent *Agent, context *AgentContext) (bool, *Agent, error) {
	// Simple strategy: don't switch unless there's a compelling reason
	// This could be enhanced with more sophisticated logic
	return false, nil, nil
}

// AgentManager orchestrates agent lifecycle and selection
type AgentManager struct {
	registry           AgentRegistry
	loader             AgentLoader
	selectionStrategy  AgentSelectionStrategy
	
	// Configuration
	agentDirectory     string
	autoRefresh        bool
	refreshInterval    time.Duration
	
	// Runtime state
	currentAgent       *Agent
	sessionContext     *AgentContext
	agentSwitchCount   int
	
	// Event handling
	eventHandlers      []AgentEventHandler
	
	mutex              sync.RWMutex
}

// NewAgentManager creates a new agent manager
func NewAgentManager() *AgentManager {
	return &AgentManager{
		registry:          NewSimpleAgentRegistry(),
		selectionStrategy: NewSimpleSelectionStrategy(),
		autoRefresh:       false,
		refreshInterval:   5 * time.Minute,
		eventHandlers:     []AgentEventHandler{},
	}
}

// WithRegistry sets the agent registry
func (am *AgentManager) WithRegistry(registry AgentRegistry) *AgentManager {
	am.registry = registry
	return am
}

// WithLoader sets the agent loader
func (am *AgentManager) WithLoader(loader AgentLoader) *AgentManager {
	am.loader = loader
	return am
}

// WithSelectionStrategy sets the agent selection strategy
func (am *AgentManager) WithSelectionStrategy(strategy AgentSelectionStrategy) *AgentManager {
	am.selectionStrategy = strategy
	return am
}

// WithAgentDirectory sets the agent directory path
func (am *AgentManager) WithAgentDirectory(dir string) *AgentManager {
	am.agentDirectory = dir
	return am
}

// WithAutoRefresh enables/disables auto-refresh of agents
func (am *AgentManager) WithAutoRefresh(enabled bool, interval time.Duration) *AgentManager {
	am.autoRefresh = enabled
	am.refreshInterval = interval
	return am
}

// AddEventHandler adds an agent event handler
func (am *AgentManager) AddEventHandler(handler AgentEventHandler) {
	am.mutex.Lock()
	defer am.mutex.Unlock()
	am.eventHandlers = append(am.eventHandlers, handler)
}

// LoadAgents loads agents using the configured loader and registry
func (am *AgentManager) LoadAgents(ctx context.Context) error {
	if am.loader != nil && am.agentDirectory != "" {
		agents, err := am.loader.LoadFromDirectory(am.agentDirectory)
		if err != nil {
			return fmt.Errorf("failed to load agents from directory: %w", err)
		}
		
		for _, agent := range agents {
			if err := am.registry.RegisterAgent(agent); err != nil {
				return fmt.Errorf("failed to register agent %s: %w", agent.Metadata.Name, err)
			}
			
			// Notify event handlers
			for _, handler := range am.eventHandlers {
				handler.OnAgentLoaded(agent)
			}
		}
	}
	
	return am.registry.LoadAgents(ctx)
}

// RegisterAgent registers a new agent
func (am *AgentManager) RegisterAgent(agent *Agent) error {
	err := am.registry.RegisterAgent(agent)
	if err == nil {
		// Notify event handlers
		for _, handler := range am.eventHandlers {
			handler.OnAgentLoaded(agent)
		}
	}
	return err
}

// GetCurrentAgent returns the currently selected agent
func (am *AgentManager) GetCurrentAgent() *Agent {
	am.mutex.RLock()
	defer am.mutex.RUnlock()
	return am.currentAgent
}

// SelectAgent selects the best agent for the given context
func (am *AgentManager) SelectAgent(context *AgentContext) (*Agent, error) {
	availableAgents := am.registry.ListAgents()
	if len(availableAgents) == 0 {
		return nil, fmt.Errorf("no agents available")
	}
	
	agent, confidence, err := am.selectionStrategy.SelectAgent(context, availableAgents)
	if err != nil {
		return nil, err
	}
	
	// Update current agent and context
	am.mutex.Lock()
	previousAgent := am.currentAgent
	am.currentAgent = agent
	am.sessionContext = context
	am.mutex.Unlock()
	
	// Activate the selected agent
	agent.Activate()
	agent.Use()
	
	// Notify event handlers
	for _, handler := range am.eventHandlers {
		handler.OnAgentSelected(agent, context)
		if previousAgent != nil && previousAgent != agent {
			am.agentSwitchCount++
			handler.OnAgentSwitched(previousAgent, agent, fmt.Sprintf("Automatic selection (confidence: %.2f)", confidence))
		}
	}
	
	return agent, nil
}

// ProcessWithAgent processes a query using a specific agent
func (am *AgentManager) ProcessWithAgent(agent *Agent, messages []Message, engine Engine) (*Response, error) {
	if agent == nil {
		return nil, fmt.Errorf("agent cannot be nil")
	}
	
	// Prepare messages with agent's system prompt
	agentMessages := []Message{
		NewSystemMessage(agent.SystemPrompt),
	}
	agentMessages = append(agentMessages, messages...)
	
	// Process with the engine
	response, err := engine.Chat(agentMessages)
	if err != nil {
		// Notify event handlers of error
		for _, handler := range am.eventHandlers {
			handler.OnAgentError(agent, err)
		}
		return nil, err
	}
	
	// Update agent usage
	agent.Use()
	
	return response, nil
}

// ListAgents returns all available agents
func (am *AgentManager) ListAgents() []*Agent {
	return am.registry.ListAgents()
}

// GetAgent retrieves an agent by name
func (am *AgentManager) GetAgent(name string) (*Agent, error) {
	return am.registry.GetAgent(name)
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

// SimpleEventHandler provides a basic implementation of AgentEventHandler
type SimpleEventHandler struct{}

// NewSimpleEventHandler creates a new simple event handler
func NewSimpleEventHandler() *SimpleEventHandler {
	return &SimpleEventHandler{}
}

// OnAgentLoaded handles agent loaded events
func (h *SimpleEventHandler) OnAgentLoaded(agent *Agent) {
	// Default implementation - could be enhanced with logging
}

// OnAgentUnloaded handles agent unloaded events
func (h *SimpleEventHandler) OnAgentUnloaded(agent *Agent) {
	// Default implementation - could be enhanced with logging
}

// OnAgentSelected handles agent selected events
func (h *SimpleEventHandler) OnAgentSelected(agent *Agent, context *AgentContext) {
	// Default implementation - could be enhanced with logging
}

// OnAgentSwitched handles agent switched events
func (h *SimpleEventHandler) OnAgentSwitched(from *Agent, to *Agent, reason string) {
	// Default implementation - could be enhanced with logging
}

// OnAgentError handles agent error events
func (h *SimpleEventHandler) OnAgentError(agent *Agent, err error) {
	// Default implementation - could be enhanced with logging
}