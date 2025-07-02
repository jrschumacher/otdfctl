package llm

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// InMemoryAgentRegistry implements AgentRegistry using in-memory storage
type InMemoryAgentRegistry struct {
	agents          map[string]*Agent // agents by name
	agentsByType    map[AgentType][]*Agent
	loader          AgentLoader
	agentDirectory  string
	
	// Runtime state
	mu              sync.RWMutex
	lastRefresh     time.Time
	refreshInterval time.Duration
	autoRefresh     bool
	
	// Event handling
	eventHandlers   []AgentEventHandler
}

// NewInMemoryAgentRegistry creates a new in-memory agent registry
func NewInMemoryAgentRegistry(loader AgentLoader) *InMemoryAgentRegistry {
	return &InMemoryAgentRegistry{
		agents:          make(map[string]*Agent),
		agentsByType:    make(map[AgentType][]*Agent),
		loader:          loader,
		refreshInterval: 5 * time.Minute,
		autoRefresh:     false,
		eventHandlers:   make([]AgentEventHandler, 0),
	}
}

// WithDirectory sets the agent directory path
func (r *InMemoryAgentRegistry) WithDirectory(dirPath string) *InMemoryAgentRegistry {
	r.agentDirectory = dirPath
	return r
}

// WithAutoRefresh enables automatic agent refreshing
func (r *InMemoryAgentRegistry) WithAutoRefresh(interval time.Duration) *InMemoryAgentRegistry {
	r.autoRefresh = true
	r.refreshInterval = interval
	return r
}

// WithEventHandler adds an event handler
func (r *InMemoryAgentRegistry) WithEventHandler(handler AgentEventHandler) *InMemoryAgentRegistry {
	r.eventHandlers = append(r.eventHandlers, handler)
	return r
}

// LoadAgents loads agents from the configured directory
func (r *InMemoryAgentRegistry) LoadAgents(ctx context.Context) error {
	if r.agentDirectory == "" {
		return fmt.Errorf("agent directory not configured")
	}
	
	if r.loader == nil {
		return fmt.Errorf("agent loader not configured")
	}
	
	// Load agents from directory
	agents, err := r.loader.LoadFromDirectory(r.agentDirectory)
	if err != nil {
		return fmt.Errorf("failed to load agents: %w", err)
	}
	
	r.mu.Lock()
	defer r.mu.Unlock()
	
	// Clear existing agents
	r.agents = make(map[string]*Agent)
	r.agentsByType = make(map[AgentType][]*Agent)
	
	// Register all loaded agents
	for _, agent := range agents {
		if err := r.registerAgentUnsafe(agent); err != nil {
			// Log warning but continue
			fmt.Printf("Warning: failed to register agent %s: %v\n", agent.Metadata.Name, err)
			continue
		}
		
		// Notify event handlers
		r.notifyAgentLoaded(agent)
	}
	
	r.lastRefresh = time.Now()
	
	// Start auto-refresh if enabled
	if r.autoRefresh {
		go r.autoRefreshLoop(ctx)
	}
	
	return nil
}

// RefreshAgents reloads agents from the directory
func (r *InMemoryAgentRegistry) RefreshAgents(ctx context.Context) error {
	return r.LoadAgents(ctx)
}

// GetAgent retrieves an agent by name
func (r *InMemoryAgentRegistry) GetAgent(name string) (*Agent, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	agent, exists := r.agents[name]
	if !exists {
		return nil, fmt.Errorf("agent not found: %s", name)
	}
	
	// Update usage statistics
	agent.LastUsed = time.Now()
	agent.UsageCount++
	
	return agent, nil
}

// GetAgentByType retrieves the first agent of a specific type
func (r *InMemoryAgentRegistry) GetAgentByType(agentType AgentType) (*Agent, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	agents, exists := r.agentsByType[agentType]
	if !exists || len(agents) == 0 {
		return nil, fmt.Errorf("no agents found for type: %s", agentType)
	}
	
	// Return the most recently used agent of this type
	agent := agents[0]
	for _, a := range agents {
		if a.LastUsed.After(agent.LastUsed) {
			agent = a
		}
	}
	
	// Update usage statistics
	agent.LastUsed = time.Now()
	agent.UsageCount++
	
	return agent, nil
}

// ListAgents returns all registered agents
func (r *InMemoryAgentRegistry) ListAgents() []*Agent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	agents := make([]*Agent, 0, len(r.agents))
	for _, agent := range r.agents {
		agents = append(agents, agent)
	}
	
	// Sort by usage count (most used first)
	sort.Slice(agents, func(i, j int) bool {
		return agents[i].UsageCount > agents[j].UsageCount
	})
	
	return agents
}

// ListAgentsByType returns all agents of a specific type
func (r *InMemoryAgentRegistry) ListAgentsByType(agentType AgentType) []*Agent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	agents, exists := r.agentsByType[agentType]
	if !exists {
		return []*Agent{}
	}
	
	// Return a copy to prevent external modification
	result := make([]*Agent, len(agents))
	copy(result, agents)
	
	return result
}

// RegisterAgent registers a new agent
func (r *InMemoryAgentRegistry) RegisterAgent(agent *Agent) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	return r.registerAgentUnsafe(agent)
}

// registerAgentUnsafe registers an agent without locking (internal use)
func (r *InMemoryAgentRegistry) registerAgentUnsafe(agent *Agent) error {
	if agent == nil {
		return fmt.Errorf("agent cannot be nil")
	}
	
	if agent.Metadata.Name == "" {
		return fmt.Errorf("agent name cannot be empty")
	}
	
	// Check for duplicate names
	if _, exists := r.agents[agent.Metadata.Name]; exists {
		return fmt.Errorf("agent with name %s already exists", agent.Metadata.Name)
	}
	
	// Register agent
	r.agents[agent.Metadata.Name] = agent
	
	// Add to type index
	if r.agentsByType[agent.Type] == nil {
		r.agentsByType[agent.Type] = make([]*Agent, 0)
	}
	r.agentsByType[agent.Type] = append(r.agentsByType[agent.Type], agent)
	
	return nil
}

// UnregisterAgent removes an agent from the registry
func (r *InMemoryAgentRegistry) UnregisterAgent(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	agent, exists := r.agents[name]
	if !exists {
		return fmt.Errorf("agent not found: %s", name)
	}
	
	// Remove from main index
	delete(r.agents, name)
	
	// Remove from type index
	typeAgents := r.agentsByType[agent.Type]
	for i, a := range typeAgents {
		if a.Metadata.Name == name {
			r.agentsByType[agent.Type] = append(typeAgents[:i], typeAgents[i+1:]...)
			break
		}
	}
	
	// Notify event handlers
	r.notifyAgentUnloaded(agent)
	
	return nil
}

// FindBestAgent finds the best agent for a query using simple keyword matching
func (r *InMemoryAgentRegistry) FindBestAgent(query string, context map[string]interface{}) (*Agent, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	if len(r.agents) == 0 {
		return nil, fmt.Errorf("no agents available")
	}
	
	queryLower := strings.ToLower(query)
	
	// Score agents based on keyword matching
	type agentScore struct {
		agent *Agent
		score float64
	}
	
	scores := make([]agentScore, 0, len(r.agents))
	
	for _, agent := range r.agents {
		score := r.calculateRelevanceScore(agent, queryLower, context)
		scores = append(scores, agentScore{agent: agent, score: score})
	}
	
	// Sort by score (highest first)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	
	if len(scores) == 0 || scores[0].score == 0 {
		// Return general agent as fallback
		for _, agent := range r.agents {
			if agent.Type == AgentTypeGeneral {
				return agent, nil
			}
		}
		// If no general agent, return the first available
		return scores[0].agent, nil
	}
	
	return scores[0].agent, nil
}

// calculateRelevanceScore calculates how relevant an agent is to a query
func (r *InMemoryAgentRegistry) calculateRelevanceScore(agent *Agent, queryLower string, context map[string]interface{}) float64 {
	score := 0.0
	
	// Agent type keywords
	typeKeywords := map[AgentType][]string{
		AgentTypePolicy:     {"policy", "attribute", "abac", "permission", "access", "control", "mapping", "namespace"},
		AgentTypeDevelopment: {"code", "sdk", "api", "development", "programming", "integration", "java", "go", "javascript", "python"},
		AgentTypeDeployment: {"install", "deploy", "setup", "configure", "docker", "kubernetes", "infrastructure", "ops"},
		AgentTypeGeneral:    {"what", "explain", "help", "overview", "concept", "tdf", "opentdf"},
	}
	
	// Check type-specific keywords
	if keywords, exists := typeKeywords[agent.Type]; exists {
		for _, keyword := range keywords {
			if strings.Contains(queryLower, keyword) {
				score += 10.0
			}
		}
	}
	
	// Check agent description
	descLower := strings.ToLower(agent.Metadata.Description)
	for _, word := range strings.Fields(queryLower) {
		if len(word) > 3 && strings.Contains(descLower, word) {
			score += 5.0
		}
	}
	
	// Check agent tags/capabilities
	for _, tag := range agent.Metadata.Tags {
		if strings.Contains(queryLower, strings.ToLower(tag)) {
			score += 8.0
		}
	}
	
	for _, capability := range agent.Metadata.Capabilities {
		if strings.Contains(queryLower, strings.ToLower(capability)) {
			score += 8.0
		}
	}
	
	// Boost score based on recent usage
	if !agent.LastUsed.IsZero() {
		hoursSinceLastUse := time.Since(agent.LastUsed).Hours()
		if hoursSinceLastUse < 1 {
			score += 2.0
		} else if hoursSinceLastUse < 24 {
			score += 1.0
		}
	}
	
	// Usage count bonus
	if agent.UsageCount > 0 {
		score += float64(agent.UsageCount) * 0.1
	}
	
	return score
}

// SuggestAgent suggests an agent based on user input patterns
func (r *InMemoryAgentRegistry) SuggestAgent(userInput string) (*Agent, error) {
	// Use the same logic as FindBestAgent for now
	return r.FindBestAgent(userInput, nil)
}

// autoRefreshLoop runs automatic agent refreshing
func (r *InMemoryAgentRegistry) autoRefreshLoop(ctx context.Context) {
	ticker := time.NewTicker(r.refreshInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := r.RefreshAgents(ctx); err != nil {
				fmt.Printf("Warning: auto-refresh failed: %v\n", err)
			}
		}
	}
}

// Event notification methods
func (r *InMemoryAgentRegistry) notifyAgentLoaded(agent *Agent) {
	for _, handler := range r.eventHandlers {
		handler.OnAgentLoaded(agent)
	}
}

func (r *InMemoryAgentRegistry) notifyAgentUnloaded(agent *Agent) {
	for _, handler := range r.eventHandlers {
		handler.OnAgentUnloaded(agent)
	}
}

func (r *InMemoryAgentRegistry) notifyAgentSelected(agent *Agent, context *AgentContext) {
	for _, handler := range r.eventHandlers {
		handler.OnAgentSelected(agent, context)
	}
}

func (r *InMemoryAgentRegistry) notifyAgentSwitched(from *Agent, to *Agent, reason string) {
	for _, handler := range r.eventHandlers {
		handler.OnAgentSwitched(from, to, reason)
	}
}

func (r *InMemoryAgentRegistry) notifyAgentError(agent *Agent, err error) {
	for _, handler := range r.eventHandlers {
		handler.OnAgentError(agent, err)
	}
}

// GetRegistryStats returns statistics about the registry
func (r *InMemoryAgentRegistry) GetRegistryStats() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	stats := map[string]interface{}{
		"total_agents":   len(r.agents),
		"last_refresh":   r.lastRefresh,
		"auto_refresh":   r.autoRefresh,
		"agents_by_type": make(map[string]int),
	}
	
	for agentType, agents := range r.agentsByType {
		stats["agents_by_type"].(map[string]int)[string(agentType)] = len(agents)
	}
	
	return stats
}