package llm

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/opentdf/otdfctl/pkg/config"
)

// DefaultAgentSelectionStrategy implements basic agent selection logic
type DefaultAgentSelectionStrategy struct {
	switchThreshold float64
	confidence      float64
}

// NewDefaultAgentSelectionStrategy creates a new default selection strategy
func NewDefaultAgentSelectionStrategy() *DefaultAgentSelectionStrategy {
	return &DefaultAgentSelectionStrategy{
		switchThreshold: 7.0,  // Switch if new agent score is 7+ points higher
		confidence:      5.0,  // Minimum confidence for agent selection
	}
}

// SelectAgent selects the best agent for the given context
func (s *DefaultAgentSelectionStrategy) SelectAgent(ctx *AgentContext, availableAgents []*Agent) (*Agent, float64, error) {
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
	
	if bestScore < s.confidence {
		// Return general agent as fallback
		for _, agent := range availableAgents {
			if agent.Type == AgentTypeGeneral {
				return agent, s.confidence, nil
			}
		}
	}
	
	return bestAgent, bestScore, nil
}

// CalculateRelevanceScore calculates an agent's relevance to the context
func (s *DefaultAgentSelectionStrategy) CalculateRelevanceScore(agent *Agent, context *AgentContext) float64 {
	score := 0.0
	queryLower := strings.ToLower(context.UserQuery)
	
	// Type-based scoring
	typeKeywords := map[AgentType][]string{
		AgentTypePolicy:     {"policy", "policies", "attribute", "attributes", "abac", "permission", "permissions", "access", "control", "mapping", "namespace", "subject", "resource", "manage", "management", "create", "delete", "update", "authorization", "rbac", "security", "govern", "governance"},
		AgentTypeDevelopment: {"code", "sdk", "api", "development", "programming", "integration", "java", "go", "javascript", "python", "encrypt", "decrypt", "implement", "build", "develop"},
		AgentTypeDeployment: {"install", "deploy", "setup", "configure", "docker", "kubernetes", "infrastructure", "ops", "platform", "environment", "production", "staging"},
		AgentTypeGeneral:    {"what", "explain", "help", "overview", "concept", "tdf", "opentdf", "understand", "introduction", "basics", "fundamentals"},
	}
	
	if keywords, exists := typeKeywords[agent.Type]; exists {
		for _, keyword := range keywords {
			if strings.Contains(queryLower, keyword) {
				score += 10.0
			}
		}
	}
	
	// Check for specific mentions of agent capabilities
	for _, capability := range agent.Metadata.Capabilities {
		if strings.Contains(queryLower, strings.ToLower(capability)) {
			score += 15.0
		}
	}
	
	// Previous agent continuity bonus
	if context.PreviousAgent != nil && context.PreviousAgent.Metadata.Name == agent.Metadata.Name {
		score += 5.0
	}
	
	// Usage history bonus
	if agent.UsageCount > 0 {
		score += float64(agent.UsageCount) * 0.5
	}
	
	return score
}

// ShouldSwitchAgent determines if we should switch from current agent
func (s *DefaultAgentSelectionStrategy) ShouldSwitchAgent(currentAgent *Agent, context *AgentContext) (bool, *Agent, error) {
	if currentAgent == nil {
		return true, nil, nil
	}
	
	// For demonstration, we would need access to all agents here
	// In practice, this would be called by the manager with access to the registry
	// currentScore := s.CalculateRelevanceScore(currentAgent, context)
	
	return false, nil, nil
}

// NewAgentManager creates a new agent manager
func NewAgentManager(agentDirectory string, cfg *config.Config) *AgentManager {
	loader := NewFileSystemAgentLoader()
	registry := NewInMemoryAgentRegistry(loader).WithDirectory(agentDirectory)
	strategy := NewDefaultAgentSelectionStrategy()
	
	var resourceManager *ResourceManager
	if cfg != nil {
		resourceManager = NewResourceManagerFromConfig(cfg)
	}
	
	return &AgentManager{
		registry:           registry,
		loader:            loader,
		selectionStrategy: strategy,
		resourceManager:   resourceManager,
		agentDirectory:    agentDirectory,
		autoRefresh:       true,
		refreshInterval:   5 * time.Minute,
		sessionContext:    &AgentContext{},
		agentSwitchCount:  0,
	}
}

// Initialize loads agents and starts the manager
func (am *AgentManager) Initialize(ctx context.Context) error {
	log.Printf("Initializing agent manager with directory: %s", am.agentDirectory)
	
	// Load agents from directory
	if err := am.registry.LoadAgents(ctx); err != nil {
		return fmt.Errorf("failed to load agents: %w", err)
	}
	
	// Get loaded agent stats
	agents := am.registry.ListAgents()
	log.Printf("Successfully loaded %d agents", len(agents))
	
	for _, agent := range agents {
		log.Printf("  - %s (%s): %s", agent.Metadata.Name, agent.Type, agent.Metadata.Description)
	}
	
	return nil
}

// SelectAgent selects the best agent for a user query
func (am *AgentManager) SelectAgent(ctx context.Context, userQuery string, conversationHistory []ChatMessage) (*Agent, error) {
	// Update session context
	am.sessionContext.UserQuery = userQuery
	am.sessionContext.ConversationHistory = conversationHistory
	am.sessionContext.PreviousAgent = am.currentAgent
	
	// Get available agents
	availableAgents := am.registry.ListAgents()
	if len(availableAgents) == 0 {
		return nil, fmt.Errorf("no agents available")
	}
	
	// Use selection strategy to find best agent
	selectedAgent, confidence, err := am.selectionStrategy.SelectAgent(am.sessionContext, availableAgents)
	if err != nil {
		return nil, fmt.Errorf("agent selection failed: %w", err)
	}
	
	// Check if we need to switch agents
	if am.currentAgent != nil && selectedAgent.Metadata.Name != am.currentAgent.Metadata.Name {
		log.Printf("Switching from %s to %s (confidence: %.1f)", 
			am.currentAgent.Metadata.Name, selectedAgent.Metadata.Name, confidence)
		
		am.agentSwitchCount++
		am.currentAgent = selectedAgent
		
		// Notify about agent switch
		am.notifyAgentSwitch(am.currentAgent, selectedAgent, "query relevance")
	} else if am.currentAgent == nil {
		log.Printf("Selected agent: %s (confidence: %.1f)", selectedAgent.Metadata.Name, confidence)
		am.currentAgent = selectedAgent
	}
	
	return selectedAgent, nil
}

// GetCurrentAgent returns the currently active agent
func (am *AgentManager) GetCurrentAgent() *Agent {
	return am.currentAgent
}

// GetAgentByName retrieves a specific agent by name
func (am *AgentManager) GetAgentByName(name string) (*Agent, error) {
	return am.registry.GetAgent(name)
}

// GetAgentByType retrieves an agent by type
func (am *AgentManager) GetAgentByType(agentType AgentType) (*Agent, error) {
	return am.registry.GetAgentByType(agentType)
}

// ListAgents returns all available agents
func (am *AgentManager) ListAgents() []*Agent {
	return am.registry.ListAgents()
}

// RefreshAgents reloads agents from the directory
func (am *AgentManager) RefreshAgents(ctx context.Context) error {
	log.Printf("Refreshing agents from directory: %s", am.agentDirectory)
	
	if err := am.registry.RefreshAgents(ctx); err != nil {
		return fmt.Errorf("failed to refresh agents: %w", err)
	}
	
	agents := am.registry.ListAgents()
	log.Printf("Refreshed %d agents", len(agents))
	
	return nil
}

// SuggestAgentSwitch suggests a better agent based on conversation context
func (am *AgentManager) SuggestAgentSwitch(userQuery string) (*Agent, string, error) {
	// Create temporary context for suggestion
	tempContext := &AgentContext{
		UserQuery:           userQuery,
		ConversationHistory: am.sessionContext.ConversationHistory,
		PreviousAgent:       am.currentAgent,
	}
	
	availableAgents := am.registry.ListAgents()
	selectedAgent, _, err := am.selectionStrategy.SelectAgent(tempContext, availableAgents)
	if err != nil {
		return nil, "", err
	}
	
	// Only suggest switch if significantly better
	if am.currentAgent != nil && selectedAgent.Metadata.Name != am.currentAgent.Metadata.Name {
		currentScore := am.selectionStrategy.CalculateRelevanceScore(am.currentAgent, tempContext)
		newScore := am.selectionStrategy.CalculateRelevanceScore(selectedAgent, tempContext)
		
		if newScore > currentScore+3.0 { // Switch threshold (lowered for better responsiveness)
			reason := fmt.Sprintf("Better match for your query (score: %.1f vs %.1f)", newScore, currentScore)
			return selectedAgent, reason, nil
		}
	}
	
	return nil, "", nil
}

// ForceAgentSwitch manually switches to a specific agent
func (am *AgentManager) ForceAgentSwitch(agentName string) error {
	agent, err := am.registry.GetAgent(agentName)
	if err != nil {
		return fmt.Errorf("failed to get agent %s: %w", agentName, err)
	}
	
	previousAgent := am.currentAgent
	am.currentAgent = agent
	am.agentSwitchCount++
	
	log.Printf("Manually switched to agent: %s", agent.Metadata.Name)
	
	// Notify about agent switch
	am.notifyAgentSwitch(previousAgent, agent, "manual switch")
	
	return nil
}

// GetAgentStats returns statistics about agent usage
func (am *AgentManager) GetAgentStats() map[string]interface{} {
	registryStats := am.registry.(*InMemoryAgentRegistry).GetRegistryStats()
	
	stats := map[string]interface{}{
		"current_agent":      am.getCurrentAgentName(),
		"agent_switch_count": am.agentSwitchCount,
		"session_queries":    len(am.sessionContext.ConversationHistory),
		"registry_stats":     registryStats,
	}
	
	return stats
}

// getCurrentAgentName returns the current agent name or "none"
func (am *AgentManager) getCurrentAgentName() string {
	if am.currentAgent != nil {
		return am.currentAgent.Metadata.Name
	}
	return "none"
}

// notifyAgentSwitch handles agent switch notifications
func (am *AgentManager) notifyAgentSwitch(from *Agent, to *Agent, reason string) {
	event := &AgentSwitchEvent{
		Timestamp:    time.Now(),
		FromAgent:    from,
		ToAgent:      to,
		Reason:       reason,
		UserQuery:    am.sessionContext.UserQuery,
		WasAutomatic: true,
	}
	
	// In a full implementation, this would emit events to subscribers
	log.Printf("Agent switch: %s -> %s (reason: %s)", 
		am.getAgentName(from), to.Metadata.Name, reason)
	
	_ = event // Prevent unused variable warning
}

// getAgentName safely gets agent name
func (am *AgentManager) getAgentName(agent *Agent) string {
	if agent == nil {
		return "none"
	}
	return agent.Metadata.Name
}

// Close gracefully shuts down the agent manager
func (am *AgentManager) Close() error {
	log.Printf("Shutting down agent manager")
	
	// In a full implementation, this would:
	// - Stop auto-refresh goroutines
	// - Save agent usage statistics
	// - Clean up resources
	
	return nil
}

// Simple event handler implementation for logging
type LoggingAgentEventHandler struct{}

func (h *LoggingAgentEventHandler) OnAgentLoaded(agent *Agent) {
	log.Printf("Agent loaded: %s (%s)", agent.Metadata.Name, agent.Type)
}

func (h *LoggingAgentEventHandler) OnAgentUnloaded(agent *Agent) {
	log.Printf("Agent unloaded: %s", agent.Metadata.Name)
}

func (h *LoggingAgentEventHandler) OnAgentSelected(agent *Agent, context *AgentContext) {
	log.Printf("Agent selected: %s for query: %s", agent.Metadata.Name, context.UserQuery)
}

func (h *LoggingAgentEventHandler) OnAgentSwitched(from *Agent, to *Agent, reason string) {
	fromName := "none"
	if from != nil {
		fromName = from.Metadata.Name
	}
	log.Printf("Agent switched: %s -> %s (reason: %s)", fromName, to.Metadata.Name, reason)
}

func (h *LoggingAgentEventHandler) OnAgentError(agent *Agent, err error) {
	log.Printf("Agent error in %s: %v", agent.Metadata.Name, err)
}