package llm

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestAgentLoader(t *testing.T) {
	// Create temporary directory for test agents in current directory (safe)
	tempDir, err := os.MkdirTemp(".", "test_agents_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Create a test agent file
	agentContent := `---
agent_name: Test Agent
description: A test agent for unit testing
tags: ["test", "development"]
capabilities: ["testing", "validation"]
---

You are a test agent for unit testing purposes.

**Core Knowledge:**
- Unit testing best practices
- Test automation
- Quality assurance

**Communication Style:**
- Clear and concise responses
- Technical accuracy
- Helpful examples
`
	
	agentFile := filepath.Join(tempDir, "test_agent.md")
	if err := os.WriteFile(agentFile, []byte(agentContent), 0644); err != nil {
		t.Fatalf("Failed to write test agent file: %v", err)
	}
	
	// Test agent loader
	loader := NewFileSystemAgentLoader()
	
	// Test LoadFromFile
	agent, err := loader.LoadFromFile(agentFile)
	if err != nil {
		t.Errorf("LoadFromFile failed: %v", err)
	}
	
	if agent == nil {
		t.Fatal("LoadFromFile returned nil agent")
	}
	
	// Validate agent metadata
	if agent.Metadata.Name != "Test Agent" {
		t.Errorf("Expected agent name 'Test Agent', got '%s'", agent.Metadata.Name)
	}
	
	if agent.Metadata.Description == "" {
		t.Error("Agent description should not be empty")
	}
	
	if len(agent.Metadata.Tags) != 2 {
		t.Errorf("Expected 2 tags, got %d", len(agent.Metadata.Tags))
	}
	
	if len(agent.Metadata.Capabilities) != 2 {
		t.Errorf("Expected 2 capabilities, got %d", len(agent.Metadata.Capabilities))
	}
	
	// Test LoadFromDirectory
	agents, err := loader.LoadFromDirectory(tempDir)
	if err != nil {
		t.Errorf("LoadFromDirectory failed: %v", err)
	}
	
	if len(agents) != 1 {
		t.Errorf("Expected 1 agent, got %d", len(agents))
	}
	
	// Test agent validation
	if err := loader.ValidateAgent(agent); err != nil {
		t.Errorf("Agent validation failed: %v", err)
	}
}

func TestAgentRegistry(t *testing.T) {
	loader := NewFileSystemAgentLoader()
	registry := NewInMemoryAgentRegistry(loader)
	
	// Create test agents
	agent1 := &Agent{
		Metadata: AgentMetadata{
			Name:        "Test Agent 1",
			Description: "First test agent",
			Tags:        []string{"test"},
		},
		SystemPrompt:  "You are test agent 1",
		Type:          AgentTypeGeneral,
		IsActive:      true,
		ContextWindow: 4096,
		MaxTokens:     512,
	}
	
	agent2 := &Agent{
		Metadata: AgentMetadata{
			Name:        "Test Agent 2",
			Description: "Second test agent",
			Tags:        []string{"development"},
		},
		SystemPrompt:  "You are test agent 2",
		Type:          AgentTypeDevelopment,
		IsActive:      true,
		ContextWindow: 4096,
		MaxTokens:     512,
	}
	
	// Test agent registration
	if err := registry.RegisterAgent(agent1); err != nil {
		t.Errorf("Failed to register agent1: %v", err)
	}
	
	if err := registry.RegisterAgent(agent2); err != nil {
		t.Errorf("Failed to register agent2: %v", err)
	}
	
	// Test duplicate registration
	if err := registry.RegisterAgent(agent1); err == nil {
		t.Error("Expected error for duplicate agent registration")
	}
	
	// Test agent retrieval
	retrievedAgent, err := registry.GetAgent("Test Agent 1")
	if err != nil {
		t.Errorf("Failed to get agent: %v", err)
	}
	
	if retrievedAgent.Metadata.Name != "Test Agent 1" {
		t.Errorf("Expected 'Test Agent 1', got '%s'", retrievedAgent.Metadata.Name)
	}
	
	// Test agent retrieval by type
	devAgent, err := registry.GetAgentByType(AgentTypeDevelopment)
	if err != nil {
		t.Errorf("Failed to get agent by type: %v", err)
	}
	
	if devAgent.Metadata.Name != "Test Agent 2" {
		t.Errorf("Expected 'Test Agent 2', got '%s'", devAgent.Metadata.Name)
	}
	
	// Test listing agents
	agents := registry.ListAgents()
	if len(agents) != 2 {
		t.Errorf("Expected 2 agents, got %d", len(agents))
	}
	
	// Test listing agents by type
	devAgents := registry.ListAgentsByType(AgentTypeDevelopment)
	if len(devAgents) != 1 {
		t.Errorf("Expected 1 development agent, got %d", len(devAgents))
	}
	
	// Test agent unregistration
	if err := registry.UnregisterAgent("Test Agent 1"); err != nil {
		t.Errorf("Failed to unregister agent: %v", err)
	}
	
	// Verify agent was removed
	if _, err := registry.GetAgent("Test Agent 1"); err == nil {
		t.Error("Expected error for getting unregistered agent")
	}
	
	remainingAgents := registry.ListAgents()
	if len(remainingAgents) != 1 {
		t.Errorf("Expected 1 remaining agent, got %d", len(remainingAgents))
	}
}

func TestAgentSelectionStrategy(t *testing.T) {
	strategy := NewDefaultAgentSelectionStrategy()
	
	// Create test agents
	policyAgent := &Agent{
		Metadata: AgentMetadata{
			Name:         "Policy Agent",
			Description:  "OpenTDF policy specialist",
			Capabilities: []string{"policy", "attributes", "ABAC"},
		},
		Type: AgentTypePolicy,
	}
	
	devAgent := &Agent{
		Metadata: AgentMetadata{
			Name:         "Development Agent",
			Description:  "SDK development specialist",
			Capabilities: []string{"SDK", "programming", "integration"},
		},
		Type: AgentTypeDevelopment,
	}
	
	generalAgent := &Agent{
		Metadata: AgentMetadata{
			Name:        "General Agent",
			Description: "General OpenTDF information",
		},
		Type: AgentTypeGeneral,
	}
	
	agents := []*Agent{policyAgent, devAgent, generalAgent}
	
	// Test policy-related query
	policyContext := &AgentContext{
		UserQuery: "How do I create attributes and policies for access control?",
	}
	
	selectedAgent, score, err := strategy.SelectAgent(policyContext, agents)
	if err != nil {
		t.Errorf("Agent selection failed: %v", err)
	}
	
	if selectedAgent.Type != AgentTypePolicy {
		t.Errorf("Expected policy agent for policy query, got %s", selectedAgent.Type)
	}
	
	if score <= 0 {
		t.Error("Expected positive relevance score")
	}
	
	// Test development-related query
	devContext := &AgentContext{
		UserQuery: "How do I integrate the Java SDK into my application?",
	}
	
	selectedAgent, score, err = strategy.SelectAgent(devContext, agents)
	if err != nil {
		t.Errorf("Agent selection failed: %v", err)
	}
	
	if selectedAgent.Type != AgentTypeDevelopment {
		t.Errorf("Expected development agent for SDK query, got %s", selectedAgent.Type)
	}
	
	// Test general query
	generalContext := &AgentContext{
		UserQuery: "What is OpenTDF and how does it work?",
	}
	
	selectedAgent, _, err = strategy.SelectAgent(generalContext, agents)
	if err != nil {
		t.Errorf("Agent selection failed: %v", err)
	}
	
	// Should select general agent or any agent for general queries
	if selectedAgent == nil {
		t.Error("Expected an agent to be selected")
	}
}

func TestAgentManager(t *testing.T) {
	// Create temporary directory for test agents in current directory (safe)
	tempDir, err := os.MkdirTemp(".", "test_agents_*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Create test agent files
	policyAgentContent := `---
agent_name: Policy Agent
description: OpenTDF policy specialist
tags: ["policy", "ABAC"]
---

You are an OpenTDF policy specialist.
`
	
	devAgentContent := `---
agent_name: Development Agent  
description: SDK development specialist
tags: ["development", "SDK"]
---

You are an OpenTDF development specialist.
`
	
	// Write agent files
	policyFile := filepath.Join(tempDir, "policy_agent.md")
	devFile := filepath.Join(tempDir, "development_agent.md")
	
	if err := os.WriteFile(policyFile, []byte(policyAgentContent), 0644); err != nil {
		t.Fatalf("Failed to write policy agent file: %v", err)
	}
	
	if err := os.WriteFile(devFile, []byte(devAgentContent), 0644); err != nil {
		t.Fatalf("Failed to write dev agent file: %v", err)
	}
	
	// Test agent manager
	manager := NewAgentManager(tempDir, nil)
	
	ctx := context.Background()
	if err := manager.Initialize(ctx); err != nil {
		t.Errorf("Failed to initialize agent manager: %v", err)
	}
	
	// Test agent listing
	agents := manager.ListAgents()
	if len(agents) != 2 {
		t.Errorf("Expected 2 agents, got %d", len(agents))
	}
	
	// Test agent selection
	selectedAgent, err := manager.SelectAgent(ctx, "How do I create policies?", nil)
	if err != nil {
		t.Errorf("Failed to select agent: %v", err)
	}
	
	if selectedAgent == nil {
		t.Fatal("Expected an agent to be selected")
	}
	
	// Test getting current agent
	currentAgent := manager.GetCurrentAgent()
	if currentAgent == nil {
		t.Error("Expected current agent to be set")
	}
	
	// Test agent by name
	namedAgent, err := manager.GetAgentByName("Policy Agent")
	if err != nil {
		t.Errorf("Failed to get agent by name: %v", err)
	}
	
	if namedAgent.Metadata.Name != "Policy Agent" {
		t.Errorf("Expected 'Policy Agent', got '%s'", namedAgent.Metadata.Name)
	}
	
	// Test agent stats
	stats := manager.GetAgentStats()
	if stats == nil {
		t.Error("Expected agent stats to be returned")
	}
}

func TestAgentMetadataParsing(t *testing.T) {
	loader := NewFileSystemAgentLoader()
	
	testCases := []struct {
		name        string
		content     string
		expectError bool
		expectedName string
	}{
		{
			name: "valid frontmatter",
			content: `---
agent_name: Test Agent
description: A test agent
version: "1.0"
tags: ["test"]
---

Agent content here.`,
			expectError:  false,
			expectedName: "Test Agent",
		},
		{
			name: "missing frontmatter",
			content: `
Just content without frontmatter.`,
			expectError: true,
		},
		{
			name: "invalid YAML",
			content: `---
agent_name: Test Agent
description: [invalid yaml
---

Content.`,
			expectError: true,
		},
		{
			name: "missing required fields",
			content: `---
version: "1.0"
---

Content.`,
			expectError: true,
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			metadata, systemPrompt, err := loader.parseFrontmatter(tc.content)
			
			if tc.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}
			
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			
			if metadata.Name != tc.expectedName {
				t.Errorf("Expected name '%s', got '%s'", tc.expectedName, metadata.Name)
			}
			
			if systemPrompt == "" {
				t.Error("Expected system prompt to be extracted")
			}
		})
	}
}

func TestAgentTypeInference(t *testing.T) {
	loader := NewFileSystemAgentLoader()
	
	testCases := []struct {
		filepath     string
		metadata     *AgentMetadata
		expectedType AgentType
	}{
		{
			filepath:     "/path/to/policy_agent.md",
			metadata:     &AgentMetadata{Tags: []string{"policy"}},
			expectedType: AgentTypePolicy,
		},
		{
			filepath:     "/path/to/development_agent.md",
			metadata:     &AgentMetadata{},
			expectedType: AgentTypeDevelopment,
		},
		{
			filepath:     "/path/to/deployment_agent.md",
			metadata:     &AgentMetadata{},
			expectedType: AgentTypeDeployment,
		},
		{
			filepath:     "/path/to/general_agent.md",
			metadata:     &AgentMetadata{},
			expectedType: AgentTypeGeneral,
		},
		{
			filepath:     "/path/to/unknown_agent.md",
			metadata:     &AgentMetadata{},
			expectedType: AgentTypeGeneral, // Default fallback
		},
	}
	
	for _, tc := range testCases {
		actualType := loader.determineAgentType(tc.filepath, tc.metadata)
		if actualType != tc.expectedType {
			t.Errorf("For path %s, expected type %s, got %s", 
				tc.filepath, tc.expectedType, actualType)
		}
	}
}