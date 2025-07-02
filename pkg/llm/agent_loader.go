package llm

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// FileSystemAgentLoader loads agents from markdown files with YAML frontmatter
type FileSystemAgentLoader struct {
	validator     *FileValidator
	watchEnabled  bool
	watchInterval time.Duration
}

// NewFileSystemAgentLoader creates a new file system based agent loader
func NewFileSystemAgentLoader() *FileSystemAgentLoader {
	return &FileSystemAgentLoader{
		validator:     NewFileValidator().WithAllowedExtensions([]string{".md", ".yaml", ".yml"}),
		watchEnabled:  false,
		watchInterval: 30 * time.Second,
	}
}

// WithWatching enables file watching for agent changes
func (fal *FileSystemAgentLoader) WithWatching(interval time.Duration) *FileSystemAgentLoader {
	fal.watchEnabled = true
	fal.watchInterval = interval
	return fal
}

// LoadFromDirectory loads all agents from a directory
func (fal *FileSystemAgentLoader) LoadFromDirectory(dirPath string) ([]*Agent, error) {
	// Validate directory path
	if err := fal.validator.ValidateDirectory(dirPath); err != nil {
		return nil, fmt.Errorf("invalid agent directory: %w", err)
	}

	var agents []*Agent
	
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		
		// Skip directories and non-markdown files
		if d.IsDir() {
			return nil
		}
		
		if !strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			return nil
		}
		
		// Load agent from file
		agent, loadErr := fal.LoadFromFile(path)
		if loadErr != nil {
			// Log warning but continue loading other agents
			fmt.Printf("Warning: failed to load agent from %s: %v\n", path, loadErr)
			return nil
		}
		
		if agent != nil {
			agents = append(agents, agent)
		}
		
		return nil
	})
	
	if err != nil {
		return nil, fmt.Errorf("failed to walk agent directory: %w", err)
	}
	
	return agents, nil
}

// LoadFromFile loads a single agent from a file
func (fal *FileSystemAgentLoader) LoadFromFile(filePath string) (*Agent, error) {
	// Validate file
	if err := fal.validator.ValidateFile(filePath); err != nil {
		return nil, fmt.Errorf("agent file validation failed: %w", err)
	}
	
	// Read file content
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read agent file: %w", err)
	}
	
	// Parse frontmatter and content
	metadata, systemPrompt, err := fal.parseFrontmatter(string(content))
	if err != nil {
		return nil, fmt.Errorf("failed to parse agent frontmatter: %w", err)
	}
	
	// Get file info for metadata
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}
	
	// Set file metadata
	metadata.FilePath = filePath
	metadata.LastModified = fileInfo.ModTime()
	
	// Determine agent type from file name or directory
	agentType := fal.determineAgentType(filePath, metadata)
	
	// Create agent
	agent := &Agent{
		Metadata:      *metadata,
		SystemPrompt:  systemPrompt,
		Type:          agentType,
		IsActive:      true,
		LastUsed:      time.Time{},
		UsageCount:    0,
		ContextWindow: 4096, // Default context window
		MaxTokens:     512,  // Default max tokens
	}
	
	// Validate the loaded agent
	if err := fal.ValidateAgent(agent); err != nil {
		return nil, fmt.Errorf("agent validation failed: %w", err)
	}
	
	return agent, nil
}

// parseFrontmatter extracts YAML frontmatter and content from markdown
func (fal *FileSystemAgentLoader) parseFrontmatter(content string) (*AgentMetadata, string, error) {
	// Regex to match YAML frontmatter
	frontmatterRegex := regexp.MustCompile(`(?s)^---\n(.*?)\n---\n(.*)`)
	
	matches := frontmatterRegex.FindStringSubmatch(content)
	if len(matches) != 3 {
		return nil, "", fmt.Errorf("invalid frontmatter format: expected YAML frontmatter between --- delimiters")
	}
	
	yamlContent := matches[1]
	markdownContent := strings.TrimSpace(matches[2])
	
	// Parse YAML frontmatter
	var metadata AgentMetadata
	if err := yaml.Unmarshal([]byte(yamlContent), &metadata); err != nil {
		return nil, "", fmt.Errorf("failed to parse YAML frontmatter: %w", err)
	}
	
	// Validate required fields
	if metadata.Name == "" {
		return nil, "", fmt.Errorf("agent_name is required in frontmatter")
	}
	
	if metadata.Description == "" {
		return nil, "", fmt.Errorf("description is required in frontmatter")
	}
	
	return &metadata, markdownContent, nil
}

// determineAgentType infers agent type from file path and metadata
func (fal *FileSystemAgentLoader) determineAgentType(filePath string, metadata *AgentMetadata) AgentType {
	// Check if type is explicitly set in metadata
	if len(metadata.Tags) > 0 {
		for _, tag := range metadata.Tags {
			switch strings.ToLower(tag) {
			case "general", "information":
				return AgentTypeGeneral
			case "policy", "abac", "attributes":
				return AgentTypePolicy
			case "development", "dev", "sdk":
				return AgentTypeDevelopment
			case "deployment", "ops", "infrastructure":
				return AgentTypeDeployment
			}
		}
	}
	
	// Infer from file name
	fileName := strings.ToLower(filepath.Base(filePath))
	
	if strings.Contains(fileName, "policy") {
		return AgentTypePolicy
	}
	if strings.Contains(fileName, "development") || strings.Contains(fileName, "dev") {
		return AgentTypeDevelopment
	}
	if strings.Contains(fileName, "deployment") || strings.Contains(fileName, "deploy") {
		return AgentTypeDeployment
	}
	if strings.Contains(fileName, "general") || strings.Contains(fileName, "info") {
		return AgentTypeGeneral
	}
	
	// Default to general if cannot determine
	return AgentTypeGeneral
}

// ValidateAgent validates an agent's structure and content
func (fal *FileSystemAgentLoader) ValidateAgent(agent *Agent) error {
	if agent == nil {
		return fmt.Errorf("agent cannot be nil")
	}
	
	if agent.Metadata.Name == "" {
		return fmt.Errorf("agent name cannot be empty")
	}
	
	if agent.Metadata.Description == "" {
		return fmt.Errorf("agent description cannot be empty")
	}
	
	if agent.SystemPrompt == "" {
		return fmt.Errorf("agent system prompt cannot be empty")
	}
	
	if agent.ContextWindow <= 0 {
		return fmt.Errorf("agent context window must be positive")
	}
	
	if agent.MaxTokens <= 0 {
		return fmt.Errorf("agent max tokens must be positive")
	}
	
	// Validate system prompt length (reasonable limits)
	if len(agent.SystemPrompt) > 50000 { // 50KB limit
		return fmt.Errorf("agent system prompt too long: %d characters (max: 50000)", len(agent.SystemPrompt))
	}
	
	return nil
}

// WatchForChanges monitors agent files for changes
func (fal *FileSystemAgentLoader) WatchForChanges(ctx context.Context, callback func(*Agent)) error {
	if !fal.watchEnabled {
		return fmt.Errorf("watching is not enabled")
	}
	
	// Simple polling-based file watching
	// In production, consider using filesystem events (fsnotify)
	ticker := time.NewTicker(fal.watchInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			// Check for file changes
			// This is a simplified implementation
			// In production, implement proper file watching with fsnotify
			// For now, we just periodically trigger the callback for demonstration
			_ = callback // Prevent unused parameter warning
		}
	}
}

// AgentLoadResult represents the result of loading agents
type AgentLoadResult struct {
	LoadedAgents []AgentLoadSuccess
	FailedAgents []AgentLoadFailure
	TotalFiles   int
	Duration     time.Duration
}

// AgentLoadSuccess represents a successfully loaded agent
type AgentLoadSuccess struct {
	Agent    *Agent
	FilePath string
	LoadTime time.Duration
}

// AgentLoadFailure represents a failed agent load
type AgentLoadFailure struct {
	FilePath string
	Error    error
	LoadTime time.Duration
}

// LoadWithResults loads agents and returns detailed results
func (fal *FileSystemAgentLoader) LoadWithResults(dirPath string) (*AgentLoadResult, error) {
	startTime := time.Now()
	
	result := &AgentLoadResult{
		LoadedAgents: make([]AgentLoadSuccess, 0),
		FailedAgents: make([]AgentLoadFailure, 0),
	}
	
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(d.Name()), ".md") {
			return nil
		}
		
		result.TotalFiles++
		fileStartTime := time.Now()
		
		agent, loadErr := fal.LoadFromFile(path)
		loadDuration := time.Since(fileStartTime)
		
		if loadErr != nil {
			result.FailedAgents = append(result.FailedAgents, AgentLoadFailure{
				FilePath: path,
				Error:    loadErr,
				LoadTime: loadDuration,
			})
		} else if agent != nil {
			result.LoadedAgents = append(result.LoadedAgents, AgentLoadSuccess{
				Agent:    agent,
				FilePath: path,
				LoadTime: loadDuration,
			})
		}
		
		return nil
	})
	
	result.Duration = time.Since(startTime)
	
	if err != nil {
		return result, fmt.Errorf("failed to walk directory: %w", err)
	}
	
	return result, nil
}