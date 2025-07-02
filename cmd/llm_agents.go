package cmd

import (
	"context"
	"os"
	"path/filepath"

	"github.com/opentdf/otdfctl/pkg/cli"
	"github.com/opentdf/otdfctl/pkg/llm"
	"github.com/opentdf/otdfctl/pkg/man"
	"github.com/spf13/cobra"
)

var llmAgentsCmd = man.Docs.GetCommand("llm/agents", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	// Get model path
	var modelPath string
	if len(args) == 0 {
		// Try to get from configuration
		pathResolver := llm.NewModelPathResolver(&OtdfctlCfg)
		resolvedPath, err := pathResolver.ResolveChatModelPath("")
		if err != nil {
			c.ExitWithError("Model path is required (provide as argument or set in configuration)", err)
		}
		modelPath = resolvedPath
	} else {
		// Validate user-provided path
		pathResolver := llm.NewModelPathResolver(&OtdfctlCfg)
		resolvedPath, err := pathResolver.ResolveChatModelPath(args[0])
		if err != nil {
			c.ExitWithError("Invalid model path", err)
		}
		modelPath = resolvedPath
	}
	
	// Get flag values
	agentDirectory := c.Flags.GetOptionalString("agent-dir")
	enableRAG := c.Flags.GetOptionalBool("rag")
	ragIndexPath := c.Flags.GetOptionalString("index-path")
	contextSize := int(c.Flags.GetOptionalInt32("context-size"))
	temperature, _ := cmd.Flags().GetFloat64("temperature")
	systemPrompt := c.Flags.GetOptionalString("system-prompt")
	
	// Set default agent directory if not provided
	if agentDirectory == "" {
		// Look for agents in pkg/llm/agents relative to current directory
		defaultDir := filepath.Join("pkg", "llm", "agents")
		if _, err := os.Stat(defaultDir); err == nil {
			agentDirectory = defaultDir
		} else {
			// Fallback to working directory
			if wd, err := os.Getwd(); err == nil {
				agentDirectory = filepath.Join(wd, "agents")
			} else {
				agentDirectory = "./agents"
			}
		}
	}
	
	// Set default RAG index path
	if enableRAG && ragIndexPath == "" {
		homeDir, _ := os.UserHomeDir()
		ragIndexPath = filepath.Join(homeDir, ".otdfctl", "simple_rag_index.json")
	}
	
	// Check if JSON output is requested
	if jsonFlag, _ := cmd.Flags().GetBool("json"); jsonFlag {
		// For JSON mode, just output agent information
		agentCommand := llm.NewAgentChatCommand(&OtdfctlCfg)
		
		// Validate and list agents
		if err := agentCommand.ValidateAgentDirectory(agentDirectory); err != nil {
			c.ExitWithError("Agent directory validation failed", err)
		}
		
		agents, err := agentCommand.ListAvailableAgents(agentDirectory)
		if err != nil {
			c.ExitWithError("Failed to list agents", err)
		}
		
		output := map[string]interface{}{
			"agent_directory": agentDirectory,
			"model_path":     modelPath,
			"agents":         agents,
			"rag_enabled":    enableRAG,
			"context_size":   contextSize,
		}
		
		c.ExitWithJSON(output)
		return
	}
	
	// Show startup information
	c.Printf("🚀 Starting OpenTDF Agent Chat Interface...\n")
	c.Printf("📁 Agent Directory: %s\n", agentDirectory)
	c.Printf("🤖 Model: %s\n", modelPath)
	
	// Validate agent directory
	agentCommand := llm.NewAgentChatCommand(&OtdfctlCfg)
	
	if err := agentCommand.ValidateAgentDirectory(agentDirectory); err != nil {
		c.Printf("⚠️  Warning: %v\n", err)
		c.Printf("💡 The chat will work with a basic system prompt\n")
	} else {
		// Show available agents
		agents, err := agentCommand.ListAvailableAgents(agentDirectory)
		if err != nil {
			c.Printf("⚠️  Warning: Failed to list agents: %v\n", err)
		} else {
			c.Printf("✅ Found %d agents:\n", len(agents))
			for _, agent := range agents {
				c.Printf("   - %s (%s): %s\n", agent.Name, agent.Type, agent.Description)
			}
		}
	}
	
	c.Printf("\n")
	
	// Prepare agent chat options
	opts := llm.AgentChatOptions{
		ModelPath:      modelPath,
		AgentDirectory: agentDirectory,
		EnableRAG:      enableRAG,
		RAGIndexPath:   ragIndexPath,
		SystemPrompt:   systemPrompt,
		ContextSize:    contextSize,
		Temperature:    temperature,
	}
	
	// Start the agent chat TUI
	ctx := context.Background()
	if err := agentCommand.RunAgentChatWithOptions(ctx, opts); err != nil {
		c.ExitWithError("Failed to start agent chat", err)
	}
}))

// Command to list available agents
var llmAgentsListCmd = man.Docs.GetCommand("llm/agents/list", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	agentDirectory := c.Flags.GetOptionalString("agent-dir")
	if agentDirectory == "" {
		agentDirectory = filepath.Join("pkg", "llm", "agents")
	}
	
	agentCommand := llm.NewAgentChatCommand(&OtdfctlCfg)
	
	agents, err := agentCommand.ListAvailableAgents(agentDirectory)
	if err != nil {
		c.ExitWithError("Failed to list agents", err)
	}
	
	if jsonFlag, _ := cmd.Flags().GetBool("json"); jsonFlag {
		output := map[string]interface{}{
			"agent_directory": agentDirectory,
			"total_agents":    len(agents),
			"agents":          agents,
		}
		c.ExitWithJSON(output)
		return
	}
	
	// Styled output
	c.Printf("📁 Agent Directory: %s\n", agentDirectory)
	c.Printf("🤖 Available Agents (%d):\n\n", len(agents))
	
	if len(agents) == 0 {
		c.Printf("   No agents found. Create agent files in the directory with YAML frontmatter.\n")
		c.Printf("   Example agent file structure:\n")
		c.Printf("   ---\n")
		c.Printf("   agent_name: My Agent\n")
		c.Printf("   description: A helpful agent\n")
		c.Printf("   tags: [\"development\"]\n")
		c.Printf("   ---\n")
		c.Printf("   \n")
		c.Printf("   You are a helpful agent...\n")
		return
	}
	
	for i, agent := range agents {
		c.Printf("  %d. %s (%s)\n", i+1, agent.Name, agent.Type)
		c.Printf("     📝 %s\n", agent.Description)
		c.Printf("     📍 %s\n", agent.FilePath)
		
		if len(agent.Tags) > 0 {
			c.Printf("     🏷️  Tags: %v\n", agent.Tags)
		}
		
		if len(agent.Capabilities) > 0 {
			c.Printf("     ⚡ Capabilities: %v\n", agent.Capabilities)
		}
		
		c.Printf("\n")
	}
}))

// Command to validate agent configuration
var llmAgentsValidateCmd = man.Docs.GetCommand("llm/agents/validate", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	agentDirectory := c.Flags.GetOptionalString("agent-dir")
	if agentDirectory == "" {
		agentDirectory = filepath.Join("pkg", "llm", "agents")
	}
	
	c.Printf("🔍 Validating agents in: %s\n", agentDirectory)
	
	// Use the loader to get detailed results
	loader := llm.NewFileSystemAgentLoader()
	result, err := loader.LoadWithResults(agentDirectory)
	if err != nil {
		c.ExitWithError("Failed to load agents", err)
	}
	
	// Report results
	c.Printf("\n📊 Validation Results:\n")
	c.Printf("   📁 Files processed: %d\n", result.TotalFiles)
	c.Printf("   ✅ Successfully loaded: %d agents\n", len(result.LoadedAgents))
	c.Printf("   ❌ Failed to load: %d agents\n", len(result.FailedAgents))
	c.Printf("   ⏱️  Total time: %v\n", result.Duration)
	
	if jsonFlag, _ := cmd.Flags().GetBool("json"); jsonFlag {
		output := map[string]interface{}{
			"agent_directory":   agentDirectory,
			"total_files":       result.TotalFiles,
			"loaded_agents":     len(result.LoadedAgents),
			"failed_agents":     len(result.FailedAgents),
			"validation_time":   result.Duration.String(),
			"loaded_details":    result.LoadedAgents,
			"failed_details":    result.FailedAgents,
		}
		c.ExitWithJSON(output)
		return
	}
	
	// Show detailed results
	if len(result.LoadedAgents) > 0 {
		c.Printf("\n✅ Successfully Loaded Agents:\n")
		for i, success := range result.LoadedAgents {
			agent := success.Agent
			c.Printf("  %d. %s (%s)\n", i+1, agent.Metadata.Name, agent.Type)
			c.Printf("     📍 %s (loaded in %v)\n", success.FilePath, success.LoadTime)
		}
	}
	
	if len(result.FailedAgents) > 0 {
		c.Printf("\n❌ Failed to Load:\n")
		for i, failure := range result.FailedAgents {
			c.Printf("  %d. %s\n", i+1, failure.FilePath)
			c.Printf("     ❌ Error: %v\n", failure.Error)
			c.Printf("     ⏱️  Time: %v\n", failure.LoadTime)
		}
	}
	
	if len(result.FailedAgents) > 0 {
		c.Printf("\n💡 Tips for fixing agent files:\n")
		c.Printf("   - Ensure YAML frontmatter is properly formatted\n")
		c.Printf("   - Required fields: agent_name, description\n")
		c.Printf("   - Check file permissions and encoding\n")
		c.Printf("   - Validate YAML syntax with yamllint\n")
	}
}))

func init() {
	// Configure llm agents command
	llmAgentsCmd.Flags().Bool("stream", true, "Enable streaming responses")
	llmAgentsCmd.Flags().Int32("context-size", 4096, "Maximum context window size")
	llmAgentsCmd.Flags().Float64("temperature", 0.7, "Sampling temperature (0.0-1.0)")
	llmAgentsCmd.Flags().String("system-prompt", "", "Custom system prompt")
	llmAgentsCmd.Flags().Bool("rag", false, "Enable RAG (Retrieval-Augmented Generation)")
	llmAgentsCmd.Flags().String("index-path", "", "Path to RAG vector index (default: ~/.otdfctl/rag_index.json)")
	llmAgentsCmd.Flags().String("agent-dir", "", "Directory containing agent files (default: pkg/llm/agents)")
	llmAgentsCmd.Flags().Bool("json", false, "Output in JSON format")
	
	// Configure agents list command
	llmAgentsListCmd.Flags().String("agent-dir", "", "Directory containing agent files")
	llmAgentsListCmd.Flags().Bool("json", false, "Output in JSON format")
	
	// Configure agents validate command
	llmAgentsValidateCmd.Flags().String("agent-dir", "", "Directory containing agent files")
	llmAgentsValidateCmd.Flags().Bool("json", false, "Output in JSON format")
	
	// Add subcommands to agents command
	llmAgentsCmd.AddCommand(&llmAgentsListCmd.Command)
	llmAgentsCmd.AddCommand(&llmAgentsValidateCmd.Command)
	
	// Add agents command to llm parent
	llmCmd.AddCommand(&llmAgentsCmd.Command)
}