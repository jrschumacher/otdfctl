package llm

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/opentdf/otdfctl/pkg/config"
)

// SimpleChatInterfaceV2 provides a non-blocking chat interface using the new engine architecture
type SimpleChatInterfaceV2 struct {
	engineClient *EngineClient
	agentManager *AgentManager
	currentAgent *Agent
	config       *config.Config
	
	// State management
	interrupted  bool
	cancelFunc   context.CancelFunc
}

// NewSimpleChatInterfaceV2 creates a new simple chat interface using engine V2
func NewSimpleChatInterfaceV2(agentManager *AgentManager, engineClient *EngineClient, cfg *config.Config) *SimpleChatInterfaceV2 {
	return &SimpleChatInterfaceV2{
		engineClient: engineClient,
		agentManager: agentManager,
		config:       cfg,
	}
}

// Run starts the simple chat interface
func (sci *SimpleChatInterfaceV2) Run() error {
	for {
		// Show agent selection menu
		agent, exit := sci.showAgentMenu()
		if exit {
			return nil
		}
		
		if agent == nil {
			continue
		}

		sci.currentAgent = agent
		
		// Configure the engine with the agent's system prompt
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		err := sci.engineClient.ConfigurePrompt(ctx, agent.SystemPrompt)
		cancel()
		
		if err != nil {
			fmt.Printf("⚠️ Warning: Failed to configure agent prompt: %v\n", err)
			fmt.Println("Press Enter to continue...")
			bufio.NewReader(os.Stdin).ReadLine()
			continue
		}
		
		// Start chat with selected agent
		if sci.startChat() {
			return nil // User pressed Ctrl+D to exit
		}
		// User pressed Ctrl+C to go back to menu, continue loop
	}
}

// showAgentMenu displays the agent selection menu
func (sci *SimpleChatInterfaceV2) showAgentMenu() (*Agent, bool) {
	for {
		sci.clearScreen()
		fmt.Println("🤖 OpenTDF Agent Chat V2")
		fmt.Println("========================")
		fmt.Println()
		
		agents := sci.agentManager.ListAgents()
		if len(agents) == 0 {
			fmt.Println("❌ No agents found. Please check your agent directory.")
			fmt.Println()
			fmt.Println("Press Enter to exit...")
			bufio.NewReader(os.Stdin).ReadLine()
			return nil, true
		}

		fmt.Println("Available Agents:")
		fmt.Println()
		
		for i, agent := range agents {
			fmt.Printf("  %d. %s (%s)\n", i+1, agent.Metadata.Name, agent.Type)
			fmt.Printf("     %s\n", agent.Metadata.Description)
			fmt.Println()
		}
		
		fmt.Printf("Select an agent (1-%d) or 'q' to quit: ", len(agents))
		
		reader := bufio.NewReader(os.Stdin)
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("Error reading input: %v\n", err)
			continue
		}
		
		input = strings.TrimSpace(input)
		
		// Handle quit
		if input == "q" || input == "quit" || input == "exit" {
			return nil, true
		}
		
		// Parse selection
		if len(input) == 1 && input[0] >= '1' && input[0] <= '9' {
			selection := int(input[0] - '0')
			if selection >= 1 && selection <= len(agents) {
				return agents[selection-1], false
			}
		}
		
		fmt.Printf("Invalid selection. Please enter a number between 1 and %d.\n", len(agents))
		fmt.Println("Press Enter to continue...")
		reader.ReadLine()
	}
}

// startChat begins a chat session with the selected agent
func (sci *SimpleChatInterfaceV2) startChat() bool {
	sci.clearScreen()
	fmt.Printf("🤖 Chatting with: %s (%s)\n", sci.currentAgent.Metadata.Name, sci.currentAgent.Type)
	fmt.Println("====================================")
	fmt.Printf("📝 %s\n", sci.currentAgent.Metadata.Description)
	fmt.Println()
	fmt.Println("💡 Tips:")
	fmt.Println("  • Type your message and press Enter")
	fmt.Println("  • Press Ctrl+C to return to agent menu")
	fmt.Println("  • Press Ctrl+D to exit")
	fmt.Println("  • Type '/switch' to return to menu")
	fmt.Println("  • Type '/metrics' to show engine metrics")
	fmt.Println()

	// Set up cancellation context for the entire chat session
	ctx, cancel := context.WithCancel(context.Background())
	sci.cancelFunc = cancel
	defer cancel()
	
	// Set up signal handling
	sci.setupSignalHandling()
	defer sci.cleanupSignalHandling()

	reader := bufio.NewReader(os.Stdin)

	for {
		// Check if we were interrupted
		if sci.interrupted {
			return false // Go back to menu
		}
		
		fmt.Print("You: ")
		
		input, err := reader.ReadString('\n')
		if err != nil {
			if err.Error() == "EOF" { // Ctrl+D
				fmt.Println("\n\n👋 Goodbye!")
				return true // Exit app
			}
			fmt.Printf("Error reading input: %v\n", err)
			continue
		}
		
		// Check again after input in case Ctrl+C was pressed during input
		if sci.interrupted {
			return false // Go back to menu
		}
		
		userInput := strings.TrimSpace(input)
		
		if userInput == "" {
			continue
		}
		
		// Handle special commands
		if sci.handleSpecialCommands(userInput) {
			continue
		}
		
		// Send query to engine (non-blocking with proper cancellation)
		sci.handleUserQuery(ctx, userInput)
	}
}

// handleSpecialCommands processes special chat commands
func (sci *SimpleChatInterfaceV2) handleSpecialCommands(input string) bool {
	switch input {
	case "/help":
		sci.showChatHelp()
		return true
		
	case "/switch":
		fmt.Println("\n🔄 Returning to agent menu...")
		return false // This will exit the function and return to menu
		
	case "/quit", "/exit":
		fmt.Println("\n👋 Goodbye!")
		os.Exit(0)
		return true
		
	case "/metrics":
		sci.showEngineMetrics()
		return true
		
	case "/health":
		sci.showEngineHealth()
		return true
		
	default:
		return false
	}
}

// handleUserQuery processes user input and gets response from engine
func (sci *SimpleChatInterfaceV2) handleUserQuery(ctx context.Context, query string) {
	fmt.Printf("\n🤖 %s:\n", sci.currentAgent.Metadata.Name)
	
	// Create a timeout context for this specific query
	queryCtx, queryCancel := context.WithTimeout(ctx, 30*time.Second)
	defer queryCancel()
	
	// Check for agent suggestions before sending to engine
	sci.checkAgentSuggestion(query)
	
	// Stream response from engine with proper cancellation handling
	err := sci.engineClient.StreamQuery(queryCtx, query, func(token string) {
		// Check for interruption during streaming
		if sci.interrupted {
			queryCancel() // Cancel the query context
			return
		}
		fmt.Print(token)
	})
	
	if err != nil {
		if err == context.Canceled {
			if sci.interrupted {
				// User interrupted, this is expected
				return
			}
		}
		if !sci.interrupted {
			fmt.Printf("\n❌ Error: %v\n", err)
		}
	}
	
	fmt.Println() // New line after response
	fmt.Println() // Extra line for readability
}

// checkAgentSuggestion checks if a different agent would be better
func (sci *SimpleChatInterfaceV2) checkAgentSuggestion(userInput string) {
	suggestedAgent, reason, err := sci.agentManager.SuggestAgentSwitch(userInput)
	if err == nil && suggestedAgent != nil {
		fmt.Printf("💡 Suggestion: %s might be better for this query. %s\n", suggestedAgent.Metadata.Name, reason)
		fmt.Println("   Type '/switch' to return to agent menu")
		fmt.Println()
	}
}

// setupSignalHandling configures signal handling for Ctrl+C
func (sci *SimpleChatInterfaceV2) setupSignalHandling() {
	sci.interrupted = false
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT)
	
	go func() {
		<-signalChan
		sci.interrupted = true
		if sci.cancelFunc != nil {
			sci.cancelFunc() // Cancel any ongoing operations
		}
		fmt.Println("\n\n🔙 Returning to agent menu...")
		// The main loop will detect sci.interrupted and exit
	}()
}

// cleanupSignalHandling stops signal handling
func (sci *SimpleChatInterfaceV2) cleanupSignalHandling() {
	signal.Reset(syscall.SIGINT)
}

// showChatHelp displays help information during chat
func (sci *SimpleChatInterfaceV2) showChatHelp() {
	fmt.Println("\n📚 Chat Commands:")
	fmt.Println("  /help     - Show this help")
	fmt.Println("  /switch   - Return to agent menu")
	fmt.Println("  /quit     - Exit the application")
	fmt.Println("  /metrics  - Show engine metrics")
	fmt.Println("  /health   - Show engine health")
	fmt.Println("  Ctrl+C    - Return to agent menu")
	fmt.Println("  Ctrl+D    - Exit the application")
	fmt.Println()
}

// showEngineMetrics displays current engine metrics
func (sci *SimpleChatInterfaceV2) showEngineMetrics() {
	fmt.Println("\n📊 Engine Metrics:")
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	metrics, err := sci.engineClient.GetMetrics(ctx)
	if err != nil {
		fmt.Printf("❌ Failed to get metrics: %v\n", err)
		return
	}
	
	fmt.Printf("  📈 Total Requests: %d\n", metrics.TotalRequests)
	fmt.Printf("  ✅ Successful: %d\n", metrics.SuccessfulRequests)
	fmt.Printf("  ❌ Failed: %d\n", metrics.FailedRequests)
	fmt.Printf("  ⏱️ Avg Response Time: %v\n", metrics.AverageResponseTime)
	fmt.Printf("  🔄 Active Requests: %d\n", metrics.ActiveRequests)
	fmt.Printf("  📝 Tokens Generated: %d\n", metrics.TokensGenerated)
	fmt.Printf("  💾 Memory Usage: %.2f MB\n", float64(metrics.MemoryUsage)/1024/1024)
	fmt.Printf("  🕐 Uptime: %v\n", metrics.Uptime)
	
	if metrics.ModelLoaded {
		fmt.Printf("  🤖 Model: %s (loaded)\n", metrics.ModelPath)
	} else {
		fmt.Println("  🤖 Model: Not loaded")
	}
	
	if metrics.LastError != "" {
		fmt.Printf("  ⚠️ Last Error: %s\n", metrics.LastError)
	}
	
	fmt.Println()
}

// showEngineHealth displays engine health status
func (sci *SimpleChatInterfaceV2) showEngineHealth() {
	fmt.Println("\n🏥 Engine Health:")
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	health, err := sci.engineClient.HealthCheck(ctx)
	if err != nil {
		fmt.Printf("❌ Failed to get health status: %v\n", err)
		return
	}
	
	var statusIcon string
	switch health.Status {
	case "healthy":
		statusIcon = "🟢"
	case "degraded":
		statusIcon = "🟡"
	case "unhealthy":
		statusIcon = "🔴"
	default:
		statusIcon = "⚪"
	}
	
	fmt.Printf("  %s Status: %s\n", statusIcon, health.Status)
	fmt.Printf("  🕐 Uptime: %v\n", health.Uptime)
	fmt.Printf("  🤖 Model Loaded: %v\n", health.ModelLoaded)
	fmt.Printf("  📅 Last Check: %v\n", health.Timestamp.Format("15:04:05"))
	
	if health.LastError != "" {
		fmt.Printf("  ⚠️ Last Error: %s\n", health.LastError)
	}
	
	fmt.Println()
}

// clearScreen clears the terminal screen
func (sci *SimpleChatInterfaceV2) clearScreen() {
	fmt.Print("\033[2J\033[H")
}

// printWrapped prints text with word wrapping at the specified width
func (sci *SimpleChatInterfaceV2) printWrapped(text string, width int) {
	// Split into paragraphs first
	paragraphs := strings.Split(text, "\n\n")
	
	for i, paragraph := range paragraphs {
		if i > 0 {
			fmt.Println() // Add blank line between paragraphs
		}
		
		// Split into lines (respect existing line breaks)
		lines := strings.Split(paragraph, "\n")
		
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				fmt.Println()
				continue
			}
			
			// Word wrap this line
			words := strings.Fields(line)
			currentLine := ""
			
			for _, word := range words {
				// Check if adding this word would exceed the width
				testLine := currentLine
				if testLine != "" {
					testLine += " "
				}
				testLine += word
				
				if len(testLine) <= width {
					currentLine = testLine
				} else {
					// Print current line and start a new one
					if currentLine != "" {
						fmt.Println(currentLine)
					}
					currentLine = word
				}
			}
			
			// Print any remaining content
			if currentLine != "" {
				fmt.Println(currentLine)
			}
		}
	}
}