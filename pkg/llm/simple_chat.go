package llm

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/opentdf/otdfctl/pkg/config"
)

// SimpleChatInterface provides a basic chat interface without Bubbletea
type SimpleChatInterface struct {
	agentManager *AgentManager
	chatEngine   *SimpleChatEngine
	currentAgent *Agent
	config       *config.Config
	interrupted  bool
}

// NewSimpleChatInterface creates a new simple chat interface
func NewSimpleChatInterface(agentManager *AgentManager, chatEngine *SimpleChatEngine, cfg *config.Config) *SimpleChatInterface {
	return &SimpleChatInterface{
		agentManager: agentManager,
		chatEngine:   chatEngine,
		config:       cfg,
	}
}

// Run starts the simple chat interface
func (sci *SimpleChatInterface) Run() error {
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
		
		// Start chat with selected agent
		if sci.startChat() {
			return nil // User pressed Ctrl+D to exit
		}
		// User pressed Ctrl+C to go back to menu, continue loop
	}
}

// showAgentMenu displays the agent selection menu
func (sci *SimpleChatInterface) showAgentMenu() (*Agent, bool) {
	for {
		sci.clearScreen()
		fmt.Println("🤖 OpenTDF Agent Chat")
		fmt.Println("=====================")
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
		selection, err := strconv.Atoi(input)
		if err != nil || selection < 1 || selection > len(agents) {
			fmt.Printf("Invalid selection. Please enter a number between 1 and %d.\n", len(agents))
			fmt.Println("Press Enter to continue...")
			reader.ReadLine()
			continue
		}
		
		return agents[selection-1], false
	}
}

// startChat begins a chat session with the selected agent
func (sci *SimpleChatInterface) startChat() bool {
	sci.clearScreen()
	fmt.Printf("🤖 Chatting with: %s (%s)\n", sci.currentAgent.Metadata.Name, sci.currentAgent.Type)
	fmt.Println("=====================================")
	fmt.Printf("📝 %s\n", sci.currentAgent.Metadata.Description)
	fmt.Println()
	fmt.Println("💡 Tips:")
	fmt.Println("  • Type your message and press Enter")
	fmt.Println("  • Press Ctrl+C to return to agent menu")
	fmt.Println("  • Press Ctrl+D to exit")
	fmt.Println("  • Type '/switch' to return to menu")
	fmt.Println()

	// Set up signal handling for Ctrl+C
	sci.interrupted = false
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT)
	
	go func() {
		<-signalChan
		sci.interrupted = true
		fmt.Println("\n\n🔙 Returning to agent menu...")
		// Note: This will be detected on next input attempt
	}()
	
	defer func() {
		signal.Stop(signalChan)
		close(signalChan)
	}()

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
		if userInput == "/help" {
			sci.showChatHelp()
			continue
		}
		
		if userInput == "/switch" {
			fmt.Println("\n🔄 Returning to agent menu...")
			return false
		}
		
		if userInput == "/quit" || userInput == "/exit" {
			fmt.Println("\n👋 Goodbye!")
			return true
		}
		
		// Check for agent suggestions
		sci.checkAgentSuggestion(userInput)
		
		// Generate response
		fmt.Printf("\n🤖 %s:\n", sci.currentAgent.Metadata.Name)
		err = sci.generateAndStreamResponse(userInput)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		}
		fmt.Println() // Extra line for readability
	}
}

// checkAgentSuggestion checks if a different agent would be better
func (sci *SimpleChatInterface) checkAgentSuggestion(userInput string) {
	suggestedAgent, reason, err := sci.agentManager.SuggestAgentSwitch(userInput)
	if err == nil && suggestedAgent != nil {
		fmt.Printf("💡 Suggestion: %s might be better for this query. %s\n", suggestedAgent.Metadata.Name, reason)
		fmt.Println("   Type '/switch' to return to agent menu")
		fmt.Println()
	}
}

// generateAndStreamResponse generates a response from the current agent
func (sci *SimpleChatInterface) generateAndStreamResponse(userInput string) error {
	if sci.currentAgent == nil || sci.chatEngine == nil {
		return fmt.Errorf("no agent selected or chat engine not available")
	}

	// Build messages with current agent's system prompt and user input only
	messages := []ChatMessage{
		{Role: "system", Content: sci.currentAgent.SystemPrompt},
		{Role: "user", Content: userInput},
	}

	// Generate response
	response := sci.chatEngine.Chat(messages)
	if response.Error != nil {
		return response.Error
	}

	// Clean up the response content
	content := strings.TrimSpace(response.Content)
	
	// Remove common model artifacts
	content = strings.ReplaceAll(content, "<|im_start|>", "")
	content = strings.ReplaceAll(content, "<|im_end|>", "")
	content = strings.ReplaceAll(content, "<|start|>", "")
	content = strings.ReplaceAll(content, "<|end|>", "")
	content = strings.ReplaceAll(content, "assistant\n", "")
	content = strings.ReplaceAll(content, "Assistant:", "")
	content = strings.TrimSpace(content)
	
	if content == "" {
		fmt.Println("(No response generated)")
		return nil
	}
	
	// Print response with proper formatting and word wrapping
	sci.printWrapped(content, 80)

	return nil
}

// showChatHelp displays help information during chat
func (sci *SimpleChatInterface) showChatHelp() {
	fmt.Println("\n📚 Chat Commands:")
	fmt.Println("  /help     - Show this help")
	fmt.Println("  /switch   - Return to agent menu")
	fmt.Println("  /quit     - Exit the application")
	fmt.Println("  Ctrl+C    - Return to agent menu")
	fmt.Println("  Ctrl+D    - Exit the application")
	fmt.Println()
}

// clearScreen clears the terminal screen
func (sci *SimpleChatInterface) clearScreen() {
	fmt.Print("\033[2J\033[H")
}

// printWrapped prints text with word wrapping at the specified width
func (sci *SimpleChatInterface) printWrapped(text string, width int) {
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