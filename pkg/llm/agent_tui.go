package llm

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Streaming configuration constants
const (
	ChannelBufferSize = 100                      // Tokens buffered
	TickInterval      = 50 * time.Millisecond    // UI update frequency  
	MaxTokensPerTick  = 10                       // Throttling limit
	StreamingTimeout  = 30 * time.Second         // Overall timeout
)

// RunAgentChatTUI starts the enhanced Bubbletea TUI interface
func RunAgentChatTUI(model *AgentChatModel) error {
	program := tea.NewProgram(model, tea.WithAltScreen(), tea.WithMouseCellMotion())
	_, err := program.Run()
	return err
}

// AgentChatModel represents the main TUI model for agent-based chat
type AgentChatModel struct {
	// Core components
	agentManager    *AgentManager
	chatEngine      *SimpleChatEngine
	
	// UI components
	viewport        viewport.Model
	textarea        textarea.Model
	agentList       list.Model
	help           help.Model
	
	// State
	messages        []ChatMessage
	currentAgent    *Agent
	showAgentList   bool
	showHelp        bool
	
	// Context management
	contextUsage    ContextUsage
	conversationTokens int
	maxContextTokens   int
	summarizationManager *AutoSummarizationManager
	conversationID     string
	
	// UI state
	width           int
	height          int
	ready           bool
	quitting        bool
	
	// Channel-based streaming state
	streamingCh       chan string
	streamingDone     chan error
	streamingActive   bool
	streamingContent  strings.Builder
	streamingAgent    *Agent
	responseStartTime time.Time
	responseEndTime   time.Time
	tokenBatch        []string
	
	// Keybindings
	keys            AgentChatKeyMap
}

// ContextUsage tracks context window usage (Claude Code style)
type ContextUsage struct {
	UsedTokens    int
	MaxTokens     int
	Percentage    float64
	NeedsCleanup  bool
}

// AgentChatKeyMap defines keyboard shortcuts
type AgentChatKeyMap struct {
	Send         key.Binding
	SwitchAgent  key.Binding
	ShowAgents   key.Binding
	ClearHistory key.Binding
	Summarize    key.Binding
	ShowHelp     key.Binding
	Quit         key.Binding
	Up           key.Binding
	Down         key.Binding
	Enter        key.Binding
	Escape       key.Binding
}

// DefaultAgentChatKeyMap returns the default keybindings
func DefaultAgentChatKeyMap() AgentChatKeyMap {
	return AgentChatKeyMap{
		Send: key.NewBinding(
			key.WithKeys("enter"),
			key.WithHelp("enter", "send message"),
		),
		SwitchAgent: key.NewBinding(
			key.WithKeys("ctrl+s"),
			key.WithHelp("ctrl+s", "switch agent"),
		),
		ShowAgents: key.NewBinding(
			key.WithKeys("ctrl+a"),
			key.WithHelp("ctrl+a", "show agents"),
		),
		ClearHistory: key.NewBinding(
			key.WithKeys("ctrl+l"),
			key.WithHelp("ctrl+l", "clear history"),
		),
		Summarize: key.NewBinding(
			key.WithKeys("ctrl+u"),
			key.WithHelp("ctrl+u", "summarize"),
		),
		ShowHelp: key.NewBinding(
			key.WithKeys("ctrl+h", "?"),
			key.WithHelp("ctrl+h/?", "show help"),
		),
		Quit: key.NewBinding(
			key.WithKeys("ctrl+c", "esc"),
			key.WithHelp("ctrl+c/esc", "quit"),
		),
		Up: key.NewBinding(key.WithKeys("up")),
		Down: key.NewBinding(key.WithKeys("down")),
		Enter: key.NewBinding(key.WithKeys("enter")),
		Escape: key.NewBinding(key.WithKeys("esc")),
	}
}

// ShortHelp returns key help
func (k AgentChatKeyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Send, k.SwitchAgent, k.ShowAgents, k.Quit}
}

// FullHelp returns full key help
func (k AgentChatKeyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Send, k.SwitchAgent, k.ShowAgents},
		{k.ClearHistory, k.Summarize, k.ShowHelp, k.Quit},
	}
}

// NewAgentChatModel creates a new agent chat TUI model
func NewAgentChatModel(agentManager *AgentManager, chatEngine *SimpleChatEngine) *AgentChatModel {
	// Initialize textarea for input
	ta := textarea.New()
	ta.Placeholder = "Ask me anything about OpenTDF... (Enter to send, Shift+Enter for new line)"
	ta.Focus()
	ta.Prompt = "┃ "
	ta.CharLimit = 2000
	ta.SetWidth(80)
	ta.SetHeight(3)
	ta.ShowLineNumbers = false
	// Allow multi-line input but we'll handle Enter key manually
	ta.KeyMap.InsertNewline.SetKeys("shift+enter")
	
	// Set transparent/default colors for textarea
	ta.FocusedStyle.Base = lipgloss.NewStyle()
	ta.BlurredStyle.Base = lipgloss.NewStyle()
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()
	ta.BlurredStyle.CursorLine = lipgloss.NewStyle()
	
	// Initialize viewport for chat history
	vp := viewport.New(80, 20)
	
	// Initialize agent list
	agents := agentManager.ListAgents()
	agentItems := make([]list.Item, len(agents))
	for i, agent := range agents {
		agentItems[i] = AgentListItem{agent: agent}
	}
	
	agentList := list.New(agentItems, NewAgentItemDelegate(), 30, 10)
	agentList.Title = "Available Agents"
	agentList.SetShowStatusBar(false)
	agentList.SetFilteringEnabled(true)
	
	return &AgentChatModel{
		agentManager:       agentManager,
		chatEngine:        chatEngine,
		viewport:          vp,
		textarea:          ta,
		agentList:         agentList,
		help:             help.New(),
		messages:          []ChatMessage{},
		showAgentList:     false,
		showHelp:         false,
		maxContextTokens: 4096,
		keys:             DefaultAgentChatKeyMap(),
		summarizationManager: NewAutoSummarizationManager(),
		conversationID:     fmt.Sprintf("chat_%d", time.Now().Unix()),
	}
}

// AgentListItem represents an agent in the selection list
type AgentListItem struct {
	agent *Agent
}

func (i AgentListItem) FilterValue() string {
	return i.agent.Metadata.Name + " " + i.agent.Metadata.Description
}

func (i AgentListItem) Title() string {
	return i.agent.Metadata.Name
}

func (i AgentListItem) Description() string {
	return fmt.Sprintf("%s • %s", i.agent.Type, i.agent.Metadata.Description)
}

// AgentItemDelegate handles rendering of agent list items
type AgentItemDelegate struct{}

func NewAgentItemDelegate() AgentItemDelegate {
	return AgentItemDelegate{}
}

func (d AgentItemDelegate) Height() int                               { return 2 }
func (d AgentItemDelegate) Spacing() int                             { return 1 }
func (d AgentItemDelegate) Update(msg tea.Msg, m *list.Model) tea.Cmd { return nil }

func (d AgentItemDelegate) Render(w io.Writer, m list.Model, index int, listItem list.Item) {
	i, ok := listItem.(AgentListItem)
	if !ok {
		return
	}
	
	agent := i.agent
	title := agent.Metadata.Name
	desc := fmt.Sprintf("%s • %s", agent.Type, agent.Metadata.Description)
	
	// Style based on selection
	titleStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#FAFAFA"))
	descStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#626262"))
	
	if index == m.Index() {
		titleStyle = titleStyle.Foreground(lipgloss.Color("#EE6FF8")).Bold(true)
		descStyle = descStyle.Foreground(lipgloss.Color("#AD58B4"))
	}
	
	fmt.Fprintf(w, "%s\n%s", titleStyle.Render(title), descStyle.Render(desc))
}

// Init initializes the model
func (m *AgentChatModel) Init() tea.Cmd {
	return tea.Batch(
		textarea.Blink,
		m.selectDefaultAgent(),
	)
}

// selectDefaultAgent selects the general agent as default
func (m *AgentChatModel) selectDefaultAgent() tea.Cmd {
	return func() tea.Msg {
		agent, err := m.agentManager.GetAgentByType(AgentTypeGeneral)
		if err != nil {
			// Fallback to first available agent
			agents := m.agentManager.ListAgents()
			if len(agents) > 0 {
				agent = agents[0]
			}
		}
		
		if agent != nil {
			return AgentSelectedMsg{agent: agent}
		}
		
		return nil
	}
}

// AgentSelectedMsg is sent when an agent is selected
type AgentSelectedMsg struct {
	agent *Agent
}

// ResponseMsg is sent when we receive a chat response
type ResponseMsg struct {
	response string
	agent    *Agent
	tokens   int
}

// StreamingTokenMsg is sent for each token during streaming
type StreamingTokenMsg struct {
	token string
	agent *Agent
}

// StreamingCompleteMsg is sent when streaming is finished
type StreamingCompleteMsg struct {
	fullResponse string
	agent        *Agent
	tokens       int
	error        error
}

// StreamingTickMsg is sent periodically to check for new tokens
type StreamingTickMsg struct{}

// StreamingTokensMsg is sent with batched tokens
type StreamingTokensMsg struct {
	Tokens []string
}

// StreamingStartMsg is sent when streaming begins
type StreamingStartMsg struct {
	agent *Agent
}

// Update handles messages and updates the model
func (m *AgentChatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		cmd  tea.Cmd
		cmds []tea.Cmd
	)
	
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case key.Matches(msg, m.keys.Quit):
			m.quitting = true
			return m, tea.Quit
			
		case key.Matches(msg, m.keys.ShowHelp):
			m.showHelp = !m.showHelp
			return m, nil
			
		case key.Matches(msg, m.keys.ShowAgents):
			m.showAgentList = !m.showAgentList
			if !m.showAgentList {
				m.textarea.Focus()
			}
			return m, nil
			
		case key.Matches(msg, m.keys.ClearHistory):
			m.messages = []ChatMessage{}
			m.conversationTokens = 0
			m.updateContextUsage()
			m.updateViewport()
			return m, nil
			
		case key.Matches(msg, m.keys.Summarize):
			return m.handleManualSummarization()
			
			
		// Handle Enter key based on context and modifiers
		case msg.String() == "enter":
			if m.showAgentList {
				return m.handleAgentSelection()
			} else {
				// Plain Enter sends message, Shift+Enter is handled by textarea for new lines
				return m.handleSendMessage()
			}
			
		case key.Matches(msg, m.keys.Escape):
			if m.showAgentList {
				m.showAgentList = false
				m.textarea.Focus()
				return m, nil
			}
			if m.showHelp {
				m.showHelp = false
				return m, nil
			}
		}
		
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		
		if !m.ready {
			m.ready = true
		}
		
		// Update component sizes
		m.updateSizes()
		return m, nil
		
	case AgentSelectedMsg:
		m.currentAgent = msg.agent
		// Only show switch message if this isn't the initial agent selection
		if len(m.messages) > 0 {
			m.addSystemMessage(fmt.Sprintf("🤖 Switched to %s", msg.agent.Metadata.Name))
		}
		return m, nil
		
	case ResponseMsg:
		m.addMessage(ChatMessage{
			Role:    "assistant",
			Content: msg.response,
		})
		m.conversationTokens += msg.tokens
		m.updateContextUsage()
		m.updateViewport()
		return m, nil
		
	case StreamingTickMsg:
		if !m.streamingActive {
			// Not streaming, ignore tick
			return m, nil
		}
		
		// Check for completion first
		select {
		case err := <-m.streamingDone:
			// Streaming completed
			m.streamingActive = false
			m.responseEndTime = time.Now()
			
			// Streaming completed
			fullResponse := m.streamingContent.String()
			
			if err != nil {
				m.addSystemMessage(fmt.Sprintf("❌ Error: %v", err))
			} else {
				// Add the complete response
				m.addMessage(ChatMessage{
					Role:    "assistant",
					Content: fullResponse,
				})
				m.conversationTokens += len(fullResponse) / 4 // Rough token estimate
				m.updateContextUsage()
				
				// Add timing info
				duration := m.responseEndTime.Sub(m.responseStartTime)
				m.addTimingMessage(fmt.Sprintf("Response generated in %v", duration.Round(time.Millisecond)))
			}
			m.cleanupStreaming()
			return m, nil
		default:
			// Continue streaming
		}
		
		// Drain available tokens
		tokens := m.drainAvailableTokens()
		if len(tokens) > 0 {
			// Send token batch and schedule next tick
			return m, tea.Batch(
				func() tea.Msg {
					return StreamingTokensMsg{Tokens: tokens}
				},
				m.createTokenTick(),
			)
		}
		
		// No tokens available, just schedule next tick
		return m, m.createTokenTick()
		
	case StreamingTokensMsg:
		if m.streamingActive {
			// Process batched tokens
			for _, token := range msg.Tokens {
				m.streamingContent.WriteString(token)
			}
			// Process batched tokens
			// Update the viewport with current streaming content
			m.updateStreamingViewport()
		}
		return m, nil
		
	case StreamingStartMsg:
		m.streamingActive = true
		m.streamingContent.Reset()
		m.streamingAgent = msg.agent
		m.responseStartTime = time.Now()
		
		// Start streaming
		// Update viewport to show streaming cursor immediately
		m.updateStreamingViewport()
		return m, nil
		
	case StreamingCompleteMsg:
		// Legacy message type - handle for backward compatibility but this shouldn't be used
		m.streamingActive = false
		m.responseEndTime = time.Now()
		
		if msg.error != nil {
			m.addSystemMessage(fmt.Sprintf("❌ Error: %v", msg.error))
		} else {
			m.addMessage(ChatMessage{
				Role:    "assistant",
				Content: msg.fullResponse,
			})
			m.conversationTokens += msg.tokens
			m.updateContextUsage()
			
			duration := m.responseEndTime.Sub(m.responseStartTime)
			m.addTimingMessage(fmt.Sprintf("Response generated in %v", duration.Round(time.Millisecond)))
		}
		m.streamingContent.Reset()
		return m, nil
		
	case SummarizationCompleteMsg:
		if msg.success {
			m.messages = msg.summarizedMessages
			m.conversationTokens -= msg.summary.TokensSaved // Reduce token count
			m.updateContextUsage()
			m.addSystemMessage(fmt.Sprintf("✅ Conversation summarized (saved ~%d tokens)", msg.summary.TokensSaved))
			m.addSystemMessage("💡 Summary preserved for context continuity")
		} else {
			m.addSystemMessage(fmt.Sprintf("❌ Summarization failed: %v", msg.error))
		}
		return m, nil
	}
	
	// Update components based on current state
	if m.showAgentList {
		m.agentList, cmd = m.agentList.Update(msg)
		cmds = append(cmds, cmd)
	} else {
		// Don't pass Enter key to textarea if we handled it for sending
		if keyMsg, ok := msg.(tea.KeyMsg); ok {
			switch keyMsg.String() {
			case "enter":
				// We already handled Enter for sending, don't pass to textarea
				// This prevents newlines from being added after sending
			default:
				m.textarea, cmd = m.textarea.Update(msg)
				cmds = append(cmds, cmd)
			}
		} else {
			m.textarea, cmd = m.textarea.Update(msg)
			cmds = append(cmds, cmd)
		}
	}
	
	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)
	
	return m, tea.Batch(cmds...)
}

// handleSendMessage processes user input and sends to agent
func (m *AgentChatModel) handleSendMessage() (tea.Model, tea.Cmd) {
	userInput := strings.TrimSpace(m.textarea.Value())
	if userInput == "" {
		return m, nil
	}
	
	// Add user message
	m.addMessage(ChatMessage{
		Role:    "user", 
		Content: userInput,
	})
	
	// Clear textarea
	m.textarea.SetValue("")
	
	// Recalculate sizes after adding message
	m.updateSizes()
	
	// Check if we need to suggest agent switch
	if suggestedAgent, reason, err := m.agentManager.SuggestAgentSwitch(userInput); err == nil && suggestedAgent != nil {
		m.addSystemMessage(fmt.Sprintf("💡 Suggestion: Switch to %s? %s (Press Ctrl+S to switch)", 
			suggestedAgent.Metadata.Name, reason))
	}
	
	// Send to current agent
	return m, m.generateResponse(userInput)
}

// handleAgentSelection processes agent selection from the list
func (m *AgentChatModel) handleAgentSelection() (tea.Model, tea.Cmd) {
	if selectedItem, ok := m.agentList.SelectedItem().(AgentListItem); ok {
		m.currentAgent = selectedItem.agent
		m.showAgentList = false
		m.textarea.Focus()
		
		m.addSystemMessage(fmt.Sprintf("🤖 Switched to %s", selectedItem.agent.Metadata.Name))
		
		return m, nil
	}
	return m, nil
}

// handleManualSummarization processes manual summarization request
func (m *AgentChatModel) handleManualSummarization() (tea.Model, tea.Cmd) {
	if len(m.messages) < 4 { // Need at least a few messages to summarize
		m.addSystemMessage("⚠️ Need more conversation to summarize")
		return m, nil
	}
	
	if m.summarizationManager == nil || m.chatEngine == nil {
		m.addSystemMessage("⚠️ Summarization not available")
		return m, nil
	}
	
	m.addSystemMessage("📝 Creating conversation summary...")
	
	return m, m.performSummarization()
}

// performSummarization executes the summarization process
func (m *AgentChatModel) performSummarization() tea.Cmd {
	return func() tea.Msg {
		summarizer := m.summarizationManager.GetSummarizer(m.conversationID, m.chatEngine)
		
		// Force summarization regardless of context usage
		summary, err := summarizer.SummarizeConversation(m.messages, m.currentAgent)
		if err != nil {
			return SummarizationCompleteMsg{
				success: false,
				error:   err,
			}
		}
		
		// Apply summary
		summarizedMessages := summarizer.ApplySummary(m.messages, summary)
		
		return SummarizationCompleteMsg{
			success:           true,
			summary:           summary,
			summarizedMessages: summarizedMessages,
		}
	}
}

// SummarizationCompleteMsg represents completion of summarization
type SummarizationCompleteMsg struct {
	success           bool
	summary           *ConversationSummary
	summarizedMessages []ChatMessage
	error             error
}

// generateResponse sends the query to the current agent with streaming
func (m *AgentChatModel) generateResponse(userInput string) tea.Cmd {
	return tea.Batch(
		// First, send the streaming start message
		func() tea.Msg {
			return StreamingStartMsg{agent: m.currentAgent}
		},
		// Start channels and tick processing
		m.startStreaming(),
		// Then start the streaming generation
		m.generateStreamingResponse(userInput),
	)
}

// generateStreamingResponse handles the actual streaming generation
func (m *AgentChatModel) generateStreamingResponse(userInput string) tea.Cmd {
	return func() tea.Msg {
		if m.currentAgent == nil || m.chatEngine == nil {
			if m.streamingDone != nil {
				m.streamingDone <- fmt.Errorf("no agent selected or chat engine not available")
			}
			return nil
		}
		
		// Build messages with current agent's system prompt and ONLY the current user input
		// DO NOT send conversation history to prevent feedback loops
		messages := []ChatMessage{
			{Role: "system", Content: m.currentAgent.SystemPrompt},
			{Role: "user", Content: userInput},
		}
		
		// Start the streaming in a goroutine with timeout
		go func() {
			// Create context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), StreamingTimeout)
			defer cancel()
			
			var fullResponse strings.Builder
			tokenCount := 0
			
			// Channel to collect result
			resultCh := make(chan error, 1)
			
			// Start the actual streaming in another goroutine
			go func() {
				defer func() {
					if r := recover(); r != nil {
						resultCh <- fmt.Errorf("streaming panic: %v", r)
					}
				}()
				
				// Streaming callback that sends tokens to the model's channel
				callback := func(token string) {
					tokenCount++
					fullResponse.WriteString(token)
					
					// Try to send token to the model's channel (non-blocking)
					if m.streamingCh != nil {
						select {
						case m.streamingCh <- token:
							// Token sent successfully to channel
						default:
							// Channel full, skip this token to prevent blocking
						}
					}
				}
				
				// Generate response using streaming chat engine
				response := m.chatEngine.ChatStream(messages, callback)
				resultCh <- response.Error
			}()
			
			// Wait for completion or timeout
			select {
			case err := <-resultCh:
				// Streaming completed
				if m.streamingDone != nil {
					m.streamingDone <- err
				}
			case <-ctx.Done():
				// Timeout occurred
				if m.streamingDone != nil {
					m.streamingDone <- fmt.Errorf("streaming timeout after %v", StreamingTimeout)
				}
			}
		}()
		
		// Return nil since the tick system will handle the streaming
		return nil
	}
}


// Channel management methods

// startStreaming creates channels and starts the streaming process
func (m *AgentChatModel) startStreaming() tea.Cmd {
	// Create buffered channels with capacity from constants
	m.streamingCh = make(chan string, ChannelBufferSize)
	m.streamingDone = make(chan error, 1)
	m.streamingActive = true
	m.tokenBatch = make([]string, 0, MaxTokensPerTick)
	
	// Return first tick command
	return m.createTokenTick()
}

// drainAvailableTokens reads available tokens from channel (non-blocking)
func (m *AgentChatModel) drainAvailableTokens() []string {
	var tokens []string
	
	for len(tokens) < MaxTokensPerTick {
		select {
		case token, ok := <-m.streamingCh:
			if !ok {
				// Channel closed
				return tokens
			}
			tokens = append(tokens, token)
		default:
			// No more tokens available, return what we have
			return tokens
		}
	}
	
	return tokens
}

// cleanupStreaming safely closes channels and resets state
func (m *AgentChatModel) cleanupStreaming() {
	if m.streamingCh != nil {
		close(m.streamingCh)
		m.streamingCh = nil
	}
	if m.streamingDone != nil {
		close(m.streamingDone)
		m.streamingDone = nil
	}
	m.streamingActive = false
	m.tokenBatch = nil
}

// createTokenTick creates a periodic tick command for checking tokens
func (m *AgentChatModel) createTokenTick() tea.Cmd {
	return tea.Tick(TickInterval, func(time.Time) tea.Msg {
		return StreamingTickMsg{}
	})
}

// addMessage adds a message to the conversation
func (m *AgentChatModel) addMessage(msg ChatMessage) {
	m.messages = append(m.messages, msg)
	
	// Check for auto-summarization after adding messages
	if msg.Role == "assistant" {
		m.checkAutoSummarization()
	}
	
	// Recalculate sizes when content changes
	m.updateSizes()
	m.updateViewport()
}

// addSystemMessage adds a system notification
func (m *AgentChatModel) addSystemMessage(text string) {
	m.addMessage(ChatMessage{
		Role:    "system",
		Content: text,
	})
}

// addTimingMessage adds a low-focus timing message
func (m *AgentChatModel) addTimingMessage(text string) {
	m.addMessage(ChatMessage{
		Role:    "timing",
		Content: text,
	})
}

// updateViewport refreshes the chat display
func (m *AgentChatModel) updateViewport() {
	var content strings.Builder
	
	for _, msg := range m.messages {
		switch msg.Role {
		case "user":
			content.WriteString(m.renderUserMessage(msg.Content))
		case "assistant":
			content.WriteString(m.renderAssistantMessage(msg.Content))
		case "system":
			content.WriteString(m.renderSystemMessage(msg.Content))
		case "timing":
			content.WriteString(m.renderTimingMessage(msg.Content))
		}
		content.WriteString("\n\n")
	}
	
	m.viewport.SetContent(content.String())
	m.viewport.GotoBottom()
}

// updateStreamingViewport refreshes the chat display with streaming content
func (m *AgentChatModel) updateStreamingViewport() {
	var content strings.Builder
	
	// Render all existing messages
	for _, msg := range m.messages {
		switch msg.Role {
		case "user":
			content.WriteString(m.renderUserMessage(msg.Content))
		case "assistant":
			content.WriteString(m.renderAssistantMessage(msg.Content))
		case "system":
			content.WriteString(m.renderSystemMessage(msg.Content))
		case "timing":
			content.WriteString(m.renderTimingMessage(msg.Content))
		}
		content.WriteString("\n\n")
	}
	
	// Add streaming content if we're currently streaming
	if m.streamingActive {
		// Show streaming even if no tokens yet
		streamingResponse := m.streamingContent.String() + "▊" // Typing cursor
		content.WriteString(m.renderStreamingAssistantMessage(streamingResponse))
		content.WriteString("\n\n")
	}
	
	m.viewport.SetContent(content.String())
	m.viewport.GotoBottom()
}

// updateContextUsage calculates context window usage
func (m *AgentChatModel) updateContextUsage() {
	m.contextUsage.UsedTokens = m.conversationTokens
	m.contextUsage.MaxTokens = m.maxContextTokens
	m.contextUsage.Percentage = float64(m.contextUsage.UsedTokens) / float64(m.contextUsage.MaxTokens) * 100
	m.contextUsage.NeedsCleanup = m.contextUsage.Percentage > 80.0
}

// checkAutoSummarization handles automatic conversation summarization
func (m *AgentChatModel) checkAutoSummarization() {
	if m.summarizationManager == nil || m.chatEngine == nil {
		return
	}
	
	m.updateContextUsage()
	
	// Check if auto-summarization is needed
	summarizedMessages, summary, err := m.summarizationManager.HandleAutoSummarization(
		m.conversationID,
		m.messages,
		m.contextUsage,
		m.currentAgent,
		m.chatEngine,
	)
	
	if err != nil {
		m.addSystemMessage(fmt.Sprintf("⚠️ Auto-summarization failed: %v", err))
		return
	}
	
	if summary != nil {
		// Summarization occurred
		m.messages = summarizedMessages
		m.conversationTokens = summary.TokensSaved // Rough approximation
		m.updateContextUsage()
		
		m.addSystemMessage(fmt.Sprintf("📝 Auto-summarized conversation (saved ~%d tokens)", summary.TokensSaved))
		m.addSystemMessage("💡 Previous conversation summary preserved for context")
	}
}

// updateSizes adjusts component sizes based on window size
func (m *AgentChatModel) updateSizes() {
	if m.width <= 0 || m.height <= 0 {
		return
	}
	
	// Reserve space for header (3), input area (6), footer (2), and padding
	headerHeight := 3
	inputAreaHeight := 6  // Input box with border + separator + margin
	footerHeight := 2
	padding := 2
	
	availableHeight := m.height - headerHeight - inputAreaHeight - footerHeight - padding
	availableWidth := m.width - 2 // Less padding for full width usage
	
	// Ensure minimum viewport height
	viewportHeight := availableHeight
	if viewportHeight < 6 {
		viewportHeight = 6
	}
	
	// Account for chat view border (border adds 4 to height: 2 top + 2 bottom)
	chatViewInnerHeight := viewportHeight - 4
	if chatViewInnerHeight < 4 {
		chatViewInnerHeight = 4
	}
	
	// Account for chat view border (border adds 4 to width: 2 left + 2 right) 
	chatViewInnerWidth := availableWidth - 4
	if chatViewInnerWidth < 20 {
		chatViewInnerWidth = 20
	}
	
	// Set viewport size for chat history (inner content)
	m.viewport.Width = chatViewInnerWidth
	m.viewport.Height = chatViewInnerHeight
	
	// Set textarea size for input (accounts for input border)
	m.textarea.SetWidth(availableWidth - 4) // Account for border padding
	m.textarea.SetHeight(3) // Fixed height for input
	
	// Agent list sizing
	if m.showAgentList {
		listWidth := availableWidth / 3
		if listWidth < 30 {
			listWidth = 30
		}
		m.agentList.SetSize(listWidth, chatViewInnerHeight)
	}
}

// View renders the complete interface
func (m *AgentChatModel) View() string {
	if !m.ready {
		return "Initializing..."
	}
	
	if m.quitting {
		return "Thanks for using OpenTDF Agent Chat! 👋\n"
	}
	
	// Main layout
	var sections []string
	
	// Header with current agent and context usage
	sections = append(sections, m.renderHeader())
	
	// Main content area
	if m.showAgentList {
		sections = append(sections, m.renderWithAgentList())
	} else {
		sections = append(sections, m.renderChatView())
	}
	
	// Input area
	sections = append(sections, m.renderInputArea())
	
	// Footer with help or status
	if m.showHelp {
		sections = append(sections, m.renderHelp())
	} else {
		sections = append(sections, m.renderFooter())
	}
	
	return lipgloss.JoinVertical(lipgloss.Left, sections...)
}

// renderHeader shows agent info and context usage
func (m *AgentChatModel) renderHeader() string {
	headerStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FAFAFA")).
		Background(lipgloss.Color("#874BFD")).
		Padding(0, 1).
		Bold(true)
	
	agentInfo := "No Agent"
	if m.currentAgent != nil {
		agentInfo = fmt.Sprintf("🤖 %s (%s)", m.currentAgent.Metadata.Name, m.currentAgent.Type)
	}
	
	contextInfo := m.renderContextUsage()
	
	// Create a full-width header
	headerWidth := m.width
	if headerWidth <= 0 {
		headerWidth = 80
	}
	
	leftPart := headerStyle.Render(agentInfo)
	rightPart := headerStyle.Render(contextInfo)
	
	// Calculate padding to right-align context info
	leftWidth := lipgloss.Width(leftPart)
	rightWidth := lipgloss.Width(rightPart)
	padding := headerWidth - leftWidth - rightWidth
	if padding < 0 {
		padding = 0
	}
	
	paddingStyle := headerStyle.Copy()
	paddingStr := paddingStyle.Render(strings.Repeat(" ", padding))
	
	return leftPart + paddingStr + rightPart
}

// renderContextUsage shows Claude Code style context usage
func (m *AgentChatModel) renderContextUsage() string {
	percentage := m.contextUsage.Percentage
	
	var color lipgloss.Color
	var indicator string
	
	switch {
	case percentage < 50:
		color = lipgloss.Color("#50C878") // Green
		indicator = "●"
	case percentage < 80:
		color = lipgloss.Color("#FFD700") // Yellow
		indicator = "●"
	default:
		color = lipgloss.Color("#FF6B6B") // Red
		indicator = "⚠"
	}
	
	usageStyle := lipgloss.NewStyle().Foreground(color)
	
	return fmt.Sprintf("%s Context: %d/%d (%.0f%%)", 
		usageStyle.Render(indicator),
		m.contextUsage.UsedTokens, 
		m.contextUsage.MaxTokens, 
		percentage)
}

// renderChatView shows the main chat interface with border
func (m *AgentChatModel) renderChatView() string {
	chatStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#626262")).
		Padding(1, 2)
	
	return chatStyle.Render(m.viewport.View())
}

// renderWithAgentList shows chat with agent selection sidebar
func (m *AgentChatModel) renderWithAgentList() string {
	// Get the chat view with proper styling
	chatView := m.renderChatView()
	agentListView := m.agentList.View()
	
	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		chatView,
		lipgloss.NewStyle().MarginLeft(2).Render(agentListView),
	)
}

// renderInputArea shows the text input with clear separation
func (m *AgentChatModel) renderInputArea() string {
	// Create a separator line above the input
	separatorStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#626262")).
		Bold(true)
	
	separator := separatorStyle.Render(strings.Repeat("─", m.width-2))
	
	// Style the input area
	inputStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#874BFD")).
		Padding(0, 1).
		MarginTop(1)
	
	return lipgloss.JoinVertical(
		lipgloss.Left,
		separator,
		inputStyle.Render(m.textarea.View()),
	)
}

// renderFooter shows help hints
func (m *AgentChatModel) renderFooter() string {
	footerStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#626262")).
		Padding(0, 1)
	
	help := m.help.ShortHelpView(m.keys.ShortHelp())
	return footerStyle.Render(help)
}

// renderHelp shows full help
func (m *AgentChatModel) renderHelp() string {
	helpStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#874BFD")).
		Padding(1, 2)
	
	help := m.help.FullHelpView(m.keys.FullHelp())
	return helpStyle.Render(help)
}

// renderUserMessage styles user messages
func (m *AgentChatModel) renderUserMessage(content string) string {
	userStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#7D56F4")).
		Bold(true).
		MarginBottom(1)
	
	timestamp := time.Now().Format("15:04")
	header := userStyle.Render(fmt.Sprintf("👤 You (%s)", timestamp))
	
	contentStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FAFAFA")).
		PaddingLeft(3)
	
	return header + "\n" + contentStyle.Render(content)
}

// renderAssistantMessage styles assistant messages  
func (m *AgentChatModel) renderAssistantMessage(content string) string {
	assistantStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#50C878")).
		Bold(true).
		MarginBottom(1)
	
	agentName := "Assistant"
	if m.currentAgent != nil {
		agentName = m.currentAgent.Metadata.Name
	}
	
	timestamp := time.Now().Format("15:04")
	header := assistantStyle.Render(fmt.Sprintf("🤖 %s (%s)", agentName, timestamp))
	
	contentStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FAFAFA")).
		PaddingLeft(3)
	
	return header + "\n" + contentStyle.Render(content)
}

// renderStreamingAssistantMessage styles streaming assistant messages with typing indicator
func (m *AgentChatModel) renderStreamingAssistantMessage(content string) string {
	assistantStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#50C878")).
		Bold(true).
		MarginBottom(1)
	
	agentName := "Assistant"
	if m.streamingAgent != nil {
		agentName = m.streamingAgent.Metadata.Name
	} else if m.currentAgent != nil {
		agentName = m.currentAgent.Metadata.Name
	}
	
	timestamp := time.Now().Format("15:04")
	// Add typing indicator to header
	header := assistantStyle.Render(fmt.Sprintf("🤖 %s (%s) ⏳", agentName, timestamp))
	
	contentStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#FAFAFA")).
		PaddingLeft(3)
	
	return header + "\n" + contentStyle.Render(content)
}

// renderSystemMessage styles system notifications
func (m *AgentChatModel) renderSystemMessage(content string) string {
	systemStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#626262")).
		Italic(true).
		PaddingLeft(1)
	
	return systemStyle.Render("ℹ " + content)
}

// renderTimingMessage styles timing information with low focus
func (m *AgentChatModel) renderTimingMessage(content string) string {
	timingStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#444444")).
		Italic(true).
		Align(lipgloss.Right).
		Width(m.viewport.Width)
	
	return timingStyle.Render("⏱️ " + content)
}