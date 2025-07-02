package llm

import (
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
	return []key.Binding{k.Send, k.SwitchAgent, k.ShowAgents, k.Summarize, k.Quit}
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
			} else if !m.showAgentList {
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
		if keyMsg, ok := msg.(tea.KeyMsg); ok && keyMsg.String() == "enter" {
			// We already handled Enter for sending, don't pass to textarea
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

// generateResponse sends the query to the current agent
func (m *AgentChatModel) generateResponse(userInput string) tea.Cmd {
	return func() tea.Msg {
		if m.currentAgent == nil || m.chatEngine == nil {
			return ResponseMsg{
				response: "No agent selected or chat engine not available.",
				agent:    nil,
				tokens:   0,
			}
		}
		
		// Build messages with current agent's system prompt and ONLY the current user input
		// DO NOT send conversation history to prevent feedback loops
		messages := []ChatMessage{
			{Role: "system", Content: m.currentAgent.SystemPrompt},
			{Role: "user", Content: userInput},
		}
		
		// Generate response using simple chat engine
		response := m.chatEngine.Chat(messages)
		if response.Error != nil {
			return ResponseMsg{
				response: fmt.Sprintf("Error: %v", response.Error),
				agent:    m.currentAgent,
				tokens:   0,
			}
		}
		
		// Estimate token usage (rough approximation)
		tokens := len(response.Content) / 4
		
		return ResponseMsg{
			response: response.Content,
			agent:    m.currentAgent,
			tokens:   tokens,
		}
	}
}

// addMessage adds a message to the conversation
func (m *AgentChatModel) addMessage(msg ChatMessage) {
	m.messages = append(m.messages, msg)
	
	// Check for auto-summarization after adding messages
	if msg.Role == "assistant" {
		m.checkAutoSummarization()
	}
	
	m.updateViewport()
}

// addSystemMessage adds a system notification
func (m *AgentChatModel) addSystemMessage(text string) {
	m.addMessage(ChatMessage{
		Role:    "system",
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
		}
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
	
	// Reserve space for header, footer, and context bar
	availableHeight := m.height - 8
	availableWidth := m.width - 4
	
	// Viewport takes most of the space
	viewportHeight := availableHeight - 5
	if viewportHeight < 10 {
		viewportHeight = 10
	}
	
	m.viewport.Width = availableWidth
	m.viewport.Height = viewportHeight
	
	// Textarea gets remaining space
	m.textarea.SetWidth(availableWidth)
	
	// Agent list sizing
	if m.showAgentList {
		listWidth := availableWidth / 3
		if listWidth < 30 {
			listWidth = 30
		}
		m.agentList.SetSize(listWidth, viewportHeight)
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

// renderChatView shows the main chat interface
func (m *AgentChatModel) renderChatView() string {
	return m.viewport.View()
}

// renderWithAgentList shows chat with agent selection sidebar
func (m *AgentChatModel) renderWithAgentList() string {
	chatView := m.viewport.View()
	agentListView := m.agentList.View()
	
	return lipgloss.JoinHorizontal(
		lipgloss.Top,
		chatView,
		lipgloss.NewStyle().MarginLeft(2).Render(agentListView),
	)
}

// renderInputArea shows the text input
func (m *AgentChatModel) renderInputArea() string {
	inputStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#874BFD")).
		Padding(0, 1)
	
	return inputStyle.Render(m.textarea.View())
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

// renderSystemMessage styles system notifications
func (m *AgentChatModel) renderSystemMessage(content string) string {
	systemStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#626262")).
		Italic(true).
		PaddingLeft(1)
	
	return systemStyle.Render("ℹ " + content)
}