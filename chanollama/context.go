package chanollama

import (
	"fmt"
	"strings"
	"time"
)

// ContextUsage tracks context window usage
type ContextUsage struct {
	UsedTokens    int     `json:"used_tokens"`
	MaxTokens     int     `json:"max_tokens"`
	Percentage    float64 `json:"percentage"`
	NeedsCleanup  bool    `json:"needs_cleanup"`
}

// CalculateContextUsage computes context usage statistics
func CalculateContextUsage(usedTokens, maxTokens int) ContextUsage {
	percentage := float64(usedTokens) / float64(maxTokens) * 100
	return ContextUsage{
		UsedTokens:   usedTokens,
		MaxTokens:    maxTokens,
		Percentage:   percentage,
		NeedsCleanup: percentage > 80.0,
	}
}

// AutoSummarizer handles automatic conversation summarization
type AutoSummarizer struct {
	engine           Engine
	enabled          bool
	summaryThreshold float64 // Context usage percentage to trigger summarization
	maxSummaryLength int     // Maximum length of generated summaries
}

// NewAutoSummarizer creates a new auto-summarization system
func NewAutoSummarizer(engine Engine) *AutoSummarizer {
	return &AutoSummarizer{
		engine:           engine,
		enabled:          true,
		summaryThreshold: 75.0, // Trigger at 75% context usage
		maxSummaryLength: 500,  // 500 tokens max for summaries
	}
}

// WithThreshold sets the context usage threshold for triggering summarization
func (as *AutoSummarizer) WithThreshold(threshold float64) *AutoSummarizer {
	as.summaryThreshold = threshold
	return as
}

// WithMaxLength sets the maximum summary length
func (as *AutoSummarizer) WithMaxLength(maxLength int) *AutoSummarizer {
	as.maxSummaryLength = maxLength
	return as
}

// SetEnabled enables or disables auto-summarization
func (as *AutoSummarizer) SetEnabled(enabled bool) {
	as.enabled = enabled
}

// ShouldSummarize determines if summarization is needed based on context usage
func (as *AutoSummarizer) ShouldSummarize(contextUsage ContextUsage) bool {
	if !as.enabled || as.engine == nil {
		return false
	}
	
	return contextUsage.Percentage >= as.summaryThreshold
}

// ConversationSummary represents a summarized conversation
type ConversationSummary struct {
	Summary            string    `json:"summary"`
	OriginalMessages   int       `json:"original_messages"`
	SummarizedMessages int       `json:"summarized_messages"`
	CreatedAt          time.Time `json:"created_at"`
	Context            string    `json:"context,omitempty"` // Optional context info
	TokensSaved        int       `json:"tokens_saved"`
}

// SummarizeConversation creates a summary of the conversation history
func (as *AutoSummarizer) SummarizeConversation(messages []Message, context string) (*ConversationSummary, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages to summarize")
	}
	
	// Filter out system messages and previous summaries
	relevantMessages := as.filterMessagesForSummary(messages)
	if len(relevantMessages) == 0 {
		return nil, fmt.Errorf("no relevant messages to summarize")
	}
	
	// Create summarization prompt
	summaryPrompt := as.buildSummaryPrompt(relevantMessages, context)
	
	// Generate summary using the chat engine
	summary, err := as.generateSummary(summaryPrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate summary: %w", err)
	}
	
	// Create conversation summary
	conversationSummary := &ConversationSummary{
		Summary:            summary,
		OriginalMessages:   len(messages),
		SummarizedMessages: len(relevantMessages),
		CreatedAt:          time.Now(),
		Context:            context,
		TokensSaved:        as.estimateTokensSaved(relevantMessages, summary),
	}
	
	return conversationSummary, nil
}

// ApplySummary replaces conversation history with a summary
func (as *AutoSummarizer) ApplySummary(messages []Message, summary *ConversationSummary) []Message {
	// Keep the system message (first message) if present
	var newMessages []Message
	
	if len(messages) > 0 && messages[0].Role == "system" {
		newMessages = append(newMessages, messages[0])
	}
	
	// Add summary as a system message
	summaryMessage := Message{
		Role:    "system",
		Content: fmt.Sprintf("Previous conversation summary: %s", summary.Summary),
	}
	newMessages = append(newMessages, summaryMessage)
	
	// Keep the last few messages for continuity (last 3 exchanges)
	recentMessages := as.getRecentMessages(messages, 6) // 3 user + 3 assistant
	newMessages = append(newMessages, recentMessages...)
	
	return newMessages
}

// filterMessagesForSummary removes system messages and keeps user/assistant exchanges
func (as *AutoSummarizer) filterMessagesForSummary(messages []Message) []Message {
	var filtered []Message
	
	for _, msg := range messages {
		// Skip system messages and previous summaries
		if msg.Role == "system" {
			continue
		}
		
		// Skip messages that look like summaries
		if strings.Contains(strings.ToLower(msg.Content), "previous conversation summary") {
			continue
		}
		
		filtered = append(filtered, msg)
	}
	
	return filtered
}

// buildSummaryPrompt creates a prompt for generating conversation summary
func (as *AutoSummarizer) buildSummaryPrompt(messages []Message, context string) string {
	var prompt strings.Builder
	
	prompt.WriteString("Please create a concise summary of this conversation. ")
	prompt.WriteString("Focus on the main topics discussed, key information shared, and important decisions or conclusions. ")
	prompt.WriteString("Keep the summary under 200 words and maintain the essential context for continuing the conversation.\n\n")
	
	// Add context if available
	if context != "" {
		prompt.WriteString(fmt.Sprintf("Context: %s\n\n", context))
	}
	
	prompt.WriteString("Conversation to summarize:\n")
	
	// Add the conversation messages
	for i, msg := range messages {
		speaker := "User"
		if msg.Role == "assistant" {
			speaker = "Assistant"
		}
		
		// Truncate very long messages
		content := msg.Content
		if len(content) > 300 {
			content = content[:300] + "..."
		}
		
		prompt.WriteString(fmt.Sprintf("%d. %s: %s\n", i+1, speaker, content))
	}
	
	prompt.WriteString("\nSummary:")
	
	return prompt.String()
}

// generateSummary uses the chat engine to create a summary
func (as *AutoSummarizer) generateSummary(summaryPrompt string) (string, error) {
	// Create a simple summarization request
	messages := []Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant that creates concise, accurate summaries of conversations. Focus on preserving key information and context.",
		},
		{
			Role:    "user",
			Content: summaryPrompt,
		},
	}
	
	// Generate summary
	response, err := as.engine.Chat(messages)
	if err != nil {
		return "", err
	}
	
	if response.Error != nil {
		return "", response.Error
	}
	
	// Clean up the summary
	summary := strings.TrimSpace(response.Content)
	
	// Enforce length limits
	if len(summary) > as.maxSummaryLength*4 { // Rough token estimation
		// Truncate if too long
		words := strings.Fields(summary)
		if len(words) > as.maxSummaryLength {
			summary = strings.Join(words[:as.maxSummaryLength], " ") + "..."
		}
	}
	
	return summary, nil
}

// getRecentMessages returns the last N messages for continuity
func (as *AutoSummarizer) getRecentMessages(messages []Message, count int) []Message {
	if len(messages) <= count {
		return messages
	}
	
	// Return the last 'count' messages
	return messages[len(messages)-count:]
}

// estimateTokensSaved calculates approximate tokens saved by summarization
func (as *AutoSummarizer) estimateTokensSaved(originalMessages []Message, summary string) int {
	// Rough token estimation: 1 token ≈ 4 characters
	originalTokens := 0
	for _, msg := range originalMessages {
		originalTokens += len(msg.Content) / 4
	}
	
	summaryTokens := len(summary) / 4
	
	saved := originalTokens - summaryTokens
	if saved < 0 {
		saved = 0
	}
	
	return saved
}

// SmartConversation combines Conversation with automatic summarization
type SmartConversation struct {
	*Conversation
	summarizer   *AutoSummarizer
	tokenLimit   int
	context      string
}

// NewSmartConversation creates a conversation with automatic summarization
func NewSmartConversation(engine Engine, tokenLimit int) *SmartConversation {
	return &SmartConversation{
		Conversation: NewConversation(),
		summarizer:   NewAutoSummarizer(engine),
		tokenLimit:   tokenLimit,
	}
}

// WithContext sets context information for summarization
func (sc *SmartConversation) WithContext(context string) *SmartConversation {
	sc.context = context
	return sc
}

// AddUserMessage adds a user message and checks for auto-summarization
func (sc *SmartConversation) AddUserMessage(content string) error {
	sc.Conversation.AddUserMessage(content)
	return sc.checkAndSummarize()
}

// AddAssistantMessage adds an assistant message and checks for auto-summarization
func (sc *SmartConversation) AddAssistantMessage(content string) error {
	sc.Conversation.AddAssistantMessage(content)
	return sc.checkAndSummarize()
}

// checkAndSummarize checks if summarization is needed and applies it
func (sc *SmartConversation) checkAndSummarize() error {
	currentTokens := sc.Conversation.EstimateTokens()
	contextUsage := CalculateContextUsage(currentTokens, sc.tokenLimit)
	
	if sc.summarizer.ShouldSummarize(contextUsage) {
		summary, err := sc.summarizer.SummarizeConversation(sc.Conversation.Messages, sc.context)
		if err != nil {
			return fmt.Errorf("auto-summarization failed: %w", err)
		}
		
		// Apply the summary
		sc.Conversation.Messages = sc.summarizer.ApplySummary(sc.Conversation.Messages, summary)
		
		return nil
	}
	
	return nil
}

// GetContextUsage returns current context usage statistics
func (sc *SmartConversation) GetContextUsage() ContextUsage {
	currentTokens := sc.Conversation.EstimateTokens()
	return CalculateContextUsage(currentTokens, sc.tokenLimit)
}