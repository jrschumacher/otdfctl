package llm

import (
	"fmt"
	"log"
	"strings"
	"time"
)

// AutoSummarizer handles automatic conversation summarization
type AutoSummarizer struct {
	chatEngine       *SimpleChatEngine
	enabled          bool
	summaryThreshold float64 // Context usage percentage to trigger summarization
	maxSummaryLength int     // Maximum length of generated summaries
}

// NewAutoSummarizer creates a new auto-summarization system
func NewAutoSummarizer(chatEngine *SimpleChatEngine) *AutoSummarizer {
	return &AutoSummarizer{
		chatEngine:       chatEngine,
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

// ShouldSummarize determines if summarization is needed based on context usage
func (as *AutoSummarizer) ShouldSummarize(contextUsage ContextUsage) bool {
	if !as.enabled || as.chatEngine == nil {
		return false
	}
	
	return contextUsage.Percentage >= as.summaryThreshold
}

// SummarizeConversation creates a summary of the conversation history
func (as *AutoSummarizer) SummarizeConversation(messages []ChatMessage, currentAgent *Agent) (*ConversationSummary, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages to summarize")
	}
	
	// Filter out system messages and previous summaries
	relevantMessages := as.filterMessagesForSummary(messages)
	if len(relevantMessages) == 0 {
		return nil, fmt.Errorf("no relevant messages to summarize")
	}
	
	// Create summarization prompt
	summaryPrompt := as.buildSummaryPrompt(relevantMessages, currentAgent)
	
	// Generate summary using the chat engine
	summary, err := as.generateSummary(summaryPrompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate summary: %w", err)
	}
	
	// Create conversation summary
	conversationSummary := &ConversationSummary{
		Summary:           summary,
		OriginalMessages:  len(messages),
		SummarizedMessages: len(relevantMessages),
		CreatedAt:         time.Now(),
		Agent:            currentAgent,
		TokensSaved:      as.estimateTokensSaved(relevantMessages, summary),
	}
	
	return conversationSummary, nil
}

// ConversationSummary represents a summarized conversation
type ConversationSummary struct {
	Summary            string    `json:"summary"`
	OriginalMessages   int       `json:"original_messages"`
	SummarizedMessages int       `json:"summarized_messages"`
	CreatedAt          time.Time `json:"created_at"`
	Agent             *Agent    `json:"agent,omitempty"`
	TokensSaved       int       `json:"tokens_saved"`
}

// ApplySummary replaces conversation history with a summary
func (as *AutoSummarizer) ApplySummary(messages []ChatMessage, summary *ConversationSummary) []ChatMessage {
	// Keep the system message (first message) if present
	var newMessages []ChatMessage
	
	if len(messages) > 0 && messages[0].Role == "system" {
		newMessages = append(newMessages, messages[0])
	}
	
	// Add summary as a system message
	summaryMessage := ChatMessage{
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
func (as *AutoSummarizer) filterMessagesForSummary(messages []ChatMessage) []ChatMessage {
	var filtered []ChatMessage
	
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
func (as *AutoSummarizer) buildSummaryPrompt(messages []ChatMessage, currentAgent *Agent) string {
	var prompt strings.Builder
	
	prompt.WriteString("Please create a concise summary of this conversation. ")
	prompt.WriteString("Focus on the main topics discussed, key information shared, and important decisions or conclusions. ")
	prompt.WriteString("Keep the summary under 200 words and maintain the essential context for continuing the conversation.\n\n")
	
	// Add agent context if available
	if currentAgent != nil {
		prompt.WriteString(fmt.Sprintf("Context: This conversation involved the %s agent, which specializes in %s.\n\n", 
			currentAgent.Metadata.Name, currentAgent.Metadata.Description))
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
	messages := []ChatMessage{
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
	response := as.chatEngine.Chat(messages)
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
func (as *AutoSummarizer) getRecentMessages(messages []ChatMessage, count int) []ChatMessage {
	if len(messages) <= count {
		return messages
	}
	
	// Return the last 'count' messages
	return messages[len(messages)-count:]
}

// estimateTokensSaved calculates approximate tokens saved by summarization
func (as *AutoSummarizer) estimateTokensSaved(originalMessages []ChatMessage, summary string) int {
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

// SummarizationStats provides statistics about summarization effectiveness
type SummarizationStats struct {
	TotalSummarizations int     `json:"total_summarizations"`
	TokensSaved         int     `json:"tokens_saved"`
	AverageCompression  float64 `json:"average_compression_ratio"`
	LastSummarization   time.Time `json:"last_summarization"`
}

// AutoSummarizationManager handles multiple conversations and statistics
type AutoSummarizationManager struct {
	summarizers map[string]*AutoSummarizer // Key: conversation ID
	stats       SummarizationStats
	enabled     bool
}

// NewAutoSummarizationManager creates a new summarization manager
func NewAutoSummarizationManager() *AutoSummarizationManager {
	return &AutoSummarizationManager{
		summarizers: make(map[string]*AutoSummarizer),
		enabled:     true,
	}
}

// GetSummarizer returns a summarizer for a specific conversation
func (asm *AutoSummarizationManager) GetSummarizer(conversationID string, chatEngine *SimpleChatEngine) *AutoSummarizer {
	if summarizer, exists := asm.summarizers[conversationID]; exists {
		return summarizer
	}
	
	// Create new summarizer
	summarizer := NewAutoSummarizer(chatEngine)
	asm.summarizers[conversationID] = summarizer
	
	return summarizer
}

// HandleAutoSummarization checks if summarization is needed and applies it
func (asm *AutoSummarizationManager) HandleAutoSummarization(
	conversationID string,
	messages []ChatMessage,
	contextUsage ContextUsage,
	currentAgent *Agent,
	chatEngine *SimpleChatEngine,
) ([]ChatMessage, *ConversationSummary, error) {
	
	if !asm.enabled {
		return messages, nil, nil
	}
	
	summarizer := asm.GetSummarizer(conversationID, chatEngine)
	
	// Check if summarization is needed
	if !summarizer.ShouldSummarize(contextUsage) {
		return messages, nil, nil
	}
	
	log.Printf("Auto-summarization triggered at %.1f%% context usage", contextUsage.Percentage)
	
	// Create summary
	summary, err := summarizer.SummarizeConversation(messages, currentAgent)
	if err != nil {
		log.Printf("Warning: Auto-summarization failed: %v", err)
		return messages, nil, err
	}
	
	// Apply summary to reduce context
	summarizedMessages := summarizer.ApplySummary(messages, summary)
	
	// Update statistics
	asm.updateStats(summary)
	
	log.Printf("Auto-summarization complete: %d messages -> %d messages (saved %d tokens)",
		len(messages), len(summarizedMessages), summary.TokensSaved)
	
	return summarizedMessages, summary, nil
}

// updateStats updates summarization statistics
func (asm *AutoSummarizationManager) updateStats(summary *ConversationSummary) {
	asm.stats.TotalSummarizations++
	asm.stats.TokensSaved += summary.TokensSaved
	asm.stats.LastSummarization = summary.CreatedAt
	
	// Calculate average compression ratio
	if summary.OriginalMessages > 0 {
		compressionRatio := float64(summary.SummarizedMessages) / float64(summary.OriginalMessages)
		asm.stats.AverageCompression = (asm.stats.AverageCompression + compressionRatio) / 2.0
	}
}

// GetStats returns summarization statistics
func (asm *AutoSummarizationManager) GetStats() SummarizationStats {
	return asm.stats
}

// SetEnabled enables or disables auto-summarization
func (asm *AutoSummarizationManager) SetEnabled(enabled bool) {
	asm.enabled = enabled
}

// SummarizationResult represents the result of a summarization operation
type SummarizationResult struct {
	Success           bool                `json:"success"`
	Summary          *ConversationSummary `json:"summary,omitempty"`
	UpdatedMessages  []ChatMessage        `json:"updated_messages,omitempty"`
	Error            string               `json:"error,omitempty"`
	PerformanceInfo  map[string]interface{} `json:"performance_info"`
}