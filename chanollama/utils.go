package chanollama

import (
	"regexp"
	"runtime"
	"strings"
	"unicode"
)

// TokenEstimator provides token estimation utilities
type TokenEstimator struct {
	CharsPerToken float64 // Average characters per token (model-dependent)
}

// NewTokenEstimator creates a token estimator with default ratios
func NewTokenEstimator() *TokenEstimator {
	return &TokenEstimator{
		CharsPerToken: 4.0, // Conservative estimate for most models
	}
}

// WithCharsPerToken sets a custom character-to-token ratio
func (te *TokenEstimator) WithCharsPerToken(ratio float64) *TokenEstimator {
	te.CharsPerToken = ratio
	return te
}

// EstimateTokens estimates token count from text
func (te *TokenEstimator) EstimateTokens(text string) int {
	if text == "" {
		return 0
	}
	return int(float64(len(text)) / te.CharsPerToken)
}

// EstimateMessageTokens estimates tokens for a message including role overhead
func (te *TokenEstimator) EstimateMessageTokens(msg Message) int {
	// Add overhead for role and formatting
	roleOverhead := 10 // Approximate tokens for role formatting
	contentTokens := te.EstimateTokens(msg.Content)
	return contentTokens + roleOverhead
}

// EstimateMessagesTokens estimates total tokens for a slice of messages
func (te *TokenEstimator) EstimateMessagesTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		total += te.EstimateMessageTokens(msg)
	}
	return total
}

// TextProcessor provides text processing utilities for LLM inputs
type TextProcessor struct {
	MaxLength    int  // Maximum text length to process
	CleanText    bool // Whether to clean whitespace and formatting
	PreserveCode bool // Whether to preserve code blocks
}

// NewTextProcessor creates a text processor with defaults
func NewTextProcessor() *TextProcessor {
	return &TextProcessor{
		MaxLength:    50000, // 50k characters default
		CleanText:    true,
		PreserveCode: true,
	}
}

// WithMaxLength sets the maximum text length
func (tp *TextProcessor) WithMaxLength(length int) *TextProcessor {
	tp.MaxLength = length
	return tp
}

// WithCleanText enables or disables text cleaning
func (tp *TextProcessor) WithCleanText(clean bool) *TextProcessor {
	tp.CleanText = clean
	return tp
}

// WithPreserveCode enables or disables code block preservation
func (tp *TextProcessor) WithPreserveCode(preserve bool) *TextProcessor {
	tp.PreserveCode = preserve
	return tp
}

// ProcessText cleans and prepares text for LLM processing
func (tp *TextProcessor) ProcessText(text string) string {
	if text == "" {
		return text
	}

	// Apply length limit
	if len(text) > tp.MaxLength {
		text = text[:tp.MaxLength] + "..."
	}

	if !tp.CleanText {
		return text
	}

	// Clean text while preserving structure
	result := text

	// Normalize line endings
	result = strings.ReplaceAll(result, "\r\n", "\n")
	result = strings.ReplaceAll(result, "\r", "\n")

	// Clean excessive whitespace while preserving code blocks
	if tp.PreserveCode {
		result = tp.cleanWithCodePreservation(result)
	} else {
		result = tp.cleanGeneral(result)
	}

	return strings.TrimSpace(result)
}

// cleanWithCodePreservation cleans text while preserving code block formatting
func (tp *TextProcessor) cleanWithCodePreservation(text string) string {
	// Regex to match code blocks (``` or ``` language)
	codeBlockRegex := regexp.MustCompile("(?s)```[\\w]*\\s*\\n(.*?)\\n```")
	
	// Find all code blocks and their positions
	codeBlocks := codeBlockRegex.FindAllString(text, -1)
	codeBlockPositions := codeBlockRegex.FindAllStringIndex(text, -1)
	
	// Process text outside code blocks
	result := text
	offset := 0
	
	for i, pos := range codeBlockPositions {
		start := pos[0] + offset
		end := pos[1] + offset
		
		// Clean text before this code block
		if start > 0 {
			beforeBlock := result[:start]
			cleanedBefore := tp.cleanGeneral(beforeBlock)
			lengthDiff := len(cleanedBefore) - len(beforeBlock)
			result = cleanedBefore + result[start:]
			offset += lengthDiff
			start += lengthDiff
			end += lengthDiff
		}
		
		// Keep code block as-is (already clean)
		// Move to after this code block
		if i < len(codeBlockPositions)-1 {
			continue
		} else {
			// Clean text after the last code block
			if end < len(result) {
				afterBlock := result[end:]
				cleanedAfter := tp.cleanGeneral(afterBlock)
				result = result[:end] + cleanedAfter
			}
		}
	}
	
	// If no code blocks found, clean normally
	if len(codeBlocks) == 0 {
		result = tp.cleanGeneral(result)
	}
	
	return result
}

// cleanGeneral performs general text cleaning
func (tp *TextProcessor) cleanGeneral(text string) string {
	// Remove excessive consecutive newlines (keep max 2)
	multiNewlineRegex := regexp.MustCompile(`\n{3,}`)
	result := multiNewlineRegex.ReplaceAllString(text, "\n\n")
	
	// Remove excessive spaces (keep single spaces)
	multiSpaceRegex := regexp.MustCompile(` {2,}`)
	result = multiSpaceRegex.ReplaceAllString(result, " ")
	
	// Remove trailing spaces on lines
	lines := strings.Split(result, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRightFunc(line, unicode.IsSpace)
	}
	result = strings.Join(lines, "\n")
	
	return result
}

// ExtractKeywords extracts important keywords from text
func (tp *TextProcessor) ExtractKeywords(text string, maxKeywords int) []string {
	if text == "" || maxKeywords <= 0 {
		return nil
	}

	// Convert to lowercase for processing
	lowerText := strings.ToLower(text)
	
	// Remove punctuation and split into words
	wordRegex := regexp.MustCompile(`[^\w\s]`)
	cleanText := wordRegex.ReplaceAllString(lowerText, " ")
	words := strings.Fields(cleanText)
	
	// Common stop words to filter out
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
		"with": true, "by": true, "is": true, "are": true, "was": true, "were": true,
		"be": true, "been": true, "have": true, "has": true, "had": true, "do": true,
		"does": true, "did": true, "will": true, "would": true, "could": true, "should": true,
		"this": true, "that": true, "these": true, "those": true, "i": true, "you": true,
		"he": true, "she": true, "it": true, "we": true, "they": true, "me": true,
		"him": true, "her": true, "us": true, "them": true, "my": true, "your": true,
		"his": true, "its": true, "our": true, "their": true,
	}
	
	// Count word frequencies
	wordFreq := make(map[string]int)
	for _, word := range words {
		// Filter short words and stop words
		if len(word) < 3 || stopWords[word] {
			continue
		}
		wordFreq[word]++
	}
	
	// Sort by frequency and return top keywords
	type wordCount struct {
		word  string
		count int
	}
	
	var wordCounts []wordCount
	for word, count := range wordFreq {
		wordCounts = append(wordCounts, wordCount{word, count})
	}
	
	// Simple bubble sort by frequency (desc)
	for i := 0; i < len(wordCounts)-1; i++ {
		for j := 0; j < len(wordCounts)-i-1; j++ {
			if wordCounts[j].count < wordCounts[j+1].count {
				wordCounts[j], wordCounts[j+1] = wordCounts[j+1], wordCounts[j]
			}
		}
	}
	
	// Extract top keywords
	var keywords []string
	maxResults := maxKeywords
	if len(wordCounts) < maxResults {
		maxResults = len(wordCounts)
	}
	
	for i := 0; i < maxResults; i++ {
		keywords = append(keywords, wordCounts[i].word)
	}
	
	return keywords
}

// TruncateText truncates text to specified length while preserving word boundaries
func (tp *TextProcessor) TruncateText(text string, maxLength int) string {
	if len(text) <= maxLength {
		return text
	}
	
	// Find last space before maxLength
	truncated := text[:maxLength]
	lastSpace := strings.LastIndex(truncated, " ")
	
	if lastSpace > 0 && lastSpace > maxLength/2 {
		// Truncate at last space if it's not too early
		return text[:lastSpace] + "..."
	}
	
	// Hard truncate if no good space found
	return text[:maxLength-3] + "..."
}

// ResourceMonitor provides system resource monitoring for LLM operations
type ResourceMonitor struct {
	enabled bool
}

// NewResourceMonitor creates a resource monitor
func NewResourceMonitor() *ResourceMonitor {
	return &ResourceMonitor{enabled: true}
}

// SetEnabled enables or disables resource monitoring
func (rm *ResourceMonitor) SetEnabled(enabled bool) {
	rm.enabled = enabled
}

// ResourceUsage represents current resource usage
type ResourceUsage struct {
	MemoryMB     uint64  `json:"memory_mb"`
	GoroutineCount int   `json:"goroutine_count"`
	CPUPercent   float64 `json:"cpu_percent,omitempty"`
}

// GetCurrentUsage returns current resource usage
func (rm *ResourceMonitor) GetCurrentUsage() ResourceUsage {
	if !rm.enabled {
		return ResourceUsage{}
	}
	
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	return ResourceUsage{
		MemoryMB:       memStats.Alloc / 1024 / 1024,
		GoroutineCount: runtime.NumGoroutine(),
		// CPU monitoring would require additional dependencies
	}
}

// ShouldLimitProcessing determines if processing should be limited due to resource usage
func (rm *ResourceMonitor) ShouldLimitProcessing(maxMemoryMB uint64, maxGoroutines int) bool {
	if !rm.enabled {
		return false
	}
	
	usage := rm.GetCurrentUsage()
	
	if maxMemoryMB > 0 && usage.MemoryMB > maxMemoryMB {
		return true
	}
	
	if maxGoroutines > 0 && usage.GoroutineCount > maxGoroutines {
		return true
	}
	
	return false
}