package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"log"
)

// SimpleDocument represents a document for basic text matching
type SimpleDocument struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	Content  string `json:"content"`
	URL      string `json:"url"`
	FilePath string `json:"file_path"`
	Keywords []string `json:"keywords"`
}

// SimpleRAGStore provides basic keyword-based document retrieval
type SimpleRAGStore struct {
	documents []SimpleDocument
	indexPath string
}

// NewSimpleRAGStore creates a new simple RAG store
func NewSimpleRAGStore(indexPath string) *SimpleRAGStore {
	return &SimpleRAGStore{
		documents: make([]SimpleDocument, 0),
		indexPath: indexPath,
	}
}

// LoadIndex loads documents from the simple index
func (s *SimpleRAGStore) LoadIndex() error {
	if _, err := os.Stat(s.indexPath); os.IsNotExist(err) {
		log.Printf("Simple RAG index not found at %s, will create new one", s.indexPath)
		return nil
	}

	data, err := os.ReadFile(s.indexPath)
	if err != nil {
		return fmt.Errorf("failed to read simple index: %v", err)
	}

	var indexData struct {
		Documents []SimpleDocument `json:"documents"`
	}

	if err := json.Unmarshal(data, &indexData); err != nil {
		return fmt.Errorf("failed to unmarshal simple index: %v", err)
	}

	s.documents = indexData.Documents
	log.Printf("Loaded %d documents from simple RAG index", len(s.documents))
	return nil
}

// SaveIndex saves documents to the simple index
func (s *SimpleRAGStore) SaveIndex() error {
	indexData := struct {
		Documents []SimpleDocument `json:"documents"`
	}{
		Documents: s.documents,
	}

	data, err := json.MarshalIndent(indexData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal simple index: %v", err)
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(s.indexPath), 0755); err != nil {
		return fmt.Errorf("failed to create index directory: %v", err)
	}

	if err := os.WriteFile(s.indexPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write simple index: %v", err)
	}

	log.Printf("Saved simple RAG index with %d documents", len(s.documents))
	return nil
}

// AddDocument adds a document to the store
func (s *SimpleRAGStore) AddDocument(doc SimpleDocument) error {
	s.documents = append(s.documents, doc)
	return nil
}

// SearchResult represents a search result with basic scoring
type SearchResult struct {
	Document SimpleDocument `json:"document"`
	Score    float32        `json:"score"`
}

// Search finds documents using basic keyword matching
func (s *SimpleRAGStore) Search(query string, topK int) ([]SearchResult, error) {
	if len(s.documents) == 0 {
		return []SearchResult{}, nil
	}

	queryWords := extractKeywords(strings.ToLower(query))
	results := make([]SearchResult, 0)

	for _, doc := range s.documents {
		score := s.calculateScore(queryWords, doc)
		if score > 0 {
			results = append(results, SearchResult{
				Document: doc,
				Score:    score,
			})
		}
	}

	// Sort by score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topK < len(results) {
		results = results[:topK]
	}

	return results, nil
}

// GetDocumentCount returns the number of documents
func (s *SimpleRAGStore) GetDocumentCount() int {
	return len(s.documents)
}

// calculateScore computes a basic relevance score
func (s *SimpleRAGStore) calculateScore(queryWords []string, doc SimpleDocument) float32 {
	if len(queryWords) == 0 {
		return 0
	}

	docText := strings.ToLower(doc.Title + " " + doc.Content)
	docWords := extractKeywords(docText)
	
	// Create word frequency maps
	queryWordCount := make(map[string]int)
	for _, word := range queryWords {
		queryWordCount[word]++
	}
	
	docWordCount := make(map[string]int)
	for _, word := range docWords {
		docWordCount[word]++
	}
	
	// Calculate score based on common words
	var score float32
	var totalQueryWords float32 = float32(len(queryWords))
	
	for word, qCount := range queryWordCount {
		if dCount, exists := docWordCount[word]; exists {
			// Weight by frequency and relative importance
			wordScore := float32(qCount) / totalQueryWords
			if dCount > 1 {
				wordScore *= 1.5 // Boost if word appears multiple times in doc
			}
			
			// Boost for title matches
			if strings.Contains(strings.ToLower(doc.Title), word) {
				wordScore *= 2.0
			}
			
			score += wordScore
		}
	}
	
	// Boost for exact phrase matches
	queryLower := strings.ToLower(strings.Join(queryWords, " "))
	if strings.Contains(docText, queryLower) {
		score += 1.0
	}
	
	return score
}

// extractKeywords extracts meaningful keywords from text
func extractKeywords(text string) []string {
	// Remove common stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
		"in": true, "on": true, "at": true, "to": true, "for": true, "of": true,
		"with": true, "by": true, "from": true, "about": true, "into": true,
		"through": true, "during": true, "before": true, "after": true, "above": true,
		"below": true, "up": true, "down": true, "out": true, "off": true, "over": true,
		"under": true, "again": true, "further": true, "then": true, "once": true,
		"is": true, "are": true, "was": true, "were": true, "be": true, "been": true,
		"being": true, "have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"this": true, "that": true, "these": true, "those": true, "i": true, "me": true,
		"my": true, "myself": true, "we": true, "our": true, "ours": true, "ourselves": true,
		"you": true, "your": true, "yours": true, "yourself": true, "yourselves": true,
		"he": true, "him": true, "his": true, "himself": true, "she": true, "her": true,
		"hers": true, "herself": true, "it": true, "its": true, "itself": true, "they": true,
		"them": true, "their": true, "theirs": true, "themselves": true, "what": true,
		"which": true, "who": true, "whom": true, "whose": true, "where": true, "when": true,
		"why": true, "how": true,
	}

	// Split into words and filter
	words := strings.FieldsFunc(text, func(c rune) bool {
		return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
	})

	filtered := make([]string, 0)
	for _, word := range words {
		word = strings.ToLower(strings.TrimSpace(word))
		if len(word) > 2 && !stopWords[word] {
			filtered = append(filtered, word)
		}
	}

	return filtered
}

// BuildSimpleRAGContext creates context from search results
func BuildSimpleRAGContext(query string, results []SearchResult, maxTokens int) RAGContext {
	var contextBuilder strings.Builder
	contextBuilder.WriteString("# Relevant OpenTDF Documentation\n\n")
	
	tokenCount := 0
	usedResults := make([]SimilarityResult, 0)
	
	for _, result := range results {
		// Estimate token count (rough approximation: 1 token ≈ 4 characters)
		docTokens := len(result.Document.Content) / 4
		if tokenCount + docTokens > maxTokens {
			break
		}
		
		contextBuilder.WriteString(fmt.Sprintf("## %s\n", result.Document.Title))
		contextBuilder.WriteString(fmt.Sprintf("**Source:** %s\n", result.Document.URL))
		contextBuilder.WriteString(fmt.Sprintf("**Relevance:** %.3f\n\n", result.Score))
		contextBuilder.WriteString(result.Document.Content)
		contextBuilder.WriteString("\n\n---\n\n")
		
		tokenCount += docTokens
		
		// Convert to SimilarityResult for compatibility
		usedResults = append(usedResults, SimilarityResult{
			Document: Document{
				ID:       result.Document.ID,
				Title:    result.Document.Title,
				Content:  result.Document.Content,
				URL:      result.Document.URL,
				FilePath: result.Document.FilePath,
			},
			Similarity: result.Score,
		})
	}
	
	return RAGContext{
		Query:        query,
		Results:      usedResults,
		ContextText:  contextBuilder.String(),
		NumDocuments: len(usedResults),
	}
}