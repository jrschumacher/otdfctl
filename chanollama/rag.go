package chanollama

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// Document represents a piece of documentation with its embedding
type Document struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Content     string    `json:"content"`
	URL         string    `json:"url"`
	FilePath    string    `json:"file_path"`
	Embedding   []float32 `json:"embedding,omitempty"`
	ChunkIndex  int       `json:"chunk_index"`
	TotalChunks int       `json:"total_chunks"`
	Keywords    []string  `json:"keywords,omitempty"`
}

// NewDocument creates a new document
func NewDocument(id, title, content string) *Document {
	return &Document{
		ID:          id,
		Title:       title,
		Content:     content,
		ChunkIndex:  0,
		TotalChunks: 1,
		Keywords:    []string{},
	}
}

// WithURL sets the document URL
func (d *Document) WithURL(url string) *Document {
	d.URL = url
	return d
}

// WithFilePath sets the document file path
func (d *Document) WithFilePath(path string) *Document {
	d.FilePath = path
	return d
}

// WithEmbedding sets the document embedding
func (d *Document) WithEmbedding(embedding []float32) *Document {
	d.Embedding = embedding
	return d
}

// WithKeywords sets the document keywords
func (d *Document) WithKeywords(keywords []string) *Document {
	d.Keywords = keywords
	return d
}

// WithChunk sets chunk information
func (d *Document) WithChunk(index, total int) *Document {
	d.ChunkIndex = index
	d.TotalChunks = total
	return d
}

// DocumentChunk represents a smaller piece of a document for better retrieval
type DocumentChunk struct {
	Document
	ParentID string `json:"parent_id"`
}

// SimilarityResult represents a document with its similarity score
type SimilarityResult struct {
	Document   Document `json:"document"`
	Similarity float32  `json:"similarity"`
}

// SearchResult represents a search result with basic scoring
type SearchResult struct {
	Document Document `json:"document"`
	Score    float32  `json:"score"`
}

// RAGContext represents retrieved context for augmenting prompts
type RAGContext struct {
	Query        string             `json:"query"`
	Results      []SimilarityResult `json:"results"`
	ContextText  string             `json:"context_text"`
	NumDocuments int                `json:"num_documents"`
}

// VectorStore manages document embeddings and similarity search
type VectorStore struct {
	documents    []Document
	embeddingDim int
	mu           sync.RWMutex
	indexPath    string
}

// NewVectorStore creates a new vector store
func NewVectorStore(indexPath string) *VectorStore {
	return &VectorStore{
		documents:    make([]Document, 0),
		embeddingDim: 0,
		indexPath:    indexPath,
	}
}

// LoadIndex loads the vector store from disk
func (vs *VectorStore) LoadIndex() error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	if _, err := os.Stat(vs.indexPath); os.IsNotExist(err) {
		log.Printf("Vector index not found at %s, will create new one", vs.indexPath)
		return nil
	}

	data, err := os.ReadFile(vs.indexPath)
	if err != nil {
		return fmt.Errorf("failed to read index file: %v", err)
	}

	var indexData struct {
		Documents    []Document `json:"documents"`
		EmbeddingDim int        `json:"embedding_dim"`
	}

	if err := json.Unmarshal(data, &indexData); err != nil {
		return fmt.Errorf("failed to unmarshal index: %v", err)
	}

	vs.documents = indexData.Documents
	vs.embeddingDim = indexData.EmbeddingDim

	log.Printf("Loaded %d documents from vector index", len(vs.documents))
	return nil
}

// SaveIndex saves the vector store to disk
func (vs *VectorStore) SaveIndex() error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	indexData := struct {
		Documents    []Document `json:"documents"`
		EmbeddingDim int        `json:"embedding_dim"`
	}{
		Documents:    vs.documents,
		EmbeddingDim: vs.embeddingDim,
	}

	data, err := json.MarshalIndent(indexData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal index: %v", err)
	}

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(vs.indexPath), 0755); err != nil {
		return fmt.Errorf("failed to create index directory: %v", err)
	}

	if err := os.WriteFile(vs.indexPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write index file: %v", err)
	}

	log.Printf("Saved vector index with %d documents", len(vs.documents))
	return nil
}

// AddDocument adds a document with its embedding to the store
func (vs *VectorStore) AddDocument(doc Document) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	if vs.embeddingDim == 0 && len(doc.Embedding) > 0 {
		vs.embeddingDim = len(doc.Embedding)
	}

	if len(doc.Embedding) != vs.embeddingDim && vs.embeddingDim > 0 {
		return fmt.Errorf("embedding dimension mismatch: expected %d, got %d", vs.embeddingDim, len(doc.Embedding))
	}

	vs.documents = append(vs.documents, doc)
	return nil
}

// Search finds the most similar documents to a query embedding
func (vs *VectorStore) Search(queryEmbedding []float32, topK int) ([]SimilarityResult, error) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	if len(queryEmbedding) != vs.embeddingDim {
		return nil, fmt.Errorf("query embedding dimension mismatch: expected %d, got %d", vs.embeddingDim, len(queryEmbedding))
	}

	if topK > len(vs.documents) {
		topK = len(vs.documents)
	}

	results := make([]SimilarityResult, 0, len(vs.documents))

	for _, doc := range vs.documents {
		similarity := CosineSimilarity(queryEmbedding, doc.Embedding)
		results = append(results, SimilarityResult{
			Document:   doc,
			Similarity: similarity,
		})
	}

	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if topK < len(results) {
		results = results[:topK]
	}

	return results, nil
}

// GetDocumentCount returns the number of documents in the store
func (vs *VectorStore) GetDocumentCount() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.documents)
}

// SimpleRAGStore provides basic keyword-based document retrieval
type SimpleRAGStore struct {
	documents []Document
	indexPath string
	mu        sync.RWMutex
}

// NewSimpleRAGStore creates a new simple RAG store
func NewSimpleRAGStore(indexPath string) *SimpleRAGStore {
	return &SimpleRAGStore{
		documents: make([]Document, 0),
		indexPath: indexPath,
	}
}

// LoadIndex loads documents from the simple index
func (s *SimpleRAGStore) LoadIndex() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, err := os.Stat(s.indexPath); os.IsNotExist(err) {
		log.Printf("Simple RAG index not found at %s, will create new one", s.indexPath)
		return nil
	}

	data, err := os.ReadFile(s.indexPath)
	if err != nil {
		return fmt.Errorf("failed to read simple index: %v", err)
	}

	var indexData struct {
		Documents []Document `json:"documents"`
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
	s.mu.RLock()
	defer s.mu.RUnlock()

	indexData := struct {
		Documents []Document `json:"documents"`
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
func (s *SimpleRAGStore) AddDocument(doc Document) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.documents = append(s.documents, doc)
	return nil
}

// Search finds documents using basic keyword matching
func (s *SimpleRAGStore) Search(query string, topK int) ([]SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.documents) == 0 {
		return []SearchResult{}, nil
	}

	queryWords := ExtractKeywords(strings.ToLower(query))
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
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.documents)
}

// calculateScore computes a basic relevance score
func (s *SimpleRAGStore) calculateScore(queryWords []string, doc Document) float32 {
	if len(queryWords) == 0 {
		return 0
	}

	docText := strings.ToLower(doc.Title + " " + doc.Content)
	docWords := ExtractKeywords(docText)

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

// CosineSimilarity calculates the cosine similarity between two vectors
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

// ExtractKeywords extracts meaningful keywords from text
func ExtractKeywords(text string) []string {
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

// ChunkText splits text into overlapping chunks for better retrieval
func ChunkText(text string, chunkSize int, overlap int) []string {
	words := strings.Fields(text)
	if len(words) <= chunkSize {
		return []string{text}
	}

	chunks := make([]string, 0)
	start := 0

	for start < len(words) {
		end := start + chunkSize
		if end > len(words) {
			end = len(words)
		}

		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)

		if end == len(words) {
			break
		}

		start += chunkSize - overlap
	}

	return chunks
}

// BuildRAGContext creates context from similarity search results
func BuildRAGContext(query string, results []SimilarityResult, maxTokens int) RAGContext {
	var contextBuilder strings.Builder
	contextBuilder.WriteString("# Relevant Documentation\n\n")

	tokenCount := 0
	usedResults := make([]SimilarityResult, 0)

	for _, result := range results {
		// Estimate token count (rough approximation: 1 token ≈ 4 characters)
		docTokens := len(result.Document.Content) / 4
		if tokenCount+docTokens > maxTokens {
			break
		}

		contextBuilder.WriteString(fmt.Sprintf("## %s\n", result.Document.Title))
		if result.Document.URL != "" {
			contextBuilder.WriteString(fmt.Sprintf("**Source:** %s\n", result.Document.URL))
		}
		contextBuilder.WriteString(fmt.Sprintf("**Relevance:** %.3f\n\n", result.Similarity))
		contextBuilder.WriteString(result.Document.Content)
		contextBuilder.WriteString("\n\n---\n\n")

		tokenCount += docTokens
		usedResults = append(usedResults, result)
	}

	return RAGContext{
		Query:        query,
		Results:      usedResults,
		ContextText:  contextBuilder.String(),
		NumDocuments: len(usedResults),
	}
}

// BuildSimpleRAGContext creates context from search results (keyword-based)
func BuildSimpleRAGContext(query string, results []SearchResult, maxTokens int) RAGContext {
	var contextBuilder strings.Builder
	contextBuilder.WriteString("# Relevant Documentation\n\n")

	tokenCount := 0
	usedResults := make([]SimilarityResult, 0)

	for _, result := range results {
		// Estimate token count (rough approximation: 1 token ≈ 4 characters)
		docTokens := len(result.Document.Content) / 4
		if tokenCount+docTokens > maxTokens {
			break
		}

		contextBuilder.WriteString(fmt.Sprintf("## %s\n", result.Document.Title))
		if result.Document.URL != "" {
			contextBuilder.WriteString(fmt.Sprintf("**Source:** %s\n", result.Document.URL))
		}
		contextBuilder.WriteString(fmt.Sprintf("**Relevance:** %.3f\n\n", result.Score))
		contextBuilder.WriteString(result.Document.Content)
		contextBuilder.WriteString("\n\n---\n\n")

		tokenCount += docTokens

		// Convert to SimilarityResult for compatibility
		usedResults = append(usedResults, SimilarityResult{
			Document:   result.Document,
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

// RAGEngine combines document retrieval with LLM querying
type RAGEngine struct {
	engine     Engine
	vectorStore *VectorStore
	simpleStore *SimpleRAGStore
	useVector   bool
}

// NewRAGEngine creates a new RAG engine
func NewRAGEngine(engine Engine, indexPath string, useVector bool) *RAGEngine {
	var vectorStore *VectorStore
	var simpleStore *SimpleRAGStore

	if useVector {
		vectorStore = NewVectorStore(indexPath)
	} else {
		simpleStore = NewSimpleRAGStore(indexPath)
	}

	return &RAGEngine{
		engine:      engine,
		vectorStore: vectorStore,
		simpleStore: simpleStore,
		useVector:   useVector,
	}
}

// LoadIndex loads the appropriate index
func (re *RAGEngine) LoadIndex() error {
	if re.useVector && re.vectorStore != nil {
		return re.vectorStore.LoadIndex()
	} else if re.simpleStore != nil {
		return re.simpleStore.LoadIndex()
	}
	return fmt.Errorf("no store configured")
}

// Query performs RAG-enhanced querying
func (re *RAGEngine) Query(query string, systemPrompt string, maxContextTokens int) (*Response, error) {
	// Build context from documents
	var ragContext RAGContext

	if re.useVector && re.vectorStore != nil {
		// This would require an embedding for the query - simplified for now
		ragContext = RAGContext{
			Query:        query,
			Results:      []SimilarityResult{},
			ContextText:  "",
			NumDocuments: 0,
		}
	} else if re.simpleStore != nil {
		results, err := re.simpleStore.Search(query, 3)
		if err != nil {
			return nil, fmt.Errorf("failed to search documents: %w", err)
		}
		ragContext = BuildSimpleRAGContext(query, results, maxContextTokens)
	} else {
		return nil, fmt.Errorf("no store configured")
	}

	// Prepare messages with RAG context
	messages := []Message{}

	if systemPrompt != "" {
		if ragContext.NumDocuments > 0 {
			enhancedSystem := fmt.Sprintf("%s\n\n%s\n\nBased on the above documentation, please provide accurate and helpful responses.",
				systemPrompt, ragContext.ContextText)
			messages = append(messages, NewSystemMessage(enhancedSystem))
		} else {
			messages = append(messages, NewSystemMessage(systemPrompt))
		}
	}

	messages = append(messages, NewUserMessage(query))

	// Query the LLM
	return re.engine.Chat(messages)
}

// GetDocumentCount returns the total number of documents
func (re *RAGEngine) GetDocumentCount() int {
	if re.useVector && re.vectorStore != nil {
		return re.vectorStore.GetDocumentCount()
	} else if re.simpleStore != nil {
		return re.simpleStore.GetDocumentCount()
	}
	return 0
}