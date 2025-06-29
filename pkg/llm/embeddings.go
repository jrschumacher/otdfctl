package llm

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

	"github.com/ollama/ollama/llama"
)

// Document represents a piece of documentation with its embedding
type Document struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Content     string    `json:"content"`
	URL         string    `json:"url"`
	FilePath    string    `json:"file_path"`
	Embedding   []float32 `json:"embedding"`
	ChunkIndex  int       `json:"chunk_index"`
	TotalChunks int       `json:"total_chunks"`
}

// DocumentChunk represents a smaller piece of a document for better retrieval
type DocumentChunk struct {
	Document
	ParentID string `json:"parent_id"`
}

// VectorStore manages document embeddings and similarity search
type VectorStore struct {
	documents    []Document
	embeddingDim int
	mu           sync.RWMutex
	indexPath    string
}

// SimilarityResult represents a document with its similarity score
type SimilarityResult struct {
	Document   Document `json:"document"`
	Similarity float32  `json:"similarity"`
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
		similarity := cosineSimilarity(queryEmbedding, doc.Embedding)
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

// EmbeddingEngine handles text embeddings using Ollama models
type EmbeddingEngine struct {
	model   *llama.Model
	context *llama.Context
	mu      sync.Mutex
}

// NewEmbeddingEngine creates a new embedding engine
func NewEmbeddingEngine(modelPath string) (*EmbeddingEngine, error) {
	// Initialize llama backend
	llama.BackendInit()

	// Set up model parameters for embedding model
	modelParams := llama.ModelParams{
		NumGpuLayers: 0, // TODO: Add GPU support
		UseMmap:      true,
		VocabOnly:    false,
	}

	// Load model
	model, err := llama.LoadModelFromFile(modelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("failed to load embedding model: %v", err)
	}

	// Create context for embeddings
	contextParams := llama.NewContextParams(
		512,  // numCtx - smaller for embeddings
		1,    // batchSize
		1,    // numSeqMax
		4,    // threads
		false, // flashAttention
		"",   // kvCacheType
	)

	context, err := llama.NewContextWithModel(model, contextParams)
	if err != nil {
		llama.FreeModel(model)
		return nil, fmt.Errorf("failed to create embedding context: %v", err)
	}

	return &EmbeddingEngine{
		model:   model,
		context: context,
	}, nil
}

// Close cleans up the embedding engine resources
func (ee *EmbeddingEngine) Close() {
	if ee.model != nil {
		llama.FreeModel(ee.model)
	}
}

// GenerateEmbedding creates an embedding vector for the given text
func (ee *EmbeddingEngine) GenerateEmbedding(text string) ([]float32, error) {
	ee.mu.Lock()
	defer ee.mu.Unlock()

	// Tokenize the text
	tokens, err := ee.model.Tokenize(text, true, true)
	if err != nil {
		return nil, fmt.Errorf("tokenization failed: %v", err)
	}

	// Create batch for embedding
	batch, err := llama.NewBatch(len(tokens), 1, 0)
	if err != nil {
		return nil, fmt.Errorf("batch creation failed: %v", err)
	}
	defer batch.Free()

	// Add tokens to batch
	for i, token := range tokens {
		batch.Add(token, nil, i, false, 0) // No logits needed for embeddings
	}

	// Process the batch
	err = ee.context.Decode(batch)
	if err != nil {
		return nil, fmt.Errorf("context decode failed: %v", err)
	}

	// Get embeddings from the last sequence
	embeddings := ee.context.GetEmbeddingsSeq(0)
	if embeddings == nil {
		return nil, fmt.Errorf("failed to get embeddings")
	}

	return embeddings, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
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

// RAGContext represents retrieved context for augmenting prompts
type RAGContext struct {
	Query        string             `json:"query"`
	Results      []SimilarityResult `json:"results"`
	ContextText  string             `json:"context_text"`
	NumDocuments int                `json:"num_documents"`
}

// BuildRAGContext creates context from similarity search results
func BuildRAGContext(query string, results []SimilarityResult, maxTokens int) RAGContext {
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