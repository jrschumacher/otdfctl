package llm

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/llama"
)

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ChatMessage represents a single message in the conversation
type ChatMessage struct {
	Role    string `json:"role"`    // "user", "assistant", "system"  
	Content string `json:"content"`
}

// ChatRequest represents a request to the chat engine
type ChatRequest struct {
	Messages []ChatMessage          `json:"messages"`
	Stream   bool                   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

// ChatResponse represents a response from the chat engine
type ChatResponse struct {
	Message ChatMessage `json:"message"`
	Done    bool        `json:"done"`
	Error   error       `json:"error,omitempty"`
}

// ChatEngine manages the LLM inference using Ollama's internal llama bindings
type ChatEngine struct {
	modelPath       string
	model           *llama.Model
	context         *llama.Context
	requestChan     chan ChatRequest
	responseChan    chan ChatResponse
	ctx             context.Context
	cancel          context.CancelFunc
	mu              sync.RWMutex
	running         bool
	// RAG components
	vectorStore     *VectorStore
	embeddingEngine *EmbeddingEngine
	simpleRAGStore  *SimpleRAGStore
	ragEnabled      bool
	simpleRAGEnabled bool
}

// NewChatEngine creates a new chat engine instance
func NewChatEngine(modelPath string) *ChatEngine {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &ChatEngine{
		modelPath:    modelPath,
		requestChan:  make(chan ChatRequest, 10),
		responseChan: make(chan ChatResponse, 10),
		ctx:          ctx,
		cancel:       cancel,
		ragEnabled:   false,
	}
}

// EnableRAG enables Retrieval-Augmented Generation with the given vector store and embedding engine
func (ce *ChatEngine) EnableRAG(vectorStore *VectorStore, embeddingEngine *EmbeddingEngine) {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	ce.vectorStore = vectorStore
	ce.embeddingEngine = embeddingEngine
	ce.ragEnabled = true
	
	log.Printf("RAG enabled with %d documents in vector store", vectorStore.GetDocumentCount())
}

// EnableSimpleRAG enables simple keyword-based RAG with the given store
func (ce *ChatEngine) EnableSimpleRAG(simpleStore *SimpleRAGStore) {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	ce.simpleRAGStore = simpleStore
	ce.simpleRAGEnabled = true
	
	log.Printf("Simple RAG enabled with %d documents", simpleStore.GetDocumentCount())
}

// Start initializes and starts the chat engine with Ollama's llama bindings
func (ce *ChatEngine) Start() error {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	if ce.running {
		return fmt.Errorf("chat engine is already running")
	}
	
	// TODO: Complete Ollama llama.cpp integration
	// Current implementation loads the model but uses simulated responses
	// Need to implement proper sampling with SamplingContext for real inference
	
	log.Printf("Loading model from %s...", ce.modelPath)
	
	// TODO: Verify model file exists and is accessible
	
	// Initialize llama backend
	llama.BackendInit()
	
	// Set up model parameters
	modelParams := llama.ModelParams{
		NumGpuLayers: 0, // TODO: Add GPU support detection
		UseMmap:      true,
		VocabOnly:    false,
	}
	
	// Load model
	model, err := llama.LoadModelFromFile(ce.modelPath, modelParams)
	if err != nil {
		// TODO: For POC, continue without actual model loading
		log.Printf("Model loading failed (expected for POC): %v", err)
		log.Printf("Continuing with simulated responses to demonstrate architecture...")
		ce.model = nil // Will use simulation
	} else {
		ce.model = model
		
		// Create context
		contextParams := llama.NewContextParams(
			4096, // numCtx
			512,  // batchSize  
			1,    // numSeqMax
			4,    // threads
			false, // flashAttention
			"",   // kvCacheType
		)
		
		context, err := llama.NewContextWithModel(model, contextParams)
		if err != nil {
			log.Printf("Context creation failed: %v", err)
			ce.context = nil
		} else {
			ce.context = context
		}
	}
	
	ce.running = true
	
	log.Printf("Chat engine initialized, starting inference goroutine...")
	
	// Start the inference goroutine
	go ce.inferenceLoop()
	
	return nil
}

// Stop gracefully shuts down the chat engine
func (ce *ChatEngine) Stop() {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	if !ce.running {
		return
	}
	
	ce.cancel()
	ce.running = false
	
	// Clean up resources
	// Context uses finalizer, model needs explicit free
	if ce.model != nil {
		llama.FreeModel(ce.model)
	}
	
	close(ce.requestChan)
	close(ce.responseChan)
}

// Chat sends a chat request and returns a response channel
func (ce *ChatEngine) Chat(messages []ChatMessage, stream bool) <-chan ChatResponse {
	responseChan := make(chan ChatResponse, 10)
	
	go func() {
		defer close(responseChan)
		
		select {
		case ce.requestChan <- ChatRequest{
			Messages: messages,
			Stream:   stream,
		}:
			// Request sent successfully
		case <-ce.ctx.Done():
			responseChan <- ChatResponse{
				Error: fmt.Errorf("chat engine is shutting down"),
			}
			return
		}
		
		// Forward responses from the main response channel
		for {
			select {
			case response, ok := <-ce.responseChan:
				if !ok {
					return
				}
				responseChan <- response
				if response.Done || response.Error != nil {
					return
				}
			case <-ce.ctx.Done():
				responseChan <- ChatResponse{
					Error: fmt.Errorf("chat engine is shutting down"),
				}
				return
			}
		}
	}()
	
	return responseChan
}

// inferenceLoop runs the main inference logic in a separate goroutine
func (ce *ChatEngine) inferenceLoop() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Chat engine panic recovered: %v", r)
		}
	}()
	
	for {
		select {
		case request, ok := <-ce.requestChan:
			if !ok {
				return
			}
			
			ce.processRequest(request)
			
		case <-ce.ctx.Done():
			return
		}
	}
}

// processRequest handles individual chat requests using Ollama's llama bindings
func (ce *ChatEngine) processRequest(request ChatRequest) {
	// Get user query for RAG
	userQuery := ce.extractUserQuery(request.Messages)
	
	// Build prompt from messages with optional RAG context
	prompt, err := ce.buildPromptWithRAG(request.Messages, userQuery)
	if err != nil {
		log.Printf("Failed to build prompt with RAG: %v", err)
		ce.sendErrorResponse(fmt.Errorf("failed to build prompt: %v", err))
		return
	}
	
	if ce.model != nil && ce.context != nil {
		// Real inference with loaded model
		log.Printf("Starting inference for prompt: %s...", prompt[:min(50, len(prompt))])
		
		response, err := ce.performInference(prompt, request.Options)
		if err != nil {
			log.Printf("Inference failed: %v", err)
			ce.sendErrorResponse(fmt.Errorf("inference failed: %v", err))
			return
		}
		
		if request.Stream {
			ce.streamRealResponse(response)
		} else {
			ce.sendCompleteResponse(response)
		}
	} else {
		// Fallback to simulation for missing model
		log.Printf("Model not loaded, using simulation for: %s...", prompt[:min(50, len(prompt))])
		response := fmt.Sprintf("🤖 **Model Loading Failed - Using Simulation**\n\n"+
			"📝 **Your input:** %s\n\n"+
			"⚠️ **Status:** Model file could not be loaded. This could be due to:\n"+
			"- Incorrect model path\n"+
			"- Unsupported model format\n"+
			"- Insufficient memory\n\n"+
			"💡 **Try:** Use a valid GGUF model file path", 
			prompt[:min(100, len(prompt))])
		
		if request.Stream {
			ce.simulateStreamingResponse(response)
		} else {
			ce.simulateNonStreamingResponse(response)
		}
	}
}

// simulateStreamingResponse simulates streaming for demonstration
func (ce *ChatEngine) simulateStreamingResponse(response string) {
	words := strings.Fields(response)
	var fullResponse strings.Builder
	
	for _, word := range words {
		piece := word + " "
		fullResponse.WriteString(piece)
		
		// Send streaming chunk
		select {
		case ce.responseChan <- ChatResponse{
			Message: ChatMessage{
				Role:    "assistant",
				Content: piece,
			},
			Done: false,
		}:
			// Simulate natural typing speed
			time.Sleep(100 * time.Millisecond)
		case <-ce.ctx.Done():
			return
		}
	}
	
	// Send final response
	select {
	case ce.responseChan <- ChatResponse{
		Message: ChatMessage{
			Role:    "assistant",
			Content: strings.TrimSpace(fullResponse.String()),
		},
		Done: true,
	}:
	case <-ce.ctx.Done():
	}
}

// simulateNonStreamingResponse simulates non-streaming response  
func (ce *ChatEngine) simulateNonStreamingResponse(response string) {
	// Simulate processing time
	time.Sleep(500 * time.Millisecond)
	
	// Send complete response
	select {
	case ce.responseChan <- ChatResponse{
		Message: ChatMessage{
			Role:    "assistant",
			Content: response,
		},
		Done: true,
	}:
	case <-ce.ctx.Done():
	}
}


// performInference runs actual model inference using Ollama's llama bindings
func (ce *ChatEngine) performInference(prompt string, options map[string]interface{}) (string, error) {
	// Tokenize the prompt
	tokens, err := ce.model.Tokenize(prompt, true, true)
	if err != nil {
		return "", fmt.Errorf("tokenization failed: %v", err)
	}
	
	// Create batch for processing
	batch, err := llama.NewBatch(len(tokens), 1, 0)
	if err != nil {
		return "", fmt.Errorf("batch creation failed: %v", err)
	}
	defer batch.Free()
	
	// Add tokens to batch
	for i, token := range tokens {
		batch.Add(token, nil, i, i == len(tokens)-1, 0) // Only get logits for last token
	}
	
	// Process the batch
	err = ce.context.Decode(batch)
	if err != nil {
		return "", fmt.Errorf("context decode failed: %v", err)
	}
	
	// Set up sampling parameters
	samplingParams := llama.SamplingParams{
		TopK:           40,
		TopP:           0.9,
		MinP:           0.1,
		Temp:           0.7, // TODO: Use request temperature
		RepeatLastN:    64,
		PenaltyRepeat:  1.1,
		PenaltyFreq:    0.0,
		PenaltyPresent: 0.0,
		PenalizeNl:     true,
		Seed:           0,
	}
	
	// Create sampling context
	sampler, err := llama.NewSamplingContext(ce.model, samplingParams)
	if err != nil {
		return "", fmt.Errorf("sampling context creation failed: %v", err)
	}
	
	var response strings.Builder
	maxTokens := 512 // TODO: Make configurable
	
	// Generate tokens iteratively
	for i := 0; i < maxTokens; i++ {
		// Sample next token
		token := sampler.Sample(ce.context, batch.NumTokens()-1)
		
		// Check for end of generation
		if ce.model.TokenIsEog(token) {
			break
		}
		
		// Convert token to text
		piece := ce.model.TokenToPiece(token)
		response.WriteString(piece)
		
		// Accept the token for grammar/repetition tracking
		sampler.Accept(token, true)
		
		// Prepare for next iteration - add token to batch
		batch.Clear()
		batch.Add(token, nil, len(tokens)+i, true, 0)
		
		// Decode for next iteration
		err = ce.context.Decode(batch)
		if err != nil {
			log.Printf("Decode failed during generation: %v", err)
			break
		}
	}
	
	return strings.TrimSpace(response.String()), nil
}

// sendErrorResponse sends an error response
func (ce *ChatEngine) sendErrorResponse(err error) {
	select {
	case ce.responseChan <- ChatResponse{
		Error: err,
		Done:  true,
	}:
	case <-ce.ctx.Done():
	}
}

// sendCompleteResponse sends a complete non-streaming response
func (ce *ChatEngine) sendCompleteResponse(content string) {
	select {
	case ce.responseChan <- ChatResponse{
		Message: ChatMessage{
			Role:    "assistant",
			Content: content,
		},
		Done: true,
	}:
	case <-ce.ctx.Done():
	}
}

// streamRealResponse sends a real response in streaming chunks
func (ce *ChatEngine) streamRealResponse(content string) {
	words := strings.Fields(content)
	var accumulated strings.Builder
	
	for _, word := range words {
		piece := word + " "
		accumulated.WriteString(piece)
		
		select {
		case ce.responseChan <- ChatResponse{
			Message: ChatMessage{
				Role:    "assistant",
				Content: piece,
			},
			Done: false,
		}:
			time.Sleep(50 * time.Millisecond) // Natural typing speed
		case <-ce.ctx.Done():
			return
		}
	}
	
	// Send final complete response
	select {
	case ce.responseChan <- ChatResponse{
		Message: ChatMessage{
			Role:    "assistant",
			Content: strings.TrimSpace(accumulated.String()),
		},
		Done: true,
	}:
	case <-ce.ctx.Done():
	}
}

// extractUserQuery extracts the latest user message for RAG search
func (ce *ChatEngine) extractUserQuery(messages []ChatMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}

// buildPromptWithRAG builds a prompt with optional RAG context
func (ce *ChatEngine) buildPromptWithRAG(messages []ChatMessage, userQuery string) (string, error) {
	var systemMessage string
	var conversationMessages []ChatMessage
	
	// Separate system message from conversation
	for _, msg := range messages {
		if msg.Role == "system" {
			systemMessage = msg.Content
		} else {
			conversationMessages = append(conversationMessages, msg)
		}
	}
	
	// Add RAG context if enabled
	if ce.ragEnabled && userQuery != "" && ce.vectorStore != nil && ce.embeddingEngine != nil {
		ragContext, err := ce.retrieveRAGContext(userQuery)
		if err != nil {
			log.Printf("Warning: RAG retrieval failed: %v", err)
		} else if ragContext.NumDocuments > 0 {
			// Enhance system message with retrieved context
			enhancedSystem := fmt.Sprintf("%s\n\n%s\n\nBased on the above documentation, please provide accurate and helpful responses about OpenTDF.", 
				systemMessage, ragContext.ContextText)
			systemMessage = enhancedSystem
			
			log.Printf("RAG: Retrieved %d relevant documents for query", ragContext.NumDocuments)
		}
	} else if ce.simpleRAGEnabled && userQuery != "" && ce.simpleRAGStore != nil {
		ragContext, err := ce.retrieveSimpleRAGContext(userQuery)
		if err != nil {
			log.Printf("Warning: Simple RAG retrieval failed: %v", err)
		} else if ragContext.NumDocuments > 0 {
			// Enhance system message with retrieved context
			enhancedSystem := fmt.Sprintf("%s\n\n%s\n\nBased on the above documentation, please provide accurate and helpful responses about OpenTDF.", 
				systemMessage, ragContext.ContextText)
			systemMessage = enhancedSystem
			
			log.Printf("Simple RAG: Retrieved %d relevant documents for query", ragContext.NumDocuments)
		}
	}
	
	return ce.buildPrompt(systemMessage, conversationMessages), nil
}

// retrieveRAGContext performs similarity search and builds context
func (ce *ChatEngine) retrieveRAGContext(query string) (RAGContext, error) {
	// Generate embedding for the query
	queryEmbedding, err := ce.embeddingEngine.GenerateEmbedding(query)
	if err != nil {
		return RAGContext{}, fmt.Errorf("failed to generate query embedding: %v", err)
	}
	
	// Search for similar documents
	results, err := ce.vectorStore.Search(queryEmbedding, 5) // Top 5 results
	if err != nil {
		return RAGContext{}, fmt.Errorf("similarity search failed: %v", err)
	}
	
	// Filter results by similarity threshold
	var filteredResults []SimilarityResult
	for _, result := range results {
		if result.Similarity > 0.3 { // Minimum similarity threshold
			filteredResults = append(filteredResults, result)
		}
	}
	
	// Build context with max 2000 tokens to leave room for conversation
	return BuildRAGContext(query, filteredResults, 2000), nil
}

// retrieveSimpleRAGContext performs simple keyword search and builds context
func (ce *ChatEngine) retrieveSimpleRAGContext(query string) (RAGContext, error) {
	// Search for similar documents using simple keyword matching
	results, err := ce.simpleRAGStore.Search(query, 5) // Top 5 results
	if err != nil {
		return RAGContext{}, fmt.Errorf("simple search failed: %v", err)
	}
	
	// Filter results by score threshold
	var filteredResults []SearchResult
	for _, result := range results {
		if result.Score > 0.1 { // Minimum score threshold
			filteredResults = append(filteredResults, result)
		}
	}
	
	// Build context with max 2000 tokens to leave room for conversation
	return BuildSimpleRAGContext(query, filteredResults, 2000), nil
}

// buildPrompt converts chat messages to a prompt string
func (ce *ChatEngine) buildPrompt(systemMessage string, messages []ChatMessage) string {
	var prompt strings.Builder
	
	// Add system message
	if systemMessage != "" {
		prompt.WriteString(fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n", systemMessage))
	}
	
	// Add conversation messages
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			prompt.WriteString(fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n", msg.Content))
		case "assistant":
			prompt.WriteString(fmt.Sprintf("<|im_start|>assistant\n%s<|im_end|>\n", msg.Content))
		}
	}
	
	// Add the assistant prompt to start generation
	prompt.WriteString("<|im_start|>assistant\n")
	
	return prompt.String()
}