package llm

import (
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/ollama/ollama/llama"
)

// SimpleEngine is a simplified LLM engine without complex goroutine management
type SimpleChatEngine struct {
	modelPath        string
	model           *llama.Model
	context         *llama.Context
	simpleRAGStore  *SimpleRAGStore
	ragEnabled      bool
	mu              sync.Mutex
	running         bool
}

// NewSimpleChatEngine creates a new simplified chat engine
func NewSimpleChatEngine(modelPath string) *SimpleChatEngine {
	return &SimpleChatEngine{
		modelPath:  modelPath,
		ragEnabled: false,
		running:    false,
	}
}

// EnableSimpleRAG enables RAG with the simple store
func (sce *SimpleChatEngine) EnableSimpleRAG(store *SimpleRAGStore) {
	sce.mu.Lock()
	defer sce.mu.Unlock()
	
	sce.simpleRAGStore = store
	sce.ragEnabled = true
	log.Printf("Simple RAG enabled with %d documents", store.GetDocumentCount())
}

// Start initializes the model
func (sce *SimpleChatEngine) Start() error {
	sce.mu.Lock()
	defer sce.mu.Unlock()
	
	if sce.running {
		return fmt.Errorf("engine already running")
	}
	
	log.Printf("Loading model from %s...", sce.modelPath)
	
	// Initialize llama backend
	llama.BackendInit()
	
	// Set up model parameters
	modelParams := llama.ModelParams{
		NumGpuLayers: 0,
		UseMmap:      true,
		VocabOnly:    false,
	}
	
	// Load model
	model, err := llama.LoadModelFromFile(sce.modelPath, modelParams)
	if err != nil {
		log.Printf("Model loading failed: %v", err)
		log.Printf("Continuing without model (simulation mode)")
		sce.model = nil
	} else {
		sce.model = model
		
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
			sce.context = nil
		} else {
			sce.context = context
		}
	}
	
	sce.running = true
	log.Printf("Simple chat engine initialized")
	return nil
}

// Stop cleans up resources
func (sce *SimpleChatEngine) Stop() {
	sce.mu.Lock()
	defer sce.mu.Unlock()
	
	if !sce.running {
		return
	}
	
	if sce.model != nil {
		llama.FreeModel(sce.model)
		sce.model = nil
	}
	
	sce.context = nil
	sce.running = false
	log.Printf("Simple chat engine stopped")
}

// SimpleResponse represents a simple response without streaming
type SimpleResponse struct {
	Content string
	Error   error
}

// StreamingCallback is called for each generated token during streaming
type StreamingCallback func(token string)

// Chat performs a simple chat without streaming
func (sce *SimpleChatEngine) Chat(messages []ChatMessage) SimpleResponse {
	sce.mu.Lock()
	defer sce.mu.Unlock()
	
	if !sce.running {
		return SimpleResponse{Error: fmt.Errorf("engine not running")}
	}
	
	// Extract user query for RAG
	userQuery := sce.extractUserQuery(messages)
	
	// Build prompt with optional RAG context
	prompt, err := sce.buildPromptWithRAG(messages, userQuery)
	if err != nil {
		return SimpleResponse{Error: fmt.Errorf("failed to build prompt: %v", err)}
	}
	
	// Perform inference
	if sce.model == nil || sce.context == nil {
		return SimpleResponse{Error: fmt.Errorf("model or context not loaded")}
	}
	
	log.Printf("Starting inference...")
	response, err := sce.performSimpleInference(prompt)
	if err != nil {
		log.Printf("Inference failed: %v", err)
		return SimpleResponse{Error: err}
	}
	
	return SimpleResponse{Content: response}
}

// ChatStream performs a simple chat with streaming output
func (sce *SimpleChatEngine) ChatStream(messages []ChatMessage, callback StreamingCallback) SimpleResponse {
	sce.mu.Lock()
	defer sce.mu.Unlock()
	
	if !sce.running {
		return SimpleResponse{Error: fmt.Errorf("engine not running")}
	}
	
	// Extract user query for RAG
	userQuery := sce.extractUserQuery(messages)
	
	// Build prompt with optional RAG context
	prompt, err := sce.buildPromptWithRAG(messages, userQuery)
	if err != nil {
		return SimpleResponse{Error: fmt.Errorf("failed to build prompt: %v", err)}
	}
	
	// Perform streaming inference
	if sce.model == nil || sce.context == nil {
		return SimpleResponse{Error: fmt.Errorf("model or context not loaded")}
	}
	
	log.Printf("Starting streaming inference...")
	response, err := sce.performStreamingInference(prompt, callback)
	if err != nil {
		log.Printf("Streaming inference failed: %v", err)
		return SimpleResponse{Error: err}
	}
	
	return SimpleResponse{Content: response}
}

// extractUserQuery gets the latest user message
func (sce *SimpleChatEngine) extractUserQuery(messages []ChatMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}

// buildPromptWithRAG builds prompt with RAG context
func (sce *SimpleChatEngine) buildPromptWithRAG(messages []ChatMessage, userQuery string) (string, error) {
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
	if sce.ragEnabled && userQuery != "" && sce.simpleRAGStore != nil {
		results, err := sce.simpleRAGStore.Search(userQuery, 2) // Top 2 results
		if err != nil {
			log.Printf("Warning: RAG search failed: %v", err)
		} else if len(results) > 0 {
			ragContext := BuildSimpleRAGContext(userQuery, results, 800) // Reduced from 1500 to 800 tokens
			if ragContext.NumDocuments > 0 {
				enhancedSystem := fmt.Sprintf("%s\n\n%s\n\nBased on the above documentation, please provide accurate and helpful responses about OpenTDF.",
					systemMessage, ragContext.ContextText)
				systemMessage = enhancedSystem
				log.Printf("Simple RAG: Retrieved %d relevant documents", ragContext.NumDocuments)
			}
		}
	}
	
	return sce.buildPrompt(systemMessage, conversationMessages), nil
}

// buildPrompt creates the final prompt string
func (sce *SimpleChatEngine) buildPrompt(systemMessage string, messages []ChatMessage) string {
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
	
	// Add assistant prompt to start generation
	prompt.WriteString("<|im_start|>assistant\n")
	
	return prompt.String()
}

// performSimpleInference does actual model inference
func (sce *SimpleChatEngine) performSimpleInference(prompt string) (string, error) {
	// Tokenize the prompt
	tokens, err := sce.model.Tokenize(prompt, true, true)
	if err != nil {
		return "", fmt.Errorf("tokenization failed: %v", err)
	}
	
	log.Printf("Prompt tokenized to %d tokens", len(tokens))
	
	// Limit batch size to prevent assertion errors
	maxBatchSize := 512
	if len(tokens) > maxBatchSize {
		log.Printf("Truncating prompt from %d to %d tokens", len(tokens), maxBatchSize)
		tokens = tokens[:maxBatchSize]
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
	err = sce.context.Decode(batch)
	if err != nil {
		return "", fmt.Errorf("context decode failed: %v", err)
	}
	
	// Set up sampling parameters
	samplingParams := llama.SamplingParams{
		TopK:           40,
		TopP:           0.9,
		MinP:           0.1,
		Temp:           0.7,
		RepeatLastN:    64,
		PenaltyRepeat:  1.1,
		PenaltyFreq:    0.0,
		PenaltyPresent: 0.0,
		PenalizeNl:     true,
		Seed:           0,
	}
	
	// Create sampling context
	sampler, err := llama.NewSamplingContext(sce.model, samplingParams)
	if err != nil {
		return "", fmt.Errorf("sampling context creation failed: %v", err)
	}
	
	var response strings.Builder
	maxTokens := 512
	
	// Generate tokens iteratively
	for i := 0; i < maxTokens; i++ {
		// Sample next token
		token := sampler.Sample(sce.context, batch.NumTokens()-1)
		
		// Check for end of generation
		if sce.model.TokenIsEog(token) {
			break
		}
		
		// Convert token to text
		piece := sce.model.TokenToPiece(token)
		response.WriteString(piece)
		
		// Accept the token for grammar/repetition tracking
		sampler.Accept(token, true)
		
		// Prepare for next iteration - add token to batch
		batch.Clear()
		batch.Add(token, nil, len(tokens)+i, true, 0)
		
		// Decode for next iteration
		err = sce.context.Decode(batch)
		if err != nil {
			log.Printf("Decode failed during generation: %v", err)
			break
		}
	}
	
	return strings.TrimSpace(response.String()), nil
}

// performStreamingInference does actual model inference with streaming output
func (sce *SimpleChatEngine) performStreamingInference(prompt string, callback StreamingCallback) (string, error) {
	// Tokenize the prompt
	tokens, err := sce.model.Tokenize(prompt, true, true)
	if err != nil {
		return "", fmt.Errorf("tokenization failed: %v", err)
	}
	
	log.Printf("Prompt tokenized to %d tokens", len(tokens))
	
	// Limit batch size to prevent assertion errors
	maxBatchSize := 512
	if len(tokens) > maxBatchSize {
		log.Printf("Truncating prompt from %d to %d tokens", len(tokens), maxBatchSize)
		tokens = tokens[:maxBatchSize]
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
	err = sce.context.Decode(batch)
	if err != nil {
		return "", fmt.Errorf("context decode failed: %v", err)
	}
	
	// Set up sampling parameters
	samplingParams := llama.SamplingParams{
		TopK:           40,
		TopP:           0.9,
		MinP:           0.1,
		Temp:           0.7,
		RepeatLastN:    64,
		PenaltyRepeat:  1.1,
		PenaltyFreq:    0.0,
		PenaltyPresent: 0.0,
		PenalizeNl:     true,
		Seed:           0,
	}
	
	// Create sampling context
	sampler, err := llama.NewSamplingContext(sce.model, samplingParams)
	if err != nil {
		return "", fmt.Errorf("sampling context creation failed: %v", err)
	}
	
	var response strings.Builder
	maxTokens := 512
	
	// Generate tokens iteratively with streaming
	for i := 0; i < maxTokens; i++ {
		// Sample next token
		token := sampler.Sample(sce.context, batch.NumTokens()-1)
		
		// Check for end of generation
		if sce.model.TokenIsEog(token) {
			break
		}
		
		// Convert token to text
		piece := sce.model.TokenToPiece(token)
		response.WriteString(piece)
		
		// Stream the token to the callback
		if callback != nil {
			callback(piece)
		}
		
		// Accept the token for grammar/repetition tracking
		sampler.Accept(token, true)
		
		// Prepare for next iteration - add token to batch
		batch.Clear()
		batch.Add(token, nil, len(tokens)+i, true, 0)
		
		// Decode for next iteration
		err = sce.context.Decode(batch)
		if err != nil {
			log.Printf("Decode failed during generation: %v", err)
			break
		}
	}
	
	return strings.TrimSpace(response.String()), nil
}

