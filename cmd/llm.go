package cmd

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/jrschumacher/chanollama"
	"github.com/opentdf/otdfctl/pkg/cli"
	"github.com/opentdf/otdfctl/pkg/llm"
	"github.com/opentdf/otdfctl/pkg/llm/evaluation"
	"github.com/opentdf/otdfctl/pkg/man"
	"github.com/spf13/cobra"
)

var llmCmd = man.Docs.GetCommand("llm")

var llmChatCmd = man.Docs.GetCommand("llm/chat", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	// Resolve model path with local-only resolution
	var model string
	if len(args) > 0 {
		model = args[0]
	} else {
		// Try to find a local model in standard locations
		pathResolver := llm.NewModelPathResolver(&OtdfctlCfg)
		if resolvedPath, err := pathResolver.ResolveChatModelPath(""); err == nil {
			model = resolvedPath
		} else {
			// Search for any local models in standard locations
			if foundModel, err := findLocalChatModel(); err == nil {
				model = foundModel
			} else {
				c.ExitWithError("No model specified and no local models found. Please provide a model path.", nil)
			}
		}
	}
	
	// Get flag values
	stream := c.Flags.GetOptionalBool("stream")
	contextSize := int(c.Flags.GetOptionalInt32("context-size"))
	temperatureFlag, _ := cmd.Flags().GetFloat64("temperature")
	temperature := temperatureFlag
	systemPrompt := c.Flags.GetOptionalString("system-prompt")
	enableRAG := c.Flags.GetOptionalBool("rag")
	indexPath := c.Flags.GetOptionalString("index-path")
	verbose := c.Flags.GetOptionalBool("verbose")
	embeddingModel := c.Flags.GetOptionalString("embedding-model")
	
	// Initialize chanollama engine with auto-detection (offline vs HTTP)
	config := chanollama.DefaultConfig(model)
	config.Timeout = 30 * time.Second
	config.Verbose = verbose
	if temperature > 0 {
		config.Temperature = temperature
	}
	
	// Configure for longer, more complete responses
	if contextSize > 0 {
		config.ContextSize = contextSize  // Context window size
		config.MaxTokens = min(contextSize/2, 2048)  // Generation length (up to half context or 2048)
	} else {
		// Default to larger values for better responses
		config.ContextSize = 8192     // Larger context window
		config.MaxTokens = 2048       // More generous generation length
	}
	
	// Show loading indicator for model initialization
	c.Printf("🔧 Loading model...\n")
	
	// Auto-detect offline mode for file paths
	chatEngine, err := chanollama.NewEngine(config)
	if err != nil {
		c.ExitWithError("Failed to initialize LLM engine", err)
	}
	defer chatEngine.Close()
	
	c.Printf("✅ Model ready!\n")
	
	// Set up RAG if enabled
	var ragEngine *chanollama.RAGEngine
	if enableRAG {
		c.Printf("🔧 Initializing RAG support...\n")
		
		// Auto-detect index type and set default paths
		useVector := false
		if indexPath == "" {
			homeDir, _ := os.UserHomeDir()
			vectorIndexPath := filepath.Join(homeDir, ".otdfctl", "vector_index.json")
			simpleIndexPath := filepath.Join(homeDir, ".otdfctl", "simple_rag_index.json")
			
			// Check if vector index exists
			if _, err := os.Stat(vectorIndexPath); err == nil {
				indexPath = vectorIndexPath
				useVector = true
				c.Printf("📐 Found vector index, using embeddings mode\n")
			} else if _, err := os.Stat(simpleIndexPath); err == nil {
				indexPath = simpleIndexPath
				useVector = false
				c.Printf("🔤 Found simple index, using keyword mode\n")
			} else {
				// Default to simple mode if no index exists
				indexPath = simpleIndexPath
				useVector = false
				c.Printf("🔤 No existing index found, defaulting to keyword mode\n")
			}
		} else {
			// User specified index path - detect type by filename
			if strings.Contains(indexPath, "vector") {
				useVector = true
				c.Printf("📐 Using vector embeddings mode\n")
			} else {
				useVector = false
				c.Printf("🔤 Using keyword matching mode\n")
			}
		}
		
		// Set default embedding model
		if embeddingModel == "" {
			embeddingModel = "mxbai-embed-large"
		}
		
		// Create RAG engine
		ragEngine = chanollama.NewRAGEngine(chatEngine, indexPath, useVector)
		
		// If using vector mode, set up embedding client for online query processing
		if useVector && embeddingModel != "" {
			// Create embedding client (only needed for query embedding generation)
			embeddingConfig := chanollama.DefaultConfig("")
			embeddingClient := chanollama.NewWithConfig("", embeddingConfig)
			ragEngine.WithEmbeddingClient(embeddingClient, embeddingModel)
			
			c.Printf("📐 Embedding model: %s\n", embeddingModel)
		}
		
		if err := ragEngine.LoadIndex(); err != nil {
			c.ExitWithError("Failed to load RAG index", err)
		}
		
		if ragEngine.GetDocumentCount() == 0 {
			c.Printf("⚠️  Warning: No documents found in RAG index. Run 'otdfctl llm ingest' first.\n")
		} else {
			if useVector {
				c.Printf("✅ Vector RAG enabled with %d documents\n", ragEngine.GetDocumentCount())
			} else {
				c.Printf("✅ Simple RAG enabled with %d documents\n", ragEngine.GetDocumentCount())
			}
		}
	}
	
	// Check if JSON output is requested
	if jsonFlag, _ := cmd.Flags().GetBool("json"); jsonFlag {
		session := map[string]interface{}{
			"model": model,
			"config": map[string]interface{}{
				"stream":       stream,
				"context_size": contextSize,
				"temperature":  temperature,
			},
			"status": "initialized",
		}
		c.ExitWithJSON(session)
		return
	}
	
	// Start interactive chat session
	if err := startInteractiveChat(c, chatEngine, ragEngine, systemPrompt, stream); err != nil {
		c.ExitWithError("Failed to start chat session", err)
	}
}))

// llmIngestCmd adds documents to the RAG index
var llmIngestCmd = man.Docs.GetCommand("llm/ingest", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	if len(args) == 0 {
		c.ExitWithError("Document path is required", nil)
	}
	
	docPath := args[0]
	indexPath := c.Flags.GetOptionalString("index-path")
	useEmbeddings := c.Flags.GetOptionalBool("embeddings")
	embeddingModel := c.Flags.GetOptionalString("embedding-model")
	
	// Set default paths based on mode
	if indexPath == "" {
		homeDir, _ := os.UserHomeDir()
		if useEmbeddings {
			indexPath = filepath.Join(homeDir, ".otdfctl", "vector_index.json")
		} else {
			indexPath = filepath.Join(homeDir, ".otdfctl", "simple_rag_index.json")
		}
	}
	
	// Set default embedding model
	if embeddingModel == "" {
		embeddingModel = "mxbai-embed-large"
	}
	
	// Read document content
	content, err := os.ReadFile(docPath)
	if err != nil {
		c.ExitWithError("Failed to read document", err)
	}
	
	// Create document
	doc := chanollama.Document{
		ID:      filepath.Base(docPath),
		Title:   filepath.Base(docPath),
		Content: string(content),
	}
	
	if useEmbeddings {
		c.Printf("🔧 Using vector embeddings mode...\n")
		c.Printf("📐 Embedding model: %s\n", embeddingModel)
		
		// Create vector store with embedding support
		vectorStore := chanollama.NewVectorStore(indexPath)
		
		// Initialize Ollama client for embedding generation
		config := chanollama.DefaultConfig("") // No chat model needed for embeddings
		client := chanollama.NewWithConfig("", config)
		vectorStore.WithClient(client).WithEmbeddingModel(embeddingModel)
		
		// Load existing index
		if err := vectorStore.LoadIndex(); err != nil {
			c.Printf("⚠️  Could not load existing index: %v\n", err)
		}
		
		c.Printf("🔄 Generating embeddings...\n")
		
		// Add document with embedding generation
		if err := vectorStore.AddDocumentWithEmbedding(doc); err != nil {
			c.ExitWithError("Failed to add document with embeddings", err)
		}
		
		// Save updated index
		if err := vectorStore.SaveIndex(); err != nil {
			c.ExitWithError("Failed to save vector index", err)
		}
		
		c.Printf("✅ Document ingested with embeddings: %s\n", docPath)
		c.Printf("📁 Vector index: %s\n", indexPath)
		c.Printf("📊 Total documents: %d\n", vectorStore.GetDocumentCount())
		
	} else {
		c.Printf("🔧 Using simple keyword mode...\n")
		
		// Use chanollama's SimpleRAGStore for document ingestion
		ragStore := chanollama.NewSimpleRAGStore(indexPath)
		
		// Load existing index
		if err := ragStore.LoadIndex(); err != nil {
			c.Printf("⚠️  Could not load existing index: %v\n", err)
		}
		
		// Add to RAG store
		if err := ragStore.AddDocument(doc); err != nil {
			c.ExitWithError("Failed to add document to RAG index", err)
		}
		
		// Save updated index
		if err := ragStore.SaveIndex(); err != nil {
			c.ExitWithError("Failed to save simple index", err)
		}
		
		c.Printf("✅ Document ingested: %s\n", docPath)
		c.Printf("📁 Index path: %s\n", indexPath)
		c.Printf("📊 Total documents: %d\n", ragStore.GetDocumentCount())
	}
}))

var llmEvalCmd = man.Docs.GetCommand("llm/eval", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	// Get evaluation parameters
	query := c.Flags.GetOptionalString("query")
	testSuite := c.Flags.GetOptionalString("test-suite")
	baseline := c.Flags.GetOptionalString("baseline")
	outputFormat := c.Flags.GetOptionalString("output")
	model := c.Flags.GetOptionalString("model")
	verbose := c.Flags.GetOptionalBool("verbose")
	agentMode := c.Flags.GetOptionalBool("agent-mode")
	
	if query == "" && testSuite == "" {
		c.ExitWithError("Either --query or --test-suite must be provided", nil)
	}
	
	// Initialize OpenTDF evaluator
	evaluator, err := initializeEvaluator(model, agentMode, verbose)
	if err != nil {
		c.ExitWithError("Failed to initialize evaluation engine", err)
	}
	defer evaluator.Close()
	
	// Initialize output formatter
	formatter := evaluation.NewOutputFormatter(outputFormat)
	
	if query != "" {
		// Single query evaluation
		result, err := evaluator.EvaluateQuery(query)
		if err != nil {
			c.ExitWithError("Failed to evaluate query", err)
		}
		
		if verbose {
			evaluation.PrintSingleQueryResult(c, result)
		}
		
		// Create summary for consistent output handling
		summary := chanollama.EvalSummary{
			TotalTests:   1,
			PassedTests:  1,
			FailedTests:  0,
			AverageScore: result.QualityScore,
			AverageTime:  result.ResponseTime,
			AverageLength: float64(result.Metrics.ResponseLength),
			Results:      []chanollama.EvalResult{result},
		}
		
		formatter.FormatEvaluationSummary(c, summary)
	} else {
		// Test suite evaluation
		summary, err := evaluator.EvaluateSuite(testSuite)
		if err != nil {
			c.ExitWithError("Failed to evaluate test suite", err)
		}
		
		if verbose {
			evaluation.PrintProgressResults(c, summary.Results)
		}
		
		formatter.FormatEvaluationSummary(c, summary)
		formatter.FormatDetailedResults(c, summary.Results)
		
		// Handle baseline comparison if provided
		if baseline != "" {
			comparison, err := evaluator.CompareWithBaseline(summary, baseline)
			if err != nil {
				c.ExitWithError("Failed to compare with baseline", err)
			}
			
			evaluation.FormatBaselineComparison(c, comparison, outputFormat)
		}
		
		// Exit with error if quality threshold not met
		context := "development" // Could be configurable
		_, _, passes := evaluator.GetQualityAssessment(summary.AverageScore, context)
		if !passes {
			c.ExitWithError("Quality threshold not met", nil)
		}
	}
}))

// initializeEvaluator creates an OpenTDF evaluator with the specified configuration
func initializeEvaluator(model string, agentMode bool, verbose bool) (*evaluation.OpenTDFEvaluator, error) {
	config := evaluation.DefaultEvaluatorConfig(model)
	config.AgentMode = agentMode
	config.Verbose = verbose
	
	pathResolver := llm.NewModelPathResolver(&OtdfctlCfg)
	return evaluation.NewOpenTDFEvaluator(config, pathResolver)
}

func init() {
	// Chat command flags
	llmChatCmd.Flags().Bool("stream", true, "Enable streaming responses")
	llmChatCmd.Flags().Int32("context-size", 4096, "Maximum context window size")
	llmChatCmd.Flags().Float64("temperature", 0.7, "Sampling temperature (0.0-1.0)")
	llmChatCmd.Flags().String("system-prompt", "", "Custom system prompt")
	llmChatCmd.Flags().Bool("rag", false, "Enable RAG (Retrieval-Augmented Generation)")
	llmChatCmd.Flags().String("index-path", "", "Path to RAG vector index (default: ~/.otdfctl/rag_index.json)")
	llmChatCmd.Flags().String("embedding-model", "mxbai-embed-large", "Embedding model for vector RAG (default: mxbai-embed-large)")
	llmChatCmd.Flags().Bool("json", false, "Output in JSON format")
	llmChatCmd.Flags().Bool("verbose", false, "Show detailed loading logs")
	
	// Ingest command flags
	llmIngestCmd.Flags().String("index-path", "", "Path to RAG index (default: ~/.otdfctl/simple_rag_index.json or vector_index.json)")
	llmIngestCmd.Flags().Bool("embeddings", false, "Use vector embeddings instead of keyword matching")
	llmIngestCmd.Flags().String("embedding-model", "mxbai-embed-large", "Embedding model to use (requires --embeddings)")
	llmIngestCmd.Flags().Bool("json", false, "Output in JSON format")
	
	// Eval command flags
	llmEvalCmd.Flags().StringP("query", "q", "", "Single query to evaluate")
	llmEvalCmd.Flags().StringP("test-suite", "s", "", "Path to JSON test suite file")
	llmEvalCmd.Flags().StringP("baseline", "b", "", "Path to baseline results for comparison")
	llmEvalCmd.Flags().StringP("output", "o", "summary", "Output format: json, detailed, summary")
	llmEvalCmd.Flags().StringP("model", "m", "", "Path to GGUF model file or Ollama model name")
	llmEvalCmd.Flags().BoolP("verbose", "v", false, "Enable verbose logging and detailed metrics")
	llmEvalCmd.Flags().Bool("agent-mode", false, "Enable agent-based routing for queries")
	llmEvalCmd.Flags().Float64("min-score", 5.0, "Minimum acceptable quality score")
	llmEvalCmd.Flags().Duration("timeout", 0, "Maximum evaluation time per query")
	llmEvalCmd.MarkFlagRequired("model")
	
	// Add subcommands to llm parent
	llmCmd.AddCommand(&llmChatCmd.Command)
	llmCmd.AddCommand(&llmIngestCmd.Command)
	llmCmd.AddCommand(&llmEvalCmd.Command)
	
	// Add llm command to root
	RootCmd.AddCommand(&llmCmd.Command)
}

// startInteractiveChat handles the interactive chat session with chanollama
func startInteractiveChat(c *cli.Cli, engine chanollama.Engine, ragEngine *chanollama.RAGEngine, systemPrompt string, stream bool) error {
	// Set up signal handler for SIGABRT to prevent crashes
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGABRT)
	go func() {
		<-sigChan
		c.Printf("\n\n⚠️  Critical error detected - conversation too long for model. Restarting chat...\n")
		c.Printf("Use 'clear' command to reset conversation history or restart with a smaller model.\n")
		os.Exit(1)
	}()
	
	// Initialize conversation with system message
	messages := []chanollama.Message{}
	if systemPrompt != "" {
		messages = append(messages, chanollama.NewSystemMessage(systemPrompt))
	} else {
		messages = append(messages, chanollama.NewSystemMessage(getDefaultSystemPrompt()))
	}
	
	c.Printf("🤖 OpenTDF LLM Chat started! Type 'exit' to quit, 'clear' to clear history.\n")
	c.Printf("   Use '/stream' to toggle streaming mode, '/help' for commands.\n")
	c.Printf("   Powered by chanollama\n\n")
	
	scanner := bufio.NewScanner(os.Stdin)
	
	for {
		c.Printf("> ")
		
		if !scanner.Scan() {
			break
		}
		
		input := strings.TrimSpace(scanner.Text())
		
		if input == "" {
			continue
		}
		
		// Handle commands
		switch input {
		case "exit", "quit":
			c.Println("Goodbye! 👋")
			return nil
		case "clear":
			messages = messages[:1] // Keep system message
			c.Println("Chat history cleared.")
			continue
		case "/stream":
			stream = !stream
			c.Printf("Streaming mode: %v\n", stream)
			continue
		case "/help":
			printHelp(c)
			continue
		}
		
		// Add user message
		messages = append(messages, chanollama.NewUserMessage(input))
		
		// Get response
		c.Printf("🤖 ")
		
		start := time.Now()
		var fullResponse strings.Builder
		
		// Use RAG engine if available, otherwise use regular engine
		if ragEngine != nil {
			// Use RAG-enhanced querying with conversation context
			// For RAG, we need to perform document retrieval and then use regular chat
			ragResponse, err := safePerformRAGQuery(ragEngine, engine, messages, input, stream, func(token string) {
				c.Printf("%s", token)
				os.Stdout.Sync()
				fullResponse.WriteString(token)
			})
			
			if err != nil {
				c.Printf("\nError: %v\n", err)
				continue
			}
			
			if !stream {
				c.Printf("%s\n\n⏱️  Response time: %v\n", ragResponse.Content, time.Since(start))
				fullResponse.WriteString(ragResponse.Content)
			} else {
				c.Printf("\n\n⏱️  Response time: %v\n", time.Since(start))
			}
		} else {
			// Use regular engine
			if stream {
				// Use streaming inference
				response := safeEngineCallStream(func() *chanollama.Response {
					return engine.ChatStream(messages, func(token string) {
						c.Printf("%s", token)
						os.Stdout.Sync() // Force flush for real-time streaming
						fullResponse.WriteString(token)
					})
				})
				
				if response.Error != nil {
					c.Printf("\nError: %v\n", response.Error)
					continue
				}
				
				c.Printf("\n\n⏱️  Response time: %v\n", time.Since(start))
			} else {
				// Use non-streaming inference
				response, err := safeEngineCall(func() (*chanollama.Response, error) {
					return engine.Chat(messages)
				})
				
				if err != nil {
					c.Printf("\nError: %v\n", err)
					continue
				}
				
				c.Printf("%s\n\n⏱️  Response time: %v\n", response.Content, time.Since(start))
				fullResponse.WriteString(response.Content)
			}
		}
		
		// Add assistant response to history
		if fullResponse.Len() > 0 {
			messages = append(messages, chanollama.NewAssistantMessage(fullResponse.String()))
		}
		
		// Prevent conversation from growing too large (temporary safeguard)
		// Keep system message + last 6 messages (3 exchanges) - aggressive limit
		if len(messages) > 8 {
			messages = append(messages[:1], messages[len(messages)-6:]...)
			c.Printf("\n⚠️  Conversation history truncated to prevent memory issues\n")
		}
	}
	
	return nil
}

// getDefaultSystemPrompt returns the default system prompt with OpenTDF knowledge but natural responses
func getDefaultSystemPrompt() string {
	return `You are a helpful AI assistant with expertise in OpenTDF (Trusted Data Format) and related technologies. 

When asked about OpenTDF topics, you can help with:
- OpenTDF architecture and concepts
- Policy management (attributes, namespaces, values, subject mappings)
- TDF encryption/decryption workflows
- Key Access Service (KAS) configuration
- otdfctl CLI tool usage and troubleshooting
- OpenTDF Platform deployment and administration
- Data security and access control patterns

For non-OpenTDF questions, respond naturally and helpfully. Always be honest about what you do and don't know. Provide clear, accurate information without making assumptions about the user's specific context or situation.`
}

// performRAGQuery performs document retrieval and then uses regular chat with context
func performRAGQuery(ragEngine *chanollama.RAGEngine, engine chanollama.Engine, messages []chanollama.Message, input string, stream bool, streamCallback func(string)) (*chanollama.Response, error) {
	// First, perform document retrieval using the RAG engine's search capability
	// We'll use the input query to find relevant documents
	searchResults, err := ragEngine.Search(input, 3) // Get top 3 relevant documents
	if err != nil {
		// If search fails, fall back to regular chat
		if stream {
			return engine.ChatStream(messages, streamCallback), nil
		}
		return engine.Chat(messages)
	}
	
	// Build enhanced system prompt with retrieved documents
	var enhancedSystemPrompt strings.Builder
	enhancedSystemPrompt.WriteString(getDefaultSystemPrompt())
	
	if len(searchResults) > 0 {
		enhancedSystemPrompt.WriteString("\n\n# Relevant Documentation\n\n")
		for _, result := range searchResults {
			enhancedSystemPrompt.WriteString(fmt.Sprintf("## %s\n", result.Document.Title))
			enhancedSystemPrompt.WriteString(fmt.Sprintf("**Relevance:** %.3f\n\n", result.Score))
			enhancedSystemPrompt.WriteString(result.Document.Content)
			enhancedSystemPrompt.WriteString("\n\n---\n\n")
		}
		enhancedSystemPrompt.WriteString("Based on the above documentation, please provide accurate and helpful responses.\n")
	}
	
	// Create enhanced messages with RAG context
	enhancedMessages := []chanollama.Message{}
	
	// Replace system message with enhanced version
	enhancedMessages = append(enhancedMessages, chanollama.NewSystemMessage(enhancedSystemPrompt.String()))
	
	// Add conversation history (skip original system message)
	for i := 1; i < len(messages); i++ {
		enhancedMessages = append(enhancedMessages, messages[i])
	}
	
	// Add current user input
	enhancedMessages = append(enhancedMessages, chanollama.NewUserMessage(input))
	
	// Use regular chat engine with enhanced context
	if stream {
		return engine.ChatStream(enhancedMessages, streamCallback), nil
	}
	return engine.Chat(enhancedMessages)
}

// printHelp displays available commands
func printHelp(c *cli.Cli) {
	c.Println("\nAvailable commands:")
	c.Println("  exit, quit  - Exit the chat")
	c.Println("  clear       - Clear chat history")
	c.Println("  /stream     - Toggle streaming mode")
	c.Println("  /help       - Show this help")
}

// findLocalChatModel searches for chat models in standard locations
func findLocalChatModel() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}
	
	// Standard locations to search for chat models
	searchPaths := []string{
		// Ollama standard locations
		filepath.Join(homeDir, ".ollama", "models"),
		// Alternative locations
		"/usr/local/lib/ollama/models",
		"/opt/ollama/models",
		// Local model directory
		"./models",
		// Current directory
		".",
	}
	
	// Common chat model filenames to look for
	modelNames := []string{
		"llama2.gguf",
		"llama-2-7b.gguf",
		"llama-2-13b.gguf", 
		"mistral-7b.gguf",
		"codellama.gguf",
		"vicuna.gguf",
		"alpaca.gguf",
	}
	
	// First pass: Look for specific well-known models
	for _, searchPath := range searchPaths {
		if _, err := os.Stat(searchPath); err != nil {
			continue // Directory doesn't exist, skip
		}
		
		for _, modelName := range modelNames {
			modelPath := filepath.Join(searchPath, modelName)
			if err := validateModelFile(modelPath); err == nil {
				return modelPath, nil
			}
		}
	}
	
	// Second pass: Look for any .gguf files
	for _, searchPath := range searchPaths {
		if foundModel := findFirstGGUFModel(searchPath); foundModel != "" {
			return foundModel, nil
		}
	}
	
	return "", fmt.Errorf("no local chat models found in standard locations")
}

// findFirstGGUFModel finds the first .gguf file in the given directory
func findFirstGGUFModel(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		
		if strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
			modelPath := filepath.Join(dir, entry.Name())
			if err := validateModelFile(modelPath); err == nil {
				return modelPath
			}
		}
	}
	
	return ""
}

// validateModelFile validates that a model file exists and is accessible
func validateModelFile(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	
	if info.IsDir() {
		return fmt.Errorf("path is a directory, not a file")
	}
	
	if info.Size() == 0 {
		return fmt.Errorf("model file is empty")
	}
	
	return nil
}

// safeEngineCall wraps engine.Chat calls with panic recovery
func safeEngineCall(fn func() (*chanollama.Response, error)) (*chanollama.Response, error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Engine call panic recovered: %v", r)
			log.Printf("Stack trace: %s", debug.Stack())
		}
	}()
	
	response, err := fn()
	if err != nil {
		return nil, fmt.Errorf("engine call failed: %w", err)
	}
	return response, nil
}

// safeEngineCallStream wraps engine.ChatStream calls with panic recovery
func safeEngineCallStream(fn func() *chanollama.Response) *chanollama.Response {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Streaming engine call panic recovered: %v", r)
			log.Printf("Stack trace: %s", debug.Stack())
		}
	}()
	
	response := fn()
	if response == nil {
		return &chanollama.Response{
			Content: "Error: Engine call returned nil response",
			Error:   fmt.Errorf("engine call returned nil response"),
		}
	}
	return response
}

// safePerformRAGQuery wraps performRAGQuery calls with panic recovery
func safePerformRAGQuery(ragEngine *chanollama.RAGEngine, engine chanollama.Engine, messages []chanollama.Message, input string, stream bool, streamCallback func(string)) (*chanollama.Response, error) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("RAG query panic recovered: %v", r)
			log.Printf("Stack trace: %s", debug.Stack())
		}
	}()
	
	response, err := performRAGQuery(ragEngine, engine, messages, input, stream, streamCallback)
	if err != nil {
		return nil, fmt.Errorf("RAG query failed: %w", err)
	}
	return response, nil
}