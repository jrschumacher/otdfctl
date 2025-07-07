# LLM Engine Architecture Implementation Plan

## Overview
Build a completely decoupled, message-passing LLM engine that separates inference operations from interface concerns. This will solve the current Ctrl+C handling issues and create a robust, testable architecture.

## Core Design Principles

1. **Interface Agnostic** - Engine can power CLI, web API, desktop app, etc.
2. **Non-blocking Operations** - Interface never blocks on LLM operations
3. **Proper Cancellation** - All operations support context-based cancellation
4. **Resource Safety** - Single-threaded engine manages all LLM resources
5. **Observable** - Comprehensive metrics and health monitoring
6. **Testable** - Clear separation of concerns for testing

## Problem Statement

### Why Current Goroutine/Channel Solution Fails

The fundamental issue is **blocking I/O mixed with signal handling**:

1. **`bufio.ReadString('\n')` blocks the goroutine** - No amount of channel orchestration can interrupt a blocking syscall
2. **Race conditions** - Signal handler and main loop share state (`interrupted` flag)  
3. **Tight coupling** - Interface logic mixed with LLM engine management
4. **Resource leaks** - Goroutines and channels not properly cleaned up
5. **No cancellation** - Can't cancel ongoing LLM inference

## Architecture Components

### 1. Core Types and Messages

**File**: `pkg/llm/engine_types.go`

```go
type OperationType int

const (
    OpLoadModel OperationType = iota
    OpInference              // Send query and get response
    OpStream                 // Send query and get streaming response
    OpGetMetrics
    OpUnloadModel
    OpShutdown
    OpCancel
    OpConfigurePrompt
    OpConfigureRAG
    OpInspectModel
    OpHealthCheck
    OpConfigureGeneration
    OpTokenCount             // Count tokens in text
    OpWarmup                 // Pre-warm model
)

type EngineRequest struct {
    ID       string
    Type     OperationType
    Payload  interface{}
    Response chan<- EngineResponse
    Context  context.Context
}

type EngineResponse struct {
    RequestID string
    Success   bool
    Data      interface{}
    Error     error
    Metadata  map[string]interface{}
}

type StreamingResponse struct {
    RequestID string
    Token     string
    Done      bool
    Error     error
    Metadata  map[string]interface{}
}
```

### 2. Operation Payloads

```go
// Specific payload types for each operation
type LoadModelPayload struct {
    ModelPath     string
    SystemPrompt  string
    ContextSize   int
    EnableRAG     bool
    RAGIndexPath  string
    GenerationParams GenerationParams
}

type InferencePayload struct {
    Query        string
    MaxTokens    int
    Temperature  float64
    IncludeRAG   bool
}

type StreamingPayload struct {
    Query        string
    MaxTokens    int
    Temperature  float64
    IncludeRAG   bool
    ChunkSize    int // Tokens per streaming chunk
}

type ConfigurePromptPayload struct {
    SystemPrompt string
}

type ConfigureRAGPayload struct {
    Enabled   bool
    IndexPath string
}

type GenerationParams struct {
    Temperature    float64
    TopK          int
    TopP          float64
    RepeatPenalty float64
    MaxTokens     int
}
```

### 3. Engine Metrics

```go
type EngineMetrics struct {
    // Performance metrics
    TotalRequests     int64
    SuccessfulRequests int64
    FailedRequests    int64
    AverageResponseTime time.Duration
    
    // Resource metrics
    ModelLoaded       bool
    ModelPath         string
    MemoryUsage       int64
    TokensGenerated   int64
    
    // Current state
    ActiveRequests    int
    QueuedRequests    int
    LastActivity      time.Time
    Uptime           time.Duration
    
    // Error tracking
    LastError         string
    ErrorCount        map[string]int
}
```

### 4. Main Engine Implementation

**File**: `pkg/llm/engine.go`

```go
type LLMEngine struct {
    // Communication channels
    requests    chan EngineRequest
    streaming   chan StreamingResponse
    shutdown    chan struct{}
    
    // Engine state
    model          *llama.Model
    context        *llama.Context
    systemPrompt   string
    ragStore       *SimpleRAGStore
    ragEnabled     bool
    
    // Generation parameters
    genParams      GenerationParams
    
    // Metrics and monitoring
    metrics        *EngineMetrics
    metricsLock    sync.RWMutex
    
    // Cancellation tracking
    activeOps      map[string]context.CancelFunc
    opsLock        sync.RWMutex
    
    // Configuration
    config         *EngineConfig
    
    // Resource management
    resourceManager *ResourceManager
}

type EngineConfig struct {
    RequestChannelSize  int
    StreamingChannelSize int
    OperationTimeout    time.Duration
    MaxConcurrentOps    int
    MetricsInterval     time.Duration
}
```

### 5. Engine Operations

```go
// Core engine loop
func (e *LLMEngine) Run(ctx context.Context) error {
    defer e.cleanup()
    
    // Start metrics collection
    go e.collectMetrics(ctx)
    
    for {
        select {
        case req := <-e.requests:
            go e.handleRequest(req)
            
        case <-ctx.Done():
            return ctx.Err()
            
        case <-e.shutdown:
            return nil
        }
    }
}

func (e *LLMEngine) handleRequest(req EngineRequest) {
    // Track operation for cancellation
    e.trackOperation(req.ID, req.Context)
    defer e.untrackOperation(req.ID)
    
    // Update metrics
    e.updateMetrics(req.Type, time.Now())
    
    var response EngineResponse
    
    switch req.Type {
    case OpLoadModel:
        response = e.handleLoadModel(req)
    case OpInference:
        response = e.handleInference(req)
    case OpStream:
        response = e.handleStreaming(req)
    case OpGetMetrics:
        response = e.handleGetMetrics(req)
    // ... other operations
    }
    
    // Send response back
    select {
    case req.Response <- response:
    case <-req.Context.Done():
        // Request was cancelled
    }
}
```

### 6. Interface Abstraction Layer

**File**: `pkg/llm/engine_client.go`

```go
type EngineClient struct {
    engine    *LLMEngine
    responses chan EngineResponse
    streaming chan StreamingResponse
    timeout   time.Duration
}

func NewEngineClient(engine *LLMEngine) *EngineClient {
    return &EngineClient{
        engine:    engine,
        responses: make(chan EngineResponse, 10),
        streaming: make(chan StreamingResponse, 100),
        timeout:   30 * time.Second,
    }
}

func (c *EngineClient) LoadModel(ctx context.Context, payload LoadModelPayload) error {
    req := EngineRequest{
        ID:       generateRequestID(),
        Type:     OpLoadModel,
        Payload:  payload,
        Response: c.responses,
        Context:  ctx,
    }
    
    return c.sendRequestAndWait(req)
}

func (c *EngineClient) SendQuery(ctx context.Context, query string) (string, error) {
    payload := InferencePayload{Query: query}
    req := EngineRequest{
        ID:       generateRequestID(),
        Type:     OpInference,
        Payload:  payload,
        Response: c.responses,
        Context:  ctx,
    }
    
    resp, err := c.sendRequestAndWaitForResponse(req)
    if err != nil {
        return "", err
    }
    
    if result, ok := resp.Data.(string); ok {
        return result, nil
    }
    return "", fmt.Errorf("unexpected response type")
}

func (c *EngineClient) StreamQuery(ctx context.Context, query string, callback func(string)) error {
    payload := StreamingPayload{Query: query}
    req := EngineRequest{
        ID:       generateRequestID(),
        Type:     OpStream,
        Payload:  payload,
        Response: c.responses,
        Context:  ctx,
    }
    
    // Send request
    if err := c.sendRequest(req); err != nil {
        return err
    }
    
    // Handle streaming responses
    return c.handleStreamingResponse(req.ID, callback)
}
```

### 7. Refactored Chat Interface

**File**: `pkg/llm/simple_chat_v2.go`

```go
type SimpleChatInterface struct {
    engineClient   *EngineClient
    agentManager   *AgentManager
    currentAgent   *Agent
    config         *config.Config
    
    // UI state
    interrupted    bool
    cancelFunc     context.CancelFunc
}

func (sci *SimpleChatInterface) startChat() bool {
    sci.displayChatHeader()
    
    // Set up cancellation context
    ctx, cancel := context.WithCancel(context.Background())
    sci.cancelFunc = cancel
    defer cancel()
    
    // Set up signal handling
    sci.setupSignalHandling()
    
    reader := bufio.NewReader(os.Stdin)
    
    for {
        if sci.interrupted {
            return false // Return to menu
        }
        
        fmt.Print("You: ")
        input, err := reader.ReadString('\n')
        if err != nil {
            if err.Error() == "EOF" {
                return true // Exit app
            }
            continue
        }
        
        if sci.interrupted {
            return false
        }
        
        userInput := strings.TrimSpace(input)
        if userInput == "" {
            continue
        }
        
        if sci.handleSpecialCommands(userInput) {
            continue
        }
        
        // Send query to engine (non-blocking)
        sci.handleUserQuery(ctx, userInput)
    }
}

func (sci *SimpleChatInterface) handleUserQuery(ctx context.Context, query string) {
    fmt.Printf("\n🤖 %s:\n", sci.currentAgent.Metadata.Name)
    
    // Stream response from engine
    err := sci.engineClient.StreamQuery(ctx, query, func(token string) {
        // Check for interruption during streaming
        if sci.interrupted {
            return
        }
        fmt.Print(token)
    })
    
    if err != nil && !sci.interrupted {
        fmt.Printf("Error: %v\n", err)
    }
    
    fmt.Println() // New line after response
}

func (sci *SimpleChatInterface) setupSignalHandling() {
    sci.interrupted = false
    signalChan := make(chan os.Signal, 1)
    signal.Notify(signalChan, syscall.SIGINT)
    
    go func() {
        <-signalChan
        sci.interrupted = true
        if sci.cancelFunc != nil {
            sci.cancelFunc() // Cancel any ongoing operations
        }
        fmt.Println("\n\n🔙 Returning to agent menu...")
    }()
}
```

## Implementation Steps

### Phase 1: Core Engine Foundation
1. ✅ **Create engine types** (`pkg/llm/engine_types.go`)
2. **Implement basic engine** (`pkg/llm/engine.go`)
3. **Add operation handlers** for LoadModel, Inference, GetMetrics
4. **Create engine client abstraction** (`pkg/llm/engine_client.go`)

### Phase 2: Advanced Features
5. **Add streaming support** with proper cancellation
6. **Implement metrics collection** and monitoring
7. **Add RAG configuration** operations
8. **Create comprehensive error handling**

### Phase 3: Interface Integration
9. **Refactor SimpleChatInterface** to use engine client
10. **Implement proper signal handling** with cancellation
11. **Add command validation** and error display
12. **Test Ctrl+C behavior** thoroughly

### Phase 4: Testing and Polish
13. **Add unit tests** for engine operations
14. **Add integration tests** for signal handling
15. **Performance testing** and optimization
16. **Documentation** and examples

## Success Criteria

1. **Ctrl+C works instantly** - No delays or hanging
2. **Operations are cancellable** - Long-running inference can be stopped
3. **Interface never blocks** - Always responsive to user input
4. **Resource safety** - No memory leaks or resource corruption
5. **Proper error handling** - Clear error messages and recovery
6. **Comprehensive metrics** - Observable engine performance

## Key Benefits of This Architecture

1. **True Decoupling** - Interface never blocks on LLM operations
2. **Proper Cancellation** - Any operation can be cancelled via context
3. **Resource Safety** - Engine manages all resources in single goroutine
4. **Testability** - Engine can be mocked/tested independently
5. **Scalability** - Multiple interfaces can use same engine
6. **Monitoring** - Centralized metrics and health checking

## Configuration Questions

1. **Channel Buffer Sizes**: What should be the default buffer sizes for request/response channels?
   - **Decision**: RequestChannelSize: 100, StreamingChannelSize: 1000

2. **Timeout Strategy**: Should operations have individual timeouts or global engine timeout?
   - **Decision**: Individual operation timeouts with configurable defaults

3. **Error Recovery**: Should engine auto-restart on critical errors, or bubble up to caller?
   - **Decision**: Bubble up to caller for now, add auto-restart later if needed

4. **Metrics Storage**: Should metrics be in-memory only, or persisted somewhere?
   - **Decision**: In-memory for now, with option to add persistence later

5. **Testing Strategy**: Do you want integration tests with real models, or mock-based testing?
   - **Decision**: Mock-based for unit tests, optional integration tests with real models

## Design Decisions Made

- **Request/Response Pattern**: Using channels for async communication
- **Operation-Based**: Clear operation types instead of generic commands
- **Context-Based Cancellation**: Proper Go idiom for cancellation
- **Thread-Safe Metrics**: Safe concurrent access to metrics
- **Configurable Timeouts**: Flexible timeout configuration
- **Resource Management**: Single-threaded engine for resource safety

This architecture completely decouples the interface from LLM operations, ensuring Ctrl+C and other signals work reliably while maintaining clean separation of concerns.