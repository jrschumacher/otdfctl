# 🚀 Router Agent Implementation Plan

## 📝 Overview
Implement an intelligent routing system that automatically classifies user queries and directs them to the most appropriate specialized agent. This eliminates manual agent switching and provides a seamless conversational experience.

## 🎯 Goals
1. **Seamless UX**: User asks question → gets automatically routed → specialized response
2. **Performance**: Sub-200ms routing with minimal context usage
3. **Memory Efficient**: Same model for routing and answering
4. **Reliable**: Graceful fallbacks when routing fails
5. **Maintainable**: Simple, debuggable implementation

## 🏗️ Architecture Overview

### Two-Stage Inference
1. **Stage 1**: Router classification (fast, 1 token response)
2. **Stage 2**: Specialized agent response (full context)

### Context Budget Allocation
```
Router Phase:
- Router system prompt: ~200 tokens
- User query: ~50 tokens  
- Response: 1 token
Total: ~251 tokens (6% of budget)

Agent Phase:
- Agent system prompt: ~300 tokens
- Conversation history: ~3000+ tokens (94% of budget)
- User query: ~50 tokens
```

## 📋 Implementation Phases

### Phase 0: Extract Reusable Streaming Library

**Create standalone Go module: `github.com/jrschumacher/chanollama`**

**Why "chanollama"?** Combines "channel" + "ollama" - emphasizing the core value: channel-based streaming with Ollama models.

**Extract Core Streaming Components:**
```
chanollama/
├── go.mod
├── README.md
├── LICENSE
├── client.go                 # HTTP client for Ollama API
├── engine.go                 # Core chat engine with streaming
├── types.go                  # Message types, Response types
├── streaming.go              # Channel-based streaming logic
├── config.go                 # Configuration options
└── examples/
    ├── 01_basic_chat/           # Simple request-response
    ├── 02_streaming/            # Real-time token streaming  
    ├── 03_system_prompts/       # Using system prompts effectively
    ├── 04_context_aware/        # Maintaining conversation history
    ├── 05_context_summarization/ # Managing context window limits
    ├── 06_concurrent_streaming/ # Multiple concurrent conversations
    ├── 07_error_handling/       # Robust error patterns
    └── 08_performance_tips/     # Optimization techniques
```

**Stay in otdfctl:**
```
pkg/llm/
├── agent_tui.go              # Bubbletea TUI integration
├── agent_manager.go          # Multi-agent system
├── agent_types.go            # Agent definitions  
├── auto_summarize.go         # Conversation summarization
├── handler.go                # CLI integration
├── agents/                   # OpenTDF-specific agent prompts
│   ├── general_agent.md
│   ├── policy_agent.md
│   ├── development_agent.md
│   └── deployment_agent.md
├── ingestion.go              # OpenTDF doc ingestion
└── resource_manager.go       # OpenTDF-specific features
```

**Module Interface Design:**
```go
// Core engine interface
type Engine interface {
    Chat(messages []Message) (*Response, error)
    ChatStream(messages []Message, callback StreamCallback) *Response
    IsHealthy() bool
    Close() error
}

// Streaming callback
type StreamCallback func(token string)

// Message types
type Message struct {
    Role    string `json:"role"`    // "system", "user", "assistant"
    Content string `json:"content"`
}

type Response struct {
    Content string `json:"content"`
    Error   error  `json:"error,omitempty"`
    Tokens  int    `json:"tokens,omitempty"`
}

// Engine configuration (simplified)
type Config struct {
    Model       string        // Required: e.g. "llama2:7b"
    Timeout     time.Duration // Default: 30s
    MaxTokens   int          // Default: 2048
    Temperature float64      // Default: 0.7
}

// Helper patterns for common use cases
type Conversation struct {
    Messages    []Message
    MaxMessages int // Auto-trim when exceeded
    TokenLimit  int // Auto-summarize when exceeded
}

type ContextSummarizer interface {
    SummarizeHistory(messages []Message) (string, error)
}
```

**Example Use Cases:**

### **03_system_prompts/main.go**
```go
// Specialized assistant with system prompt
messages := []chanollama.Message{
    {Role: "system", Content: "You are a helpful coding assistant specialized in Go programming."},
    {Role: "user", Content: "How do I handle errors in Go?"},
}

response := engine.Chat(messages)
```

### **04_context_aware/main.go**  
```go
// Maintaining conversation history
conversation := &chanollama.Conversation{
    MaxMessages: 20, // Keep last 20 messages
    TokenLimit:  4000, // Auto-summarize at 4k tokens
}

// Add messages and maintain context
conversation.AddUserMessage("Hello!")
response := engine.Chat(conversation.Messages)
conversation.AddAssistantMessage(response.Content)
```

### **05_context_summarization/main.go**
```go
// Smart context window management
summarizer := &AutoSummarizer{engine: engine}
conversation := &SmartConversation{
    Summarizer: summarizer,
    TokenLimit: 3000,
}

// Automatically summarizes when approaching limit
response := conversation.Chat("Long question...")
```

### **06_concurrent_streaming/main.go**
```go
// Multiple concurrent conversations
type ChatRoom struct {
    conversations map[string]*Conversation
    engine       *chanollama.Engine
}

// Handle multiple users streaming simultaneously
go room.HandleUser("user1", "What is Go?")
go room.HandleUser("user2", "Explain Docker")
```

### **08_performance_tips/main.go**
```go
// Efficient patterns
// 1. Reuse engine instances
// 2. Pool connections
// 3. Batch similar requests
// 4. Optimize token usage
// 5. Cache common responses
```

**Key Value Propositions Demonstrated:**

1. **🚀 Simple Setup** - Just `chanollama.New("llama2:7b")` and you're streaming
2. **📡 Real-time Streaming** - Channel-based, non-blocking token delivery  
3. **🧠 Context Management** - Built-in conversation history and summarization
4. **🔧 System Prompts** - Easy specialization of AI behavior
5. **⚡ Performance** - Concurrent conversations, efficient token handling
6. **🛡️ Production Ready** - Error handling, timeouts, goroutine safety

**Migration Strategy:**
1. Create new module structure
2. Move core files to new module
3. Update otdfctl to import from new module
4. Test compatibility
5. Publish initial version

**Deliverables:**
- [ ] Create `chanollama` repository structure
- [ ] Extract HTTP client for Ollama API (localhost:11434 only)
- [ ] Extract core chat engine (non-streaming + streaming)
- [ ] Extract channel-based streaming logic with goroutine safety
- [ ] Extract message and response types
- [ ] Create clean, minimal public interface
- [ ] Implement helper types (Conversation, ContextSummarizer)
- [ ] Create comprehensive examples:
  - [ ] Basic chat example
  - [ ] Real-time streaming example
  - [ ] System prompt patterns
  - [ ] Context-aware conversations
  - [ ] Context summarization techniques
  - [ ] Concurrent streaming patterns
  - [ ] Error handling best practices
  - [ ] Performance optimization tips
- [ ] Update otdfctl to import chanollama
- [ ] Write detailed README with usage patterns
- [ ] Test streaming performance and reliability
- [ ] Publish initial v0.1.0 release

### Phase 1: Router System Prompt & Mapping

**Router Prompt Design:**
```go
const ROUTER_SYSTEM_PROMPT = `You are a query classifier for OpenTDF assistance. Respond with ONLY ONE DIGIT:

1 = General questions, introductions, basic concepts, unclear intent
2 = Policy: attributes, namespaces, subject mappings, access control, ABAC
3 = Development: SDKs, APIs, integration, code examples, TDF creation/decryption
4 = Deployment: installation, configuration, infrastructure, Docker, Kubernetes

Examples:
"What is OpenTDF?" → 1
"How do I create attribute namespaces?" → 2
"Show me Python TDF encryption" → 3
"Deploy OpenTDF with Docker" → 4

Respond with ONLY the digit.`
```

**Agent ID Mapping:**
```go
var routerToAgentMap = map[string]AgentType{
    "1": AgentTypeGeneral,
    "2": AgentTypePolicy, 
    "3": AgentTypeDevelopment,
    "4": AgentTypeDeployment,
}
```

**Deliverables:**
- [ ] Define router system prompt
- [ ] Create agent mapping logic
- [ ] Test classification accuracy manually

### Phase 2: New Message Types

```go
// New routing messages
type RoutingStartMsg struct {
    query string
}

type RoutingCompleteMsg struct {
    selectedAgent *Agent
    query         string
    routingToken  string
}

type RoutingFailedMsg struct {
    error error
    query string
}
```

**Deliverables:**
- [ ] Add routing message types to agent_tui.go
- [ ] Update message handling in Update() method
- [ ] Prepare TUI for routing flow

### Phase 3: Core Routing Logic

**New methods in agent_tui.go:**
```go
// Main routing entry point
func (m *AgentChatModel) generateResponseWithRouting(userInput string) tea.Cmd

// Execute router inference
func (m *AgentChatModel) routeQuery(userInput string) tea.Cmd

// Parse router response and select agent
func (m *AgentChatModel) parseRoutingResponse(response string) (*Agent, error)

// Router-specific message construction
func (m *AgentChatModel) buildRouterMessages(userInput string) []ChatMessage
```

**Router execution flow:**
1. User submits query → `RoutingStartMsg`
2. Quick router inference (minimal context)
3. Parse single digit response → agent selection
4. `RoutingCompleteMsg` → proceed with full agent conversation
5. Fallback to current agent on routing failure

**Deliverables:**
- [ ] Implement routing logic methods
- [ ] Create router message construction
- [ ] Add routing response parsing
- [ ] Update generateResponse() to use routing

### Phase 4: TUI Integration

**Update message handling in `Update()` method:**
- `RoutingStartMsg`: Show "🔄 Routing..." status
- `RoutingCompleteMsg`: Switch to selected agent, start normal flow
- `RoutingFailedMsg`: Fallback to current agent with error message

**Status indicators:**
- Brief "Routing to Development Agent..." message
- Seamless transition to agent response
- Optional routing decision visibility for debugging

**Deliverables:**
- [ ] Add routing message handlers to Update()
- [ ] Implement routing status indicators
- [ ] Update UI to show routing progress
- [ ] Test user experience flow

### Phase 5: Fallback & Error Handling

**Fallback scenarios:**
- Invalid routing token → Default to General agent
- Router model failure → Use current agent
- Empty/malformed response → Use current agent
- Routing timeout → Use current agent

**Error handling strategy:**
- Subtle: Don't overwhelm user with routing failures
- Log routing decisions for debugging
- Graceful degradation to manual agent selection

**Deliverables:**
- [ ] Implement fallback logic
- [ ] Add error handling for router failures
- [ ] Create routing decision logging
- [ ] Test failure scenarios

### Phase 6: Agent System Prompt Updates

**Current agents need optimization for routing:**

**General Agent:** 
- Remove development/policy specifics
- Focus on introductions and basic concepts
- Handle unclear/ambiguous queries

**Policy Agent:** 
- Enhance with policy-specific context
- Add ABAC and access control focus
- Include attribute/namespace examples

**Development Agent:** 
- Strengthen with SDK examples and code patterns
- Focus on integration and implementation
- Add TDF creation/decryption specifics

**Deployment Agent:**
- Add infrastructure and ops focus
- Include Docker/Kubernetes examples
- Cover installation and configuration

**Deliverables:**
- [ ] Review and update agent system prompts
- [ ] Test agent specialization effectiveness
- [ ] Ensure no overlap between agent domains
- [ ] Validate routing accuracy with updated prompts

## 🔄 Message Flow

```
User Input → RoutingStartMsg
  ↓
Router Inference (minimal context)
  ↓
Parse Response → Select Agent
  ↓
RoutingCompleteMsg → Set Agent
  ↓
Normal Agent Flow (full context)
  ↓
Streaming Response to User
```

## 🎛️ Configuration

```go
const (
    ROUTER_TIMEOUT = 5 * time.Second    // Router inference timeout
    ROUTER_MAX_TOKENS = 1               // Single digit response
    ROUTER_FALLBACK_AGENT = "1"         // Default to General
)
```

## 📊 Success Criteria

**Functional:**
- Router classifies queries accurately (>80% for clear queries)
- Seamless agent switching without user intervention
- Graceful fallback when routing fails
- Preserved conversation context and history

**Performance:**
- Router inference <200ms for most queries
- Context usage <10% for routing phase
- No memory overhead (same model)
- Maintains current streaming performance

**User Experience:**
- Transparent routing (brief status indicator)
- No disruption to conversation flow
- Optional manual agent override
- Clear feedback when routing fails

## 📁 Files to Create/Modify

### New Module (`chanollama`)
1. **`client.go`** - Extracted HTTP client from pkg/llm/simple_engine.go
2. **`engine.go`** - Extracted core chat functionality
3. **`streaming.go`** - Channel-based streaming logic
4. **`types.go`** - Message and Response types
5. **`config.go`** - Engine configuration

### Updated in otdfctl
1. **`pkg/llm/agent_tui.go`** - Update imports to use chanollama, integrate routing
2. **`pkg/llm/simple_engine.go`** - Becomes thin wrapper around chanollama
3. **`pkg/llm/agents/general_agent.md`** - Updated system prompt for routing
4. **`pkg/llm/agents/policy_agent.md`** - Enhanced policy focus
5. **`pkg/llm/agents/development_agent.md`** - Strengthened dev context
6. **`pkg/llm/agents/deployment_agent.md`** - Added ops focus
7. **`go.mod`** - Add chanollama dependency

## 🧪 Testing Strategy

### Manual Testing
- [ ] Test router prompt classification manually
- [ ] Verify agent selection accuracy
- [ ] Test fallback scenarios
- [ ] Validate user experience flow

### Integration Testing
- [ ] Test with various query types
- [ ] Verify context preservation
- [ ] Test performance characteristics
- [ ] Validate error handling

### Edge Cases
- [ ] Ambiguous queries
- [ ] Multi-domain queries
- [ ] Very short queries
- [ ] Router model failures

## 🚀 Future Enhancements

### Phase 7: Enhanced Features (Future)
- **Learning System**: Track routing accuracy and user corrections
- **Multi-Intent Detection**: Handle queries spanning multiple domains
- **Confidence Scoring**: Router uncertainty handling
- **User Preferences**: Learn individual user patterns
- **Performance Optimization**: Dedicated lightweight router model

### Phase 8: Advanced Routing (Future)
- **Context-Aware Routing**: Use conversation history in routing decisions
- **Dynamic Agent Creation**: Route to specialized sub-agents
- **Query Decomposition**: Break complex queries into multiple routing decisions
- **Federated Routing**: Route to external specialized systems

## 📝 Implementation Notes

### Critical Details
1. **Context Preservation**: Router uses minimal context, agent gets full history
2. **Error Boundaries**: Always have fallback to current agent selection
3. **Performance First**: Router optimized for speed over accuracy
4. **User Control**: Maintain manual agent switching as backup
5. **Debugging**: Log routing decisions for troubleshooting

### Potential Pitfalls
- **Router Prompt Drift**: Classification accuracy degrades over time
- **Context Bleeding**: Router accidentally gets too much context
- **Performance Regression**: Router adds noticeable latency
- **Fallback Loops**: Error handling creates infinite fallback cycles
- **Agent Overlap**: Multiple agents claim same query domain

---

**Implementation Status**: Planning Phase
**Target Completion**: TBD
**Owner**: LLM Integration Team