# LLM Chat Integration - Implementation Plan

## Executive Summary

This implementation plan addresses critical security vulnerabilities in the LLM integration while adding a sophisticated multi-agent system for specialized OpenTDF assistance. The plan is structured in phases to prioritize security fixes while building toward an enhanced user experience.

## Phase 1: Critical Security Fixes (Week 1-2) 🔴

### **Task 1.1: Remove Hardcoded Paths**
**Priority:** Critical  
**Files:** `cmd/llm_ingest.go:24`

**Current Vulnerability:**
```go
// Line 24 - SECURITY RISK
embeddingModelPath = "/Users/ryan/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
```

**Implementation:**
```go
// New: pkg/llm/paths.go
func getEmbeddingModelPath(cfg *config.Config) (string, error) {
    if cfg.LLM.DefaultEmbeddingModelPath != "" {
        return cfg.LLM.DefaultEmbeddingModelPath, nil
    }
    
    // Try standard locations
    locations := []string{
        filepath.Join(os.Getenv("HOME"), ".ollama", "models"),
        "/usr/local/lib/ollama/models",
        "./models",
    }
    
    for _, loc := range locations {
        if _, err := os.Stat(loc); err == nil {
            return loc, nil
        }
    }
    
    return "", fmt.Errorf("no embedding model found in standard locations")
}
```

**Config Changes:**
```yaml
# Add to config.go LLM struct:
DefaultEmbeddingModelPath string `yaml:"default_embedding_model_path" default:""`
```

### **Task 1.2: Input Validation & Sanitization**
**Priority:** Critical  
**Files:** `cmd/llm_ingest_simple.go`, `pkg/llm/ingestion.go`

**Implementation:**
```go
// New: pkg/llm/validation.go
package llm

type FileValidator struct {
    MaxFileSize   int64    // bytes
    MaxFiles      int    
    AllowedExts   []string
    ForbiddenPaths []string
}

func (v *FileValidator) ValidateFile(path string) error {
    // Path traversal protection
    cleanPath := filepath.Clean(path)
    if strings.Contains(cleanPath, "..") {
        return fmt.Errorf("path traversal detected: %s", path)
    }
    
    // Extension validation
    ext := strings.ToLower(filepath.Ext(cleanPath))
    if !v.isAllowedExtension(ext) {
        return fmt.Errorf("forbidden file extension: %s", ext)
    }
    
    // Size validation
    if info, err := os.Stat(cleanPath); err == nil {
        if info.Size() > v.MaxFileSize {
            return fmt.Errorf("file too large: %d bytes", info.Size())
        }
    }
    
    return nil
}
```

### **Task 1.3: Resource Limits**
**Priority:** Critical  
**Files:** `pkg/llm/engine.go`, `pkg/llm/simple_engine.go`

**Implementation:**
```go
// Add to config.go LLM struct:
type LLM struct {
    // ... existing fields
    MaxMemoryMB     int    `yaml:"max_memory_mb" default:"2048"`
    MaxFileSize     int64  `yaml:"max_file_size" default:"10485760"` // 10MB
    MaxFiles        int    `yaml:"max_files" default:"1000"`
    ProcessTimeout  int    `yaml:"process_timeout" default:"300"`   // seconds
}

// Add to engine initialization:
func (sce *SimpleChatEngine) Chat(messages []ChatMessage) SimpleResponse {
    ctx, cancel := context.WithTimeout(context.Background(), 
        time.Duration(sce.config.ProcessTimeout)*time.Second)
    defer cancel()
    
    // Existing implementation with context...
}
```

## Phase 2: Multi-Agent System Foundation (Week 3-4) 🚀

### **Task 2.1: Agent Infrastructure**
**Priority:** High  
**New Files:** `pkg/llm/agents/`

**Agent Manager Architecture:**
```go
// pkg/llm/agents/manager.go
type AgentManager struct {
    agents          map[string]*Agent
    currentAgent    *Agent
    conversationCtx *ConversationContext
    chatEngine      *SimpleChatEngine
}

type Agent struct {
    ID           string
    Name         string
    Description  string
    SystemPrompt string
    Metadata     AgentMetadata
}

type AgentMetadata struct {
    Specialties     []string
    RedirectRules   []RedirectRule
    ContextLimits   ContextLimits
    Keywords        []string
}

type ConversationContext struct {
    Messages    []ChatMessage
    AgentID     string
    TokenCount  int
    MaxTokens   int
}
```

### **Task 2.2: Agent Discovery & Loading**
**Priority:** High  
**Files:** `pkg/llm/agents/loader.go`

**Implementation:**
```go
func LoadAgents(agentsDir string) (map[string]*Agent, error) {
    agents := make(map[string]*Agent)
    
    files, err := filepath.Glob(filepath.Join(agentsDir, "*.md"))
    if err != nil {
        return nil, err
    }
    
    for _, file := range files {
        agent, err := parseAgentFile(file)
        if err != nil {
            log.Printf("Warning: failed to load agent %s: %v", file, err)
            continue
        }
        agents[agent.ID] = agent
    }
    
    return agents, nil
}

func parseAgentFile(path string) (*Agent, error) {
    content, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }
    
    // Parse YAML frontmatter
    var metadata AgentMetadata
    frontmatter, body, err := extractFrontmatter(content)
    if err != nil {
        return nil, err
    }
    
    if err := yaml.Unmarshal(frontmatter, &metadata); err != nil {
        return nil, err
    }
    
    return &Agent{
        ID:           strings.TrimSuffix(filepath.Base(path), ".md"),
        Name:         metadata.AgentName,
        Description:  metadata.Description,
        SystemPrompt: string(body),
        Metadata:     metadata,
    }, nil
}
```

### **Task 2.3: Enhanced Agent Metadata**
**Priority:** Medium  
**Files:** `pkg/llm/agents/*.md`

**Enhanced YAML Structure:**
```yaml
---
agent_name: Development Agent
description: >
  An OpenTDF development agent specializing in helping developers integrate 
  OpenTDF into their applications using various programming languages and SDKs.
  
version: "1.0"
specialties:
  - "java-sdk"
  - "go-sdk" 
  - "javascript-sdk"
  - "integration-patterns"
  - "code-examples"

context_limits:
  max_tokens: 2048
  max_messages: 20
  
redirect_rules:
  - pattern: "install|setup|deploy"
    suggest_agent: "deployment_agent"
    message: "For installation and deployment questions, I recommend switching to the Deployment Agent who specializes in setup procedures."
    
  - pattern: "policy|attribute|access control"
    suggest_agent: "policy_agent"
    message: "For policy design questions, the Policy Agent would be more helpful."

keywords:
  - "SDK"
  - "integration"
  - "code"
  - "development"
  - "programming"
---
```

## Phase 3: Bubbletea Interface (Week 5-6) 🎨

### **Task 3.1: Agent Selector Interface**
**Priority:** High  
**Files:** `pkg/llm/tui/agent_selector.go`

**Implementation:**
```go
type AgentSelector struct {
    list        list.Model
    agents      map[string]*Agent
    selected    *Agent
    manager     *AgentManager
}

func NewAgentSelector(manager *AgentManager) *AgentSelector {
    agents := manager.GetAvailableAgents()
    
    items := make([]list.Item, 0, len(agents))
    for _, agent := range agents {
        items = append(items, AgentItem{
            id:          agent.ID,
            name:        agent.Name,
            description: agent.Description,
        })
    }
    
    l := list.New(items, list.NewDefaultDelegate(), 0, 0)
    l.Title = "🤖 Select an OpenTDF Agent"
    l.SetShowStatusBar(false)
    l.SetFilteringEnabled(true)
    
    return &AgentSelector{
        list:    l,
        agents:  agents,
        manager: manager,
    }
}

func (m *AgentSelector) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.KeyMsg:
        switch msg.String() {
        case "enter":
            selected := m.list.SelectedItem().(AgentItem)
            m.selected = m.agents[selected.id]
            return NewChatInterface(m.manager, m.selected), nil
        case "q", "ctrl+c":
            return m, tea.Quit
        }
    }
    
    var cmd tea.Cmd
    m.list, cmd = m.list.Update(msg)
    return m, cmd
}
```

### **Task 3.2: Chat Interface with Agent Switching**
**Priority:** High  
**Files:** `pkg/llm/tui/chat_interface.go`

**Implementation:**
```go
type ChatInterface struct {
    manager      *AgentManager
    currentAgent *Agent
    messages     []ChatMessage
    input        textinput.Model
    viewport     viewport.Model
    showAgents   bool
    width        int
    height       int
}

func NewChatInterface(manager *AgentManager, agent *Agent) *ChatInterface {
    ti := textinput.New()
    ti.Placeholder = "Ask me anything about OpenTDF..."
    ti.Focus()
    ti.CharLimit = 500
    ti.Width = 80
    
    vp := viewport.New(80, 20)
    vp.SetContent(fmt.Sprintf("🤖 **%s** is ready to help!\n\n%s\n\nType your question below. Press Ctrl+A to switch agents, Ctrl+C to quit.\n", 
        agent.Name, agent.Description))
    
    return &ChatInterface{
        manager:      manager,
        currentAgent: agent,
        messages:     []ChatMessage{},
        input:        ti,
        viewport:     vp,
        showAgents:   false,
    }
}

func (m *ChatInterface) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        m.width = msg.Width
        m.height = msg.Height
        m.viewport.Width = msg.Width
        m.viewport.Height = msg.Height - 4 // Leave space for input
        m.input.Width = msg.Width - 2
        return m, nil
        
    case tea.KeyMsg:
        switch msg.String() {
        case "ctrl+a": // Switch agents
            return NewAgentSelector(m.manager), nil
        case "ctrl+c":
            return m, tea.Quit
        case "enter":
            return m.handleUserInput()
        }
    }
    
    var cmd tea.Cmd
    m.input, cmd = m.input.Update(msg)
    return m, cmd
}

func (m *ChatInterface) handleUserInput() (tea.Model, tea.Cmd) {
    userInput := m.input.Value()
    if userInput == "" {
        return m, nil
    }
    
    m.input.SetValue("")
    
    // Add user message
    userMsg := ChatMessage{Role: "user", Content: userInput}
    m.messages = append(m.messages, userMsg)
    
    // Check if agent should redirect
    if shouldRedirect, suggestedAgent := m.checkRedirectNeeds(userInput); shouldRedirect {
        return m.suggestAgentSwitch(suggestedAgent)
    }
    
    // Build messages with current agent's system prompt
    agentMessages := []ChatMessage{
        {Role: "system", Content: m.currentAgent.SystemPrompt},
    }
    agentMessages = append(agentMessages, m.messages...)
    
    // Get response from current agent
    response := m.manager.chatEngine.Chat(agentMessages)
    if response.Error != nil {
        m.addToConversation("❌ Error: " + response.Error.Error())
    } else {
        m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: response.Content})
        m.addToConversation("🤖 " + response.Content)
    }
    
    return m, nil
}
```

### **Task 3.3: Command Integration**
**Priority:** Medium  
**Files:** `cmd/llm.go`

**Enhanced Command Structure:**
```go
var llmAgentCmd = man.Docs.GetCommand("llm/agent", man.WithRun(func(cmd *cobra.Command, args []string) {
    c := cli.New(cmd, args)
    
    // Initialize agent manager
    agentManager, err := llm.NewAgentManager(c.Config)
    if err != nil {
        c.ExitWithError("Failed to initialize agent manager", err)
    }
    
    // Check for direct agent selection
    if agentFlag := c.Flags.GetOptionalString("agent"); agentFlag != "" {
        agent, exists := agentManager.GetAgent(agentFlag)
        if !exists {
            c.ExitWithError("Unknown agent", fmt.Errorf("agent '%s' not found", agentFlag))
        }
        
        // Start directly with selected agent
        chatInterface := tui.NewChatInterface(agentManager, agent)
        program := tea.NewProgram(chatInterface, tea.WithAltScreen())
        
        if _, err := program.Run(); err != nil {
            c.ExitWithError("Failed to start chat interface", err)
        }
        return
    }
    
    // Start with agent selector
    agentSelector := tui.NewAgentSelector(agentManager)
    program := tea.NewProgram(agentSelector, tea.WithAltScreen())
    
    if _, err := program.Run(); err != nil {
        c.ExitWithError("Failed to start agent selector", err)
    }
}))

func init() {
    llmAgentCmd.Flags().String("agent", "", "Direct agent selection (development, policy, general, deployment)")
    llmAgentCmd.Flags().String("model", "", "Path to LLM model file")
    llmAgentCmd.Flags().Bool("rag", false, "Enable RAG support")
    
    llmCmd.AddCommand(&llmAgentCmd.Command)
}
```

### **Task 3.4: Context Management & Usage Display**
**Priority:** High  
**Files:** `pkg/llm/context/`, `pkg/llm/tui/chat_interface.go`

**Context Window Detection:**
```go
// pkg/llm/context/detector.go
type ContextDetector struct {
    model *llama.Model
}

func (cd *ContextDetector) DetectContextSize() (int, error) {
    // Strategy 1: Try to read from model metadata (GGUF)
    if size := cd.readModelMetadata(); size > 0 {
        return size, nil
    }
    
    // Strategy 2: Infer from model filename patterns
    // Examples: "model-4k.gguf", "llama-2-7b-chat-8k.gguf"
    if size := cd.inferFromFilename(); size > 0 {
        return size, nil
    }
    
    // Strategy 3: Use conservative defaults by model type
    return cd.getDefaultByType(), nil // 4096 default
}
```

**Real-time Context Tracking:**
```go
// pkg/llm/context/tracker.go
type ContextTracker struct {
    maxTokens     int
    currentTokens int
    messages      []ChatMessage
    model         *llama.Model
}

type ContextUsage struct {
    Current    int
    Maximum    int
    Percentage float64
    Status     ContextStatus
}

func (ct *ContextTracker) GetUsage() ContextUsage {
    pct := float64(ct.currentTokens) / float64(ct.maxTokens) * 100
    status := StatusOK
    switch {
    case pct >= 90: status = StatusCritical  // Red - Auto-summarize
    case pct >= 75: status = StatusWarning   // Yellow - Warn user
    case pct >= 50: status = StatusCaution   // Orange - Show usage
    }
    
    return ContextUsage{
        Current: ct.currentTokens,
        Maximum: ct.maxTokens,
        Percentage: pct,
        Status: status,
    }
}
```

**Enhanced TUI Header with Context Display:**
```go
func (m *ChatInterface) renderContextHeader() string {
    usage := m.contextTracker.GetUsage()
    
    // Visual progress bar (Claude Code style)
    barWidth := 40
    filledWidth := int(float64(barWidth) * usage.Percentage / 100)
    
    var barColor lipgloss.Color
    var statusIcon string
    switch usage.Status {
    case StatusOK:        barColor, statusIcon = "#00FF00", "✅"
    case StatusCaution:   barColor, statusIcon = "#FFA500", "⚠️"
    case StatusWarning:   barColor, statusIcon = "#FFFF00", "⚠️"
    case StatusCritical:  barColor, statusIcon = "#FF0000", "🔴"
    }
    
    bar := strings.Repeat("█", filledWidth) + strings.Repeat("░", barWidth-filledWidth)
    
    contextInfo := fmt.Sprintf("%s Context: %d/%d tokens (%.1f%%) [%s]", 
        statusIcon, usage.Current, usage.Maximum, usage.Percentage, bar)
    
    agentInfo := fmt.Sprintf("🤖 %s | Ctrl+A: Switch | /context: Info | /clear: Reset", 
        m.currentAgent.Name)
    
    return lipgloss.JoinVertical(lipgloss.Left,
        lipgloss.NewStyle().Foreground(barColor).Bold(true).Render(contextInfo),
        lipgloss.NewStyle().Faint(true).Render(agentInfo),
        strings.Repeat("─", m.width),
    )
}
```

**Enhanced Commands:**
```go
// Additional commands in chat interface
func (m *ChatInterface) handleCommand(command string) (tea.Model, tea.Cmd) {
    switch command {
    case "context":
        usage := m.contextTracker.GetUsage()
        info := fmt.Sprintf("📊 **Context Usage**\n"+
            "Current: %d tokens\n"+
            "Maximum: %d tokens\n"+
            "Usage: %.1f%%\n"+
            "Model: %s", 
            usage.Current, usage.Maximum, usage.Percentage, m.modelPath)
        m.addSystemMessage(info)
        
    case "clear":
        m.messages = m.messages[:1] // Keep system message
        m.contextTracker.Reset()
        m.addSystemMessage("📝 Chat history cleared.")
        
    case "summarize":
        if usage := m.contextTracker.GetUsage(); usage.Percentage < 50 {
            m.addSystemMessage("💡 Context usage is low (%.1f%%). Summarization not needed.", usage.Percentage)
        } else {
            return m.triggerSummarization()
        }
    }
    return m, nil
}
```

## Phase 4: Advanced Features (Week 7-8) 🔧

### **Task 4.1: Auto-Summarization System**
**Priority:** High  
**Files:** `pkg/llm/context/summarizer.go`

**Implementation:**
```go
type ContextSummarizer struct {
    engine    *SimpleChatEngine
    threshold float64 // Trigger at 85% capacity
}

func (cs *ContextSummarizer) ShouldSummarize(usage ContextUsage) bool {
    return usage.Percentage >= cs.threshold
}

func (cs *ContextSummarizer) SummarizeConversation(messages []ChatMessage) ([]ChatMessage, error) {
    // Smart summarization strategy:
    // 1. Always keep system prompt
    // 2. Keep last 3-5 exchanges (recent context)
    // 3. Summarize middle conversation
    // 4. Preserve agent-specific important context
    
    systemMsg := messages[0]
    if len(messages) <= 11 { // Minimal conversation
        return messages, nil
    }
    
    // Identify conversation segments
    middleStart := 1
    recentStart := len(messages) - 8 // Keep last 4 exchanges
    middleMessages := messages[middleStart:recentStart]
    recentMessages := messages[recentStart:]
    
    // Generate intelligent summary
    summary, err := cs.generateAgentAwareSummary(middleMessages)
    if err != nil {
        return messages, err // Graceful fallback
    }
    
    // Construct optimized conversation
    return []ChatMessage{
        systemMsg,
        {Role: "assistant", Content: fmt.Sprintf("📋 **Context Summary**: %s\n\n*Continuing conversation...*", summary)},
        append(recentMessages...)...,
    }, nil
}

func (cs *ContextSummarizer) generateAgentAwareSummary(messages []ChatMessage) (string, error) {
    // Use specialized prompts based on conversation type
    summaryMessages := []ChatMessage{
        {Role: "system", Content: cs.getAgentSpecificSummaryPrompt()},
        {Role: "user", Content: cs.buildConversationForSummary(messages)},
    }
    
    response := cs.engine.Chat(summaryMessages)
    return response.Content, response.Error
}

func (cs *ContextSummarizer) getAgentSpecificSummaryPrompt() string {
    return `You are an intelligent conversation summarizer for OpenTDF technical discussions. 

Create a concise summary that preserves:
- Key technical concepts discussed
- Code examples or configurations mentioned  
- Specific OpenTDF components (KAS, policies, etc.)
- User's current problem or goal
- Important decisions or solutions reached

Format as a brief technical summary suitable for continuing the conversation.`
}
```

### **Task 4.2: Smart Redirect System**
**Priority:** Medium  
**Files:** `pkg/llm/agents/redirect.go`

**Implementation:**
```go
type RedirectAnalyzer struct {
    currentAgent *Agent
}

func (ra *RedirectAnalyzer) AnalyzeQuery(query string) (shouldRedirect bool, suggestedAgent string, confidence float64) {
    query = strings.ToLower(query)
    
    for _, rule := range ra.currentAgent.Metadata.RedirectRules {
        if matched, _ := regexp.MatchString(rule.Pattern, query); matched {
            return true, rule.SuggestAgent, 0.8
        }
    }
    
    // Keyword-based analysis
    for agentID, agent := range ra.availableAgents {
        if agentID == ra.currentAgent.ID {
            continue
        }
        
        score := ra.calculateKeywordScore(query, agent.Metadata.Keywords)
        if score > 0.6 {
            return true, agentID, score
        }
    }
    
    return false, "", 0.0
}

func (ra *RedirectAnalyzer) calculateKeywordScore(query string, keywords []string) float64 {
    matches := 0
    for _, keyword := range keywords {
        if strings.Contains(query, strings.ToLower(keyword)) {
            matches++
        }
    }
    return float64(matches) / float64(len(keywords))
}
```

### **Task 4.2: Cross-Agent Memory**
**Priority:** Low  
**Files:** `pkg/llm/agents/memory.go`

**Implementation:**
```go
type CrossAgentMemory struct {
    SharedContext map[string]interface{}
    UserProfile   UserProfile
    SessionData   SessionData
}

type UserProfile struct {
    CurrentProject string
    Preferences    map[string]string
    ExpertiseLevel string
}

func (cam *CrossAgentMemory) BuildTransferContext(fromAgent, toAgent string, lastQuery string) string {
    context := fmt.Sprintf("Previous conversation with %s about: %s\n", fromAgent, lastQuery)
    
    if cam.UserProfile.CurrentProject != "" {
        context += fmt.Sprintf("User is working on project: %s\n", cam.UserProfile.CurrentProject)
    }
    
    if expertise := cam.UserProfile.ExpertiseLevel; expertise != "" {
        context += fmt.Sprintf("User expertise level: %s\n", expertise)
    }
    
    return context
}
```

### **Task 4.3: Configuration Enhancement**
**Priority:** Medium  
**Files:** `pkg/config/config.go`, `otdfctl-example.yaml`

**Enhanced Configuration:**
```yaml
llm:
  default_model_path: "/path/to/model.gguf"
  default_embedding_model_path: "/path/to/embeddings.gguf"
  context_size: 4096
  temperature: 0.7
  stream: true
  
  # Security settings
  max_memory_mb: 2048
  max_file_size: 10485760  # 10MB
  max_files: 1000
  process_timeout: 300     # 5 minutes
  
  # Agent system
  agents:
    enabled: true
    directory: "~/.otdfctl/agents"
    default_agent: "general"
    context_management:
      max_tokens_per_agent: 2048
      max_total_tokens: 8192
      auto_switch_threshold: 0.7
      
  # Context management
  context:
    auto_detect_size: true          # Try to detect model context window
    fallback_size: 4096            # Default if detection fails
    usage_display: true            # Show context usage in TUI
    auto_summarize: true           # Enable auto-summarization
    summarize_threshold: 0.85      # Trigger at 85% usage
    preserve_recent_messages: 8    # Keep last N messages during summarization
    warning_threshold: 0.75        # Show warning at 75% usage
    
  # Agent tools (API integration)
  tools:
    enabled: true                  # Enable API tool capabilities
    auto_approve: false            # Require user confirmation for API calls
    policy_agent:
      create_operations: true      # Allow creating attributes, namespaces, etc.
      read_operations: true        # Allow listing and getting resources
      update_operations: false     # Disable updates for safety
      delete_operations: false     # Disable deletions for safety
    
  # RAG settings
  rag:
    enabled: false
    index_path: "~/.otdfctl/rag_index.json"
    max_documents: 1000
    similarity_threshold: 0.3
```

## Phase 5: API-Enabled Policy Agent (Week 9-10) 🔧

### **Task 5.1: Policy Agent Tool System Foundation**
**Priority:** High  
**Files:** `pkg/llm/agents/tools/`, `pkg/llm/agents/policy_agent.go`

**Design Principles:**
- **Leverage existing handlers**: Reuse proven SDK integration patterns
- **Security-first**: User confirmation required for API calls
- **Progressive permissions**: Start with safe operations (create/read only)
- **Clean tool interface**: Designed for future MCP compatibility

**Tool Architecture:**
```go
// pkg/llm/agents/tools/interface.go
type Tool interface {
    Name() string
    Description() string
    Parameters() ToolParameters
    Execute(ctx context.Context, params map[string]interface{}) (ToolResult, error)
}

type ToolParameters struct {
    Type       string                 `json:"type"`
    Properties map[string]interface{} `json:"properties"`
    Required   []string              `json:"required"`
}

type ToolResult struct {
    Success bool                   `json:"success"`
    Data    interface{}           `json:"data,omitempty"`
    Error   string                `json:"error,omitempty"`
    Message string                `json:"message,omitempty"`
}

// pkg/llm/agents/tools/policy_tools.go
type PolicyToolSet struct {
    handler handlers.Handler
}

func NewPolicyToolSet(handler handlers.Handler) *PolicyToolSet {
    return &PolicyToolSet{handler: handler}
}

func (pts *PolicyToolSet) GetTools() []Tool {
    return []Tool{
        // Phase 5a: Read-only tools (safe implementation first)
        &ListAttributesTool{handler: pts.handler},
        &GetAttributeTool{handler: pts.handler},
        &ListNamespacesTool{handler: pts.handler},
        &GetNamespaceTool{handler: pts.handler},
        
        // Phase 5b: Create operations (with user confirmation)
        &CreateAttributeTool{handler: pts.handler},
        &CreateNamespaceTool{handler: pts.handler},
        &CreateSubjectMappingTool{handler: pts.handler},
        
        // Future phases: Update/delete operations (disabled by default)
        // &UpdateAttributeTool{handler: pts.handler, enabled: false},
        // &DeleteAttributeTool{handler: pts.handler, enabled: false},
    }
}
```

**Create Attribute Tool:**
```go
// pkg/llm/agents/tools/create_attribute.go
type CreateAttributeTool struct {
    handler handlers.Handler
}

func (cat *CreateAttributeTool) Name() string {
    return "create_attribute"
}

func (cat *CreateAttributeTool) Description() string {
    return "Create a new OpenTDF attribute with specified name, namespace, rule, and values"
}

func (cat *CreateAttributeTool) Parameters() ToolParameters {
    return ToolParameters{
        Type: "object",
        Properties: map[string]interface{}{
            "namespace": map[string]interface{}{
                "type":        "string",
                "description": "Namespace for the attribute (e.g., 'https://example.com/attr')",
            },
            "name": map[string]interface{}{
                "type":        "string", 
                "description": "Name of the attribute (e.g., 'classification')",
            },
            "rule": map[string]interface{}{
                "type":        "string",
                "enum":        []string{"ALL_OF", "ANY_OF", "HIERARCHY"},
                "description": "Attribute rule type",
            },
            "values": map[string]interface{}{
                "type":        "array",
                "items":       map[string]interface{}{"type": "string"},
                "description": "Array of attribute values (e.g., ['public', 'internal', 'secret'])",
            },
        },
        Required: []string{"namespace", "name", "rule", "values"},
    }
}

func (cat *CreateAttributeTool) Execute(ctx context.Context, params map[string]interface{}) (ToolResult, error) {
    namespace := params["namespace"].(string)
    name := params["name"].(string)
    rule := params["rule"].(string)
    values := params["values"].([]interface{})
    
    // Convert values to strings
    valueStrings := make([]string, len(values))
    for i, v := range values {
        valueStrings[i] = v.(string)
    }
    
    // Validate parameters against existing handler patterns
    if err := cat.validateParameters(namespace, name, rule, valueStrings); err != nil {
        return ToolResult{
            Success: false,
            Error:   err.Error(),
            Message: "Parameter validation failed",
        }, nil
    }
    
    // Use existing handler - leverages authentication, validation, error handling
    attribute, err := cat.handler.CreateAttribute(ctx, namespace, name, rule, valueStrings, nil)
    if err != nil {
        return ToolResult{
            Success: false,
            Error:   err.Error(),
            Message: fmt.Sprintf("Failed to create attribute %s.%s", namespace, name),
        }, nil
    }
    
    return ToolResult{
        Success: true,
        Data:    attribute,
        Message: fmt.Sprintf("✅ Created attribute: %s.%s with %d values", namespace, name, len(valueStrings)),
    }, nil
}

func (cat *CreateAttributeTool) validateParameters(namespace, name, rule string, values []string) error {
    // Reuse existing validation patterns from handlers
    if namespace == "" || name == "" {
        return fmt.Errorf("namespace and name are required")
    }
    
    validRules := []string{"ALL_OF", "ANY_OF", "HIERARCHY"}
    if !contains(validRules, rule) {
        return fmt.Errorf("rule must be one of: %s", strings.Join(validRules, ", "))
    }
    
    if len(values) == 0 {
        return fmt.Errorf("at least one value is required")
    }
    
    return nil
}
```

**Read-Only Tools (Phase 5a - Week 9):**
```go
// pkg/llm/agents/tools/list_attributes.go
type ListAttributesTool struct {
    handler handlers.Handler
}

func (lat *ListAttributesTool) Execute(ctx context.Context, params map[string]interface{}) (ToolResult, error) {
    // Optional namespace filter
    namespace := ""
    if ns, ok := params["namespace"]; ok {
        namespace = ns.(string)
    }
    
    // Use existing handler for listing
    attributes, err := lat.handler.ListAttributes(ctx, namespace)
    if err != nil {
        return ToolResult{Success: false, Error: err.Error()}, nil
    }
    
    // Format for agent consumption
    summary := fmt.Sprintf("Found %d attributes", len(attributes))
    if namespace != "" {
        summary += fmt.Sprintf(" in namespace %s", namespace)
    }
    
    return ToolResult{
        Success: true,
        Data:    attributes,
        Message: summary,
    }, nil
}
```

### **Task 5.2: Enhanced Policy Agent with Tool Integration**
**Priority:** High  
**Files:** `pkg/llm/agents/policy_agent_enhanced.go`

**Agent Integration Strategy:**
- **Phase 5a**: Read-only tools for safe exploration
- **Phase 5b**: Create operations with user confirmation
- **Authentication**: Leverage existing profile/handler system
- **Error handling**: Consistent with CLI patterns

**Tool-Enabled Agent:**
```go
type EnhancedPolicyAgent struct {
    *Agent
    toolSet     *PolicyToolSet
    handler     handlers.Handler
    interactive bool
}

func NewEnhancedPolicyAgent(handler handlers.Handler, config *config.Config) *EnhancedPolicyAgent {
    agent := &EnhancedPolicyAgent{
        Agent:   loadPolicyAgent(),
        toolSet: NewPolicyToolSet(handler),
        handler: handler,
        interactive: config.LLM.Tools.PolicyAgent.CreateOperations, // Configurable tool access
    }
    
    // Update system prompt to include tool capabilities
    agent.SystemPrompt = agent.buildToolAwareSystemPrompt()
    
    return agent
}

func (epa *EnhancedPolicyAgent) ProcessMessage(ctx context.Context, message string) (string, error) {
    // Enhanced system prompt with tool capabilities
    systemPrompt := epa.buildToolAwareSystemPrompt()
    
    messages := []ChatMessage{
        {Role: "system", Content: systemPrompt},
        {Role: "user", Content: message},
    }
    
    response := epa.chatEngine.Chat(messages)
    if response.Error != nil {
        return "", response.Error
    }
    
    // Check if response contains tool calls
    if toolCalls := epa.extractToolCalls(response.Content); len(toolCalls) > 0 {
        return epa.executeToolCalls(ctx, toolCalls)
    }
    
    return response.Content, nil
}

func (epa *EnhancedPolicyAgent) buildToolAwareSystemPrompt() string {
    basePrompt := epa.Agent.SystemPrompt
    
    toolDescriptions := ""
    for _, tool := range epa.toolSet.GetTools() {
        toolDescriptions += fmt.Sprintf("\n- **%s**: %s", tool.Name(), tool.Description())
    }
    
    return fmt.Sprintf(`%s

## Available Tools
You can perform actions on the OpenTDF Platform using these tools:%s

When you want to use a tool, format your response like this:
```tool_call
{
  "tool": "create_attribute",
  "parameters": {
    "namespace": "https://example.com/attr",
    "name": "classification", 
    "rule": "HIERARCHY",
    "values": ["public", "internal", "secret"]
  }
}
```

**Usage Guidelines:**
- Always explain what you're going to do before using a tool
- Use read-only tools (list_*, get_*) freely to explore the current state
- For create operations, ask for confirmation and explain the impact
- Provide clear summaries after tool execution
- If a tool fails, suggest alternative approaches

**Current Profile:** Connected to %s`, basePrompt, toolDescriptions, epa.handler.GetEndpoint())
}
```

### **Task 5.3: Interactive Tool Confirmation**
**Priority:** Medium  
**Files:** `pkg/llm/tui/tool_confirmation.go`

**User Consent System:**
```go
// Enhanced chat interface with tool confirmation
func (m *ChatInterface) handleToolCall(toolCall ToolCall) (tea.Model, tea.Cmd) {
    if m.autoApprove {
        return m.executeToolCall(toolCall)
    }
    
    // Show confirmation dialog
    confirmDialog := NewToolConfirmationDialog(toolCall)
    return confirmDialog, nil
}

type ToolConfirmationDialog struct {
    toolCall    ToolCall
    confirmed   bool
    cancelled   bool
    width       int
    height      int
}

func (tcd *ToolConfirmationDialog) View() string {
    style := lipgloss.NewStyle().
        Border(lipgloss.RoundedBorder()).
        BorderForeground(lipgloss.Color("#FFA500")).
        Padding(1).
        Width(60)
    
    // Different colors for different operation types
    var borderColor lipgloss.Color
    switch {
    case strings.HasPrefix(tcd.toolCall.Tool, "create_"):
        borderColor = lipgloss.Color("#FF6B35") // Orange for creates
    case strings.HasPrefix(tcd.toolCall.Tool, "update_"):
        borderColor = lipgloss.Color("#F7931E") // Yellow for updates  
    case strings.HasPrefix(tcd.toolCall.Tool, "delete_"):
        borderColor = lipgloss.Color("#FF0000") // Red for deletes
    default:
        borderColor = lipgloss.Color("#00FF00") // Green for reads
    }
    
    style = style.BorderForeground(borderColor)
    
    content := fmt.Sprintf(`🔧 **Tool Call Requested**

**Tool:** %s
**Action:** %s
**Profile:** %s
**Endpoint:** %s

**Parameters:**
%s

Do you want to execute this action?

[Y] Yes, execute   [N] No, cancel   [A] Auto-approve all   [?] Help`, 
        tcd.toolCall.Tool,
        tcd.toolCall.Description,
        tcd.profileName,
        tcd.endpoint,
        tcd.formatParameters())
    
    return lipgloss.Place(tcd.width, tcd.height, lipgloss.Center, lipgloss.Center, style.Render(content))
}
```

### **Task 5.4: Tool Result Integration**
**Priority:** High  
**Files:** `pkg/llm/agents/result_formatter.go`

**Result Presentation:**
```go
func (epa *EnhancedPolicyAgent) formatToolResult(result ToolResult) string {
    if !result.Success {
        return fmt.Sprintf("❌ **Tool Error**: %s\n\n%s", result.Error, result.Message)
    }
    
    switch result.Data.(type) {
    case *policy.Attribute:
        attr := result.Data.(*policy.Attribute)
        return fmt.Sprintf(`✅ **Attribute Created Successfully**

**FQN:** %s
**Rule:** %s  
**Values:** %s
**ID:** %s

%s

💡 **Next Steps:**
- Use this attribute in TDF policies
- Map subjects to these values: \`otdfctl policy subject-mappings create\`
- View in TUI: \`otdfctl policy attributes get %s\``, 
        attr.GetFqn(), attr.GetRule(), 
        strings.Join(extractValues(attr.GetValues()), ", "),
        attr.GetId(), result.Message, attr.GetId())
        
    case []*policy.Attribute:
        attrs := result.Data.([]*policy.Attribute)
        summary := fmt.Sprintf("📋 **Found %d attributes:**\n\n", len(attrs))
        for _, attr := range attrs {
            summary += fmt.Sprintf("- **%s** (%s) - %d values\n", 
                attr.GetFqn(), attr.GetRule(), len(attr.GetValues()))
        }
        return summary + "\n" + result.Message
        
    default:
        return fmt.Sprintf("✅ **Success**: %s\n\nResult: %+v", result.Message, result.Data)
    }
}
```

## Phase 6: Testing & Security Validation (Week 11-12) 🧪

### **Task 5.5: User Experience Flow**
**Priority:** High  
**Files:** `docs/user-experience-examples.md`

**Example Conversation Flow:**
```
User: "I need to set up data classification for my company"

Agent: "I'll help you create a data classification system. Let me first check what 
       attributes already exist, then create a classification attribute.

🔍 Checking existing attributes...

📋 Found 3 attributes:
- https://opentdf.io/attr/departments (ALL_OF) - 5 values
- https://opentdf.io/attr/clearance (HIERARCHY) - 3 values  
- https://opentdf.io/attr/projects (ANY_OF) - 12 values

I don't see a classification attribute yet. I recommend creating one with 
hierarchical values. Here's what I'll create:

🔧 Tool Call: create_attribute
- Namespace: https://yourcompany.com/attr
- Name: classification
- Rule: HIERARCHY  
- Values: [public, internal, confidential, secret]

[Confirmation dialog shows]

✅ Attribute Created Successfully
FQN: https://yourcompany.com/attr/classification
Rule: HIERARCHY
Values: public, internal, confidential, secret
ID: attr-7f8e9d2c

💡 Next Steps:
- Use this attribute in TDF policies
- Map subjects to these values: `otdfctl policy subject-mappings create`
- View in TUI: `otdfctl policy attributes get attr-7f8e9d2c`

Your classification system is ready! Would you like me to help you create 
subject mappings for different user groups?"
```

### **Task 5.6: Configuration Integration**
**Priority:** Medium  
**Files:** `pkg/config/config.go`

**Enhanced Configuration Structure:**
```go
type LLMTools struct {
    Enabled     bool                 `yaml:"enabled" default:"true"`
    AutoApprove bool                 `yaml:"auto_approve" default:"false"`
    PolicyAgent LLMPolicyAgentTools  `yaml:"policy_agent"`
}

type LLMPolicyAgentTools struct {
    CreateOperations bool `yaml:"create_operations" default:"true"`
    ReadOperations   bool `yaml:"read_operations" default:"true"`
    UpdateOperations bool `yaml:"update_operations" default:"false"`
    DeleteOperations bool `yaml:"delete_operations" default:"false"`
}
```

## Phase 6: Testing & Security Validation (Week 11-12) 🧪

### **Task 6.1: Tool Security Test Suite**
**Priority:** Critical  
**Files:** `pkg/llm/tools/security_test.go`

**Tool-Specific Security Tests:**
```go
func TestToolAuthentication(t *testing.T) {
    // Test that tools respect profile authentication
    // Test that tools fail with invalid credentials
    // Test that tools use correct endpoint from profile
}

func TestToolAuthorization(t *testing.T) {
    // Test that disabled operations are blocked
    // Test that configuration controls tool availability
    // Test that user confirmation is required for create operations
}

func TestToolParameterValidation(t *testing.T) {
    // Test injection prevention in tool parameters
    // Test that malformed parameters are rejected
    // Test that tools validate against OpenTDF constraints
}
```

### **Task 6.2: Integration Test Suite**
**Priority:** High  
**Files:** `pkg/llm/integration_test.go`

**End-to-End Tool Workflows:**
```go
func TestPolicyAgentWorkflow(t *testing.T) {
    // Test complete workflow: list -> create -> verify
    // Test error handling when platform is unavailable
    // Test tool confirmation dialog interactions
}

func TestMultiAgentToolHandoff(t *testing.T) {
    // Test switching from Policy Agent to Development Agent
    // Verify context preservation across agent switches
    // Test tool availability per agent type
}
```

### **Task 6.3: Security Test Suite**
**Priority:** Critical  
**Files:** `pkg/llm/security_test.go`

**Implementation:**
```go
func TestPathTraversalPrevention(t *testing.T) {
    validator := &FileValidator{
        MaxFileSize: 1024 * 1024,
        AllowedExts: []string{".md", ".txt"},
    }
    
    tests := []struct {
        input    string
        expected error
    }{
        {"../../../etc/passwd", ErrPathTraversal},
        {"..\\windows\\system32", ErrPathTraversal},
        {"normal/file.md", nil},
        {"./docs/readme.md", nil},
    }
    
    for _, test := range tests {
        err := validator.ValidateFile(test.input)
        if test.expected == nil {
            assert.NoError(t, err, "Expected no error for %s", test.input)
        } else {
            assert.Error(t, err, "Expected error for %s", test.input)
        }
    }
}

func TestResourceLimits(t *testing.T) {
    // Test file size limits
    // Test memory limits  
    // Test timeout handling
}

func TestAgentSwitching(t *testing.T) {
    // Test agent loading
    // Test redirect logic
    // Test context preservation
}
```

### **Task 5.2: Integration Testing**
**Priority:** High  
**Files:** `pkg/llm/integration_test.go`

**Test Categories:**
- Agent loading and discovery
- Chat interface functionality
- Redirect system accuracy
- Memory management
- Configuration validation

### **Task 5.3: Documentation**
**Priority:** Medium  
**Files:** `docs/llm-agents.md`, `docs/security/llm-security.md`

## Implementation Timeline

| Phase | Duration | Deliverables | Risk Level |
|-------|----------|--------------|------------|
| **Phase 1** | Week 1-2 | Security fixes, input validation | 🔴 Critical |
| **Phase 2** | Week 3-4 | Agent infrastructure, loading system | 🟡 Medium |
| **Phase 3** | Week 5-6 | Bubbletea interfaces, context management | 🟡 Medium |
| **Phase 4** | Week 7-8 | Advanced features, smart redirects | 🟢 Low |
| **Phase 5** | Week 9-10 | **API-enabled Policy Agent, tool system** | 🟡 Medium |
| **Phase 6** | Week 11-12 | Testing, security validation | 🔴 Critical |

## Success Criteria

### **Phase 1 Success (Must Have):**
- [ ] All hardcoded paths removed
- [ ] Input validation prevents path traversal
- [ ] Resource limits prevent DoS
- [ ] Security scan passes (no critical/high findings)

### **Phase 2 Success (Must Have):**
- [ ] Agent loading system functional
- [ ] Agent metadata parsing works
- [ ] Basic agent switching implemented
- [ ] Configuration integration complete

### **Phase 3 Success (Should Have):**
- [ ] Intuitive agent selection interface
- [ ] Smooth chat experience with agent context
- [ ] Agent switching preserves conversation flow
- [ ] Integration with existing CLI patterns
- [ ] **Real-time context usage display** (Claude Code style)
- [ ] **Context window auto-detection** for common models
- [ ] **Enhanced commands** (/context, /clear, /summarize)
- [ ] **Visual context warnings** when approaching limits

### **Phase 4 Success (Nice to Have):**
- [ ] **Auto-summarization system** preserving technical context
- [ ] **Intelligent conversation compression** at 85% capacity
- [ ] **Agent-aware summarization** maintaining domain expertise
- [ ] Smart redirect suggestions based on query analysis
- [ ] Cross-agent memory preservation
- [ ] Advanced configuration options
- [ ] Performance optimization

### **Phase 5 Success (Must Have):**
- [ ] **Policy Agent tool system** with create/read operations  
- [ ] **Interactive tool confirmation** for user consent
- [ ] **Secure tool execution** with proper authentication
- [ ] **Tool result formatting** for clear feedback
- [ ] **Integration with existing handlers** and SDK

### **Phase 6 Success (Must Have):**
- [ ] 80%+ test coverage for LLM package including tool system
- [ ] Security tests prevent known vulnerabilities
- [ ] Integration tests validate multi-agent workflows
- [ ] Tool execution security testing
- [ ] Documentation covers all features including API capabilities

## Command Structure

```bash
# Interactive agent selection with TUI
otdfctl llm

# Direct agent selection
otdfctl llm --agent development
otdfctl llm --agent policy
otdfctl llm --agent deployment
otdfctl llm --agent general

# Legacy direct chat (bypass agent system)
otdfctl llm chat /path/to/model

# Agent management
otdfctl llm agents list
otdfctl llm agents info development

# Interactive commands within chat
/context     # Show detailed context usage
/clear       # Clear conversation history  
/summarize   # Manually trigger summarization
/tools       # List available tools for current agent
/help        # Show available commands

# Tool interaction examples  
User: "Create an attribute for data classification"
Agent: [Explains approach, shows tool call, requests confirmation]
User: "y" [Confirms tool execution]
Agent: [Executes tool, shows results, suggests next steps]
```

## Risk Mitigation

**High Risk Items:**
1. **Security vulnerabilities**: Address in Phase 1 before any new features
2. **API tool security**: Extensive testing of tool authentication and authorization
3. **User confusion with tools**: Clear UX for tool confirmation and results
4. **Bubbletea integration complexity**: Leverage existing TUI patterns
5. **Performance impact**: Benchmark agent switching overhead
6. **Model compatibility**: Test with multiple model formats

**Rollback Plan:**
- Each phase includes rollback commits
- Feature flags for agent system vs. legacy chat
- Gradual rollout with monitoring
- Fallback to simple engine if agent system fails

## Future Roadmap: MCP Integration (Phase 7+)

### **Why MCP for the Future?**
After establishing the direct integration, MCP offers valuable benefits:
- **Ecosystem compatibility** with other AI tools (Cline, Cursor, etc.)
- **Standardized protocol** for tool capabilities  
- **Decoupled architecture** allowing otdfctl to serve multiple AI systems
- **Industry alignment** with emerging MCP ecosystem

### **MCP Implementation Strategy**
```go
// Future: pkg/mcp/server.go
type MCPServer struct {
    policyTools *PolicyToolSet
    handler     handlers.Handler
}

func (s *MCPServer) ListTools(ctx context.Context) ([]mcp.Tool, error) {
    tools := s.policyTools.GetTools()
    mcpTools := make([]mcp.Tool, len(tools))
    
    for i, tool := range tools {
        mcpTools[i] = mcp.Tool{
            Name:        tool.Name(),
            Description: tool.Description(),
            InputSchema: tool.Parameters(),
        }
    }
    
    return mcpTools, nil
}

func (s *MCPServer) CallTool(ctx context.Context, req mcp.CallToolRequest) (mcp.CallToolResult, error) {
    tool := s.findTool(req.Name)
    result, err := tool.Execute(ctx, req.Arguments)
    
    return mcp.CallToolResult{
        Content: []mcp.Content{{
            Type: "text",
            Text: s.formatResult(result),
        }},
    }, err
}
```

### **Command Integration**
```bash
# Future MCP server mode
otdfctl llm mcp-server --port 8080
otdfctl llm mcp-server --stdio

# Register with MCP clients
# ~/.config/claude-desktop/claude_desktop_config.json
{
  "mcpServers": {
    "otdfctl": {
      "command": "otdfctl",
      "args": ["llm", "mcp-server", "--stdio"]
    }
  }
}
```

This comprehensive plan addresses immediate security concerns, implements the innovative multi-agent system, adds powerful API capabilities, and provides a roadmap for future ecosystem integration through MCP.