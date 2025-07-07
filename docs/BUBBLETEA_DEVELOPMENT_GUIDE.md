# Bubbletea Development Guide for OpenTDF otdfctl

## Overview

This guide outlines best practices for developing Bubbletea interfaces in the otdfctl project, based on the existing sophisticated implementations and established patterns.

## Core Principles

### 1. **Modular Component Architecture**
- Use composition over inheritance
- Create reusable UI components
- Separate concerns (data, view, update logic)
- Follow the Elm Architecture pattern

### 2. **Consistent Styling System**
- Use lipgloss for all styling
- Maintain consistent color palette
- Create style constants for reuse
- Support responsive design

### 3. **Professional UX Patterns**
- Provide clear visual feedback
- Support keyboard navigation
- Show loading states
- Handle errors gracefully

## Project Structure

### Current Architecture
```
/pkg/llm/
├── agent_tui.go           # Main chat TUI implementation
├── simple_chat.go         # Terminal fallback interface
└── agent_*.go             # Supporting agent logic

/tui/
├── common.go              # Shared utilities and styles
├── appMenu.go             # Main application menu
└── [domain-specific UIs]  # Attribute, policy management
```

### File Organization Rules
1. **Domain-specific TUIs** go in their respective packages (`/pkg/llm/`, `/pkg/policy/`)
2. **Shared components** go in `/tui/` 
3. **Common utilities** centralized in `/tui/common.go`
4. **Integration** happens in `/cmd/` command files

## Component Development Patterns

### 1. **Model Structure Template**

```go
type MyTUIModel struct {
    // Core dependencies
    manager     *MyManager
    engine      *SomeEngine
    
    // UI components (bubbletea built-ins)
    viewport    viewport.Model
    textarea    textarea.Model
    list        list.Model
    help        help.Model
    
    // Application state
    data        []MyData
    currentItem *MyItem
    
    // UI state
    width       int
    height      int
    ready       bool
    quitting    bool
    
    // UI modes/views
    showList    bool
    showHelp    bool
    
    // Keybindings
    keys        MyKeyMap
}
```

### 2. **Essential Methods Implementation**

```go
// Required bubbletea interface methods
func (m *MyTUIModel) Init() tea.Cmd
func (m *MyTUIModel) Update(msg tea.Msg) (tea.Model, tea.Cmd)
func (m *MyTUIModel) View() string

// Standard utility methods
func (m *MyTUIModel) updateSizes()
func (m *MyTUIModel) handleKeyPress(key tea.KeyMsg) (tea.Model, tea.Cmd)
func (m *MyTUIModel) renderHeader() string
func (m *MyTUIModel) renderFooter() string
```

### 3. **Message Pattern**

```go
// Custom messages for async operations
type MyResponseMsg struct {
    data   MyData
    error  error
}

type MyStateChangeMsg struct {
    newState MyState
}

// Command functions return these messages
func performAsyncOperation() tea.Cmd {
    return func() tea.Msg {
        result, err := doSomething()
        return MyResponseMsg{data: result, error: err}
    }
}
```

## Streaming and Async Patterns

### Current Streaming Implementation

The project uses callback-based streaming with `ChatStream()`:

```go
// Current pattern - callback based
func (engine *SimpleChatEngine) ChatStream(messages []ChatMessage, callback StreamingCallback) SimpleResponse

// Usage in TUI:
func (m *Model) generateStreamingResponse() tea.Cmd {
    return func() tea.Msg {
        var accumulator strings.Builder
        
        callback := func(token string) {
            accumulator.WriteString(token)
            // Send intermediate updates
        }
        
        response := m.engine.ChatStream(messages, callback)
        return ResponseCompleteMsg{content: accumulator.String()}
    }
}
```

### Enhanced Channel-Based Streaming Pattern

For better TUI integration, consider this channel-based approach:

```go
// Enhanced streaming with channels
type StreamingResponseMsg struct {
    token    string
    done     bool
    error    error
}

func (m *Model) generateStreamingResponse() tea.Cmd {
    return func() tea.Msg {
        // Create a channel for streaming tokens
        tokenCh := make(chan string, 10)
        doneCh := make(chan error, 1)
        
        go func() {
            defer close(tokenCh)
            defer close(doneCh)
            
            callback := func(token string) {
                select {
                case tokenCh <- token:
                case <-time.After(100 * time.Millisecond):
                    // Prevent blocking
                }
            }
            
            response := m.engine.ChatStream(messages, callback)
            doneCh <- response.Error
        }()
        
        return m.listenForTokens(tokenCh, doneCh)
    }
}

func (m *Model) listenForTokens(tokenCh <-chan string, doneCh <-chan error) tea.Cmd {
    return func() tea.Msg {
        select {
        case token, ok := <-tokenCh:
            if !ok {
                return StreamingResponseMsg{done: true}
            }
            return StreamingResponseMsg{token: token}
        case err := <-doneCh:
            return StreamingResponseMsg{done: true, error: err}
        }
    }
}
```

## Key Development Guidelines

### 1. **State Management**

```go
// Good: Clear state transitions
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case MyDataMsg:
        m.data = msg.data
        m.updateView()
        return m, nil
    }
}

// Bad: Direct state mutation in view
func (m *Model) View() string {
    m.data = someCalculation() // Don't do this!
}
```

### 2. **Responsive Design**

```go
func (m *Model) updateSizes() {
    if m.width <= 0 || m.height <= 0 {
        return
    }
    
    // Calculate available space
    headerHeight := 3
    footerHeight := 2
    availableHeight := m.height - headerHeight - footerHeight
    
    // Responsive breakpoints
    if m.width < 80 {
        m.useCompactLayout()
    } else {
        m.useFullLayout()
    }
    
    // Update component sizes
    m.viewport.Width = m.width - 4
    m.viewport.Height = availableHeight - 5
}
```

### 3. **Error Handling Patterns**

```go
// Show errors in UI, don't crash
func (m *Model) handleError(err error) {
    m.addSystemMessage(fmt.Sprintf("❌ Error: %v", err))
    // Log for debugging
    log.Printf("TUI Error: %v", err)
}

// Graceful degradation
func (m *Model) performOperation() tea.Cmd {
    return func() tea.Msg {
        result, err := riskyOperation()
        if err != nil {
            return ErrorMsg{error: err}
        }
        return SuccessMsg{data: result}
    }
}
```

### 4. **Keyboard Handling Best Practices**

```go
// Define clear keymaps
type KeyMap struct {
    Send    key.Binding
    Quit    key.Binding
    Help    key.Binding
    // ... more bindings
}

// Context-aware key handling
func (m *Model) handleKeyPress(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
    // Handle global keys first
    switch {
    case key.Matches(msg, m.keys.Quit):
        return m, tea.Quit
    case key.Matches(msg, m.keys.Help):
        m.showHelp = !m.showHelp
        return m, nil
    }
    
    // Handle mode-specific keys
    if m.showList {
        return m.handleListKeys(msg)
    }
    
    return m.handleDefaultKeys(msg)
}
```

## Styling Guidelines

### 1. **Color Palette** (from existing implementation)

```go
var (
    PrimaryColor    = lipgloss.Color("#874BFD")  // Purple
    SuccessColor    = lipgloss.Color("#50C878")  // Green  
    WarningColor    = lipgloss.Color("#FFD700")  // Yellow
    ErrorColor      = lipgloss.Color("#FF6B6B")  // Red
    TextColor       = lipgloss.Color("#FAFAFA")  // Light gray
    DimColor        = lipgloss.Color("#626262")  // Dark gray
    AccentColor     = lipgloss.Color("#EE6FF8")  // Light purple
)
```

### 2. **Style Consistency**

```go
// Create reusable style functions
func headerStyle() lipgloss.Style {
    return lipgloss.NewStyle().
        Foreground(TextColor).
        Background(PrimaryColor).
        Padding(0, 1).
        Bold(true)
}

func borderStyle() lipgloss.Style {
    return lipgloss.NewStyle().
        Border(lipgloss.RoundedBorder()).
        BorderForeground(PrimaryColor)
}
```

### 3. **Component Integration**

```go
// Integrate with existing components
func NewMyComponent() MyComponent {
    // Follow existing patterns from agent_tui.go
    component := SomeComponent.New()
    component.Title = "My Feature"
    component.SetShowStatusBar(false)
    component.SetFilteringEnabled(true)
    
    return MyComponent{
        component: component,
        help:     help.New(),
        keys:     DefaultMyKeyMap(),
    }
}
```

## Testing Patterns

### 1. **Model Testing**

```go
func TestModelUpdate(t *testing.T) {
    model := NewMyModel()
    
    // Test message handling
    updatedModel, cmd := model.Update(MyTestMsg{})
    
    assert.NotNil(t, cmd)
    assert.Equal(t, expectedState, updatedModel.state)
}
```

### 2. **Integration Testing**

```go
func TestTUIIntegration(t *testing.T) {
    // Test with real program
    program := tea.NewProgram(NewMyModel())
    
    // Send test inputs
    program.Send(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("test")})
    
    // Verify state changes
}
```

## Performance Considerations

### 1. **Efficient Rendering**

```go
// Cache expensive calculations
func (m *Model) View() string {
    if m.cachedView == "" || m.dirty {
        m.cachedView = m.renderView()
        m.dirty = false
    }
    return m.cachedView
}

// Minimize lipgloss operations
func (m *Model) renderMessages() string {
    var builder strings.Builder
    for _, msg := range m.messages {
        builder.WriteString(m.renderMessage(msg))
    }
    return builder.String()
}
```

### 2. **Memory Management**

```go
// Limit message history
func (m *Model) addMessage(msg Message) {
    m.messages = append(m.messages, msg)
    
    // Keep only recent messages
    maxMessages := 100
    if len(m.messages) > maxMessages {
        m.messages = m.messages[len(m.messages)-maxMessages:]
    }
}
```

## Integration with otdfctl Patterns

### 1. **Configuration Integration**

```go
// Use existing config patterns
func NewMyTUIWithConfig(cfg *config.Config) *MyTUI {
    return &MyTUI{
        config: cfg,
        // ... other fields
    }
}
```

### 2. **Command Integration**

```go
// Follow existing command patterns in /cmd/
func NewMyTUICommand() *cobra.Command {
    return &cobra.Command{
        Use:   "mytui",
        Short: "Interactive TUI for my feature",
        RunE: func(cmd *cobra.Command, args []string) error {
            model := NewMyTUI()
            program := tea.NewProgram(model, tea.WithAltScreen())
            _, err := program.Run()
            return err
        },
    }
}
```

## Future Enhancements

### 1. **Advanced Streaming**
- Real-time token display with typewriter effect
- Cancellable long-running operations
- Progress indicators for model loading

### 2. **Enhanced Interactivity**
- Mouse support for clickable elements
- Multi-pane layouts with focus management
- Drag-and-drop file handling

### 3. **Accessibility**
- Screen reader support
- High contrast themes
- Keyboard-only navigation

## Summary

The otdfctl project has excellent Bubbletea foundations with sophisticated implementations already in place. When developing new TUI components:

1. **Follow existing patterns** in `agent_tui.go` and `/tui/` directory
2. **Use the established styling** and color palette
3. **Implement proper streaming** for real-time interactions
4. **Maintain keyboard-first** interaction design
5. **Handle errors gracefully** with user-friendly messaging
6. **Test thoroughly** with various terminal sizes

The existing agent chat TUI serves as an excellent reference implementation showcasing professional-grade Bubbletea development patterns.