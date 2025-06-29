package cmd

import (
	"fmt"

	"github.com/opentdf/otdfctl/pkg/cli"
	"github.com/opentdf/otdfctl/pkg/llm"
	"github.com/opentdf/otdfctl/pkg/man"
	"github.com/spf13/cobra"
)

var llmCmd = man.Docs.GetCommand("llm")

var llmChatCmd = man.Docs.GetCommand("llm/chat", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)
	
	if len(args) == 0 {
		c.ExitWithError("Model path is required", nil)
	}
	
	modelPath := args[0]
	
	// Get flag values
	stream := c.Flags.GetOptionalBool("stream")
	contextSize := int(c.Flags.GetOptionalInt32("context-size"))
	temperatureFlag, _ := cmd.Flags().GetFloat64("temperature")
	temperature := temperatureFlag
	systemPrompt := c.Flags.GetOptionalString("system-prompt")
	
	// Create LLM handler with injected functions to avoid import cycles
	printfFunc := func(format string, args ...interface{}) { c.Printf(format, args...) }
	printlnFunc := func(format string, args ...interface{}) { c.Println(fmt.Sprintf(format, args...)) }
	
	handler := llm.NewHandler(
		&OtdfctlCfg,
		printfFunc,
		printlnFunc,
		c.ExitWithJSON,
		c.Flags.GetOptionalBool("json"),
	)
	defer handler.Close()
	
	if err := handler.StartChat(modelPath, stream, contextSize, temperature, systemPrompt); err != nil {
		c.ExitWithError("Failed to start chat session", err)
	}
}))

func init() {
	// TODO: Fix flag documentation parsing and use proper doc-driven flags
	// For POC, hardcode flags temporarily
	llmChatCmd.Flags().Bool("stream", true, "Enable streaming responses")
	llmChatCmd.Flags().Int32("context-size", 4096, "Maximum context window size")
	llmChatCmd.Flags().Float64("temperature", 0.7, "Sampling temperature (0.0-1.0)")
	llmChatCmd.Flags().String("system-prompt", "", "Custom system prompt")
	
	// Add chat command to llm parent
	llmCmd.AddCommand(&llmChatCmd.Command)
	
	// Add llm command to root
	RootCmd.AddCommand(&llmCmd.Command)
}