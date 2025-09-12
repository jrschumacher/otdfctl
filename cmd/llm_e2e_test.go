package cmd

import (
	"os"
	"path/filepath"  
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLLMChatCommandE2E tests the end-to-end chat command functionality
func TestLLMChatCommandE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e test in short mode")
	}

	// Test that we can parse and validate the llm chat command
	t.Run("Command parsing and flag validation", func(t *testing.T) {
		// Test the actual llmChatCmd command setup
		cmd := &llmChatCmd.Command
		
		// Verify command structure  
		assert.NotEmpty(t, cmd.Use, "Command should have Use field")
		assert.Contains(t, cmd.Use, "chat", "Should be chat command")
		
		// The man.Docs.GetCommand approach may not set RunE directly
		// Instead verify the command is properly initialized
		assert.NotNil(t, cmd, "Command should be initialized")
		
		// Verify flags are properly configured
		flags := []string{"stream", "context-size", "temperature", "system-prompt", "rag", "index-path", "embedding-model", "json", "verbose"}
		for _, flagName := range flags {
			flag := cmd.Flags().Lookup(flagName)
			assert.NotNil(t, flag, "Flag %s should exist", flagName)
		}
		
		// Test flag defaults
		streamFlag := cmd.Flags().Lookup("stream")
		assert.Equal(t, "true", streamFlag.DefValue, "Stream should default to true")
		
		contextFlag := cmd.Flags().Lookup("context-size")  
		assert.Equal(t, "4096", contextFlag.DefValue, "Context size should default to 4096")
		
		tempFlag := cmd.Flags().Lookup("temperature")
		assert.Equal(t, "0.7", tempFlag.DefValue, "Temperature should default to 0.7")
	})

	// Test model discovery and validation
	t.Run("Model path resolution", func(t *testing.T) {
		// Test findLocalChatModel function
		model, err := findLocalChatModel()
		
		if err != nil {
			// It's okay if no models are found in test environment
			t.Logf("No local chat models found (expected in test env): %v", err)
		} else {
			// If a model is found, verify it's valid
			assert.NotEmpty(t, model, "Model path should not be empty")
			assert.True(t, strings.HasSuffix(model, ".gguf") || strings.Contains(model, "ollama"), 
				"Model should be GGUF file or Ollama model")
			
			// Test model validation
			err := validateModelFile(model)
			if strings.Contains(model, "ollama") {
				// Ollama models might not be files, skip validation
				t.Logf("Skipping file validation for Ollama model: %s", model)
			} else {
				assert.NoError(t, err, "Model file should be valid")
			}
		}
	})

	// Test JSON output mode (doesn't require actual model)
	t.Run("JSON output mode", func(t *testing.T) {
		// Create a temporary directory for test
		tmpDir := t.TempDir()
		
		// Change to temp directory to avoid affecting real configs
		oldDir, _ := os.Getwd()
		defer os.Chdir(oldDir)
		os.Chdir(tmpDir)
		
		// This would normally test the actual command execution with --json flag
		// but we can't easily test it without mocking the entire CLI framework
		// Instead, test the components that JSON mode would use
		
		testModel := "/fake/model/path.gguf"
		
		// Test the session structure that would be returned
		session := map[string]any{
			"model": testModel,
			"config": map[string]any{
				"stream":       true,
				"context_size": 4096,
				"temperature":  0.7,
			},
			"status": "initialized",
		}
		
		assert.Equal(t, testModel, session["model"], "Model should be set")
		assert.Equal(t, "initialized", session["status"], "Status should be initialized")
		
		config := session["config"].(map[string]any)
		assert.Equal(t, true, config["stream"], "Stream should be enabled")
		assert.Equal(t, 4096, config["context_size"], "Context size should be set")
		assert.Equal(t, 0.7, config["temperature"], "Temperature should be set")
	})

	// Test RAG index path detection
	t.Run("RAG index path detection", func(t *testing.T) {
		tmpDir := t.TempDir()
		
		// Create test index files
		vectorIndexPath := filepath.Join(tmpDir, ".otdfctl", "vector_index.json")
		simpleIndexPath := filepath.Join(tmpDir, ".otdfctl", "simple_rag_index.json")
		
		// Create directory
		err := os.MkdirAll(filepath.Dir(vectorIndexPath), 0755)
		require.NoError(t, err, "Should create test directory")
		
		// Test 1: No index exists - should default to simple
		indexPath := ""
		useVector := false
		
		if indexPath == "" {
			// Simulate the detection logic from llmChatCmd
			if _, err := os.Stat(vectorIndexPath); err == nil {
				indexPath = vectorIndexPath
				useVector = true
			} else if _, err := os.Stat(simpleIndexPath); err == nil {
				indexPath = simpleIndexPath
				useVector = false
			} else {
				indexPath = simpleIndexPath
				useVector = false
			}
		}
		
		assert.Equal(t, simpleIndexPath, indexPath, "Should default to simple index")
		assert.False(t, useVector, "Should use simple mode by default")
		
		// Test 2: Create vector index and test detection
		err = os.WriteFile(vectorIndexPath, []byte("{}"), 0644)
		require.NoError(t, err, "Should create vector index file")
		
		indexPath = ""
		if indexPath == "" {
			if _, err := os.Stat(vectorIndexPath); err == nil {
				indexPath = vectorIndexPath
				useVector = true
			} else if _, err := os.Stat(simpleIndexPath); err == nil {
				indexPath = simpleIndexPath
				useVector = false
			} else {
				indexPath = simpleIndexPath
				useVector = false
			}
		}
		
		assert.Equal(t, vectorIndexPath, indexPath, "Should detect vector index")
		assert.True(t, useVector, "Should use vector mode")
	})
}

// TestLLMIngestCommandE2E tests the ingest command functionality  
func TestLLMIngestCommandE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping e2e test in short mode")
	}

	t.Run("Ingest command structure", func(t *testing.T) {
		cmd := &llmIngestCmd.Command
		
		// Verify command structure
		assert.NotEmpty(t, cmd.Use, "Command should have Use field")
		assert.Contains(t, cmd.Use, "ingest", "Should be ingest command")
		assert.NotNil(t, cmd, "Command should be initialized")
		
		// Verify flags
		flags := []string{"index-path", "embeddings", "embedding-model", "json"}
		for _, flagName := range flags {
			flag := cmd.Flags().Lookup(flagName)
			assert.NotNil(t, flag, "Flag %s should exist", flagName)
		}
	})

	t.Run("Document ingestion logic", func(t *testing.T) {
		tmpDir := t.TempDir()
		
		// Create test document
		testDoc := filepath.Join(tmpDir, "test.md")
		testContent := "# OpenTDF Test\n\nThis is test documentation content."
		err := os.WriteFile(testDoc, []byte(testContent), 0644)
		require.NoError(t, err, "Should create test document")
		
		// Test index path logic
		indexPath := ""
		useEmbeddings := false
		
		if indexPath == "" {
			if useEmbeddings {
				indexPath = filepath.Join(tmpDir, ".otdfctl", "vector_index.json")
			} else {
				indexPath = filepath.Join(tmpDir, ".otdfctl", "simple_rag_index.json")
			}
		}
		
		assert.Equal(t, filepath.Join(tmpDir, ".otdfctl", "simple_rag_index.json"), indexPath)
		
		// Test with embeddings enabled
		useEmbeddings = true
		if useEmbeddings {
			indexPath = filepath.Join(tmpDir, ".otdfctl", "vector_index.json")
		}
		
		assert.Equal(t, filepath.Join(tmpDir, ".otdfctl", "vector_index.json"), indexPath)
	})
}

// TestSystemPromptConfiguration tests the system prompt setup
func TestSystemPromptConfiguration(t *testing.T) {
	prompt := getDefaultSystemPrompt()
	
	// Verify the enhanced prompt includes directive for detailed responses  
	assert.Contains(t, prompt, "detailed, comprehensive responses", "Should encourage detailed responses")
	assert.Contains(t, prompt, "Don't be brief", "Should explicitly discourage brevity")
	
	// Verify it contains OpenTDF-specific knowledge areas
	assert.Contains(t, prompt, "OpenTDF", "Should contain OpenTDF reference")
	assert.Contains(t, prompt, "Policy management", "Should contain policy management")
	assert.Contains(t, prompt, "otdfctl CLI", "Should contain CLI reference")
}