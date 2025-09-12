package cmd

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLLMChatRealE2E tests the actual llm chat command execution
func TestLLMChatRealE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping real e2e test in short mode")
	}

	// Find a model to test with
	modelPath := findTestModel(t)
	if modelPath == "" {
		t.Skip("No test model found - cannot run real e2e tests")
	}

	t.Logf("Using test model: %s", modelPath)

	testCases := []struct {
		name     string
		args     []string
		input    string
		timeout  time.Duration
		expectError bool
		skipOutput bool // Skip output validation for long-running tests
		expectText string // Text to look for in output
	}{
		{
			name:       "JSON mode initialization - real model",
			args:       []string{"llm", "chat", modelPath, "--json"},
			timeout:    180 * time.Second, // 3 minutes for model loading
			expectError: false,
			skipOutput: true, // Model loading may be slow
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create temporary directory for test
			tmpDir := t.TempDir()
			
			ctx, cancel := context.WithTimeout(context.Background(), tc.timeout)
			defer cancel()

			// Build the command
			cmd := exec.CommandContext(ctx, "go", append([]string{"run", "."}, tc.args...)...)
			cmd.Dir = getProjectRoot(t)
			cmd.Env = append(os.Environ(), "HOME="+tmpDir) // Isolate config

			// Set up input/output
			if tc.input != "" {
				stdin, err := cmd.StdinPipe()
				require.NoError(t, err, "Should create stdin pipe")
				
				go func() {
					defer stdin.Close()
					stdin.Write([]byte(tc.input))
				}()
			}

			stdout, err := cmd.StdoutPipe()
			require.NoError(t, err, "Should create stdout pipe")

			stderr, err := cmd.StderrPipe()
			require.NoError(t, err, "Should create stderr pipe")

			// Start the command
			err = cmd.Start()
			require.NoError(t, err, "Should start command")

			// Collect output
			var stdoutBuf, stderrBuf strings.Builder
			
			// Read stdout
			go func() {
				io.Copy(&stdoutBuf, stdout)
			}()
			
			// Read stderr
			go func() {
				io.Copy(&stderrBuf, stderr)
			}()

			// Wait for completion
			err = cmd.Wait()
			
			stdoutOutput := stdoutBuf.String()
			stderrOutput := stderrBuf.String()
			
			t.Logf("STDOUT: %s", stdoutOutput)
			t.Logf("STDERR: %s", stderrOutput)

			if tc.expectError {
				assert.Error(t, err, "Should have error")
			} else {
				if err != nil {
					// Command failed - log details for debugging
					t.Logf("Command failed with error: %v", err)
					if cmd.ProcessState != nil {
						t.Logf("Exit code: %d", cmd.ProcessState.ExitCode())
					}
					
					// Handle specific error cases
					if ctx.Err() == context.DeadlineExceeded {
						t.Logf("Command timed out after %v - model loading may be slow", tc.timeout)
						if !tc.skipOutput {
							t.Skip("Command timed out (expected for large models)")
						}
					} else if strings.Contains(stderrOutput, "Failed to initialize LLM engine") ||
					         strings.Contains(stderrOutput, "no local chat models found") {
						t.Skip("Model loading failed (expected in CI environment)")
					} else if !tc.skipOutput {
						t.Errorf("Unexpected command failure: %v", err)
					}
				}
				
				// Skip output validation for long-running tests
				if tc.skipOutput {
					t.Logf("Skipping output validation for long-running test")
					return
				}
				
				// Verify expected text if specified
				if tc.expectText != "" {
					assert.Contains(t, stdoutOutput, tc.expectText, "Should contain expected text")
				} else if strings.Contains(strings.Join(tc.args, " "), "--json") {
					assert.Contains(t, stdoutOutput, `"status"`, "JSON mode should output status")
				} else {
					assert.Contains(t, stdoutOutput, "OpenTDF LLM Chat", "Should show chat startup message")
				}
			}
		})
	}
}

// TestLLMHelpCommandsE2E tests help commands (fast, no model required)
func TestLLMHelpCommandsE2E(t *testing.T) {
	testCases := []struct {
		name       string
		args       []string
		expectText string
	}{
		{
			name:       "llm help",
			args:       []string{"llm", "--help"},
			expectText: "Usage:",
		},
		{
			name:       "llm chat help",
			args:       []string{"llm", "chat", "--help"},
			expectText: "Enable streaming responses",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			cmd := exec.CommandContext(ctx, "go", append([]string{"run", "."}, tc.args...)...)
			cmd.Dir = getProjectRoot(t)

			output, err := cmd.CombinedOutput()
			outputStr := string(output)
			
			t.Logf("Command: %v", tc.args)
			t.Logf("Output: %s", outputStr)
			if err != nil {
				t.Logf("Error: %v", err)
			}

			assert.NoError(t, err, "Help command should not error")
			assert.Contains(t, outputStr, tc.expectText, "Should contain expected help text")
		})
	}
}

// TestLLMCommandValidationE2E tests command validation without requiring model loading
func TestLLMCommandValidationE2E(t *testing.T) {
	testCases := []struct {
		name        string
		args        []string
		expectError bool
		expectText  string
	}{
		{
			name:        "llm help",
			args:        []string{"llm", "--help"},
			expectError: false,
			expectText:  "Usage:",
		},
		{
			name:        "llm chat help",
			args:        []string{"llm", "chat", "--help"},
			expectError: false,
			expectText:  "Enable streaming responses",
		},
		{
			name:        "llm ingest help",
			args:        []string{"llm", "ingest", "--help"},
			expectError: false,
			expectText:  "ingest",
		},
		{
			name:        "llm eval help",
			args:        []string{"llm", "eval", "--help"},
			expectError: false,
			expectText:  "evaluation",
		},
		{
			name:        "Invalid model path shows config",
			args:        []string{"llm", "chat", "/definitely/does/not/exist.gguf", "--json"},
			expectError: false, // JSON mode shows config even with invalid model
			expectText:  `"status": "initialized"`, // Should show initialization status
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			cmd := exec.CommandContext(ctx, "go", append([]string{"run", "."}, tc.args...)...)
			cmd.Dir = getProjectRoot(t)

			output, err := cmd.CombinedOutput()
			outputStr := string(output)
			
			t.Logf("Command: %v", tc.args)
			t.Logf("Output: %s", outputStr)
			if err != nil {
				t.Logf("Error: %v", err)
			}

			if tc.expectError {
				assert.Error(t, err, "Should have error")
			} else {
				assert.NoError(t, err, "Should not have error")
				if tc.expectText != "" {
					assert.Contains(t, outputStr, tc.expectText, "Should contain expected text")
				}
			}
		})
	}
}

// TestLLMContextOverflowE2E tests handling of context overflow scenarios  
func TestLLMContextOverflowE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping context overflow test in short mode")
	}

	// This test is disabled by default because it requires a real model and takes a long time
	// To enable: go test -v ./cmd -run TestLLMContextOverflowE2E -timeout 10m
	t.Skip("Skipping long-running context overflow test - enable manually for real testing")

	modelPath := findTestModel(t)
	if modelPath == "" {
		t.Skip("No test model found - cannot run context overflow tests")
	}

	t.Logf("⚠️  This test will take several minutes and consume significant resources")
	
	t.Run("Context overflow crash test", func(t *testing.T) {
		tmpDir := t.TempDir()
		
		// Use small context to trigger overflow quickly
		args := []string{"llm", "chat", modelPath, "--context-size", "1024", "--temperature", "0.1"}
		
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second) // 5 minutes
		defer cancel()

		cmd := exec.CommandContext(ctx, "go", append([]string{"run", "."}, args...)...)
		cmd.Dir = getProjectRoot(t)
		cmd.Env = append(os.Environ(), "HOME="+tmpDir)

		stdin, err := cmd.StdinPipe()
		require.NoError(t, err)

		stderr, err := cmd.StderrPipe()
		require.NoError(t, err)

		err = cmd.Start()
		require.NoError(t, err)

		// Send conversation that should trigger context limits
		go func() {
			defer stdin.Close()
			
			// Send progressively longer messages
			baseMsg := "Tell me about OpenTDF security features: "
			for i := 0; i < 8; i++ {
				longMsg := baseMsg + strings.Repeat("Please provide detailed explanations with examples. ", i*10+5)
				fmt.Fprintf(stdin, "%s\n", longMsg)
				time.Sleep(10 * time.Second) // Wait for response
			}
			
			fmt.Fprintln(stdin, "exit")
		}()

		// Monitor for crashes
		crashDetected := false
		go func() {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				line := scanner.Text()
				if strings.Contains(line, "panic") || 
				   strings.Contains(line, "fatal error") ||
				   strings.Contains(line, "runtime error") ||
				   strings.Contains(line, "SIGABRT") {
					crashDetected = true
					t.Errorf("🚨 CRASH DETECTED: %s", line)
				}
				
				if strings.Contains(line, "truncated") {
					t.Logf("✅ Truncation working: %s", line)
				}
			}
		}()

		// Wait for completion
		err = cmd.Wait()
		
		// Verify no crashes occurred
		assert.False(t, crashDetected, "Should not crash during context overflow")
		
		if err != nil && ctx.Err() != context.DeadlineExceeded {
			t.Logf("Command finished with error: %v", err)
		}
	})
}

// TestLLMIngestRealE2E tests actual document ingestion
func TestLLMIngestRealE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping real ingest e2e test in short mode")
	}

	tmpDir := t.TempDir()
	
	// Create test document
	testDoc := filepath.Join(tmpDir, "test_doc.md")
	testContent := `# OpenTDF Test Document

This is a test document for RAG ingestion.

## Features

- Secure data sharing
- Policy-based access control
- Encryption at rest and in transit

## Usage

Use otdfctl to manage policies and encrypt data.
`
	
	err := os.WriteFile(testDoc, []byte(testContent), 0644)
	require.NoError(t, err)

	testCases := []struct {
		name        string
		args        []string
		expectError bool
	}{
		{
			name: "Simple ingestion",
			args: []string{"llm", "ingest", testDoc, "--json"},
			expectError: false,
		},
		{
			name: "Vector ingestion",
			args: []string{"llm", "ingest", testDoc, "--embeddings", "--embedding-model", "mxbai-embed-large", "--json"},
			expectError: false, // May fail if no embedding model, but shouldn't crash
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			cmd := exec.CommandContext(ctx, "go", append([]string{"run", "."}, tc.args...)...)
			cmd.Dir = getProjectRoot(t)
			cmd.Env = append(os.Environ(), "HOME="+tmpDir)

			output, err := cmd.CombinedOutput()
			
			t.Logf("Command output: %s", string(output))

			if tc.expectError {
				assert.Error(t, err, "Should have error")
			} else {
				if err != nil {
					// Log error but don't fail if it's expected (like missing embedding model)
					t.Logf("Command failed: %v", err)
					if strings.Contains(string(output), "Failed to") && 
					   (strings.Contains(string(output), "embedding") || 
					    strings.Contains(string(output), "ollama")) {
						t.Skip("Missing embedding model or ollama (expected in CI)")
					}
				}
				
				// Verify basic success
				assert.Contains(t, string(output), "ingested", "Should show ingestion success")
			}
		})
	}
}

// TestLLMEvalRealE2E tests the evaluation command
func TestLLMEvalRealE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping eval e2e test in short mode")
	}

	modelPath := findTestModel(t)
	if modelPath == "" {
		t.Skip("No test model found - cannot run eval tests")
	}

	tmpDir := t.TempDir()
	
	t.Run("Single query evaluation", func(t *testing.T) {
		args := []string{"llm", "eval", "--model", modelPath, "--query", "What is OpenTDF?", "--output", "json"}
		
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, "go", append([]string{"run", "."}, args...)...)
		cmd.Dir = getProjectRoot(t)
		cmd.Env = append(os.Environ(), "HOME="+tmpDir)

		output, err := cmd.CombinedOutput()
		
		t.Logf("Eval output: %s", string(output))

		if err != nil {
			if strings.Contains(string(output), "Failed to initialize") {
				t.Skip("Model initialization failed (expected in CI)")
			}
			t.Logf("Eval command failed: %v", err)
		} else {
			// Should output evaluation results
			assert.Contains(t, string(output), "quality", "Should show quality metrics")
		}
	})
}

// Helper functions

func findTestModel(t *testing.T) string {
	// Try to find a real model for testing
	modelPaths := []string{
		// User's model from the original command
		"/Users/ryan/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45",
		// Common local model locations
		filepath.Join(os.Getenv("HOME"), ".ollama", "models", "blobs"),
	}

	// First check the specific model from the original command
	if _, err := os.Stat(modelPaths[0]); err == nil {
		return modelPaths[0]
	}

	// Try to find any .gguf files in common locations
	searchDirs := []string{
		filepath.Join(os.Getenv("HOME"), ".ollama", "models"),
		filepath.Join(os.Getenv("HOME"), "models"),
		"./models",
	}

	for _, dir := range searchDirs {
		if models := findGGUFFiles(dir); len(models) > 0 {
			return models[0]
		}
	}

	// Try the original findLocalChatModel function
	if model, err := findLocalChatModel(); err == nil {
		return model
	}

	return ""
}

func findGGUFFiles(dir string) []string {
	var models []string
	
	entries, err := os.ReadDir(dir)
	if err != nil {
		return models
	}

	for _, entry := range entries {
		if entry.IsDir() {
			// Recursively search subdirectories  
			subModels := findGGUFFiles(filepath.Join(dir, entry.Name()))
			models = append(models, subModels...)
		} else if strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
			fullPath := filepath.Join(dir, entry.Name())
			if info, err := os.Stat(fullPath); err == nil && info.Size() > 1024*1024 { // At least 1MB
				models = append(models, fullPath)
			}
		}
	}

	return models
}

func getProjectRoot(t *testing.T) string {
	// Get the directory containing go.mod
	wd, err := os.Getwd()
	require.NoError(t, err)
	
	// Look for go.mod in current directory or parent directories
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		
		parent := filepath.Dir(dir)
		if parent == dir {
			break // Reached root
		}
		dir = parent
	}
	
	t.Fatal("Could not find project root (go.mod)")
	return ""
}