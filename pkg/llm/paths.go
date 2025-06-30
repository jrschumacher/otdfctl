package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/opentdf/otdfctl/pkg/config"
)

// ModelPathResolver provides secure model path resolution
type ModelPathResolver struct {
	config *config.Config
}

// NewModelPathResolver creates a new path resolver
func NewModelPathResolver(cfg *config.Config) *ModelPathResolver {
	return &ModelPathResolver{config: cfg}
}

// ResolveEmbeddingModelPath safely resolves the embedding model path
func (mpr *ModelPathResolver) ResolveEmbeddingModelPath(userProvidedPath string) (string, error) {
	// Priority 1: User-provided path via flag
	if userProvidedPath != "" {
		return mpr.validateAndNormalizePath(userProvidedPath)
	}
	
	// Priority 2: Configuration file setting
	if mpr.config != nil && mpr.config.LLM.DefaultEmbeddingModelPath != "" {
		return mpr.validateAndNormalizePath(mpr.config.LLM.DefaultEmbeddingModelPath)
	}
	
	// Priority 3: Standard locations search
	return mpr.findEmbeddingModelInStandardLocations()
}

// ResolveChatModelPath safely resolves the chat model path
func (mpr *ModelPathResolver) ResolveChatModelPath(userProvidedPath string) (string, error) {
	// Priority 1: User-provided path via flag/argument
	if userProvidedPath != "" {
		return mpr.validateAndNormalizePath(userProvidedPath)
	}
	
	// Priority 2: Configuration file setting
	if mpr.config != nil && mpr.config.LLM.DefaultModelPath != "" {
		return mpr.validateAndNormalizePath(mpr.config.LLM.DefaultModelPath)
	}
	
	return "", fmt.Errorf("no chat model path provided - specify via argument or configuration")
}

// validateAndNormalizePath validates and normalizes a given path
func (mpr *ModelPathResolver) validateAndNormalizePath(path string) (string, error) {
	// Security: Path traversal prevention
	cleanPath := filepath.Clean(path)
	if strings.Contains(cleanPath, "..") {
		return "", fmt.Errorf("path traversal detected in model path: %s", path)
	}
	
	// Convert to absolute path
	absPath, err := filepath.Abs(cleanPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve absolute path for %s: %w", path, err)
	}
	
	// Verify file exists and is readable
	if err := mpr.validateModelFile(absPath); err != nil {
		return "", fmt.Errorf("model file validation failed for %s: %w", absPath, err)
	}
	
	return absPath, nil
}

// findEmbeddingModelInStandardLocations searches for embedding models in standard locations
func (mpr *ModelPathResolver) findEmbeddingModelInStandardLocations() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}
	
	// Standard locations to search (in order of preference)
	searchPaths := []string{
		// Ollama standard locations
		filepath.Join(homeDir, ".ollama", "models"),
		// Alternative locations
		"/usr/local/lib/ollama/models",
		"/opt/ollama/models",
		// Local model directory
		"./models",
	}
	
	// Common embedding model filenames to look for
	modelNames := []string{
		"nomic-embed-text.gguf",
		"all-minilm-l6-v2.gguf", 
		"embed.gguf",
		"embedding-model.gguf",
	}
	
	for _, searchPath := range searchPaths {
		if _, err := os.Stat(searchPath); err != nil {
			continue // Directory doesn't exist, skip
		}
		
		// Look for specific model files
		for _, modelName := range modelNames {
			modelPath := filepath.Join(searchPath, modelName)
			if err := mpr.validateModelFile(modelPath); err == nil {
				return modelPath, nil
			}
		}
		
		// Look for any .gguf files in the directory
		if foundModel := mpr.findFirstGGUFModel(searchPath); foundModel != "" {
			return foundModel, nil
		}
	}
	
	return "", fmt.Errorf("no embedding model found in standard locations - please specify path via --embedding-model flag or configuration")
}

// findFirstGGUFModel finds the first .gguf file in the given directory
func (mpr *ModelPathResolver) findFirstGGUFModel(dir string) string {
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
			if err := mpr.validateModelFile(modelPath); err == nil {
				return modelPath
			}
		}
	}
	
	return ""
}

// validateModelFile validates that a model file exists and is accessible
func (mpr *ModelPathResolver) validateModelFile(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("model file does not exist")
		}
		return fmt.Errorf("cannot access model file: %w", err)
	}
	
	if info.IsDir() {
		return fmt.Errorf("path is a directory, not a file")
	}
	
	if info.Size() == 0 {
		return fmt.Errorf("model file is empty")
	}
	
	// Check if file is readable
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("model file is not readable: %w", err)
	}
	file.Close()
	
	return nil
}

// GetStandardModelDirectories returns a list of standard model directories for documentation
func GetStandardModelDirectories() []string {
	homeDir, _ := os.UserHomeDir()
	return []string{
		filepath.Join(homeDir, ".ollama", "models"),
		"/usr/local/lib/ollama/models",
		"/opt/ollama/models",
		"./models",
	}
}