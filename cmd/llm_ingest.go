package cmd

import (
	"os"
	"path/filepath"

	"github.com/opentdf/otdfctl/pkg/cli"
	"github.com/opentdf/otdfctl/pkg/llm"
	"github.com/opentdf/otdfctl/pkg/man"
	"github.com/spf13/cobra"
)

var llmIngestCmd = man.Docs.GetCommand("llm/ingest", man.WithRun(func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)

	embeddingModelPath := c.Flags.GetOptionalString("embedding-model")
	indexPath := c.Flags.GetOptionalString("index-path")
	sourceType := c.Flags.GetOptionalString("source")
	sourcePath := c.Flags.GetOptionalString("path")
	cacheDir := c.Flags.GetOptionalString("cache-dir")

	// Set defaults
	if embeddingModelPath == "" {
		embeddingModelPath = "/Users/ryan/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
	}
	if indexPath == "" {
		homeDir, _ := os.UserHomeDir()
		indexPath = filepath.Join(homeDir, ".otdfctl", "rag_index.json")
	}
	if cacheDir == "" {
		homeDir, _ := os.UserHomeDir()
		cacheDir = filepath.Join(homeDir, ".otdfctl", "doc_cache")
	}

	c.Printf("🔧 Initializing RAG document ingestion...\n")
	c.Printf("   Embedding model: %s\n", embeddingModelPath)
	c.Printf("   Index path: %s\n", indexPath)
	c.Printf("   Cache directory: %s\n", cacheDir)

	// Initialize embedding engine
	c.Printf("\n📥 Loading embedding model...\n")
	embeddingEngine, err := llm.NewEmbeddingEngine(embeddingModelPath)
	if err != nil {
		c.ExitWithError("Failed to initialize embedding engine", err)
	}
	defer embeddingEngine.Close()

	// Initialize vector store
	vectorStore := llm.NewVectorStore(indexPath)
	if err := vectorStore.LoadIndex(); err != nil {
		c.ExitWithError("Failed to load vector index", err)
	}

	// Initialize document ingester
	ingester := llm.NewDocumentIngester(vectorStore, embeddingEngine, cacheDir)

	c.Printf("\n📚 Starting document ingestion...\n")

	switch sourceType {
	case "github":
		if err := ingester.IngestFromGitHub(); err != nil {
			c.ExitWithError("Failed to ingest from GitHub", err)
		}
	case "local":
		if sourcePath == "" {
			c.ExitWithError("--path is required when --source=local", nil)
		}
		if err := ingester.IngestFromLocalDirectory(sourcePath); err != nil {
			c.ExitWithError("Failed to ingest from local directory", err)
		}
	default:
		c.ExitWithError("Invalid source type. Use 'github' or 'local'", nil)
	}

	// Save the updated index
	c.Printf("\n💾 Saving vector index...\n")
	if err := vectorStore.SaveIndex(); err != nil {
		c.ExitWithError("Failed to save vector index", err)
	}

	c.Printf("\n✅ Document ingestion completed successfully!\n")
	c.Printf("   Total documents: %d\n", vectorStore.GetDocumentCount())
	c.Printf("   Index saved to: %s\n", indexPath)
}))

func init() {
	// TODO: Fix flag documentation parsing and use proper doc-driven flags
	// For now, hardcode flags temporarily
	llmIngestCmd.Flags().String("embedding-model", "", "Path to embedding model (defaults to llama3.2:1b)")
	llmIngestCmd.Flags().String("index-path", "", "Path to save vector index (default: ~/.otdfctl/rag_index.json)")
	llmIngestCmd.Flags().String("source", "github", "Source type: 'github' or 'local'")
	llmIngestCmd.Flags().String("path", "", "Path to local docs directory (required for --source=local)")
	llmIngestCmd.Flags().String("cache-dir", "", "Directory for caching downloaded docs (default: ~/.otdfctl/doc_cache)")

	// Add ingest command to llm parent
	llmCmd.AddCommand(&llmIngestCmd.Command)
}