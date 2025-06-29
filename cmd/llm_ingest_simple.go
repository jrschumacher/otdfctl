package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"bufio"

	"github.com/opentdf/otdfctl/pkg/cli"
	"github.com/opentdf/otdfctl/pkg/llm"
	"github.com/spf13/cobra"
)

var llmIngestSimpleCmd = &cobra.Command{
	Use:   "ingest-simple",
	Short: "Ingest OpenTDF documentation using simple keyword matching",
	Long:  "Ingest OpenTDF documentation into a simple keyword-based index for RAG (no embeddings required)",
	Run: func(cmd *cobra.Command, args []string) {
	c := cli.New(cmd, args)

	indexPath := c.Flags.GetOptionalString("index-path")
	sourcePath := c.Flags.GetOptionalString("path")

	// Set defaults
	if indexPath == "" {
		homeDir, _ := os.UserHomeDir()
		indexPath = filepath.Join(homeDir, ".otdfctl", "simple_rag_index.json")
	}
	if sourcePath == "" {
		sourcePath = "./docs-main"
	}

	c.Printf("🔧 Initializing Simple RAG document ingestion...\n")
	c.Printf("   Index path: %s\n", indexPath)
	c.Printf("   Source path: %s\n", sourcePath)

	// Initialize simple RAG store
	store := llm.NewSimpleRAGStore(indexPath)
	if err := store.LoadIndex(); err != nil {
		c.ExitWithError("Failed to load simple RAG index", err)
	}

	c.Printf("\n📚 Starting document ingestion...\n")

	totalProcessed := 0

	err := filepath.WalkDir(sourcePath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Only process markdown files
		if !d.IsDir() && strings.HasSuffix(strings.ToLower(path), ".md") {
			relPath, _ := filepath.Rel(sourcePath, path)
			c.Printf("Processing: %s\n", relPath)

			content, err := os.ReadFile(path)
			if err != nil {
				c.Printf("Warning: failed to read %s: %v\n", path, err)
				return nil
			}

			processed := processMarkdownSimple(string(content))
			if strings.TrimSpace(processed) == "" {
				return nil
			}

			// Generate document ID
			hash := sha256.Sum256([]byte(relPath))
			docID := hex.EncodeToString(hash[:])[:16]

			title := extractTitleSimple(string(content))
			if title == "" {
				title = filepath.Base(path)
			}

			doc := llm.SimpleDocument{
				ID:       docID,
				Title:    title,
				Content:  processed,
				URL:      "file://" + path,
				FilePath: relPath,
				Keywords: extractKeywordsSimple(processed),
			}

			if err := store.AddDocument(doc); err != nil {
				c.Printf("Warning: failed to add document to store: %v\n", err)
				return nil
			}

			totalProcessed++
		}

		return nil
	})

	if err != nil {
		c.ExitWithError("Failed to process documents", err)
	}

	// Save the updated index
	c.Printf("\n💾 Saving simple RAG index...\n")
	if err := store.SaveIndex(); err != nil {
		c.ExitWithError("Failed to save simple RAG index", err)
	}

	c.Printf("\n✅ Simple document ingestion completed successfully!\n")
	c.Printf("   Total documents: %d\n", totalProcessed)
	c.Printf("   Index saved to: %s\n", indexPath)
	},
}

// processMarkdownSimple cleans markdown content for simple text matching
func processMarkdownSimple(content string) string {
	// Remove YAML frontmatter
	frontmatterRegex := regexp.MustCompile(`(?s)^---\n.*?\n---\n`)
	content = frontmatterRegex.ReplaceAllString(content, "")

	// Remove code blocks but keep inline code
	codeBlockRegex := regexp.MustCompile("(?s)```.*?```")
	content = codeBlockRegex.ReplaceAllString(content, " [CODE_BLOCK] ")

	// Remove HTML tags
	htmlRegex := regexp.MustCompile(`<[^>]*>`)
	content = htmlRegex.ReplaceAllString(content, "")

	// Remove markdown links but keep text
	linkRegex := regexp.MustCompile(`\[([^\]]+)\]\([^)]+\)`)
	content = linkRegex.ReplaceAllString(content, "$1")

	// Remove image references
	imageRegex := regexp.MustCompile(`!\[[^\]]*\]\([^)]+\)`)
	content = imageRegex.ReplaceAllString(content, "")

	// Clean up markdown formatting
	content = regexp.MustCompile(`#{1,6}\s*`).ReplaceAllString(content, "") // Remove headers
	content = regexp.MustCompile(`\*{1,2}([^*]+)\*{1,2}`).ReplaceAllString(content, "$1") // Remove bold/italic
	content = regexp.MustCompile("`([^`]+)`").ReplaceAllString(content, "$1") // Remove inline code

	// Clean up whitespace
	content = regexp.MustCompile(`\n{3,}`).ReplaceAllString(content, "\n\n")
	content = regexp.MustCompile(`[ \t]+`).ReplaceAllString(content, " ")

	// Split into lines and clean each line
	var cleanLines []string
	scanner := bufio.NewScanner(strings.NewReader(content))

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "<!--") {
			cleanLines = append(cleanLines, line)
		}
	}

	return strings.Join(cleanLines, "\n")
}

// extractTitleSimple extracts the title from markdown content
func extractTitleSimple(content string) string {
	// Try to find first H1 header
	h1Regex := regexp.MustCompile(`(?m)^#\s+(.+)$`)
	if matches := h1Regex.FindStringSubmatch(content); len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// Try frontmatter title
	titleRegex := regexp.MustCompile(`(?m)^title:\s*(.+)$`)
	if matches := titleRegex.FindStringSubmatch(content); len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	return ""
}

// extractKeywordsSimple extracts keywords for basic search
func extractKeywordsSimple(content string) []string {
	// Simple keyword extraction
	words := strings.FieldsFunc(strings.ToLower(content), func(c rune) bool {
		return !((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))
	})

	keywordMap := make(map[string]int)
	for _, word := range words {
		if len(word) > 3 {
			keywordMap[word]++
		}
	}

	// Get most frequent words
	var keywords []string
	for word, count := range keywordMap {
		if count >= 2 { // Must appear at least twice
			keywords = append(keywords, word)
		}
	}

	return keywords
}

func init() {
	// TODO: Fix flag documentation parsing and use proper doc-driven flags
	llmIngestSimpleCmd.Flags().String("index-path", "", "Path to save simple RAG index (default: ~/.otdfctl/simple_rag_index.json)")
	llmIngestSimpleCmd.Flags().String("path", "./docs-main", "Path to local docs directory")

	// Add ingest-simple command to llm parent
	llmCmd.AddCommand(llmIngestSimpleCmd)
}