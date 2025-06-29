package llm

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"crypto/sha256"
	"encoding/hex"
)

// DocumentIngester handles downloading and processing OpenTDF documentation
type DocumentIngester struct {
	repoURL       string
	localCachDir  string
	vectorStore   *VectorStore
	embeddingEngine *EmbeddingEngine
	chunkSize     int
	chunkOverlap  int
}

// NewDocumentIngester creates a new document ingester
func NewDocumentIngester(vectorStore *VectorStore, embeddingEngine *EmbeddingEngine, cacheDir string) *DocumentIngester {
	return &DocumentIngester{
		repoURL:         "https://raw.githubusercontent.com/opentdf/docs/main",
		localCachDir:    cacheDir,
		vectorStore:     vectorStore,
		embeddingEngine: embeddingEngine,
		chunkSize:       300,  // words per chunk
		chunkOverlap:    50,   // overlapping words
	}
}

// IngestFromGitHub downloads and processes documentation from GitHub
func (di *DocumentIngester) IngestFromGitHub() error {
	log.Printf("Starting document ingestion from OpenTDF docs repository...")
	
	// List of important documentation files to ingest
	docFiles := []string{
		"README.md",
		"platform/README.md",
		"platform/getting-started.md",
		"platform/configuration.md",
		"platform/deployment.md",
		"platform/architecture.md",
		"platform/security.md",
		"sdk/README.md",
		"sdk/getting-started.md",
		"sdk/javascript.md",
		"sdk/python.md",
		"sdk/go.md",
		"sdk/java.md",
		"protocol/README.md",
		"protocol/tdf-spec.md",
		"protocol/kas.md",
		"protocol/policy.md",
		"protocol/attributes.md",
		"spec/README.md",
		"spec/ztdf.md",
		"spec/nano-tdf.md",
	}
	
	// Create cache directory
	if err := os.MkdirAll(di.localCachDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %v", err)
	}
	
	totalProcessed := 0
	
	for _, filePath := range docFiles {
		log.Printf("Processing: %s", filePath)
		
		doc, err := di.fetchAndProcessDocument(filePath)
		if err != nil {
			log.Printf("Warning: failed to process %s: %v", filePath, err)
			continue
		}
		
		if doc != nil {
			chunks := ChunkText(doc.Content, di.chunkSize, di.chunkOverlap)
			
			for i, chunk := range chunks {
				if strings.TrimSpace(chunk) == "" {
					continue
				}
				
				chunkDoc := Document{
					ID:          fmt.Sprintf("%s_chunk_%d", doc.ID, i),
					Title:       fmt.Sprintf("%s (Part %d/%d)", doc.Title, i+1, len(chunks)),
					Content:     chunk,
					URL:         doc.URL,
					FilePath:    doc.FilePath,
					ChunkIndex:  i,
					TotalChunks: len(chunks),
				}
				
				// Generate embedding for the chunk
				embedding, err := di.embeddingEngine.GenerateEmbedding(chunk)
				if err != nil {
					log.Printf("Warning: failed to generate embedding for %s chunk %d: %v", filePath, i, err)
					continue
				}
				
				chunkDoc.Embedding = embedding
				
				if err := di.vectorStore.AddDocument(chunkDoc); err != nil {
					log.Printf("Warning: failed to add document chunk to vector store: %v", err)
					continue
				}
				
				totalProcessed++
			}
		}
	}
	
	log.Printf("Successfully processed %d document chunks", totalProcessed)
	return nil
}

// fetchAndProcessDocument downloads and processes a single document
func (di *DocumentIngester) fetchAndProcessDocument(filePath string) (*Document, error) {
	url := fmt.Sprintf("%s/%s", di.repoURL, filePath)
	
	// Check cache first
	cacheFile := filepath.Join(di.localCachDir, strings.ReplaceAll(filePath, "/", "_"))
	
	var content string
	var err error
	
	if _, statErr := os.Stat(cacheFile); statErr == nil {
		// Load from cache
		data, err := os.ReadFile(cacheFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read cached file: %v", err)
		}
		content = string(data)
		log.Printf("Loaded from cache: %s", filePath)
	} else {
		// Download from GitHub
		content, err = di.downloadFile(url)
		if err != nil {
			return nil, fmt.Errorf("failed to download file: %v", err)
		}
		
		// Save to cache
		if err := os.WriteFile(cacheFile, []byte(content), 0644); err != nil {
			log.Printf("Warning: failed to cache file %s: %v", filePath, err)
		}
		
		log.Printf("Downloaded: %s", filePath)
		time.Sleep(100 * time.Millisecond) // Be nice to GitHub
	}
	
	// Process the markdown content
	processed := di.processMarkdown(content)
	if strings.TrimSpace(processed) == "" {
		return nil, fmt.Errorf("processed content is empty")
	}
	
	// Generate document ID
	hash := sha256.Sum256([]byte(filePath))
	docID := hex.EncodeToString(hash[:])[:16]
	
	// Extract title from content or use filename
	title := di.extractTitle(content)
	if title == "" {
		title = filepath.Base(filePath)
	}
	
	return &Document{
		ID:       docID,
		Title:    title,
		Content:  processed,
		URL:      url,
		FilePath: filePath,
	}, nil
}

// downloadFile downloads a file from a URL
func (di *DocumentIngester) downloadFile(url string) (string, error) {
	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	
	return string(body), nil
}

// processMarkdown cleans and processes markdown content for embedding
func (di *DocumentIngester) processMarkdown(content string) string {
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

// extractTitle extracts the title from markdown content
func (di *DocumentIngester) extractTitle(content string) string {
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

// IngestFromLocalDirectory ingests documentation from a local directory
func (di *DocumentIngester) IngestFromLocalDirectory(dirPath string) error {
	log.Printf("Starting document ingestion from local directory: %s", dirPath)
	
	totalProcessed := 0
	
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		// Only process markdown files
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".md") {
			relPath, _ := filepath.Rel(dirPath, path)
			log.Printf("Processing: %s", relPath)
			
			content, err := os.ReadFile(path)
			if err != nil {
				log.Printf("Warning: failed to read %s: %v", path, err)
				return nil
			}
			
			processed := di.processMarkdown(string(content))
			if strings.TrimSpace(processed) == "" {
				return nil
			}
			
			// Generate document ID
			hash := sha256.Sum256([]byte(relPath))
			docID := hex.EncodeToString(hash[:])[:16]
			
			title := di.extractTitle(string(content))
			if title == "" {
				title = filepath.Base(path)
			}
			
			doc := Document{
				ID:       docID,
				Title:    title,
				Content:  processed,
				URL:      fmt.Sprintf("file://%s", path),
				FilePath: relPath,
			}
			
			chunks := ChunkText(doc.Content, di.chunkSize, di.chunkOverlap)
			
			for i, chunk := range chunks {
				if strings.TrimSpace(chunk) == "" {
					continue
				}
				
				chunkDoc := Document{
					ID:          fmt.Sprintf("%s_chunk_%d", doc.ID, i),
					Title:       fmt.Sprintf("%s (Part %d/%d)", doc.Title, i+1, len(chunks)),
					Content:     chunk,
					URL:         doc.URL,
					FilePath:    doc.FilePath,
					ChunkIndex:  i,
					TotalChunks: len(chunks),
				}
				
				// Generate embedding for the chunk
				embedding, err := di.embeddingEngine.GenerateEmbedding(chunk)
				if err != nil {
					log.Printf("Warning: failed to generate embedding for %s chunk %d: %v", relPath, i, err)
					continue
				}
				
				chunkDoc.Embedding = embedding
				
				if err := di.vectorStore.AddDocument(chunkDoc); err != nil {
					log.Printf("Warning: failed to add document chunk to vector store: %v", err)
					continue
				}
				
				totalProcessed++
			}
		}
		
		return nil
	})
	
	if err != nil {
		return fmt.Errorf("failed to walk directory: %v", err)
	}
	
	log.Printf("Successfully processed %d document chunks from local directory", totalProcessed)
	return nil
}