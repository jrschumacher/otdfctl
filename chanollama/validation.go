package chanollama

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// InputValidator provides input sanitization and validation for LLM interactions
type InputValidator struct {
	MaxLength      int      // Maximum input length in characters
	MinLength      int      // Minimum input length
	ForbiddenWords []string // Words/phrases to reject
	AllowMarkdown  bool     // Whether to allow basic markdown
	StripHTML      bool     // Whether to strip HTML tags
}

// NewInputValidator creates a validator with secure defaults
func NewInputValidator() *InputValidator {
	return &InputValidator{
		MaxLength:      10000, // 10k characters max
		MinLength:      1,     // At least 1 character
		ForbiddenWords: []string{}, // Can be configured per application
		AllowMarkdown:  true,  // Allow basic markdown
		StripHTML:      true,  // Strip HTML by default
	}
}

// WithMaxLength sets the maximum input length
func (iv *InputValidator) WithMaxLength(length int) *InputValidator {
	iv.MaxLength = length
	return iv
}

// WithMinLength sets the minimum input length
func (iv *InputValidator) WithMinLength(length int) *InputValidator {
	iv.MinLength = length
	return iv
}

// WithForbiddenWords sets words/phrases to reject
func (iv *InputValidator) WithForbiddenWords(words []string) *InputValidator {
	iv.ForbiddenWords = words
	return iv
}

// WithMarkdownAllowed controls markdown support
func (iv *InputValidator) WithMarkdownAllowed(allowed bool) *InputValidator {
	iv.AllowMarkdown = allowed
	return iv
}

// WithHTMLStripping controls HTML stripping
func (iv *InputValidator) WithHTMLStripping(strip bool) *InputValidator {
	iv.StripHTML = strip
	return iv
}

// ValidateInput validates and sanitizes user input
func (iv *InputValidator) ValidateInput(input string) (string, error) {
	if input == "" {
		return "", fmt.Errorf("input cannot be empty")
	}
	
	// Step 1: Basic length validation
	if len(input) < iv.MinLength {
		return "", fmt.Errorf("input too short (minimum %d characters)", iv.MinLength)
	}
	
	if len(input) > iv.MaxLength {
		return "", fmt.Errorf("input too long (maximum %d characters)", iv.MaxLength)
	}
	
	// Step 2: UTF-8 validation
	if !utf8.ValidString(input) {
		return "", fmt.Errorf("input contains invalid UTF-8 characters")
	}
	
	// Step 3: Control character filtering
	sanitized := iv.removeControlCharacters(input)
	
	// Step 4: HTML stripping if enabled
	if iv.StripHTML {
		sanitized = iv.stripHTML(sanitized)
	}
	
	// Step 5: Forbidden words check
	if err := iv.checkForbiddenWords(sanitized); err != nil {
		return "", err
	}
	
	// Step 6: Final length check after sanitization
	if len(sanitized) < iv.MinLength {
		return "", fmt.Errorf("input too short after sanitization")
	}
	
	return strings.TrimSpace(sanitized), nil
}

// ValidateMessage validates a complete message
func (iv *InputValidator) ValidateMessage(msg Message) (Message, error) {
	// Validate role
	if msg.Role != "system" && msg.Role != "user" && msg.Role != "assistant" {
		return Message{}, fmt.Errorf("invalid message role: %s", msg.Role)
	}
	
	// Validate and sanitize content
	sanitizedContent, err := iv.ValidateInput(msg.Content)
	if err != nil {
		return Message{}, fmt.Errorf("invalid message content: %w", err)
	}
	
	return Message{
		Role:    msg.Role,
		Content: sanitizedContent,
	}, nil
}

// ValidateMessages validates a slice of messages
func (iv *InputValidator) ValidateMessages(messages []Message) ([]Message, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("message list cannot be empty")
	}
	
	var validated []Message
	for i, msg := range messages {
		validatedMsg, err := iv.ValidateMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("message %d: %w", i, err)
		}
		validated = append(validated, validatedMsg)
	}
	
	return validated, nil
}

// removeControlCharacters removes dangerous control characters
func (iv *InputValidator) removeControlCharacters(input string) string {
	var result strings.Builder
	
	for _, r := range input {
		// Allow normal whitespace characters
		if unicode.IsSpace(r) {
			result.WriteRune(r)
			continue
		}
		
		// Allow printable characters
		if unicode.IsPrint(r) {
			result.WriteRune(r)
			continue
		}
		
		// Skip control characters
	}
	
	return result.String()
}

// stripHTML removes HTML tags and entities
func (iv *InputValidator) stripHTML(input string) string {
	// Remove HTML tags
	htmlTagRegex := regexp.MustCompile(`<[^>]*>`)
	result := htmlTagRegex.ReplaceAllString(input, "")
	
	// Decode common HTML entities
	result = strings.ReplaceAll(result, "&lt;", "<")
	result = strings.ReplaceAll(result, "&gt;", ">")
	result = strings.ReplaceAll(result, "&amp;", "&")
	result = strings.ReplaceAll(result, "&quot;", "\"")
	result = strings.ReplaceAll(result, "&#39;", "'")
	result = strings.ReplaceAll(result, "&nbsp;", " ")
	
	return result
}

// checkForbiddenWords checks for forbidden words/phrases
func (iv *InputValidator) checkForbiddenWords(input string) error {
	lowerInput := strings.ToLower(input)
	
	for _, word := range iv.ForbiddenWords {
		if strings.Contains(lowerInput, strings.ToLower(word)) {
			return fmt.Errorf("input contains forbidden content")
		}
	}
	
	return nil
}

// FileValidator provides secure file and path validation
type FileValidator struct {
	MaxFileSize    int64    // Maximum file size in bytes
	MaxFiles       int      // Maximum number of files to process
	AllowedExts    []string // Allowed file extensions
	ForbiddenPaths []string // Paths that should never be accessed
}

// NewFileValidator creates a validator with secure defaults
func NewFileValidator() *FileValidator {
	return &FileValidator{
		MaxFileSize: 10 * 1024 * 1024, // 10MB default
		MaxFiles:    1000,              // 1000 files max
		AllowedExts: []string{".md", ".txt", ".rst"}, // Only documentation files
		ForbiddenPaths: []string{
			"/etc", "/var", "/usr", "/bin", "/sbin", "/root",
			"/proc", "/sys", "/dev", "/boot",
			"C:\\Windows", "C:\\Program Files", "C:\\Users",
		},
	}
}

// WithMaxFileSize sets the maximum file size
func (fv *FileValidator) WithMaxFileSize(size int64) *FileValidator {
	fv.MaxFileSize = size
	return fv
}

// WithMaxFiles sets the maximum number of files
func (fv *FileValidator) WithMaxFiles(count int) *FileValidator {
	fv.MaxFiles = count
	return fv
}

// WithAllowedExtensions sets allowed file extensions
func (fv *FileValidator) WithAllowedExtensions(exts []string) *FileValidator {
	fv.AllowedExts = exts
	return fv
}

// ValidateFile validates a single file path and content
func (fv *FileValidator) ValidateFile(path string) error {
	// Step 1: Path traversal protection
	cleanPath := filepath.Clean(path)
	if strings.Contains(cleanPath, "..") {
		return fmt.Errorf("path traversal detected: %s", path)
	}
	
	// Step 2: Absolute path resolution and validation
	absPath, err := filepath.Abs(cleanPath)
	if err != nil {
		return fmt.Errorf("failed to resolve absolute path: %w", err)
	}
	
	// Step 3: Forbidden path protection
	if err := fv.checkForbiddenPaths(absPath); err != nil {
		return err
	}
	
	// Step 4: File extension validation
	if err := fv.checkFileExtension(absPath); err != nil {
		return err
	}
	
	// Step 5: File existence and accessibility
	if err := fv.checkFileAccess(absPath); err != nil {
		return err
	}
	
	// Step 6: File size validation
	if err := fv.checkFileSize(absPath); err != nil {
		return err
	}
	
	return nil
}

// ValidateDirectory validates a directory for safe traversal
func (fv *FileValidator) ValidateDirectory(dirPath string) error {
	// Clean and resolve path
	cleanPath := filepath.Clean(dirPath)
	if strings.Contains(cleanPath, "..") {
		return fmt.Errorf("path traversal detected in directory: %s", dirPath)
	}
	
	absPath, err := filepath.Abs(cleanPath)
	if err != nil {
		return fmt.Errorf("failed to resolve directory path: %w", err)
	}
	
	// Check forbidden paths
	if err := fv.checkForbiddenPaths(absPath); err != nil {
		return err
	}
	
	// Verify directory exists and is accessible
	info, err := os.Stat(absPath)
	if err != nil {
		return fmt.Errorf("cannot access directory: %w", err)
	}
	
	if !info.IsDir() {
		return fmt.Errorf("path is not a directory: %s", absPath)
	}
	
	return nil
}

// CountFilesInDirectory counts files in directory with validation
func (fv *FileValidator) CountFilesInDirectory(dirPath string) (int, error) {
	if err := fv.ValidateDirectory(dirPath); err != nil {
		return 0, err
	}
	
	count := 0
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err // Stop on any error
		}
		
		if info.IsDir() {
			return nil // Skip directories
		}
		
		// Check if file extension is allowed
		if fv.IsAllowedExtension(filepath.Ext(path)) {
			count++
			
			// Stop if we exceed max files
			if count > fv.MaxFiles {
				return fmt.Errorf("directory contains too many files (>%d)", fv.MaxFiles)
			}
		}
		
		return nil
	})
	
	return count, err
}

// ValidateFileContent validates file content for safety
func (fv *FileValidator) ValidateFileContent(content string) error {
	// Check for suspicious content patterns
	suspiciousPatterns := []string{
		// Potential script execution
		`<script`,
		`javascript:`,
		`data:text/html`,
		// Command injection attempts
		`$(`,
		`\${`,
		`eval\(`,
		// Path traversal in content
		`\.\./`,
		`\.\.\\`,
	}
	
	lowerContent := strings.ToLower(content)
	for _, pattern := range suspiciousPatterns {
		if strings.Contains(lowerContent, pattern) {
			return fmt.Errorf("suspicious content pattern detected: %s", pattern)
		}
	}
	
	// Content size validation
	if len(content) > int(fv.MaxFileSize) {
		return fmt.Errorf("content too large: %d bytes (max: %d)", len(content), fv.MaxFileSize)
	}
	
	return nil
}

// checkForbiddenPaths ensures we don't access system directories
func (fv *FileValidator) checkForbiddenPaths(absPath string) error {
	for _, forbidden := range fv.ForbiddenPaths {
		// Check if path starts with forbidden directory
		if strings.HasPrefix(strings.ToLower(absPath), strings.ToLower(forbidden)) {
			return fmt.Errorf("access to forbidden path: %s (matches: %s)", absPath, forbidden)
		}
	}
	return nil
}

// checkFileExtension validates file extension against allowed list
func (fv *FileValidator) checkFileExtension(path string) error {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == "" {
		return fmt.Errorf("file has no extension: %s", path)
	}
	
	if !fv.IsAllowedExtension(ext) {
		return fmt.Errorf("forbidden file extension: %s (allowed: %v)", ext, fv.AllowedExts)
	}
	
	return nil
}

// IsAllowedExtension checks if extension is in allowed list
func (fv *FileValidator) IsAllowedExtension(ext string) bool {
	ext = strings.ToLower(ext)
	for _, allowed := range fv.AllowedExts {
		if ext == strings.ToLower(allowed) {
			return true
		}
	}
	return false
}

// checkFileAccess verifies file exists and is readable
func (fv *FileValidator) checkFileAccess(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("file does not exist: %s", path)
		}
		return fmt.Errorf("cannot access file: %w", err)
	}
	
	if info.IsDir() {
		return fmt.Errorf("path is a directory, not a file: %s", path)
	}
	
	// Test file readability
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("file is not readable: %w", err)
	}
	file.Close()
	
	return nil
}

// checkFileSize validates file size against limits
func (fv *FileValidator) checkFileSize(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("cannot stat file: %w", err)
	}
	
	if info.Size() == 0 {
		return fmt.Errorf("file is empty: %s", path)
	}
	
	if info.Size() > fv.MaxFileSize {
		return fmt.Errorf("file too large: %d bytes (max: %d)", info.Size(), fv.MaxFileSize)
	}
	
	return nil
}

// URLValidator validates URLs for safety
type URLValidator struct{}

// NewURLValidator creates a new URL validator
func NewURLValidator() *URLValidator {
	return &URLValidator{}
}

// ValidateURL validates a URL for safe access
func (uv *URLValidator) ValidateURL(url string) error {
	// Basic URL format validation
	if url == "" {
		return fmt.Errorf("URL is empty")
	}
	
	// Length check
	if len(url) > 2048 {
		return fmt.Errorf("URL too long: %d characters (max: 2048)", len(url))
	}
	
	// Protocol validation
	if !strings.HasPrefix(url, "https://") && !strings.HasPrefix(url, "http://") {
		return fmt.Errorf("invalid protocol: URL must start with http:// or https://")
	}
	
	// Block local/private network access
	forbiddenPatterns := []string{
		`://localhost`,
		`://127\.`,
		`://192\.168\.`,
		`://10\.`,
		`://172\.(1[6-9]|2[0-9]|3[01])\.`,
		`://169\.254\.`, // Link-local
		`://0\.`,
		`://::1`,       // IPv6 localhost
	}
	
	for _, pattern := range forbiddenPatterns {
		matched, _ := regexp.MatchString(pattern, strings.ToLower(url))
		if matched {
			return fmt.Errorf("access to local/private networks is forbidden")
		}
	}
	
	return nil
}

// SafeInputSanitizer provides basic input sanitization for common use cases
type SafeInputSanitizer struct {
	validator *InputValidator
}

// NewSafeInputSanitizer creates a sanitizer with safe defaults
func NewSafeInputSanitizer() *SafeInputSanitizer {
	validator := NewInputValidator().
		WithMaxLength(5000).   // 5k character limit
		WithMinLength(1).      // At least 1 character
		WithHTMLStripping(true). // Strip HTML
		WithMarkdownAllowed(true) // Allow markdown
	
	return &SafeInputSanitizer{
		validator: validator,
	}
}

// SanitizeUserInput sanitizes user input for safe LLM processing
func (sis *SafeInputSanitizer) SanitizeUserInput(input string) (string, error) {
	return sis.validator.ValidateInput(input)
}

// SanitizeMessage sanitizes a complete message
func (sis *SafeInputSanitizer) SanitizeMessage(msg Message) (Message, error) {
	return sis.validator.ValidateMessage(msg)
}

// SanitizeMessages sanitizes a slice of messages
func (sis *SafeInputSanitizer) SanitizeMessages(messages []Message) ([]Message, error) {
	return sis.validator.ValidateMessages(messages)
}

// ContentFilter provides additional content filtering capabilities
type ContentFilter struct {
	BlockProfanity   bool
	BlockPII         bool // Personally Identifiable Information
	BlockCodeInjection bool
}

// NewContentFilter creates a filter with security defaults
func NewContentFilter() *ContentFilter {
	return &ContentFilter{
		BlockProfanity:      false, // Application-specific
		BlockPII:           true,   // Block PII by default
		BlockCodeInjection: true,   // Block potential code injection
	}
}

// FilterContent applies content filtering rules
func (cf *ContentFilter) FilterContent(input string) (string, error) {
	result := input
	
	// Check for potential code injection patterns
	if cf.BlockCodeInjection {
		if err := cf.checkCodeInjection(result); err != nil {
			return "", err
		}
	}
	
	// Check for PII patterns
	if cf.BlockPII {
		result = cf.maskPII(result)
	}
	
	return result, nil
}

// checkCodeInjection looks for potential code injection patterns
func (cf *ContentFilter) checkCodeInjection(input string) error {
	// Basic patterns that might indicate code injection attempts
	dangerousPatterns := []string{
		"<script",
		"javascript:",
		"eval(",
		"function(",
		"import ",
		"require(",
		"exec(",
		"system(",
		"os.system",
		"subprocess",
		"shell_exec",
	}
	
	lowerInput := strings.ToLower(input)
	for _, pattern := range dangerousPatterns {
		if strings.Contains(lowerInput, pattern) {
			return fmt.Errorf("potentially dangerous content detected")
		}
	}
	
	return nil
}

// maskPII attempts to mask common PII patterns
func (cf *ContentFilter) maskPII(input string) string {
	// Email pattern
	emailRegex := regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`)
	result := emailRegex.ReplaceAllString(input, "[EMAIL]")
	
	// Phone number pattern (basic)
	phoneRegex := regexp.MustCompile(`\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`)
	result = phoneRegex.ReplaceAllString(result, "[PHONE]")
	
	// SSN pattern (basic)
	ssnRegex := regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)
	result = ssnRegex.ReplaceAllString(result, "[SSN]")
	
	return result
}