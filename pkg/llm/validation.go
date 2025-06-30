package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// FileValidator provides secure file and path validation
type FileValidator struct {
	MaxFileSize   int64    // Maximum file size in bytes
	MaxFiles      int      // Maximum number of files to process
	AllowedExts   []string // Allowed file extensions
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
		`\.\.\/`,
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

// InputSanitizer provides general input sanitization
type InputSanitizer struct{}

// NewInputSanitizer creates a new input sanitizer
func NewInputSanitizer() *InputSanitizer {
	return &InputSanitizer{}
}

// SanitizeUserInput sanitizes user input for safe processing
func (is *InputSanitizer) SanitizeUserInput(input string) (string, error) {
	// Basic length check
	if len(input) > 10000 { // 10KB max for user input
		return "", fmt.Errorf("input too long: %d characters (max: 10000)", len(input))
	}
	
	// Remove null bytes and other dangerous characters
	cleaned := strings.ReplaceAll(input, "\x00", "")
	cleaned = strings.ReplaceAll(cleaned, "\r", "")
	
	// Check for injection attempts
	if err := is.checkForInjection(cleaned); err != nil {
		return "", err
	}
	
	return cleaned, nil
}

// checkForInjection detects potential injection attempts
func (is *InputSanitizer) checkForInjection(input string) error {
	// SQL injection patterns
	sqlPatterns := []string{
		`(?i)(union\s+select)`,
		`(?i)(drop\s+table)`,
		`(?i)(delete\s+from)`,
		`(?i)(insert\s+into)`,
		`(?i)(update\s+.*set)`,
	}
	
	// Command injection patterns
	cmdPatterns := []string{
		`[;&|]`,          // Command separators
		`\$\(.*\)`,       // Command substitution
		`\x60.*\x60`,     // Backticks
		`(?i)(rm\s+-rf)`, // Dangerous commands
		`(?i)(curl\s+)`,  // Network commands
		`(?i)(wget\s+)`,  // Network commands
	}
	
	allPatterns := append(sqlPatterns, cmdPatterns...)
	
	for _, pattern := range allPatterns {
		matched, err := regexp.MatchString(pattern, input)
		if err != nil {
			continue // Skip invalid regex
		}
		if matched {
			return fmt.Errorf("potential injection detected (pattern: %s)", pattern)
		}
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