package llm

import (
	"os"
	"strings"
	"testing"

	"github.com/opentdf/otdfctl/pkg/config"
)

func TestPathTraversalPrevention(t *testing.T) {
	validator := NewFileValidator()
	
	// Create a temporary test file in current directory (safe location)
	tmpFile, err := os.CreateTemp(".", "test_*.md")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.WriteString("# Test content\nThis is a test file.")
	tmpFile.Close()
	
	tests := []struct {
		name     string
		path     string
		wantErr  bool
		errMsg   string
	}{
		{
			name:    "valid file",
			path:    tmpFile.Name(),
			wantErr: false,
		},
		{
			name:    "path traversal with ../",
			path:    "../../../etc/passwd",
			wantErr: true,
			errMsg:  "path traversal detected",
		},
		{
			name:    "path traversal with ..\\",
			path:    "..\\..\\windows\\system32",
			wantErr: true,
			errMsg:  "path traversal detected",
		},
		{
			name:    "forbidden system directory",
			path:    "/etc/passwd",
			wantErr: true,
			errMsg:  "forbidden path",
		},
		{
			name:    "invalid extension",
			path:    "test.exe",
			wantErr: true,
			errMsg:  "forbidden file extension",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateFile(tt.path)
			
			if tt.wantErr {
				if err == nil {
					t.Errorf("ValidateFile() expected error for %s, got nil", tt.path)
					return
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("ValidateFile() error = %v, want error containing %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("ValidateFile() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestInputSanitization(t *testing.T) {
	sanitizer := NewInputSanitizer()
	
	tests := []struct {
		name    string
		input   string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "normal text",
			input:   "This is normal markdown content",
			wantErr: false,
		},
		{
			name:    "command injection",
			input:   "rm -rf /",
			wantErr: true,
			errMsg:  "potential injection detected",
		},
		{
			name:    "sql injection",
			input:   "DROP TABLE users",
			wantErr: true,
			errMsg:  "potential injection detected",
		},
		{
			name:    "script tag",
			input:   "<script>alert('xss')</script>",
			wantErr: false, // Should be sanitized, not rejected
		},
		{
			name:    "too long input",
			input:   strings.Repeat("a", 20000),
			wantErr: true,
			errMsg:  "input too long",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := sanitizer.SanitizeUserInput(tt.input)
			
			if tt.wantErr {
				if err == nil {
					t.Errorf("SanitizeUserInput() expected error for %s, got nil", tt.name)
					return
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("SanitizeUserInput() error = %v, want error containing %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("SanitizeUserInput() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestURLValidation(t *testing.T) {
	validator := NewURLValidator()
	
	tests := []struct {
		name    string
		url     string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "valid https URL",
			url:     "https://github.com/opentdf/docs",
			wantErr: false,
		},
		{
			name:    "valid http URL",
			url:     "http://example.com",
			wantErr: false,
		},
		{
			name:    "localhost blocked",
			url:     "https://localhost:8080",
			wantErr: true,
			errMsg:  "local/private networks is forbidden",
		},
		{
			name:    "private IP blocked",
			url:     "https://192.168.1.1",
			wantErr: true,
			errMsg:  "local/private networks is forbidden",
		},
		{
			name:    "invalid protocol",
			url:     "ftp://example.com",
			wantErr: true,
			errMsg:  "invalid protocol",
		},
		{
			name:    "empty URL",
			url:     "",
			wantErr: true,
			errMsg:  "URL is empty",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validator.ValidateURL(tt.url)
			
			if tt.wantErr {
				if err == nil {
					t.Errorf("ValidateURL() expected error for %s, got nil", tt.url)
					return
				}
				if !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("ValidateURL() error = %v, want error containing %s", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("ValidateURL() unexpected error = %v", err)
				}
			}
		})
	}
}

func TestResourceManager(t *testing.T) {
	cfg := &config.ResourceLimits{
		MaxMemoryMB:         100, // Small limit for testing
		MaxConcurrentOps:    2,
		ProcessTimeoutSec:   1,
		InferenceTimeoutSec: 1,
		MaxFileSize:         1024,
		MaxFiles:            10,
	}
	
	rm := NewResourceManager(cfg)
	
	// Test operation acquisition
	err1 := rm.AcquireOperation()
	if err1 != nil {
		t.Errorf("First AcquireOperation() failed: %v", err1)
	}
	
	err2 := rm.AcquireOperation()
	if err2 != nil {
		t.Errorf("Second AcquireOperation() failed: %v", err2)
	}
	
	// Third should fail
	err3 := rm.AcquireOperation()
	if err3 == nil {
		t.Error("Third AcquireOperation() should have failed")
	}
	
	// Release one and try again
	rm.ReleaseOperation()
	err4 := rm.AcquireOperation()
	if err4 != nil {
		t.Errorf("AcquireOperation() after release failed: %v", err4)
	}
	
	// Test file size validation
	err := rm.ValidateFileSize(2048) // Exceeds limit
	if err == nil {
		t.Error("ValidateFileSize() should have failed for oversized file")
	}
	
	err = rm.ValidateFileSize(512) // Within limit
	if err != nil {
		t.Errorf("ValidateFileSize() failed for valid size: %v", err)
	}
}

func TestModelPathResolver(t *testing.T) {
	// Create temporary config
	cfg := &config.Config{
		LLM: config.LLM{
			DefaultModelPath:          "/tmp/test-model.gguf",
			DefaultEmbeddingModelPath: "/tmp/test-embed.gguf",
		},
	}
	
	resolver := NewModelPathResolver(cfg)
	
	// Test with empty user path - should use config
	_, err := resolver.ResolveChatModelPath("")
	// This will fail because file doesn't exist, but should not be path traversal error
	if err != nil && strings.Contains(err.Error(), "path traversal") {
		t.Errorf("ResolveChatModelPath() should not fail with path traversal for config path: %v", err)
	}
	
	// Test path traversal prevention
	_, err = resolver.ResolveChatModelPath("../../../etc/passwd")
	if err == nil || !strings.Contains(err.Error(), "path traversal") {
		t.Error("ResolveChatModelPath() should prevent path traversal")
	}
}

func TestValidateFileContent(t *testing.T) {
	validator := NewFileValidator()
	
	// Test normal content
	err := validator.ValidateFileContent("# Normal markdown\n\nThis is fine.")
	if err != nil {
		t.Errorf("ValidateFileContent() failed for normal content: %v", err)
	}
	
	// Test suspicious content
	suspiciousContent := "<script>alert('xss')</script>"
	err = validator.ValidateFileContent(suspiciousContent)
	if err == nil {
		t.Error("ValidateFileContent() should detect suspicious script content")
	}
	
	// Test oversized content
	largeContent := strings.Repeat("a", 20*1024*1024) // 20MB
	err = validator.ValidateFileContent(largeContent)
	if err == nil {
		t.Error("ValidateFileContent() should reject oversized content")
	}
}