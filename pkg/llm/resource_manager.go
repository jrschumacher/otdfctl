package llm

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/opentdf/otdfctl/pkg/config"
)

// ResourceManager manages system resources and enforces limits
type ResourceManager struct {
	config           *config.ResourceLimits
	activeOperations int
	opMutex          sync.Mutex
	memoryBaseline   uint64
}

// NewResourceManager creates a new resource manager
func NewResourceManager(cfg *config.ResourceLimits) *ResourceManager {
	// Get baseline memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return &ResourceManager{
		config:         cfg,
		memoryBaseline: m.Alloc,
	}
}

// NewResourceManagerFromConfig creates a resource manager from LLM config
func NewResourceManagerFromConfig(cfg *config.Config) *ResourceManager {
	if cfg == nil {
		// Use secure defaults
		return NewResourceManager(&config.ResourceLimits{
			MaxMemoryMB:         2048,
			MaxFileSize:         10 * 1024 * 1024, // 10MB
			MaxFiles:            1000,
			ProcessTimeoutSec:   300, // 5 minutes
			InferenceTimeoutSec: 120, // 2 minutes
			MaxConcurrentOps:    3,
		})
	}
	return NewResourceManager(&cfg.LLM.ResourceLimits)
}

// CheckMemoryUsage validates current memory usage against limits
func (rm *ResourceManager) CheckMemoryUsage() error {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	// Convert to MB for comparison
	currentMB := (m.Alloc - rm.memoryBaseline) / 1024 / 1024
	maxMB := uint64(rm.config.MaxMemoryMB)
	
	if currentMB > maxMB {
		return fmt.Errorf("memory usage exceeded limit: %d MB (max: %d MB)", currentMB, maxMB)
	}
	
	return nil
}

// AcquireOperation tries to acquire a slot for a new operation
func (rm *ResourceManager) AcquireOperation() error {
	rm.opMutex.Lock()
	defer rm.opMutex.Unlock()
	
	if rm.activeOperations >= rm.config.MaxConcurrentOps {
		return fmt.Errorf("too many concurrent operations: %d (max: %d)", 
			rm.activeOperations, rm.config.MaxConcurrentOps)
	}
	
	rm.activeOperations++
	return nil
}

// ReleaseOperation releases an operation slot
func (rm *ResourceManager) ReleaseOperation() {
	rm.opMutex.Lock()
	defer rm.opMutex.Unlock()
	
	if rm.activeOperations > 0 {
		rm.activeOperations--
	}
}

// CreateInferenceContext creates a context with inference timeout
func (rm *ResourceManager) CreateInferenceContext() (context.Context, context.CancelFunc) {
	timeout := time.Duration(rm.config.InferenceTimeoutSec) * time.Second
	return context.WithTimeout(context.Background(), timeout)
}

// CreateProcessContext creates a context with process timeout
func (rm *ResourceManager) CreateProcessContext() (context.Context, context.CancelFunc) {
	timeout := time.Duration(rm.config.ProcessTimeoutSec) * time.Second
	return context.WithTimeout(context.Background(), timeout)
}

// ValidateFileSize checks if file size is within limits
func (rm *ResourceManager) ValidateFileSize(size int64) error {
	if size > rm.config.MaxFileSize {
		return fmt.Errorf("file size exceeds limit: %d bytes (max: %d)", size, rm.config.MaxFileSize)
	}
	return nil
}

// ValidateFileCount checks if file count is within limits
func (rm *ResourceManager) ValidateFileCount(count int) error {
	if count > rm.config.MaxFiles {
		return fmt.Errorf("file count exceeds limit: %d files (max: %d)", count, rm.config.MaxFiles)
	}
	return nil
}

// GetLimits returns the current resource limits for display
func (rm *ResourceManager) GetLimits() config.ResourceLimits {
	return *rm.config
}

// GetMemoryStatus returns current memory usage information
func (rm *ResourceManager) GetMemoryStatus() MemoryStatus {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	currentMB := (m.Alloc - rm.memoryBaseline) / 1024 / 1024
	maxMB := uint64(rm.config.MaxMemoryMB)
	
	return MemoryStatus{
		UsedMB:     currentMB,
		LimitMB:    maxMB,
		Percentage: float64(currentMB) / float64(maxMB) * 100,
	}
}

// GetOperationStatus returns current operation status
func (rm *ResourceManager) GetOperationStatus() OperationStatus {
	rm.opMutex.Lock()
	defer rm.opMutex.Unlock()
	
	return OperationStatus{
		Active: rm.activeOperations,
		Limit:  rm.config.MaxConcurrentOps,
	}
}

// MemoryStatus represents current memory usage
type MemoryStatus struct {
	UsedMB     uint64
	LimitMB    uint64
	Percentage float64
}

// OperationStatus represents current operation status
type OperationStatus struct {
	Active int
	Limit  int
}

// OperationWrapper wraps operations with resource management
type OperationWrapper struct {
	rm     *ResourceManager
	ctx    context.Context
	cancel context.CancelFunc
}

// NewOperationWrapper creates a new operation wrapper with resource management
func (rm *ResourceManager) NewOperationWrapper(operationType string) (*OperationWrapper, error) {
	// Check memory before starting
	if err := rm.CheckMemoryUsage(); err != nil {
		return nil, fmt.Errorf("memory check failed: %w", err)
	}
	
	// Acquire operation slot
	if err := rm.AcquireOperation(); err != nil {
		return nil, fmt.Errorf("failed to acquire operation slot: %w", err)
	}
	
	// Create appropriate context based on operation type
	var ctx context.Context
	var cancel context.CancelFunc
	
	switch operationType {
	case "inference":
		ctx, cancel = rm.CreateInferenceContext()
	case "process":
		ctx, cancel = rm.CreateProcessContext()
	default:
		ctx, cancel = rm.CreateProcessContext() // Default to process timeout
	}
	
	return &OperationWrapper{
		rm:     rm,
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

// Context returns the operation context
func (ow *OperationWrapper) Context() context.Context {
	return ow.ctx
}

// Close releases resources and cancels the operation
func (ow *OperationWrapper) Close() {
	if ow.cancel != nil {
		ow.cancel()
	}
	if ow.rm != nil {
		ow.rm.ReleaseOperation()
	}
}

// CheckProgress validates that the operation can continue
func (ow *OperationWrapper) CheckProgress() error {
	// Check if context is cancelled
	select {
	case <-ow.ctx.Done():
		return fmt.Errorf("operation cancelled or timed out: %w", ow.ctx.Err())
	default:
		// Continue
	}
	
	// Check memory usage
	return ow.rm.CheckMemoryUsage()
}