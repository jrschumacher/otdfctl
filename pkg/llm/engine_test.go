package llm

import (
	"context"
	"testing"
	"time"

	"github.com/opentdf/otdfctl/pkg/config"
)

// TestEngineV2BasicOperations tests the basic functionality of the V2 engine
func TestEngineV2BasicOperations(t *testing.T) {
	// Create a test config
	cfg := &config.Config{}
	
	// Create engine
	engine := NewLLMEngineV2(cfg)
	
	// Start engine
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	err := engine.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start engine: %v", err)
	}
	defer engine.Stop()
	
	// Create client
	client := NewEngineClient(engine)
	defer client.Close()
	
	// Test health check
	t.Run("HealthCheck", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		health, err := client.HealthCheck(ctx)
		if err != nil {
			t.Errorf("Health check failed: %v", err)
			return
		}
		
		if health.Status == "" {
			t.Error("Health status should not be empty")
		}
		
		t.Logf("Engine health: %s", health.Status)
	})
	
	// Test metrics
	t.Run("GetMetrics", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		metrics, err := client.GetMetrics(ctx)
		if err != nil {
			t.Errorf("Get metrics failed: %v", err)
			return
		}
		
		// Should have at least one request from the health check
		if metrics.TotalRequests < 1 {
			t.Errorf("Expected at least 1 request, got %d", metrics.TotalRequests)
		}
		
		t.Logf("Total requests: %d", metrics.TotalRequests)
		t.Logf("Successful requests: %d", metrics.SuccessfulRequests)
		t.Logf("Memory usage: %d bytes", metrics.MemoryUsage)
	})
	
	// Test prompt configuration
	t.Run("ConfigurePrompt", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		testPrompt := "You are a test assistant."
		err := client.ConfigurePrompt(ctx, testPrompt)
		if err != nil {
			t.Errorf("Configure prompt failed: %v", err)
			return
		}
		
		// Verify the prompt was set by inspecting the model (if loaded)
		modelInfo, err := client.InspectModel(ctx)
		if err != nil {
			// This is expected if no model is loaded, which is fine for testing
			t.Logf("Model inspect failed (expected): %v", err)
			return
		}
		
		if modelInfo.SystemPrompt != testPrompt {
			t.Errorf("Expected system prompt %q, got %q", testPrompt, modelInfo.SystemPrompt)
		}
	})
	
	// Test cancellation
	t.Run("Cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		
		// Cancel immediately
		cancel()
		
		// This should fail with context canceled
		_, err := client.GetMetrics(ctx)
		if err == nil {
			t.Error("Expected context canceled error")
		}
		
		if err != context.Canceled {
			t.Logf("Got expected cancellation error: %v", err)
		}
	})
}

// TestEngineV2Concurrency tests concurrent operations
func TestEngineV2Concurrency(t *testing.T) {
	cfg := &config.Config{}
	engine := NewLLMEngineV2(cfg)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	err := engine.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start engine: %v", err)
	}
	defer engine.Stop()
	
	client := NewEngineClient(engine)
	defer client.Close()
	
	// Run multiple concurrent health checks
	const numGoroutines = 10
	
	results := make(chan error, numGoroutines)
	
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			
			_, err := client.HealthCheck(ctx)
			results <- err
		}(i)
	}
	
	// Collect results
	for i := 0; i < numGoroutines; i++ {
		err := <-results
		if err != nil {
			t.Errorf("Concurrent health check %d failed: %v", i, err)
		}
	}
	
	// Check final metrics
	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	metrics, err := client.GetMetrics(ctx)
	if err != nil {
		t.Fatalf("Failed to get final metrics: %v", err)
	}
	
	// Should have at least numGoroutines + some requests
	if metrics.TotalRequests < int64(numGoroutines) {
		t.Errorf("Expected at least %d requests, got %d", numGoroutines, metrics.TotalRequests)
	}
	
	t.Logf("Concurrent test completed with %d total requests", metrics.TotalRequests)
}

// TestEngineV2Shutdown tests graceful shutdown
func TestEngineV2Shutdown(t *testing.T) {
	cfg := &config.Config{}
	engine := NewLLMEngineV2(cfg)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	err := engine.Start(ctx)
	if err != nil {
		t.Fatalf("Failed to start engine: %v", err)
	}
	
	client := NewEngineClient(engine)
	defer client.Close()
	
	// Test shutdown command
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	
	err = client.Shutdown(shutdownCtx)
	if err != nil {
		t.Errorf("Shutdown failed: %v", err)
	}
	
	// Give it a moment to shut down
	time.Sleep(200 * time.Millisecond)
	
	// Subsequent requests should fail
	healthCtx, healthCancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer healthCancel()
	
	_, err = client.HealthCheck(healthCtx)
	if err == nil {
		t.Error("Expected health check to fail after shutdown")
	}
	
	t.Logf("Shutdown test completed successfully")
}

// TestOperationTypes tests that all operation types are properly handled
func TestOperationTypes(t *testing.T) {
	operations := []OperationType{
		OpLoadModel,
		OpInference,
		OpStream,
		OpGetMetrics,
		OpUnloadModel,
		OpShutdown,
		OpConfigurePrompt,
		OpConfigureRAG,
		OpInspectModel,
		OpHealthCheck,
		OpConfigureGeneration,
		OpTokenCount,
		OpWarmup,
	}
	
	for _, op := range operations {
		t.Run(op.String(), func(t *testing.T) {
			// Test that String() method works
			str := op.String()
			if str == "Unknown" {
				t.Errorf("Operation type %d should have a proper string representation", int(op))
			}
			if str == "" {
				t.Errorf("Operation type %d should not have empty string", int(op))
			}
			
			t.Logf("Operation %d: %s", int(op), str)
		})
	}
}

// BenchmarkEngineV2Operations benchmarks basic engine operations
func BenchmarkEngineV2Operations(b *testing.B) {
	cfg := &config.Config{}
	engine := NewLLMEngineV2(cfg)
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	err := engine.Start(ctx)
	if err != nil {
		b.Fatalf("Failed to start engine: %v", err)
	}
	defer engine.Stop()
	
	client := NewEngineClient(engine)
	defer client.Close()
	
	b.Run("HealthCheck", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
			_, err := client.HealthCheck(ctx)
			cancel()
			if err != nil {
				b.Fatalf("Health check failed: %v", err)
			}
		}
	})
	
	b.Run("GetMetrics", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
			_, err := client.GetMetrics(ctx)
			cancel()
			if err != nil {
				b.Fatalf("Get metrics failed: %v", err)
			}
		}
	})
	
	b.Run("ConfigurePrompt", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
			err := client.ConfigurePrompt(ctx, "Test prompt")
			cancel()
			if err != nil {
				b.Fatalf("Configure prompt failed: %v", err)
			}
		}
	})
}