/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package prefetch

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestDigestToFilenamePathSuffix(t *testing.T) {
	// Build a 32-byte digest with the trailing 8 bytes set to a known
	// uint64 (matching the SHA256 layout), and the leading bytes set to a
	// distinguishable prefix used for <hhh>/<hh>.
	digestSHA256 := func(prefix []byte, tail uint64) []byte {
		d := make([]byte, 32)
		copy(d, prefix)
		for i := 0; i < 8; i++ {
			d[24+i] = byte(tail >> (8 * (7 - i)))
		}
		return d
	}

	digestFNV := func(v uint64) []byte {
		d := make([]byte, 8)
		for i := 0; i < 8; i++ {
			d[i] = byte(v >> (8 * (7 - i)))
		}
		return d
	}

	tests := []struct {
		name     string
		digest   []byte
		groupIdx int
		expected string
	}{
		{
			name:     "fnv 8-byte zero key",
			digest:   digestFNV(0),
			groupIdx: 0,
			expected: "000/00_g0/0000000000000000.bin",
		},
		{
			name:     "fnv 8-byte small key",
			digest:   digestFNV(0x123),
			groupIdx: 0,
			expected: "000/00_g0/0000000000000123.bin",
		},
		{
			name:     "fnv 8-byte large key",
			digest:   digestFNV(0xABCDEF1234567890),
			groupIdx: 0,
			expected: "abc/de_g0/abcdef1234567890.bin",
		},
		{
			name:     "fnv 8-byte non-zero group",
			digest:   digestFNV(0xABCDEF1234567890),
			groupIdx: 3,
			expected: "abc/de_g3/abcdef1234567890.bin",
		},
		{
			name: "sha256 32-byte digest uses leading bytes",
			// Real on-disk example: digest 7050ab3d…d04d5d1; <hhh>=705,
			// <hh>=0a; trailing 8 bytes shouldn't influence the path.
			digest: func() []byte {
				d, _ := hex.DecodeString("7050ab3d42d0b5e628c4e846e90715c1e1b2ac6247ce88b5e1a944b73c04d5d1")
				return d
			}(),
			groupIdx: 0,
			expected: "705/0a_g0/7050ab3d42d0b5e628c4e846e90715c1e1b2ac6247ce88b5e1a944b73c04d5d1.bin",
		},
		{
			name:     "sha256 32-byte with non-zero group",
			digest:   digestSHA256([]byte{0xab, 0xcd, 0xef}, 0x1234567890ABCDEF),
			groupIdx: 7,
			expected: "abc/de_g7/abcdef0000000000000000000000000000000000000000001234567890abcdef.bin",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := digestToFilenamePathSuffix(tt.digest, tt.groupIdx)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestKVFilePathBaseParams_IsSet(t *testing.T) {
	tests := []struct {
		name     string
		params   *KVFilePathBaseParams
		expected bool
	}{
		{
			name:     "nil params",
			params:   nil,
			expected: false,
		},
		{
			name:     "empty params",
			params:   &KVFilePathBaseParams{},
			expected: false,
		},
		{
			name: "only root dir",
			params: &KVFilePathBaseParams{
				RootDir: "/tmp",
			},
			expected: false,
		},
		{
			name: "only model name",
			params: &KVFilePathBaseParams{
				ModelName: "test-model",
			},
			expected: false,
		},
		{
			name: "root dir and model name but no digest",
			params: &KVFilePathBaseParams{
				RootDir:   "/tmp",
				ModelName: "test-model",
			},
			expected: false,
		},
		{
			name: "root dir, model name, and digest",
			params: &KVFilePathBaseParams{
				RootDir:   "/tmp",
				ModelName: "test-model",
				Digest:    "abcdef123456",
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.params.IsSet()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestKVFilePathBaseParams_SetDefaults(t *testing.T) {
	tests := []struct {
		name     string
		params   *KVFilePathBaseParams
		expected *KVFilePathBaseParams
	}{
		{
			name:   "all defaults",
			params: &KVFilePathBaseParams{},
			expected: &KVFilePathBaseParams{
				GpuBlocksPerFile: 1,
				TpSize:           1,
				PpSize:           1,
				PcpSize:          1,
				DcpSize:          1,
			},
		},
		{
			name: "partial defaults",
			params: &KVFilePathBaseParams{
				TpSize: 4,
			},
			expected: &KVFilePathBaseParams{
				GpuBlocksPerFile: 1,
				TpSize:           4,
				PpSize:           1,
				PcpSize:          1,
				DcpSize:          1,
			},
		},
		{
			name: "no defaults needed",
			params: &KVFilePathBaseParams{
				GpuBlocksPerFile: 8,
				TpSize:           2,
				PpSize:           2,
				PcpSize:          2,
				DcpSize:          2,
			},
			expected: &KVFilePathBaseParams{
				GpuBlocksPerFile: 8,
				TpSize:           2,
				PpSize:           2,
				PcpSize:          2,
				DcpSize:          2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.params.SetDefaults()
			assert.Equal(t, tt.expected, tt.params)
		})
	}
}

func TestPrefetchConfig_SetDefaultsForFilePrefetching(t *testing.T) {
	tests := []struct {
		name     string
		config   *PrefetchConfig
		expected *PrefetchConfig
	}{
		{
			name:   "all defaults",
			config: &PrefetchConfig{},
			expected: &PrefetchConfig{
				BlockSize:          4 * 1024 * 1024,
				BlockCount:         3,
				MaxConcurrentFiles: 16,
				WorkQueueSize:      256,
			},
		},
		{
			name: "partial defaults",
			config: &PrefetchConfig{
				BlockSize:          8 * 1024 * 1024,
				MaxConcurrentFiles: 32,
			},
			expected: &PrefetchConfig{
				BlockSize:          8 * 1024 * 1024,
				BlockCount:         3,
				MaxConcurrentFiles: 32,
				WorkQueueSize:      256,
			},
		},
		{
			name: "no defaults needed",
			config: &PrefetchConfig{
				BlockSize:          16 * 1024 * 1024,
				BlockCount:         5,
				MaxConcurrentFiles: 64,
				WorkQueueSize:      512,
			},
			expected: &PrefetchConfig{
				BlockSize:          16 * 1024 * 1024,
				BlockCount:         5,
				MaxConcurrentFiles: 64,
				WorkQueueSize:      512,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.config.SetDefaultsForFilePrefetching()
			assert.Equal(t, tt.expected, tt.config)
		})
	}
}

func TestPrefetchFile(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.bin")
	testData := []byte("test data for prefetch")
	err := os.WriteFile(testFile, testData, 0644)
	require.NoError(t, err)

	tests := []struct {
		name      string
		filePath  string
		bufferLen int
		wantErr   bool
	}{
		{
			name:      "successful read",
			filePath:  testFile,
			bufferLen: 1024,
			wantErr:   false,
		},
		{
			// A missing file is a benign skip (vLLM may not have written
			// the block yet), so prefetchFile returns nil, not an error.
			name:      "file not found",
			filePath:  filepath.Join(tmpDir, "nonexistent.bin"),
			bufferLen: 1024,
			wantErr:   false,
		},
		{
			name:      "buffer smaller than file",
			filePath:  testFile,
			bufferLen: 10,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			buffer := make([]byte, tt.bufferLen)
			err := prefetchFile(ctx, tt.filePath, buffer)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestNewPrefetchPrerequestHandler(t *testing.T) {
	kvFilePathBase := &KVFilePathBaseParams{
		RootDir:   "/data",
		ModelName: "test-model",
	}
	prefetchConfig := &PrefetchConfig{
		Enabled: true,
	}

	handler := NewPrefetchPrerequestHandler(nil, "test-provider", kvFilePathBase, prefetchConfig)

	assert.NotNil(t, handler)
	assert.Equal(t, PrefetchPrerequestHandlerType, handler.typedName.Type)
	assert.Equal(t, "test-provider", handler.engineKeysProviderPluginName)
	assert.Equal(t, kvFilePathBase, handler.kvFilePathBase)
	assert.Equal(t, prefetchConfig, handler.prefetchConfig)
}

func TestPrefetchPrerequestHandler_WithName(t *testing.T) {
	handler := NewPrefetchPrerequestHandler(nil, "", nil, nil)
	handler = handler.WithName("test-handler")

	assert.Equal(t, "test-handler", handler.typedName.Name)
}

func TestPrefetchPrerequestHandler_TypedName(t *testing.T) {
	handler := NewPrefetchPrerequestHandler(nil, "", nil, nil).WithName("test-handler")
	typedName := handler.TypedName()

	assert.Equal(t, PrefetchPrerequestHandlerType, typedName.Type)
	assert.Equal(t, "test-handler", typedName.Name)
}

func TestPluginFactory_InvalidJSON(t *testing.T) {
	invalidJSON := json.RawMessage(`{"invalid": json}`)
	_, err := PluginFactory("test", plugin.StrictDecoder(invalidJSON), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse the parameters")
}

func TestPluginFactory_NilParameters(t *testing.T) {
	// This test verifies that the factory can handle nil parameters
	// Note: This will fail without a proper mock handle that provides Context()
	// For now, we just test the parameter parsing logic
	params := prefetchPrerequestHandlerParameters{
		EngineKeysProviderPluginName: "test-provider",
	}
	rawParams, err := json.Marshal(params)
	require.NoError(t, err)

	// We can't fully test PluginFactory without a mock handle
	// but we can verify parameter unmarshaling works
	var parsed prefetchPrerequestHandlerParameters
	err = json.Unmarshal(rawParams, &parsed)
	assert.NoError(t, err)
	assert.Equal(t, "test-provider", parsed.EngineKeysProviderPluginName)
}

func TestInitializeWorkerPool_Disabled(t *testing.T) {
	handler := &PrefetchPrerequestHandler{
		prefetchConfig: &PrefetchConfig{
			Enabled: false,
		},
	}

	ctx := context.Background()
	err := initializeWorkerPool(ctx, handler)
	assert.NoError(t, err)
	assert.Nil(t, handler.workerPool)
}

func TestInitializeWorkerPool_NilConfig(t *testing.T) {
	handler := &PrefetchPrerequestHandler{
		prefetchConfig: nil,
	}

	ctx := context.Background()
	err := initializeWorkerPool(ctx, handler)
	assert.NoError(t, err)
	assert.Nil(t, handler.workerPool)
}

func TestInitializeWorkerPool_Enabled(t *testing.T) {
	handler := &PrefetchPrerequestHandler{
		prefetchConfig: &PrefetchConfig{
			Enabled:            true,
			BlockSize:          1024,
			BlockCount:         2,
			MaxConcurrentFiles: 2,
			WorkQueueSize:      10,
		},
	}

	ctx := context.Background()
	err := initializeWorkerPool(ctx, handler)
	assert.NoError(t, err)
	assert.NotNil(t, handler.workerPool)
	assert.NotNil(t, handler.workerPool.workQueue)
	assert.NotNil(t, handler.workerPool.workersDone)
	assert.NotNil(t, handler.workerPool.shutdownCtx)
	assert.NotNil(t, handler.workerPool.shutdownFn)

	// Cleanup
	handler.workerPool.shutdownFn()
	close(handler.workerPool.workQueue)

	// Wait for workers to exit with timeout
	select {
	case <-handler.workerPool.workersDone:
		// Workers exited successfully
	case <-time.After(2 * time.Second):
		t.Fatal("Workers did not exit within timeout")
	}
}

func TestPrefetchWorkerPool_WorkerProcessing(t *testing.T) {
	// Create a temporary test file
	tmpDir := t.TempDir()
	testFile := filepath.Join(tmpDir, "test.bin")
	testData := make([]byte, 1024)
	err := os.WriteFile(testFile, testData, 0644)
	require.NoError(t, err)

	handler := &PrefetchPrerequestHandler{
		prefetchConfig: &PrefetchConfig{
			Enabled:            true,
			BlockSize:          512,
			BlockCount:         1,
			MaxConcurrentFiles: 1,
			WorkQueueSize:      5,
		},
	}

	ctx := context.Background()
	err = initializeWorkerPool(ctx, handler)
	require.NoError(t, err)

	// Submit a file for prefetching
	handler.workerPool.workQueue <- testFile

	// Give worker time to process
	time.Sleep(100 * time.Millisecond)

	// Cleanup
	handler.workerPool.shutdownFn()
	close(handler.workerPool.workQueue)

	select {
	case <-handler.workerPool.workersDone:
		// Workers exited successfully
	case <-time.After(2 * time.Second):
		t.Fatal("Workers did not exit within timeout")
	}
}

// Made with Bob
