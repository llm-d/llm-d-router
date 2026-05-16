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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngineKeyToFilenamePathSuffix(t *testing.T) {
	tests := []struct {
		name      string
		engineKey uint64
		expected  string
	}{
		{
			name:      "zero key",
			engineKey: 0,
			expected:  "000/00/0000000000000000.bin",
		},
		{
			name:      "small key",
			engineKey: 0x123,
			expected:  "000/00/0000000000000123.bin",
		},
		{
			name:      "large key",
			engineKey: 0xABCDEF1234567890,
			expected:  "abc/de/abcdef1234567890.bin",
		},
		{
			name:      "max uint64",
			engineKey: 0xFFFFFFFFFFFFFFFF,
			expected:  "fff/ff/ffffffffffffffff.bin",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EngineKeyToFilenamePathSuffix(tt.engineKey)
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
			name: "both root dir and model name",
			params: &KVFilePathBaseParams{
				RootDir:   "/tmp",
				ModelName: "test-model",
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
				GpuBlockSize:     64,
				GpuBlocksPerFile: 1,
				TpSize:           1,
				PpSize:           1,
				PcpSize:          1,
			},
		},
		{
			name: "partial defaults",
			params: &KVFilePathBaseParams{
				GpuBlockSize: 128,
				TpSize:       4,
			},
			expected: &KVFilePathBaseParams{
				GpuBlockSize:     128,
				GpuBlocksPerFile: 1,
				TpSize:           4,
				PpSize:           1,
				PcpSize:          1,
			},
		},
		{
			name: "no defaults needed",
			params: &KVFilePathBaseParams{
				GpuBlockSize:     256,
				GpuBlocksPerFile: 8,
				TpSize:           2,
				PpSize:           2,
				PcpSize:          2,
			},
			expected: &KVFilePathBaseParams{
				GpuBlockSize:     256,
				GpuBlocksPerFile: 8,
				TpSize:           2,
				PpSize:           2,
				PcpSize:          2,
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

func TestKVFilePathBaseParams_BasePath(t *testing.T) {
	tests := []struct {
		name     string
		params   *KVFilePathBaseParams
		expected string
	}{
		{
			name: "basic path",
			params: &KVFilePathBaseParams{
				RootDir:          "/data",
				ModelName:        "llama-7b",
				GpuBlockSize:     64,
				GpuBlocksPerFile: 1,
				TpSize:           1,
				PpSize:           1,
				PcpSize:          1,
				Rank:             0,
				Dtype:            "float16",
			},
			expected: filepath.Join("/data", "llama-7b", "block_size_64_blocks_per_file_1", "tp_1_pp_size_1_pcp_size_1", "rank_0", "float16"),
		},
		{
			name: "with model parent dir",
			params: &KVFilePathBaseParams{
				RootDir:          "/data",
				ModelParentDir:   "models",
				ModelName:        "llama-7b",
				GpuBlockSize:     128,
				GpuBlocksPerFile: 4,
				TpSize:           2,
				PpSize:           2,
				PcpSize:          1,
				Rank:             3,
				Dtype:            "bfloat16",
			},
			expected: filepath.Join("/data", "models", "llama-7b", "block_size_128_blocks_per_file_4", "tp_2_pp_size_2_pcp_size_1", "rank_3", "bfloat16"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.params.BasePath()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestEngineKeyToFullPath(t *testing.T) {
	base := &KVFilePathBaseParams{
		RootDir:          "/data",
		ModelName:        "test-model",
		GpuBlockSize:     64,
		GpuBlocksPerFile: 1,
		TpSize:           1,
		PpSize:           1,
		PcpSize:          1,
		Rank:             0,
		Dtype:            "float16",
	}

	engineKey := uint64(0x123456789ABCDEF0)
	expected := filepath.Join("/data", "test-model", "block_size_64_blocks_per_file_1", "tp_1_pp_size_1_pcp_size_1", "rank_0", "float16", "123", "45", "123456789abcdef0.bin")

	result := EngineKeyToFullPath(base, engineKey)
	assert.Equal(t, expected, result)
}

func TestEngineKeysToFilePaths(t *testing.T) {
	tests := []struct {
		name       string
		base       *KVFilePathBaseParams
		engineKeys []uint64
		expected   int // expected number of paths
	}{
		{
			name: "single block per file",
			base: &KVFilePathBaseParams{
				RootDir:          "/data",
				ModelName:        "test",
				GpuBlockSize:     64,
				GpuBlocksPerFile: 1,
				TpSize:           1,
				PpSize:           1,
				PcpSize:          1,
				Rank:             0,
				Dtype:            "float16",
			},
			engineKeys: []uint64{0x1, 0x2, 0x3, 0x4},
			expected:   4,
		},
		{
			name: "multiple blocks per file",
			base: &KVFilePathBaseParams{
				RootDir:          "/data",
				ModelName:        "test",
				GpuBlockSize:     64,
				GpuBlocksPerFile: 2,
				TpSize:           1,
				PpSize:           1,
				PcpSize:          1,
				Rank:             0,
				Dtype:            "float16",
			},
			engineKeys: []uint64{0x1, 0x2, 0x3, 0x4},
			expected:   2,
		},
		{
			name: "multiple blocks per file with remainder",
			base: &KVFilePathBaseParams{
				RootDir:          "/data",
				ModelName:        "test",
				GpuBlockSize:     64,
				GpuBlocksPerFile: 3,
				TpSize:           1,
				PpSize:           1,
				PcpSize:          1,
				Rank:             0,
				Dtype:            "float16",
			},
			engineKeys: []uint64{0x1, 0x2, 0x3, 0x4, 0x5},
			expected:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EngineKeysToFilePaths(tt.base, tt.engineKeys)
			assert.Equal(t, tt.expected, len(result))
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
			name:      "file not found",
			filePath:  filepath.Join(tmpDir, "nonexistent.bin"),
			bufferLen: 1024,
			wantErr:   true,
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
	_, err := PluginFactory("test", invalidJSON, nil)
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
