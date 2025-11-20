// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//nolint:testpackage // testing unexported functions
package metrics

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"k8s.io/klog/v2"
)

func TestLogMetrics(t *testing.T) {
	// Set up a buffer to capture log output
	var buf bytes.Buffer
	handler := slog.NewTextHandler(&buf, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})
	logrLogger := logr.FromSlogHandler(handler)
	klog.SetLogger(logrLogger)

	ctx := context.Background()

	t.Run("no_latency", func(t *testing.T) {
		// Reset buffer
		buf.Reset()

		// Set test values for metrics
		Admissions.Inc()       // 1 admission
		Evictions.Add(2)       // 2 evictions
		LookupRequests.Add(10) // 10 lookups
		MaxPodHitCount.Add(5)  // 5 hits

		// Call logMetrics
		logMetrics(ctx)

		// Get the logged output
		output := buf.String()

		// Verify that the log contains expected key-value pairs
		expectedParts := []string{
			"admissions=1",
			"evictions=2",
			"lookups=10",
			"hits=5",
			"latency_count=0",
			"latency_sum=0",
			"latency_avg=0",
		}

		for _, part := range expectedParts {
			if !strings.Contains(output, part) {
				t.Errorf("Expected '%s' in log output, but it was not found. Full output: %s", part, output)
			}
		}
	})

	t.Run("with_latency", func(t *testing.T) {
		// Reset buffer
		buf.Reset()

		LookupLatency.Observe(0.1) // Observe latency
		LookupLatency.Observe(0.2) // Another observation

		logMetrics(ctx)
		// Get the logged output
		output := buf.String()
		// Verify that the log contains expected key-value pairs
		expectedParts := []string{
			"latency_count=2",
			"latency_sum=0.3",
			"latency_avg=0.15",
		}

		for _, part := range expectedParts {
			if !strings.Contains(output, part) {
				t.Errorf("Expected '%s' in log output, but it was not found. Full output: %s", part, output)
			}
		}
	})
}
