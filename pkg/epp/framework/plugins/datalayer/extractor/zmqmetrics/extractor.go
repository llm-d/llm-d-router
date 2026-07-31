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

package zmqmetrics

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/vmihailenco/msgpack/v5"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	sourcezmq "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/zmqmetrics"
)

const ZMQExtractorType = "zmq-state-extractor"

type Config struct {
	PrefillBufferCapacity int `json:"prefillBufferCapacity,omitempty"`
}

var DefaultConfig = Config{
	PrefillBufferCapacity: 5,
}

// Extractor implements ZMQ msgpack metrics extraction.
type Extractor struct {
	typedName fwkplugin.TypedName
	config    Config
	tracker   *prefillTracker
}

var _ fwkdl.StreamingExtractor[[]byte] = (*Extractor)(nil)

// NewZMQMetricsExtractor returns a new ZMQ metrics extractor.
func NewZMQMetricsExtractor(name string, config Config) *Extractor {
	if name == "" {
		name = ZMQExtractorType
	}
	if config.PrefillBufferCapacity <= 0 {
		config.PrefillBufferCapacity = DefaultConfig.PrefillBufferCapacity
	}
	return &Extractor{
		typedName: fwkplugin.TypedName{
			Type: ZMQExtractorType,
			Name: name,
		},
		config:  config,
		tracker: newPrefillTracker(config.PrefillBufferCapacity),
	}
}

// ZMQExtractorFactory is a factory function used to instantiate ZMQ extractor plugins.
func ZMQExtractorFactory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	return NewZMQMetricsExtractor(name, config), nil
}

// TypedName returns the type and name of the Extractor.
func (ext *Extractor) TypedName() fwkplugin.TypedName {
	return ext.typedName
}

// Extract decodes the msgpack payload and updates endpoint metrics.
func (ext *Extractor) Extract(ctx context.Context, in fwkdl.StreamInput[[]byte]) error {
	payload := in.Payload
	var stats sourcezmq.ZmqMetricsStats

	if err := decodeMsgpackPayload(payload, &stats); err != nil {
		return fmt.Errorf("failed to unmarshal msgpack (len=%d, hex=%x): %w", len(payload), truncateBytes(payload, 32), err)
	}

	ep := in.Endpoint
	epKey := ep.GetMetadata().NamespacedName.String()
	current := ep.GetMetrics()
	clone := current.Clone()

	clone.RunningRequestsSize = stats.NumRequestsRunning
	clone.WaitingQueueSize = stats.NumRequestsWaiting
	clone.KVCacheUsagePercent = stats.KVCacheUsagePerc
	clone.NumPrefillComputedTokens = stats.NumPrefillComputedTokens
	clone.NumPrefillCachedTokens = stats.NumPrefillCachedTokens
	clone.NumPrefillTotalTokens = stats.NumPrefillTotalTokens
	clone.BatchExecutionLatencyMs = stats.BatchExecutionLatencyMs

	if stats.CacheConfigInfo != nil {
		if val, ok := parseToInt(stats.CacheConfigInfo["block_size"]); ok {
			clone.CacheBlockSize = val
		}
		if val, ok := parseToInt(stats.CacheConfigInfo["num_gpu_blocks"]); ok {
			clone.CacheNumBlocks = val
		}
		if val, ok := parseToInt(stats.CacheConfigInfo["kv_cache_size_tokens"]); ok {
			clone.KvCacheMaxTokenCapacity = val
		} else if clone.CacheBlockSize > 0 && clone.CacheNumBlocks > 0 {
			clone.KvCacheMaxTokenCapacity = clone.CacheBlockSize * clone.CacheNumBlocks
		}
	}

	if stats.NumPrefillComputedTokens > 0 {
		rate := ext.tracker.AddReading(epKey, stats.NumPrefillComputedTokens, stats.BatchExecutionLatencyMs)
		clone.ComputePrefillThroughput = rate
	}

	clone.UpdateTime = time.Now()

	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().NamespacedName)
	logger.V(logutil.DEBUG).Info("Refreshed metrics via ZMQ", "updated", clone)

	ep.UpdateMetrics(clone)
	return nil
}

func decodeMsgpackPayload(payload []byte, target *sourcezmq.ZmqMetricsStats) error {
	if len(payload) == 0 {
		return fmt.Errorf("empty payload")
	}

	// 1. Try direct struct unmarshal
	if err := msgpack.Unmarshal(payload, target); err == nil {
		return nil
	}

	// 2. Try generic map[string]any
	var rawMap map[string]any
	if err := msgpack.Unmarshal(payload, &rawMap); err == nil {
		populateStatsFromMap(rawMap, target)
		return nil
	}

	// 3. Try slice/tuple unmarshal ([]any)
	var rawSlice []any
	if err := msgpack.Unmarshal(payload, &rawSlice); err == nil {
		populateStatsFromSlice(rawSlice, target)
		return nil
	}

	return fmt.Errorf("invalid msgpack format")
}

func populateStatsFromMap(m map[string]any, target *sourcezmq.ZmqMetricsStats) {
	if val, ok := parseToInt(m["num_requests_running"]); ok {
		target.NumRequestsRunning = val
	}
	if val, ok := parseToInt(m["num_requests_waiting"]); ok {
		target.NumRequestsWaiting = val
	}
	if val, ok := parseToFloat(m["kv_cache_usage_perc"]); ok {
		target.KVCacheUsagePerc = val
	}
	if cfgMap, ok := m["cache_config_info"].(map[string]any); ok {
		target.CacheConfigInfo = cfgMap
	}
	if engineID, ok := m["engine_id"].(string); ok {
		target.EngineID = engineID
	}
	if val, ok := parseToInt(m["num_prefill_computed_tokens"]); ok {
		target.NumPrefillComputedTokens = val
	}
	if val, ok := parseToInt(m["num_prefill_cached_tokens"]); ok {
		target.NumPrefillCachedTokens = val
	}
	if val, ok := parseToInt(m["num_prefill_total_tokens"]); ok {
		target.NumPrefillTotalTokens = val
	}
	if val, ok := parseToFloat(m["batch_execution_latency_ms"]); ok {
		target.BatchExecutionLatencyMs = val
	}
}

func populateStatsFromSlice(s []any, target *sourcezmq.ZmqMetricsStats) {
	if len(s) > 0 {
		if val, ok := parseToInt(s[0]); ok {
			target.NumRequestsRunning = val
		}
	}
	if len(s) > 1 {
		if val, ok := parseToInt(s[1]); ok {
			target.NumRequestsWaiting = val
		}
	}
	if len(s) > 2 {
		if val, ok := parseToFloat(s[2]); ok {
			target.KVCacheUsagePerc = val
		}
	}
	if len(s) > 3 {
		if cfgMap, ok := s[3].(map[string]any); ok {
			target.CacheConfigInfo = cfgMap
		}
	}
	if len(s) > 4 {
		if engineID, ok := s[4].(string); ok {
			target.EngineID = engineID
		}
	}
	if len(s) > 5 {
		if val, ok := parseToInt(s[5]); ok {
			target.NumPrefillComputedTokens = val
		}
	}
	if len(s) > 6 {
		if val, ok := parseToInt(s[6]); ok {
			target.NumPrefillCachedTokens = val
		}
	}
	if len(s) > 7 {
		if val, ok := parseToInt(s[7]); ok {
			target.NumPrefillTotalTokens = val
		}
	}
	if len(s) > 8 {
		if val, ok := parseToFloat(s[8]); ok {
			target.BatchExecutionLatencyMs = val
		}
	}
}

func truncateBytes(b []byte, n int) []byte {
	if len(b) > n {
		return b[:n]
	}
	return b
}

func parseToFloat(val any) (float64, bool) {
	if val == nil {
		return 0, false
	}
	switch v := val.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

func parseToInt(val any) (int, bool) {
	if val == nil {
		return 0, false
	}
	switch v := val.(type) {
	case int:
		return v, true
	case int8:
		return int(v), true
	case int16:
		return int(v), true
	case int32:
		return int(v), true
	case int64:
		return int(v), true
	case uint:
		return int(v), true
	case uint8:
		return int(v), true
	case uint16:
		return int(v), true
	case uint32:
		return int(v), true
	case uint64:
		return int(v), true
	case float32:
		return int(v), true
	case float64:
		return int(v), true
	case string:
		if i, err := strconv.Atoi(v); err == nil {
			return i, true
		}
	}
	return 0, false
}
