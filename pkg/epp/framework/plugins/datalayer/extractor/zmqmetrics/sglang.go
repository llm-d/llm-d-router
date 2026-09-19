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
	"errors"
	"fmt"
	"time"

	"github.com/vmihailenco/msgpack/v5"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// loadStatTag is the msgspec tag of SGLang's LoadStat struct. Each SGLang
// scheduler publishes a LoadStat snapshot on a dedicated PUB socket (enabled
// with --load-publish-endpoint). The wire shape (msgspec array_like + tag) is:
//
//	["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens, max_total_num_tokens, attn_dp_rank]
//
// attn_dp_rank is informational and may be null; decoders must tolerate
// trailing fields being appended.
const loadStatTag = "LoadStat"

const loadStatMinFields = 5 // tag plus the four required counts

// extractSGLang decodes an SGLang LoadStat msgpack payload and updates endpoint metrics.
func extractSGLang(ctx context.Context, in fwkdl.StreamInput[[]byte]) error {
	stat, err := decodeLoadStat(in.Payload)
	if err != nil {
		return fmt.Errorf("failed to decode LoadStat (len=%d): %w", len(in.Payload), err)
	}

	ep := in.Endpoint
	clone := ep.GetMetrics().Clone()

	clone.RunningRequestsSize = stat.numRunningReqs
	clone.WaitingQueueSize = stat.numWaitingReqs
	if stat.maxTotalNumTokens > 0 {
		clone.KvCacheMaxTokenCapacity = stat.maxTotalNumTokens
		clone.KVCacheUsagePercent = float64(stat.numTokens) / float64(stat.maxTotalNumTokens)
	}
	clone.UpdateTime = time.Now()

	logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetID())
	logger.V(logutil.DEBUG).Info("Refreshed metrics via SGLang load stream", "updated", clone)

	ep.UpdateMetrics(clone)
	return nil
}

type loadStat struct {
	numRunningReqs    int
	numWaitingReqs    int
	numTokens         int
	maxTotalNumTokens int
}

func decodeLoadStat(payload []byte) (*loadStat, error) {
	if len(payload) == 0 {
		return nil, errors.New("empty payload")
	}

	var raw []any
	if err := msgpack.Unmarshal(payload, &raw); err != nil {
		return nil, fmt.Errorf("invalid msgpack array: %w", err)
	}
	if len(raw) < loadStatMinFields {
		return nil, fmt.Errorf("expected at least %d fields, got %d", loadStatMinFields, len(raw))
	}

	tag, ok := raw[0].(string)
	if !ok || tag != loadStatTag {
		return nil, fmt.Errorf("unexpected tag %v, want %q", raw[0], loadStatTag)
	}

	stat := &loadStat{}
	fields := []struct {
		name string
		dst  *int
	}{
		{"num_running_reqs", &stat.numRunningReqs},
		{"num_waiting_reqs", &stat.numWaitingReqs},
		{"num_tokens", &stat.numTokens},
		{"max_total_num_tokens", &stat.maxTotalNumTokens},
	}
	for i, f := range fields {
		val, ok := asInt(raw[i+1])
		if !ok {
			return nil, fmt.Errorf("field %s: expected integer, got %T", f.name, raw[i+1])
		}
		*f.dst = val
	}
	return stat, nil
}

// asInt converts strictly integer-typed msgpack values; unlike parseToInt it
// rejects floats and strings so malformed LoadStat fields surface as errors.
func asInt(val any) (int, bool) {
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
	}
	return 0, false
}
