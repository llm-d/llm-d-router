/*
Copyright 2026 The llm-d Authors.

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

package wspt

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

var testFlowKey = flowcontrol.FlowKey{ID: "test-flow", Priority: 0}

func TestWSPTPolicy_Name(t *testing.T) {
	t.Parallel()
	policy := newWSPTPolicy()
	assert.Equal(t, WSPTOrderingPolicyType, policy.Name())
	assert.Equal(t, "custom", newWSPTPolicy().withName("custom").Name())
}

func TestWSPT_Less(t *testing.T) {
	t.Parallel()
	policy := newWSPTPolicy()

	now := time.Now()

	small := mocks.NewMockQueueItemAccessor(100, "small", testFlowKey)
	small.EnqueueTimeV = now.Add(2 * time.Second) // arrived last, but smallest

	large := mocks.NewMockQueueItemAccessor(10_000, "large", testFlowKey)
	large.EnqueueTimeV = now

	sameSizeEarly := mocks.NewMockQueueItemAccessor(500, "sameSizeEarly", testFlowKey)
	sameSizeEarly.EnqueueTimeV = now

	sameSizeLate := mocks.NewMockQueueItemAccessor(500, "sameSizeLate", testFlowKey)
	sameSizeLate.EnqueueTimeV = now.Add(time.Second)

	noRequest := noRequestItem{&mocks.MockQueueItemAccessor{EnqueueTimeV: now.Add(-time.Hour)}}

	testCases := []struct {
		name     string
		a        flowcontrol.QueueItemAccessor
		b        flowcontrol.QueueItemAccessor
		expected bool // true if a should be dispatched before b
	}{
		{"smaller before larger despite later arrival", small, large, true},
		{"larger after smaller", large, small, false},

		{"same size: earlier arrival first", sameSizeEarly, sameSizeLate, true},
		{"same size: later arrival second", sameSizeLate, sameSizeEarly, false},
		{"identical item is not less than itself", sameSizeEarly, sameSizeEarly, false},

		{"item with request before item without one", large, noRequest, true},
		{"item without request after item with one", noRequest, large, false},

		{"a is nil -> b wins", nil, small, false},
		{"b is nil -> a wins", small, nil, true},
		{"both nil -> false", nil, nil, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, policy.Less(tc.a, tc.b))
		})
	}
}

func TestWSPTOrderingPolicyFactory(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name    string
		params  string
		want    Config
		wantErr string
	}{
		{name: "no parameters is unbounded", want: Config{}},
		{name: "bounded", params: `{"maxDelaySeconds": 2, "referenceBytes": 65536}`,
			want: Config{MaxDelaySeconds: 2, ReferenceBytes: 65536}},
		{name: "negative maxDelaySeconds", params: `{"maxDelaySeconds": -1, "referenceBytes": 10}`,
			wantErr: "maxDelaySeconds must be >= 0"},
		{name: "maxDelaySeconds without referenceBytes", params: `{"maxDelaySeconds": 2}`,
			wantErr: "referenceBytes must be > 0 when maxDelaySeconds is set"},
		{name: "malformed", params: `{"maxDelaySeconds": "2s"}`, wantErr: "invalid config"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var decoder *json.Decoder
			if tc.params != "" {
				decoder = json.NewDecoder(strings.NewReader(tc.params))
			}
			p, err := WSPTOrderingPolicyFactory("wspt", decoder, nil)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, p.(*WSPTPolicy).cfg)
		})
	}
}

func TestWSPT_Less_MaxDelay(t *testing.T) {
	t.Parallel()
	// A request at referenceBytes or larger is delayed by the full 2s; smaller ones proportionally.
	policy := newWSPTPolicy()
	policy.cfg = Config{MaxDelaySeconds: 2, ReferenceBytes: 1000}

	now := time.Now()
	item := func(size uint64, enqueue time.Time) flowcontrol.QueueItemAccessor {
		i := mocks.NewMockQueueItemAccessor(size, "item", testFlowKey)
		i.EnqueueTimeV = enqueue
		return i
	}

	small := item(100, now)                                // key: now + 0.2s
	largeWaitedShort := item(1000, now.Add(-time.Second))  // key: now + 1s
	largeWaitedLong := item(1000, now.Add(-3*time.Second)) // key: now - 1s
	oversized := item(5000, now)                           // clamped to referenceBytes: key now + 2s
	atReference := item(1000, now)                         // key: now + 2s

	testCases := []struct {
		name     string
		a        flowcontrol.QueueItemAccessor
		b        flowcontrol.QueueItemAccessor
		expected bool
	}{
		{"small before large that waited less than the size delay", small, largeWaitedShort, true},
		{"large that waited less than the size delay after small", largeWaitedShort, small, false},
		{"large that waited past the size delay before small", largeWaitedLong, small, true},
		{"small after large that waited past the size delay", small, largeWaitedLong, false},
		{"delay is capped at referenceBytes", oversized, atReference, false},
		{"delay is capped at referenceBytes (reverse)", atReference, oversized, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, policy.Less(tc.a, tc.b))
		})
	}
}

// noRequestItem is a queue item whose OriginalRequest is nil, which the shared mock never returns.
type noRequestItem struct {
	*mocks.MockQueueItemAccessor
}

func (noRequestItem) OriginalRequest() flowcontrol.FlowControlRequest { return nil }
