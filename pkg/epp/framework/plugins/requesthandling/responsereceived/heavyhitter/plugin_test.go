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

package heavyhitter

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// newPlugin builds a plugin through the factory from a JSON parameters blob ("" for defaults).
func newPlugin(t *testing.T, params string) *Plugin {
	t.Helper()
	var dec *json.Decoder
	if params != "" {
		dec = json.NewDecoder(bytes.NewReader([]byte(params)))
	}
	p, err := PluginFactory("test", dec, nil)
	require.NoError(t, err)
	return p.(*Plugin)
}

// charge reports a finished request for a tenant and user, as the director does at end of
// stream. Either identity may be empty.
func charge(p *Plugin, tenant, user string, promptTokens, completionTokens int) {
	req := &fwksched.InferenceRequest{Headers: map[string]string{
		metadata.FlowFairnessIDKey: tenant,
		metadata.FlowUserIDKey:     user,
	}}
	resp := &fwkrh.Response{EndOfStream: true}
	resp.Usage.PromptTokens = promptTokens
	resp.Usage.CompletionTokens = completionTokens
	p.ResponseBody(context.Background(), req, resp, nil)
}

func TestFactory_Defaults(t *testing.T) {
	p := newPlugin(t, "")

	charge(p, "tenant-a", "user-1", 100, 50)

	entry, ok := p.Lookup("tenant-a/user-1")
	require.True(t, ok)
	assert.Equal(t, int64(100+2*50), entry.Weight)
	assert.Zero(t, entry.Error)
}

func TestFactory_InvalidConfig(t *testing.T) {
	tests := []struct {
		name    string
		params  string
		wantErr string
	}{
		{"window_seconds_zero", `{"windowSeconds": 0}`, "windowSeconds must be > 0"},
		{"window_seconds_negative", `{"windowSeconds": -60}`, "windowSeconds must be > 0"},
		{"slice_seconds_zero", `{"sliceSeconds": 0}`, "sliceSeconds must be > 0"},
		{"slice_over_window", `{"windowSeconds": 10, "sliceSeconds": 20}`, "sliceSeconds must be <= windowSeconds"},
		{"single_slice", `{"windowSeconds": 10, "sliceSeconds": 10}`, "at least 2 slices"},
		{"too_many_slices", `{"windowSeconds": 3600, "sliceSeconds": 0.5}`, "at most 3600 slices"},
		{"too_many_slices_extreme", `{"windowSeconds": 1000000000, "sliceSeconds": 0.000001}`, "at most 3600 slices"},
		{"window_not_slice_multiple", `{"windowSeconds": 10, "sliceSeconds": 3}`, "must be a multiple of slice"},
		{"heavy_threshold_negative", `{"heavyThreshold": -1, "expectedWindowLoad": 100}`, "heavyThreshold must be >= 0"},
		{"expected_window_load_negative", `{"heavyThreshold": 100, "expectedWindowLoad": -1}`, "expectedWindowLoad must be >= 0"},
		{"threshold_without_sizing", `{"heavyThreshold": 100}`, "heavyThreshold needs expectedWindowLoad or slotsPerSlice"},
		{"load_without_threshold", `{"expectedWindowLoad": 100}`, "expectedWindowLoad needs heavyThreshold"},
		{"slots_negative", `{"slotsPerSlice": -1}`, "slotsPerSlice must be >= 0"},
		{"slots_too_many", `{"slotsPerSlice": 65537}`, "slotsPerSlice must be <= 65536"},
		{"tracked_slots_too_many", `{"slotsPerSlice": 6554}`, "slices*slotsPerSlice must be <= 131072"}, // 20 slices * 6554 > 1<<17
		{"cost_per_request_negative", `{"costPerRequest": -1}`, "costPerRequest must be >= 0"},
		{"cost_per_input_token_negative", `{"costPerInputToken": -1}`, "costPerInputToken must be >= 0"},
		{"cost_per_output_token_negative", `{"costPerOutputToken": -1}`, "costPerOutputToken must be >= 0"},
		{"cost_coefficients_all_zero", `{"costPerRequest": 0, "costPerInputToken": 0, "costPerOutputToken": 0}`, "cost coefficients cannot all be zero"},
		{"metric_top_k_negative", `{"metricTopK": -1}`, "metricTopK must be >= 0"},
		{"metric_top_k_too_many", `{"metricTopK": 101}`, "metricTopK must be <= 100"},
		{"unknown_field", `{"unknown": 1}`, "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(tt.params)), nil)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// TestFactory_DerivedSlotsTooLarge exercises the derivation overflow through the factory: the
// derived table size must stay within maxSlotsPerSlice.
func TestFactory_DerivedSlotsTooLarge(t *testing.T) {
	_, err := PluginFactory("test", fwkplugin.StrictDecoder(json.RawMessage(
		`{"heavyThreshold": 1, "expectedWindowLoad": 100000000}`)), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "derived slotsPerSlice")
}

// TestParameters_SlotsPerSlice covers the table sizing: the expert override, the default, and
// the threshold-based derivation with its headroom.
func TestParameters_SlotsPerSlice(t *testing.T) {
	tests := []struct {
		name   string
		params Parameters
		want   int
	}{
		{"expert_override", Parameters{SlotsPerSlice: 7}, 7},
		{"default_without_threshold", Parameters{}, defaultSlotsPerSlice},
		{"derived", Parameters{HeavyThreshold: 200, ExpectedWindowLoad: 3000}, 30},
		{"derived_rounds_up", Parameters{HeavyThreshold: 200, ExpectedWindowLoad: 3001}, 31},
		{"derived_minimum_one", Parameters{HeavyThreshold: 1000, ExpectedWindowLoad: 1}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.params.slotsPerSlice()
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestResponseBody_Weight(t *testing.T) {
	tests := []struct {
		name             string
		params           string
		promptTokens     int
		completionTokens int
		streamedEvents   int
		tokenized        [][]uint32
		want             int64
	}{
		{"usage_present", "", 100, 50, 0, nil, 100 + 2*50},
		{"streamed_events_with_tokenized_prompt", "", 0, 0, 7, [][]uint32{{1, 2, 3}}, 3 + 2*7},
		{"streamed_events_without_tokenized_prompt", "", 0, 0, 7, nil, 2 * 7},
		{"tokenized_prompt_without_usage_or_events", "", 0, 0, 0, [][]uint32{{1, 2, 3}}, 3},
		{"fractional_cost_rounded_up", `{"costPerInputToken": 0.5}`, 3, 0, 0, nil, 2},
		{"cost_per_request_only", `{"costPerRequest": 5, "costPerInputToken": 0, "costPerOutputToken": 0}`, 0, 0, 0, nil, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newPlugin(t, tt.params)
			req := &fwksched.InferenceRequest{Headers: map[string]string{
				metadata.FlowFairnessIDKey: "tenant-a",
				metadata.FlowUserIDKey:     "user-1",
			}}
			if tt.tokenized != nil {
				req.Body = &fwkrequest.InferenceRequestBody{TokenizedRequest: fwkrequest.NewTokenizedRequest(tt.tokenized)}
			}
			resp := &fwkrh.Response{EndOfStream: true, StreamedEvents: tt.streamedEvents}
			resp.Usage.PromptTokens = tt.promptTokens
			resp.Usage.CompletionTokens = tt.completionTokens

			p.ResponseBody(context.Background(), req, resp, nil)

			entry, ok := p.Lookup("tenant-a/user-1")
			require.True(t, ok)
			assert.Equal(t, tt.want, entry.Weight)
		})
	}
}

func TestResponseBody_IgnoresNonFinalChunk(t *testing.T) {
	p := newPlugin(t, "")
	req := &fwksched.InferenceRequest{Headers: map[string]string{
		metadata.FlowFairnessIDKey: "tenant-a",
		metadata.FlowUserIDKey:     "user-1",
	}}
	resp := &fwkrh.Response{EndOfStream: false, StreamedEvents: 3}
	resp.Usage.PromptTokens = 10
	resp.Usage.CompletionTokens = 5

	p.ResponseBody(context.Background(), req, resp, nil)

	_, ok := p.Lookup("tenant-a/user-1")
	assert.False(t, ok, "a non-final chunk carries no charge")

	resp.EndOfStream = true
	p.ResponseBody(context.Background(), req, resp, nil)
	entry, ok := p.Lookup("tenant-a/user-1")
	require.True(t, ok)
	assert.Equal(t, int64(10+2*5), entry.Weight)
}

func TestResponseBody_SkipsZeroCost(t *testing.T) {
	p := newPlugin(t, "")

	charge(p, "tenant-a", "user-1", 0, 0)
	p.ResponseBody(context.Background(), nil, nil, nil)

	_, ok := p.Lookup("tenant-a/user-1")
	assert.False(t, ok)
}

// TestResponseBody_TrackingKey covers the tenant/user key composition: the deprecated
// fairness-header alias, the escaping that keeps a tenant-only key distinct from a tenant/user
// pair, the default-tenant prefix for a user without a tenant, and the label-bound truncation
// of composed keys.
func TestResponseBody_TrackingKey(t *testing.T) {
	p := newPlugin(t, "")

	charge(p, "tenant-a", "user-1", 10, 0)
	entry, ok := p.Lookup("tenant-a/user-1")
	require.True(t, ok)
	assert.Equal(t, "tenant-a/user-1", entry.ID)
	assert.Equal(t, int64(10), entry.Weight)

	// The deprecated fairness-header alias resolves to the same tenant.
	chargeWithHeaders(p, map[string]string{metadata.OldFlowFairnessIDKey: "tenant-a", metadata.FlowUserIDKey: "user-1"}, 5, 0)
	entry, ok = p.Lookup("tenant-a/user-1")
	require.True(t, ok)
	assert.Equal(t, int64(15), entry.Weight)

	// A tenant containing the separator must not collide with a tenant/user pair.
	charge(p, "tenant/x", "user-1", 5, 0)
	entry, ok = p.Lookup("tenant%2Fx/user-1")
	require.True(t, ok)
	assert.Equal(t, int64(5), entry.Weight)
	charge(p, "tenant/x", "", 7, 0) // tenant-only key: escaped, no separator
	entry, ok = p.Lookup("tenant%2Fx")
	require.True(t, ok)
	assert.Equal(t, int64(7), entry.Weight)

	// A user without a tenant composes onto the default tenant, so it cannot collide with a
	// tenant-only key of the same name.
	charge(p, "", "alice", 3, 0)
	entry, ok = p.Lookup(reqcommon.DefaultTenantID + "/alice")
	require.True(t, ok)
	assert.Equal(t, int64(3), entry.Weight)
	charge(p, "alice", "", 4, 0)
	entry, ok = p.Lookup("alice")
	require.True(t, ok)
	assert.Equal(t, int64(4), entry.Weight)

	// Neither identity falls back to the default flow.
	charge(p, "", "", 2, 0)
	entry, ok = p.Lookup(reqcommon.DefaultFairnessID)
	require.True(t, ok)
	assert.Equal(t, int64(2), entry.Weight)

	// A composed key longer than the label bound truncates at a rune boundary: 254 ASCII
	// bytes, then a 3-byte rune straddling byte 256, then a tail.
	long := strings.Repeat("a", 254) + "\u20ac" + strings.Repeat("b", 50)
	charge(p, long, "user-9", 4, 0)
	entry, ok = p.Lookup(long + "/user-9")
	require.True(t, ok)
	assert.Equal(t, strings.Repeat("a", 254), entry.ID)
	assert.Equal(t, int64(4), entry.Weight)

	// A key whose whole 256-byte prefix is continuation bytes has no admissible cut and folds
	// to the overflow value instead of an empty label.
	garbage := strings.Repeat("\x80", 300)
	chargeWithHeaders(p, map[string]string{metadata.FlowFairnessIDKey: garbage}, 6, 0)
	entry, ok = p.Lookup(metricsutil.OverflowValue)
	require.True(t, ok)
	assert.Equal(t, int64(6), entry.Weight)
}

// chargeWithHeaders reports a finished request carrying exactly these headers.
func chargeWithHeaders(p *Plugin, headers map[string]string, promptTokens, completionTokens int) {
	req := &fwksched.InferenceRequest{Headers: headers}
	resp := &fwkrh.Response{EndOfStream: true}
	resp.Usage.PromptTokens = promptTokens
	resp.Usage.CompletionTokens = completionTokens
	p.ResponseBody(context.Background(), req, resp, nil)
}

// A threshold with an explicit table size needs no load estimate.
func TestFactory_ThresholdWithExplicitSlots(t *testing.T) {
	p := newPlugin(t, `{"heavyThreshold": 100, "slotsPerSlice": 8, "windowSeconds": 60, "sliceSeconds": 1}`)
	charge(p, "tenant-a", "user-1", 0, 60) // 120, at or above the threshold
	assert.True(t, p.IsHeavy("tenant-a/user-1"))
}

// TestIsHeavy covers the threshold classification: at-or-above on the lower bound, below, and
// the unset-threshold behavior.
func TestIsHeavy(t *testing.T) {
	p := newPlugin(t, `{"heavyThreshold": 100, "expectedWindowLoad": 1000, "windowSeconds": 60, "sliceSeconds": 1}`)

	charge(p, "tenant-a", "user-heavy", 40, 10) // 40 + 20 = 60, below
	charge(p, "tenant-a", "user-heavy", 40, 10) // 120, at or above

	assert.True(t, p.IsHeavy("tenant-a/user-heavy"))
	assert.False(t, p.IsHeavy("tenant-a/user-light"), "an untracked key is not heavy")

	noThreshold := newPlugin(t, "")
	charge(noThreshold, "tenant-a", "user-1", 100000, 0)
	assert.False(t, noThreshold.IsHeavy("tenant-a/user-1"), "no threshold configured")
}

func TestWindowExpiry(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		p := newPlugin(t, `{"windowSeconds": 1, "sliceSeconds": 0.5}`)
		charge(p, "tenant-a", "user-1", 100, 50)

		entry, ok := p.Lookup("tenant-a/user-1")
		require.True(t, ok)
		assert.Equal(t, int64(100+2*50), entry.Weight)

		time.Sleep(2 * time.Second)

		assert.Empty(t, p.Top(-1))
		_, ok = p.Lookup("tenant-a/user-1")
		assert.False(t, ok)
	})
}
