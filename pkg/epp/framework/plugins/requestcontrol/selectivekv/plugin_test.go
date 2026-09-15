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

package selectivekv

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	preciseproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache"
)

func TestSelectiveKVPluginFactory(t *testing.T) {
	decoder := plugin.StrictDecoder(json.RawMessage(`{
		"loadPolicy":"disable",
		"offloadPolicy":"disable"
	}`))

	created, err := PluginFactory("test", decoder, nil)
	require.NoError(t, err)

	p := created.(*Plugin)
	assert.Equal(t, PluginType, p.TypedName().Type)
	assert.Equal(t, "test", p.TypedName().Name)
	assert.Equal(t, PolicyDisable, p.loadPolicy)
	assert.Equal(t, PolicyDisable, p.offloadPolicy)
}

func TestSelectiveKVPluginFactoryConfiguresThresholdPolicy(t *testing.T) {
	decoder := plugin.StrictDecoder(json.RawMessage(`{
		"loadPolicy":"threshold",
		"minExternalReusableTokens":1024,
		"maxWaitingRequests":8
	}`))

	created, err := PluginFactory("test", decoder, nil)
	require.NoError(t, err)

	p := created.(*Plugin)
	assert.Equal(t, PolicyThreshold, p.loadPolicy)
	assert.Equal(t, 1024, p.minExternalReusableTokens)
	assert.Equal(t, 8, p.maxWaitingRequests)
	assert.Equal(t, "PrefixCacheMatchInfoDataKey/"+preciseproducer.PluginType,
		p.prefixMatchInfoDataKey.String())
}

func TestSelectiveKVPluginFactoryRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantError  string
	}{
		{
			name:      "missing parameters",
			wantError: "parameters are required",
		},
		{
			name:       "inactive policy",
			parameters: `{}`,
			wantError:  "at least one of loadPolicy or offloadPolicy must be active",
		},
		{
			name: "invalid load policy",
			parameters: `{
				"loadPolicy":"sometimes"
			}`,
			wantError: "loadPolicy must be",
		},
		{
			name: "invalid offload policy",
			parameters: `{
				"offloadPolicy":"sometimes"
			}`,
			wantError: "offloadPolicy must be",
		},
		{
			name: "threshold is load-only",
			parameters: `{
				"offloadPolicy":"threshold"
			}`,
			wantError: "offloadPolicy must be \"preserve\" or \"disable\"",
		},
		{
			name: "threshold requires external token threshold",
			parameters: `{
				"loadPolicy":"threshold"
			}`,
			wantError: "minExternalReusableTokens must be greater than zero",
		},
		{
			name: "waiting threshold cannot be negative",
			parameters: `{
				"loadPolicy":"threshold",
				"minExternalReusableTokens":1,
				"maxWaitingRequests":-1
			}`,
			wantError: "maxWaitingRequests must not be negative",
		},
		{
			name: "waiting threshold requires threshold policy",
			parameters: `{
				"loadPolicy":"disable",
				"maxWaitingRequests":8
			}`,
			wantError: "maxWaitingRequests requires threshold loadPolicy",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var decoder *json.Decoder
			if test.parameters != "" {
				decoder = plugin.StrictDecoder(json.RawMessage(test.parameters))
			}
			_, err := PluginFactory("test", decoder, nil)
			require.ErrorContains(t, err, test.wantError)
		})
	}
}

func TestSelectiveKVThresholdPolicyVetoesWaitingQueuePressure(t *testing.T) {
	p, err := New("test", Config{
		LoadPolicy:                PolicyThreshold,
		OffloadPolicy:             PolicyPreserve,
		MinExternalReusableTokens: 1024,
		MaxWaitingRequests:        8,
	})
	require.NoError(t, err)

	start := time.Unix(1, 0)
	tests := []struct {
		name         string
		waiting      int
		updateTime   time.Time
		wantDisabled bool
	}{
		{
			name:         "closes at threshold",
			waiting:      8,
			updateTime:   start,
			wantDisabled: true,
		},
		{
			name:         "stays closed above reopen threshold",
			waiting:      0,
			updateTime:   start.Add(time.Second),
			wantDisabled: true,
		},
		{
			name:         "reopens after pressure decays",
			waiting:      0,
			updateTime:   start.Add(3 * time.Second),
			wantDisabled: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := thresholdResultWithQueue(p,
				map[string]int{"gpu": 4, "cpu": 20},
				test.waiting, test.updateTime)
			payload := requesthandling.PayloadMap{"model": "test"}
			body := &requesthandling.InferenceRequestBody{Payload: payload}
			request := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.PreRequest(context.Background(), request, result))
			params, hasParams := payload["kv_transfer_params"].(map[string]any)
			if test.wantDisabled {
				require.True(t, hasParams)
				assert.Equal(t, json.Number("0"), params["max_load_tokens"])
			} else {
				assert.False(t, hasParams)
			}
		})
	}
}

func TestSelectiveKVThresholdPolicyIgnoresMissingWaitingQueueSample(t *testing.T) {
	p, err := New("test", Config{
		LoadPolicy:                PolicyThreshold,
		OffloadPolicy:             PolicyPreserve,
		MinExternalReusableTokens: 1024,
		MaxWaitingRequests:        8,
	})
	require.NoError(t, err)

	result := thresholdResultWithQueue(p,
		map[string]int{"gpu": 4, "cpu": 20}, 8, time.Time{})
	payload := requesthandling.PayloadMap{"model": "test"}
	body := &requesthandling.InferenceRequestBody{Payload: payload}

	require.NoError(t, p.PreRequest(context.Background(),
		&scheduling.InferenceRequest{Body: body}, result))
	assert.True(t, body.Mutated)
	assert.NotContains(t, payload, "kv_transfer_params")
}

func TestSelectiveKVThresholdPolicy(t *testing.T) {
	p := newThresholdPlugin(t)

	tests := []struct {
		name          string
		byTier        map[string]int
		wantDisabled  bool
		wantBodyDirty bool
	}{
		{
			name:          "small external prefix recomputes",
			byTier:        map[string]int{"gpu": 10, "cpu": 20},
			wantDisabled:  true,
			wantBodyDirty: true,
		},
		{
			name:          "threshold-sized external prefix loads",
			byTier:        map[string]int{"gpu": 4, "cpu": 20},
			wantDisabled:  false,
			wantBodyDirty: true,
		},
		{
			name: "largest external tier determines reusable tokens",
			byTier: map[string]int{
				"gpu": 4, "cpu": 12, "storage": 24,
			},
			wantDisabled:  false,
			wantBodyDirty: true,
		},
		{
			name:          "missing tier evidence fails open",
			wantDisabled:  false,
			wantBodyDirty: false,
		},
		{
			name:          "empty tier evidence recomputes",
			byTier:        map[string]int{},
			wantDisabled:  true,
			wantBodyDirty: true,
		},
		{
			name: "speculative matches are excluded",
			byTier: map[string]int{
				attrprefix.SpeculativeTierKey: 20,
			},
			wantDisabled:  true,
			wantBodyDirty: true,
		},
		{
			name: "external prefix shorter than GPU prefix clamps to zero",
			byTier: map[string]int{
				"gpu": 20, "cpu": 10,
			},
			wantDisabled:  true,
			wantBodyDirty: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := thresholdResult(p, test.byTier)
			payload := requesthandling.PayloadMap{"model": "test"}
			body := &requesthandling.InferenceRequestBody{Payload: payload}
			request := &scheduling.InferenceRequest{Body: body}

			require.NoError(t, p.PreRequest(context.Background(), request, result))
			params, hasParams := payload["kv_transfer_params"].(map[string]any)
			if test.wantDisabled {
				require.True(t, hasParams)
				assert.Equal(t, json.Number("0"), params["max_load_tokens"])
			} else {
				assert.False(t, hasParams)
			}
			assert.Equal(t, test.wantBodyDirty, body.Mutated)
		})
	}
}

func TestSelectiveKVThresholdPolicyRemovesClientOptOutWhenLoadingWins(t *testing.T) {
	p := newThresholdPlugin(t)
	payload := requesthandling.PayloadMap{
		"kv_transfer_params": map[string]any{
			"max_load_tokens":  json.Number("0"),
			"remote_engine_id": "engine-a",
		},
	}
	body := &requesthandling.InferenceRequestBody{Payload: payload}
	request := &scheduling.InferenceRequest{Body: body}
	result := thresholdResult(p, map[string]int{"cpu": 20})

	require.NoError(t, p.PreRequest(context.Background(), request, result))
	assert.Equal(t, map[string]any{"remote_engine_id": "engine-a"},
		payload["kv_transfer_params"])
	assert.True(t, body.Mutated)
}

func TestSelectiveKVThresholdPolicyConsumesConfiguredData(t *testing.T) {
	p, err := New("test", Config{
		LoadPolicy:                  PolicyThreshold,
		OffloadPolicy:               PolicyPreserve,
		MinExternalReusableTokens:   1,
		MaxWaitingRequests:          8,
		PrefixMatchInfoProducerName: "precise",
	})
	require.NoError(t, err)

	dependencies := p.Consumes()
	assert.Contains(t, dependencies.Required, p.prefixMatchInfoDataKey)
	assert.Contains(t, dependencies.Required, plugin.NewDataKey(
		extractormetrics.WaitingQueueSizeKey,
		extractormetrics.MetricsExtractorType))
	assert.Empty(t, dependencies.Optional)
}

func TestSelectiveKVPreRequestAppliesIndependentPolicies(t *testing.T) {
	tests := []struct {
		name          string
		loadPolicy    Policy
		offloadPolicy Policy
		initial       requesthandling.PayloadMap
		want          requesthandling.PayloadMap
	}{
		{
			name:          "disable load",
			loadPolicy:    PolicyDisable,
			offloadPolicy: PolicyPreserve,
			initial: requesthandling.PayloadMap{
				"kv_transfer_params": map[string]any{
					"max_offload_tokens": 64,
					"remote_engine_id":   "engine-a",
				},
			},
			want: requesthandling.PayloadMap{
				"kv_transfer_params": map[string]any{
					"max_load_tokens":    json.Number("0"),
					"max_offload_tokens": 64,
					"remote_engine_id":   "engine-a",
				},
			},
		},
		{
			name:          "disable offload",
			loadPolicy:    PolicyPreserve,
			offloadPolicy: PolicyDisable,
			initial: requesthandling.PayloadMap{
				"kv_transfer_params": map[string]any{
					"kv_load_tiers": []any{map[string]any{"medium": "CPU"}},
				},
			},
			want: requesthandling.PayloadMap{
				"kv_transfer_params": map[string]any{
					"kv_load_tiers":      []any{map[string]any{"medium": "CPU"}},
					"max_offload_tokens": json.Number("0"),
				},
			},
		},
		{
			name:          "disable both",
			loadPolicy:    PolicyDisable,
			offloadPolicy: PolicyDisable,
			initial: requesthandling.PayloadMap{
				"model": "test",
			},
			want: requesthandling.PayloadMap{
				"model": "test",
				"kv_transfer_params": map[string]any{
					"max_load_tokens":    json.Number("0"),
					"max_offload_tokens": json.Number("0"),
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			p, err := New("test", Config{
				LoadPolicy:    test.loadPolicy,
				OffloadPolicy: test.offloadPolicy,
			})
			require.NoError(t, err)

			body := &requesthandling.InferenceRequestBody{Payload: test.initial}
			request := &scheduling.InferenceRequest{RequestID: "request-1", Body: body}
			require.NoError(t, p.PreRequest(context.Background(), request, nil))

			assert.Equal(t, test.want, test.initial)
			assert.True(t, body.Mutated)
		})
	}
}

func TestSelectiveKVPreRequestOverwritesClientPolicyAndIsIdempotent(t *testing.T) {
	p, err := New("test", Config{
		LoadPolicy:    PolicyDisable,
		OffloadPolicy: PolicyDisable,
	})
	require.NoError(t, err)

	payload := requesthandling.PayloadMap{
		"model": "test",
		"kv_transfer_params": map[string]any{
			"kv_load_tiers":      []any{map[string]any{"medium": "STORAGE"}},
			"max_load_tokens":    1024,
			"max_offload_tokens": 1024,
			"remote_engine_id":   "engine-a",
		},
	}
	body := &requesthandling.InferenceRequestBody{Payload: payload}
	request := &scheduling.InferenceRequest{RequestID: "request-1", Body: body}

	require.NoError(t, p.PreRequest(context.Background(), request, nil))
	first, err := payload.Marshal()
	require.NoError(t, err)
	assert.True(t, body.Mutated)
	body.Mutated = false
	require.NoError(t, p.PreRequest(context.Background(), request, nil))
	assert.True(t, body.Mutated)

	encoded, err := payload.Marshal()
	require.NoError(t, err)
	assert.JSONEq(t, string(first), string(encoded))
	assert.JSONEq(t, `{
		"model":"test",
		"kv_transfer_params":{
			"kv_load_tiers":[{"medium":"STORAGE"}],
			"max_load_tokens":0,
			"max_offload_tokens":0,
			"remote_engine_id":"engine-a"
		}
	}`, string(encoded))
}

func TestSelectiveKVPreRequestReplacesMalformedTransferParams(t *testing.T) {
	p, err := New("test", Config{
		LoadPolicy:    PolicyDisable,
		OffloadPolicy: PolicyPreserve,
	})
	require.NoError(t, err)

	payload := requesthandling.PayloadMap{"kv_transfer_params": "invalid"}
	body := &requesthandling.InferenceRequestBody{Payload: payload}
	request := &scheduling.InferenceRequest{Body: body}

	require.NoError(t, p.PreRequest(context.Background(), request, nil))
	assert.Equal(t, map[string]any{"max_load_tokens": json.Number("0")},
		payload["kv_transfer_params"])
}

func TestSelectiveKVPreRequestSkipsUnsupportedRequestShapes(t *testing.T) {
	p, err := New("test", Config{
		LoadPolicy:    PolicyDisable,
		OffloadPolicy: PolicyPreserve,
	})
	require.NoError(t, err)

	tests := []struct {
		name    string
		request *scheduling.InferenceRequest
	}{
		{name: "nil request"},
		{
			name:    "nil body",
			request: &scheduling.InferenceRequest{},
		},
		{
			name: "raw payload",
			request: &scheduling.InferenceRequest{Body: &requesthandling.InferenceRequestBody{
				Payload: requesthandling.RawPayload(`{"model":"test"}`),
			}},
		},
		{
			name: "native generate",
			request: &scheduling.InferenceRequest{Body: &requesthandling.InferenceRequestBody{
				Generate: &requesthandling.GenerateRequest{},
				Payload:  requesthandling.PayloadMap{},
			}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.NoError(t, p.PreRequest(context.Background(), test.request, nil))
			if test.request != nil && test.request.Body != nil {
				assert.False(t, test.request.Body.Mutated)
			}
		})
	}
}

func TestSelectiveKVThresholdPolicyFailsOpenForInvalidBlockSize(t *testing.T) {
	cfg := DefaultConfig
	cfg.LoadPolicy = PolicyThreshold
	cfg.MinExternalReusableTokens = 1024
	p, err := New("test", cfg)
	require.NoError(t, err)

	endpoint := scheduling.NewEndpoint(nil, nil, nil)
	endpoint.Put(p.prefixMatchInfoDataKey,
		attrprefix.NewPrefixCacheMatchInfo(0, 0, 0).
			WithCachedBlocksByTier(map[string]int{"cpu": 20}))
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
	payload := requesthandling.PayloadMap{"model": "test"}
	body := &requesthandling.InferenceRequestBody{Payload: payload}

	require.NoError(t, p.PreRequest(context.Background(),
		&scheduling.InferenceRequest{Body: body}, result))
	assert.False(t, body.Mutated)
	assert.NotContains(t, payload, "kv_transfer_params")
}

func thresholdResult(p *Plugin, byTier map[string]int) *scheduling.SchedulingResult {
	endpoint := scheduling.NewEndpoint(nil, nil, nil)
	if byTier != nil {
		endpoint.Put(p.prefixMatchInfoDataKey,
			attrprefix.NewPrefixCacheMatchInfo(0, 0, 64).
				WithCachedBlocksByTier(byTier))
	}
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

func thresholdResultWithQueue(p *Plugin, byTier map[string]int, waiting int,
	updateTime time.Time) *scheduling.SchedulingResult {
	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID: k8stypes.NamespacedName{
				Namespace: "default",
				Name:      "model-server",
			},
		},
		&fwkdl.Metrics{
			WaitingQueueSize: waiting,
			UpdateTime:       updateTime,
		},
		nil,
	)
	if byTier != nil {
		endpoint.Put(p.prefixMatchInfoDataKey,
			attrprefix.NewPrefixCacheMatchInfo(0, 0, 64).
				WithCachedBlocksByTier(byTier))
	}
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

func newThresholdPlugin(t *testing.T) *Plugin {
	t.Helper()
	p, err := PluginFactory("test", plugin.StrictDecoder(json.RawMessage(`{"loadPolicy":"threshold","minExternalReusableTokens":1024}`)), nil)
	require.NoError(t, err)
	return p.(*Plugin)
}
