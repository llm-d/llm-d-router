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

package prefixcacheaware

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
)

// fakeDetector returns a fixed saturation and records the pool size it was asked about.
type fakeDetector struct {
	saturation float64
	poolSizes  []int
}

func (f *fakeDetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-detector", Name: "fake"}
}

func (f *fakeDetector) Saturation(_ context.Context, endpoints []fwkdl.Endpoint) float64 {
	f.poolSizes = append(f.poolSizes, len(endpoints))
	return f.saturation
}

// makeEndpoint creates an endpoint with hit blocks out of 100 and the given
// uncached token projection. hit < 0 leaves the prefix attribute absent.
func makeEndpoint(name string, hit int, uncached int64) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name, Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if hit >= 0 {
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(hit, 100, 16))
	}
	ep.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{})
	ep.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: uncached})
	return ep
}

func validConfig() Config {
	c := DefaultConfig
	c.SaturationDetectorRef = "det"
	return c
}

// denial unwraps the typed error an Admit call returned, or nil when admitted.
func denial(t *testing.T, err error) *errcommon.Error {
	t.Helper()
	if err == nil {
		return nil
	}
	var e errcommon.Error
	require.ErrorAs(t, err, &e, "denial must be a typed errcommon.Error")
	return &e
}

func TestAdmitDecisions(t *testing.T) {
	// Pool: hot has a 90% hit, warm 60%, cold 10%.
	hot := makeEndpoint("hot", 90, 1000)
	warm := makeEndpoint("warm", 60, 4000)
	cold := makeEndpoint("cold", 10, 9000)
	all := []fwksched.Endpoint{hot, warm, cold}

	tests := []struct {
		name       string
		config     func(Config) Config
		saturation float64
		endpoints  []fwksched.Endpoint
		wantCode   string // empty: admitted
	}{
		{
			name:       "below threshold admits a cold request",
			saturation: 0.84,
			endpoints:  []fwksched.Endpoint{cold},
		},
		{
			name:       "at threshold a hot request is admitted",
			saturation: 0.85,
			endpoints:  all,
		},
		{
			name:       "one warm endpoint among cold ones is enough",
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{cold, warm},
		},
		{
			name:       "no hit anywhere is a cold shed",
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{cold, makeEndpoint("colder", 0, 9000)},
			wantCode:   errcommon.ResourceExhausted,
		},
		{
			name:       "missing prefix attribute counts as cold",
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{makeEndpoint("unknown", -1, 9000)},
			wantCode:   errcommon.ResourceExhausted,
		},
		{
			name:       "503 mode selects ServiceUnavailable",
			config:     func(c Config) Config { c.ShedStatusCode = http.StatusServiceUnavailable; return c },
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{cold},
			wantCode:   errcommon.ServiceUnavailable,
		},
		{
			name:       "higher minHitRatio turns warm into cold",
			config:     func(c Config) Config { c.MinHitRatio = 0.8; return c },
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{warm, cold},
			wantCode:   errcommon.ResourceExhausted,
		},
		{
			name:       "small cold request is exempt",
			config:     func(c Config) Config { c.MinUncachedTokens = 500; return c },
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{makeEndpoint("c1", 0, 400), makeEndpoint("c2", 0, 9000)},
		},
		{
			name:       "exemption uses the cheapest endpoint",
			config:     func(c Config) Config { c.MinUncachedTokens = 500; return c },
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{makeEndpoint("c1", 0, 600), makeEndpoint("c2", 0, 9000)},
			wantCode:   errcommon.ResourceExhausted,
		},
		{
			name:       "single hot endpoint is admitted at full saturation",
			saturation: 1.0,
			endpoints:  []fwksched.Endpoint{hot},
		},
		{
			name:       "empty input is admitted",
			saturation: 1.0,
			endpoints:  []fwksched.Endpoint{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resetMetrics()
			config := validConfig()
			if tc.config != nil {
				config = tc.config(config)
			}
			require.NoError(t, config.validate())
			det := &fakeDetector{saturation: tc.saturation}
			p := New("test", config, det)
			request := &fwksched.InferenceRequest{RequestID: "r1"}

			d := denial(t, p.Admit(context.Background(), request, tc.endpoints))

			if tc.wantCode == "" {
				assert.Nil(t, d, "request should be admitted")
			} else {
				require.NotNil(t, d, "request should be denied")
				assert.Equal(t, tc.wantCode, d.Code)
				// The client sees ordinary saturation backpressure, nothing about the cache state.
				assert.Equal(t, string(errcommon.RequestDroppedReasonSaturated), d.Headers[errcommon.RequestDroppedReasonHeaderKey])
				assert.Equal(t, shedMessage, d.Msg)
			}
			if len(tc.endpoints) > 0 {
				assert.Equal(t, []int{len(tc.endpoints)}, det.poolSizes, "saturation is read over the full pool")
			}
		})
	}
}

// TestAdmitThresholdIsStateless checks that each decision depends only on the
// saturation read for that request: no hysteresis, no memory of earlier calls.
func TestAdmitThresholdIsStateless(t *testing.T) {
	resetMetrics()
	det := &fakeDetector{}
	p := New("test", validConfig(), det) // T = 0.85
	cold := []fwksched.Endpoint{makeEndpoint("cold", 0, 9000)}

	admitted := func(saturation float64) bool {
		det.saturation = saturation
		return p.Admit(context.Background(), &fwksched.InferenceRequest{}, cold) == nil
	}

	assert.True(t, admitted(0.849), "below T: admit")
	assert.False(t, admitted(0.85), "at T: shed")
	assert.True(t, admitted(0.849), "below T again: admit immediately")
	assert.False(t, admitted(0.95), "well above T: shed")
	assert.True(t, admitted(0.10), "idle: admit")
}

func TestAdmitNilRequest(t *testing.T) {
	resetMetrics()
	p := New("test", validConfig(), &fakeDetector{saturation: 1.0})
	assert.Error(t, p.Admit(context.Background(), nil, []fwksched.Endpoint{makeEndpoint("cold", 0, 1)}))
}

func TestDataDependencies(t *testing.T) {
	config := validConfig()
	config.PrefixMatchInfoProducerName = "precise"
	config.InFlightLoadProducerName = "loads"
	p := New("test", config, &fakeDetector{})

	deps := p.Consumes()
	assert.Contains(t, deps.Required, attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("precise"))
	assert.Contains(t, deps.Required, attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("loads"))
	assert.Contains(t, deps.Optional, attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName("loads"))
}

func TestFactory(t *testing.T) {
	newHandle := func(t *testing.T, withDetector bool) fwkplugin.Handle {
		handle := fwkplugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil })
		if withDetector {
			det, err := concurrency.ConcurrencyDetectorFactory("concurrency-detector",
				fwkplugin.StrictDecoder([]byte(`{"concurrencyMode":"hybrid","maxConcurrency":2,"maxTokenConcurrency":100}`)), handle)
			require.NoError(t, err)
			handle.AddPlugin("concurrency-detector", det)
		}
		handle.AddPlugin("not-a-detector", &notADetector{})
		return handle
	}

	t.Run("resolves the concurrency detector", func(t *testing.T) {
		plugin, err := Factory("shed", fwkplugin.StrictDecoder([]byte(`{"saturationDetectorRef":"concurrency-detector","shedThreshold":0.5}`)), newHandle(t, true))
		require.NoError(t, err)
		p := plugin.(*Plugin)
		assert.Equal(t, PluginType, p.TypedName().Type)
		assert.Equal(t, "shed", p.TypedName().Name)
		assert.Equal(t, errcommon.ResourceExhausted, p.errorCode)

		// End to end with the real detector: one endpoint at its request limit, one idle.
		// Pool saturation = avg(max(2/2, 0), 0) = 0.5 >= T, so shedding is active.
		busy := makeEndpoint("busy", 90, 10)
		busy.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 2})
		idle := makeEndpoint("idle", 0, 10)
		pool := []fwksched.Endpoint{busy, idle}

		assert.NoError(t, p.Admit(context.Background(), &fwksched.InferenceRequest{}, pool),
			"the prompt is cached on busy; where it is placed is the scheduler's call")

		coldIdle := makeEndpoint("idle", 0, 10)
		coldBusy := makeEndpoint("busy", 0, 10)
		coldBusy.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 2})
		d := denial(t, p.Admit(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{coldBusy, coldIdle}))
		require.NotNil(t, d, "no endpoint has the prefix")
		assert.Equal(t, errcommon.ResourceExhausted, d.Code)
		assert.Equal(t, string(errcommon.RequestDroppedReasonSaturated), d.Headers[errcommon.RequestDroppedReasonHeaderKey])
	})

	t.Run("missing ref", func(t *testing.T) {
		_, err := Factory("shed", fwkplugin.StrictDecoder([]byte(`{"saturationDetectorRef":"nope"}`)), newHandle(t, true))
		require.ErrorContains(t, err, "not found")
	})

	t.Run("ref is not a detector", func(t *testing.T) {
		_, err := Factory("shed", fwkplugin.StrictDecoder([]byte(`{"saturationDetectorRef":"not-a-detector"}`)), newHandle(t, true))
		require.ErrorContains(t, err, "must implement")
	})

	t.Run("nil parameters", func(t *testing.T) {
		_, err := Factory("shed", nil, newHandle(t, true))
		require.ErrorContains(t, err, "saturationDetectorRef is required")
	})

	t.Run("unknown field", func(t *testing.T) {
		_, err := Factory("shed", fwkplugin.StrictDecoder([]byte(`{"saturationDetectorRef":"concurrency-detector","bogus":1}`)), newHandle(t, true))
		require.Error(t, err)
	})

	t.Run("nil handle", func(t *testing.T) {
		_, err := Factory("shed", fwkplugin.StrictDecoder([]byte(`{"saturationDetectorRef":"concurrency-detector"}`)), nil)
		require.Error(t, err)
	})
}

type notADetector struct{}

func (notADetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "x", Name: "not-a-detector"}
}
