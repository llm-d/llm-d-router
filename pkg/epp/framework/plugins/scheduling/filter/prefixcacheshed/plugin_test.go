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

package prefixcacheshed

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

// fakeDetector returns a fixed saturation and marks named endpoints as full.
type fakeDetector struct {
	saturation float64
	full       map[string]bool
	poolSizes  []int
}

func (f *fakeDetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake-detector", Name: "fake"}
}

func (f *fakeDetector) Saturation(_ context.Context, endpoints []fwkdl.Endpoint) float64 {
	f.poolSizes = append(f.poolSizes, len(endpoints))
	return f.saturation
}

func (f *fakeDetector) HasCapacity(ep fwksched.Endpoint) bool {
	return !f.full[ep.GetMetadata().ID.Name]
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

func names(endpoints []fwksched.Endpoint) []string {
	out := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		out = append(out, ep.GetMetadata().ID.Name)
	}
	return out
}

func drop(request *fwksched.InferenceRequest) *fwksched.RequestDrop {
	d, ok := fwksched.ReadRequestAttribute[*fwksched.RequestDrop](request, fwksched.RequestDropDataKey)
	if !ok {
		return nil
	}
	return d
}

func validConfig() Config {
	c := DefaultConfig
	c.SaturationDetectorRef = "det"
	return c
}

func TestFilterDecisions(t *testing.T) {
	// Pool: hot has a 90% hit, warm 60%, cold 10%.
	hot := makeEndpoint("hot", 90, 1000)
	warm := makeEndpoint("warm", 60, 4000)
	cold := makeEndpoint("cold", 10, 9000)
	all := []fwksched.Endpoint{hot, warm, cold}

	tests := []struct {
		name       string
		config     func(Config) Config
		saturation float64
		full       map[string]bool
		endpoints  []fwksched.Endpoint
		want       []string
		wantReason errcommon.RequestDroppedReason
		wantCode   string
	}{
		{
			name:       "below threshold passes everything through",
			saturation: 0.84,
			full:       map[string]bool{"hot": true},
			endpoints:  all,
			want:       []string{"hot", "warm", "cold"},
		},
		{
			name:       "at threshold keeps hot endpoints with room",
			saturation: 0.85,
			endpoints:  all,
			want:       []string{"hot", "warm"},
		},
		{
			name:       "hot endpoint full, warm one has room",
			saturation: 0.9,
			full:       map[string]bool{"hot": true},
			endpoints:  all,
			want:       []string{"warm"},
		},
		{
			name:       "all hot endpoints full is a displaced hot shed",
			saturation: 0.9,
			full:       map[string]bool{"hot": true, "warm": true},
			endpoints:  all,
			want:       []string{},
			wantReason: errcommon.RequestDroppedReasonShedDisplacedHot,
			wantCode:   errcommon.ServiceUnavailable,
		},
		{
			name:       "no hit anywhere is a cold shed",
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{cold, makeEndpoint("colder", 0, 9000)},
			want:       []string{},
			wantReason: errcommon.RequestDroppedReasonShedCold,
			wantCode:   errcommon.ServiceUnavailable,
		},
		{
			name:       "missing prefix attribute counts as cold",
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{makeEndpoint("unknown", -1, 9000)},
			want:       []string{},
			wantReason: errcommon.RequestDroppedReasonShedCold,
			wantCode:   errcommon.ServiceUnavailable,
		},
		{
			name:       "429 mode selects ResourceExhausted",
			config:     func(c Config) Config { c.ShedStatusCode = http.StatusTooManyRequests; return c },
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{cold},
			want:       []string{},
			wantReason: errcommon.RequestDroppedReasonShedCold,
			wantCode:   errcommon.ResourceExhausted,
		},
		{
			name:       "higher minHitRatio turns warm into cold",
			config:     func(c Config) Config { c.MinHitRatio = 0.8; return c },
			saturation: 0.9,
			full:       map[string]bool{"hot": true},
			endpoints:  all,
			want:       []string{},
			wantReason: errcommon.RequestDroppedReasonShedDisplacedHot,
			wantCode:   errcommon.ServiceUnavailable,
		},
		{
			name:       "small cold request is exempt",
			config:     func(c Config) Config { c.MinUncachedTokens = 500; return c },
			saturation: 0.9,
			endpoints:  []fwksched.Endpoint{makeEndpoint("c1", 0, 400), makeEndpoint("c2", 0, 9000)},
			want:       []string{"c1", "c2"},
		},
		{
			name:       "small request exemption ignores endpoints without room",
			config:     func(c Config) Config { c.MinUncachedTokens = 500; return c },
			saturation: 0.9,
			full:       map[string]bool{"c1": true},
			endpoints:  []fwksched.Endpoint{makeEndpoint("c1", 0, 400), makeEndpoint("c2", 0, 9000)},
			want:       []string{},
			wantReason: errcommon.RequestDroppedReasonShedCold,
			wantCode:   errcommon.ServiceUnavailable,
		},
		{
			name:       "single hot endpoint with room is kept",
			saturation: 1.0,
			endpoints:  []fwksched.Endpoint{hot},
			want:       []string{"hot"},
		},
		{
			name:       "empty input is returned as is",
			saturation: 1.0,
			endpoints:  []fwksched.Endpoint{},
			want:       []string{},
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
			det := &fakeDetector{saturation: tc.saturation, full: tc.full}
			p := New("test", config, det)
			request := &fwksched.InferenceRequest{RequestID: "r1"}

			got := p.Filter(context.Background(), request, tc.endpoints)

			assert.Equal(t, tc.want, names(got))
			d := drop(request)
			if tc.wantReason == "" {
				assert.Nil(t, d, "no drop expected")
			} else {
				require.NotNil(t, d, "drop expected")
				assert.Equal(t, string(tc.wantReason), d.Reason)
				assert.Equal(t, tc.wantCode, d.Code)
				assert.NotEmpty(t, d.Msg)
			}
			if len(tc.endpoints) > 0 {
				assert.Equal(t, []int{len(tc.endpoints)}, det.poolSizes, "saturation is read over the full input")
			}
		})
	}
}

func TestFilterHysteresis(t *testing.T) {
	resetMetrics()
	config := validConfig() // T = 0.85, hysteresis 0.05
	det := &fakeDetector{}
	p := New("test", config, det)
	cold := []fwksched.Endpoint{makeEndpoint("cold", 0, 9000)}

	step := func(saturation float64) int {
		det.saturation = saturation
		return len(p.Filter(context.Background(), &fwksched.InferenceRequest{}, cold))
	}

	assert.Equal(t, 1, step(0.84), "below T: pass")
	assert.False(t, p.Shedding())
	assert.Equal(t, 0, step(0.85), "reach T: shed")
	assert.True(t, p.Shedding())
	assert.Equal(t, 0, step(0.82), "inside the band: still shedding")
	assert.True(t, p.Shedding())
	assert.Equal(t, 0, step(0.80), "at T - hysteresis: still shedding")
	assert.Equal(t, 1, step(0.79), "below T - hysteresis: pass")
	assert.False(t, p.Shedding())
	assert.Equal(t, 1, step(0.84), "below T again: pass")
	assert.False(t, p.Shedding())
}

func TestFilterZeroHysteresis(t *testing.T) {
	resetMetrics()
	config := validConfig()
	config.Hysteresis = 0
	det := &fakeDetector{}
	p := New("test", config, det)
	cold := []fwksched.Endpoint{makeEndpoint("cold", 0, 9000)}

	det.saturation = 0.85
	assert.Empty(t, p.Filter(context.Background(), &fwksched.InferenceRequest{}, cold))
	det.saturation = 0.849
	assert.Len(t, p.Filter(context.Background(), &fwksched.InferenceRequest{}, cold), 1)
}

func TestFilterNilRequest(t *testing.T) {
	resetMetrics()
	p := New("test", validConfig(), &fakeDetector{saturation: 1.0})
	assert.Empty(t, p.Filter(context.Background(), nil, []fwksched.Endpoint{makeEndpoint("cold", 0, 1)}))
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
	assert.Contains(t, p.Produces(), fwksched.RequestDropDataKey)
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
		assert.Equal(t, errcommon.ServiceUnavailable, p.errorCode)

		// End to end with the real detector: one endpoint at its request limit, one idle.
		busy := makeEndpoint("busy", 90, 10)
		busy.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 2})
		idle := makeEndpoint("idle", 0, 10)
		// Pool saturation = avg(max(2/2, 0), 0) = 0.5 >= T.
		got := p.Filter(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{busy, idle})
		assert.Empty(t, got, "hot endpoint is full and the idle one is cold")
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
