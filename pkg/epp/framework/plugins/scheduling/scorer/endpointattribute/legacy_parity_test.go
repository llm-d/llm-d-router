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

package endpointattribute

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/go-logr/logr"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	extractormetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/selectivekv"
	filterendpointattribute "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/endpointattribute"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/runningrequests"
)

// Behavioural parity for llm-d-router#2059 item 1 and built-in scalar metric
// attribute publication.
//
// Loading a config proves nothing on its own because endpoint-attribute-scorer
// declares its dependency as Optional, so an unpopulated attribute loads cleanly
// and scores every endpoint 0.0. The checks here verify both direct score parity
// and end-to-end parity from real core-metrics-extractor scrapes (with no
// customMetrics duplication) through datalayer.ValidateAndOrderDataDependencies
// and datalayer.Scope.

func endpointWith(queue int, kvUsage float64) fwksched.Endpoint {
	return endpointWithAll(queue, 0, kvUsage)
}

func endpointWithAll(queue, running int, kvUsage float64) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(attrmetrics.ScalarMetricDataKey(extractormetrics.WaitingQueueSizeKey),
		attrmetrics.ScalarMetricValue(float64(queue)))
	attrs.Put(attrmetrics.ScalarMetricDataKey(extractormetrics.RunningRequestsSizeKey),
		attrmetrics.ScalarMetricValue(float64(running)))
	attrs.Put(attrmetrics.ScalarMetricDataKey(extractormetrics.KVCacheUsagePercentKey),
		attrmetrics.ScalarMetricValue(kvUsage))
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{},
		&fwkdl.Metrics{
			WaitingQueueSize:    queue,
			RunningRequestsSize: running,
			KVCacheUsagePercent: kvUsage,
			UpdateTime:          time.Now(),
		},
		attrs,
	)
}

func adaptiveParams(attributeKey string) parameters {
	return parameters{
		AttributeKey: attributeKey,
		Algorithm: algorithmParameters{
			Type:          algorithmLinearLowerIsBetter,
			Normalization: normalizationParameters{AdaptiveRange: &adaptiveRangeParameters{}},
		},
	}
}

func fixedParams(attributeKey string, min, max float64) parameters {
	return parameters{
		AttributeKey: attributeKey,
		Algorithm: algorithmParameters{
			Type:          algorithmLinearLowerIsBetter,
			Normalization: normalizationParameters{FixedRange: &fixedRangeParameters{Min: min, Max: max}},
		},
	}
}

func TestParityQueueScorer(t *testing.T) {
	cases := [][]fwksched.Endpoint{
		{endpointWith(0, 0.1), endpointWith(5, 0.2), endpointWith(10, 0.3)},
		{endpointWith(3, 0.1), endpointWith(3, 0.2)}, // all equal: neutral 1.0
		{endpointWith(0, 0.0), endpointWith(1, 0.0)},
		{endpointWith(7, 0.5)}, // single endpoint
	}
	legacy := queuedepth.NewQueueScorer()
	ea, err := NewEndpointAttributeScorer("ea-queue", adaptiveParams(extractormetrics.WaitingQueueSizeKey))
	require.NoError(t, err)

	for i, endpoints := range cases {
		want := legacy.Score(t.Context(), nil, endpoints)
		got := ea.Score(t.Context(), nil, endpoints)
		require.Len(t, got, len(want), "case %d: score map size", i)
		for ep, wantScore := range want {
			assert.InDelta(t, wantScore, got[ep], 1e-9,
				"case %d: queue parity diverges", i)
		}
	}
}

func TestParityRunningRequestsScorer(t *testing.T) {
	cases := [][]fwksched.Endpoint{
		{endpointWithAll(0, 0, 0.1), endpointWithAll(0, 4, 0.2), endpointWithAll(0, 8, 0.3)},
		{endpointWithAll(0, 6, 0.1), endpointWithAll(0, 6, 0.2)}, // all equal: neutral 1.0
		{endpointWithAll(0, 0, 0.0), endpointWithAll(0, 2, 0.0)},
		{endpointWithAll(0, 9, 0.5)}, // single endpoint
	}
	legacy := runningrequests.NewRunningRequestsSizeScorer()
	ea, err := NewEndpointAttributeScorer("ea-running", adaptiveParams(extractormetrics.RunningRequestsSizeKey))
	require.NoError(t, err)

	for i, endpoints := range cases {
		want := legacy.Score(t.Context(), nil, endpoints)
		got := ea.Score(t.Context(), nil, endpoints)
		require.Len(t, got, len(want), "case %d: score map size", i)
		for ep, wantScore := range want {
			assert.InDelta(t, wantScore, got[ep], 1e-9,
				"case %d: running-requests parity diverges", i)
		}
	}
}

// TestDivergenceWhenAttributeAbsent locks in the asymmetry when an endpoint has
// Metrics populated without a matching AttributeMap entry (for instance, if a
// custom producer or synthetic fixture writes Metrics alone).
//
// The legacy scorer reads WaitingQueueSize from Metrics, sees 0, and treats an
// empty queue as the best possible endpoint. The endpoint-attribute scorer
// finds no attribute and returns 0.0, the worst.
func TestDivergenceWhenAttributeAbsent(t *testing.T) {
	noAttr := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{},
		&fwkdl.Metrics{WaitingQueueSize: 0, KVCacheUsagePercent: 0.0, UpdateTime: time.Now()},
		fwkdl.NewAttributes(),
	)
	endpoints := []fwksched.Endpoint{noAttr}

	legacy := queuedepth.NewQueueScorer()
	ea, err := NewEndpointAttributeScorer("ea-queue", adaptiveParams(extractormetrics.WaitingQueueSizeKey))
	require.NoError(t, err)

	legacyScores := legacy.Score(t.Context(), nil, endpoints)
	eaScores := ea.Score(t.Context(), nil, endpoints)

	assert.InDelta(t, 1.0, legacyScores[noAttr], 1e-9,
		"legacy reads WaitingQueueSize 0 from Metrics and scores an empty queue best")
	assert.InDelta(t, 0.0, eaScores[noAttr], 1e-9,
		"endpoint-attribute finds no attribute and scores the endpoint worst")
}

func TestParityKVCacheScorer(t *testing.T) {
	cases := [][]fwksched.Endpoint{
		{endpointWith(0, 0.0), endpointWith(0, 0.5), endpointWith(0, 1.0)},
		{endpointWith(0, 0.42), endpointWith(0, 0.42)}, // equal: fixed range keeps the value
		{endpointWith(0, 0.9)},
	}
	legacy := kvcacheutilization.NewKVCacheUtilizationScorer()
	ea, err := NewEndpointAttributeScorer("ea-kv", fixedParams(extractormetrics.KVCacheUsagePercentKey, 0.0, 1.0))
	require.NoError(t, err)

	for i, endpoints := range cases {
		want := legacy.Score(t.Context(), nil, endpoints)
		got := ea.Score(t.Context(), nil, endpoints)
		require.Len(t, got, len(want), "case %d: score map size", i)
		for ep, wantScore := range want {
			assert.InDelta(t, wantScore, got[ep], 1e-9,
				"case %d: kv-cache parity diverges", i)
		}
	}
}

type stubPrefixProducer struct{}

func (stubPrefixProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "stub-prefix", Name: "precise"}
}

func (stubPrefixProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName("precise"): attrprefix.PrefixCacheMatchInfo{},
	}
}

func gaugeMetricFamily(value float64) *dto.MetricFamily {
	return &dto.MetricFamily{
		Type:   dto.MetricType_GAUGE.Enum(),
		Metric: []*dto.Metric{{Gauge: &dto.Gauge{Value: ptr.To(value)}}},
	}
}

// TestEndToEndExtractionParityAndFilteringWithoutCustomMetrics verifies that
// the default core-metrics-extractor (with no customMetrics entries) publishes
// WaitingQueueSize, RunningRequestsSize, and KVCacheUsagePercent as
// ScalarMetricValue endpoint attributes, passes DAG type validation alongside
// legacy consumers, and yields identical scores and threshold filtering through
// datalayer.Scope.
func TestEndToEndExtractionParityAndFilteringWithoutCustomMetrics(t *testing.T) {
	ctx := t.Context()

	rawExtractor, err := extractormetrics.CoreMetricsExtractorFactory(extractormetrics.MetricsExtractorType, nil, nil)
	require.NoError(t, err)
	extractor, ok := rawExtractor.(*extractormetrics.Extractor)
	require.True(t, ok)

	legacyQueue := queuedepth.NewQueueScorer()
	legacyRunning := runningrequests.NewRunningRequestsSizeScorer()
	legacyKV := kvcacheutilization.NewKVCacheUtilizationScorer()

	eaQueue, err := NewEndpointAttributeScorer("ea-queue", adaptiveParams(extractormetrics.WaitingQueueSizeKey))
	require.NoError(t, err)
	eaRunning, err := NewEndpointAttributeScorer("ea-running", adaptiveParams(extractormetrics.RunningRequestsSizeKey))
	require.NoError(t, err)
	eaKV, err := NewEndpointAttributeScorer("ea-kv", fixedParams(extractormetrics.KVCacheUsagePercentKey, 0.0, 1.0))
	require.NoError(t, err)

	runningFilterCfg, err := json.Marshal(map[string]any{
		"attribute": extractormetrics.RunningRequestsSizeKey,
		"onMissing": "Fail",
		"algorithm": map[string]any{
			"type": "threshold",
			"threshold": map[string]any{
				"operator": "LessThan",
				"value":    8.0,
			},
		},
	})
	require.NoError(t, err)
	rawFilter, err := filterendpointattribute.EndpointAttributeFilterFactory(
		"running-filter", fwkplugin.StrictDecoder(runningFilterCfg), nil)
	require.NoError(t, err)
	runningFilter, ok := rawFilter.(fwksched.Filter)
	require.True(t, ok)

	skv, err := selectivekv.New("skv", selectivekv.Config{
		LoadPolicy:                  selectivekv.PolicyThreshold,
		OffloadPolicy:               selectivekv.PolicyPreserve,
		MinExternalReusableTokens:   1,
		MaxWaitingRequests:          8,
		PrefixMatchInfoProducerName: "precise",
	})
	require.NoError(t, err)

	plugins := []fwkplugin.Plugin{
		extractor,
		stubPrefixProducer{},
		legacyQueue,
		legacyRunning,
		legacyKV,
		eaQueue,
		eaRunning,
		eaKV,
		rawFilter,
		skv,
	}
	_, err = datalayer.ValidateAndOrderDataDependencies(plugins)
	require.NoError(t, err, "legacy and generic attribute consumers must coexist without DataKey type conflicts")
	datalayer.RegisterScopeSpecs(plugins)

	scrapes := []struct {
		id      string
		waiting float64
		running float64
		kv      float64
	}{
		{id: "pod-idle", waiting: 0, running: 2, kv: 0.15},
		{id: "pod-mid", waiting: 5, running: 6, kv: 0.50},
		{id: "pod-busy", waiting: 10, running: 10, kv: 0.85},
	}

	dlEndpoints := make([]*fwkdl.ModelServer, 0, len(scrapes)+1)
	for _, s := range scrapes {
		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
			Labels: map[string]string{extractormetrics.DefaultEngineTypeLabelKey: "vllm"},
		}, nil)
		payload := sourcemetrics.PrometheusMetricMap{
			"vllm:num_requests_waiting": gaugeMetricFamily(s.waiting),
			"vllm:num_requests_running": gaugeMetricFamily(s.running),
			"vllm:kv_cache_usage_perc":  gaugeMetricFamily(s.kv),
			"vllm:lora_requests_info":   gaugeMetricFamily(0),
			"vllm:cache_config_info":    gaugeMetricFamily(1),
		}
		require.NoError(t, extractor.Extract(ctx, fwkdl.PollInput[sourcemetrics.PrometheusMetricMap]{
			Payload:  payload,
			Endpoint: ep,
		}))
		dlEndpoints = append(dlEndpoints, ep)
	}
	// Fourth endpoint has never been scraped yet.
	unscraped := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)
	dlEndpoints = append(dlEndpoints, unscraped)

	schedEndpoints := make([]fwksched.Endpoint, len(dlEndpoints))
	for i, ep := range dlEndpoints {
		schedEndpoints[i] = fwksched.NewEndpointFromSnapshot(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes())
	}

	scoreScoped := func(scorer fwksched.Scorer) map[fwksched.Endpoint]float64 {
		scoped, violations := datalayer.Scope(logr.Discard(), "Score", scorer, schedEndpoints)
		rawScores := scorer.Score(ctx, nil, scoped)
		require.NoError(t, violations.Write())
		unscopedScores := make(map[fwksched.Endpoint]float64, len(rawScores))
		for ep, v := range rawScores {
			unscopedScores[datalayer.UnscopeEndpoint(ep)] = v
		}
		return unscopedScores
	}

	scorerPairs := []struct {
		name   string
		legacy fwksched.Scorer
		ea     fwksched.Scorer
	}{
		{name: "WaitingQueueSize", legacy: legacyQueue, ea: eaQueue},
		{name: "RunningRequestsSize", legacy: legacyRunning, ea: eaRunning},
		{name: "KVCacheUsagePercent", legacy: legacyKV, ea: eaKV},
	}

	for _, pair := range scorerPairs {
		want := scoreScoped(pair.legacy)
		got := scoreScoped(pair.ea)

		// All three scraped endpoints must agree exactly between legacy and endpoint-attribute scorers.
		for i := range scrapes {
			ep := schedEndpoints[i]
			assert.InDelta(t, want[ep], got[ep], 1e-9, "%s score mismatch on scraped endpoint %d", pair.name, i)
		}
		// Unscraped endpoint is omitted by legacy (0 contribution) and scored 0.0 by endpoint-attribute
		// without distorting adaptive min-max normalization for the scraped endpoints.
		_, legacyHasUnscraped := want[schedEndpoints[len(scrapes)]]
		assert.False(t, legacyHasUnscraped, "%s legacy scorer must omit unscraped endpoint", pair.name)
		assert.InDelta(t, 0.0, got[schedEndpoints[len(scrapes)]], 1e-9,
			"%s endpoint-attribute scorer must score unscraped endpoint 0.0", pair.name)
	}

	// Verify endpoint-attribute-filter keeps pod-idle (running=2) and pod-mid (running=6),
	// and drops pod-busy (running=10 >= 8) and unscraped (onMissing=Fail).
	scopedForFilter, filterViolations := datalayer.Scope(logr.Discard(), "Filter", runningFilter, schedEndpoints)
	kept := datalayer.Unscope(runningFilter.Filter(ctx, nil, scopedForFilter))
	require.NoError(t, filterViolations.Write())
	assert.Equal(t, []fwksched.Endpoint{schedEndpoints[0], schedEndpoints[1]}, kept)
}
