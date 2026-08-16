package endpointattribute

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	extmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/kvcacheutilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
)

// Behavioural parity for llm-d-router#2059 item 1.
//
// Loading a config proves nothing here. The endpoint-attribute scorer
// declares its dependency as Optional (Consumes(), this file's package), and
// datalayer.Scope only builds the set of keys a plugin is allowed to touch —
// it never checks that something produces what something else consumes. A
// variant whose customMetrics block is wrong, or absent, therefore loads
// cleanly, runs, and scores every endpoint 0.0. That is indistinguishable
// from a scorer that discriminates badly, and a benchmark would report it as
// a result.
//
// So the check is behavioural: both implementations get the same endpoints —
// the value on the Metrics struct the legacy scorers read AND on the
// attribute map the endpoint-attribute scorer reads — and the score maps
// must agree.

func endpointWith(queue int, kvUsage float64) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	attrs.Put(attrmetrics.ScalarMetricDataKey(extmetrics.WaitingQueueSizeKey),
		attrmetrics.ScalarMetricValue(float64(queue)))
	attrs.Put(attrmetrics.ScalarMetricDataKey(extmetrics.KVCacheUsagePercentKey),
		attrmetrics.ScalarMetricValue(kvUsage))
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{},
		&fwkdl.Metrics{WaitingQueueSize: queue, KVCacheUsagePercent: kvUsage},
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
	ea, err := NewEndpointAttributeScorer("ea-queue", adaptiveParams(extmetrics.WaitingQueueSizeKey))
	require.NoError(t, err)

	for i, endpoints := range cases {
		want := legacy.Score(context.Background(), nil, endpoints)
		got := ea.Score(context.Background(), nil, endpoints)
		require.Len(t, got, len(want), "case %d: score map size", i)
		for ep, wantScore := range want {
			assert.InDelta(t, wantScore, got[ep], 1e-9,
				"case %d: queue parity diverges", i)
		}
	}
}

func TestParityKVCacheScorer(t *testing.T) {
	cases := [][]fwksched.Endpoint{
		{endpointWith(0, 0.0), endpointWith(0, 0.5), endpointWith(0, 1.0)},
		{endpointWith(0, 0.42), endpointWith(0, 0.42)}, // equal: fixed range keeps the value
		{endpointWith(0, 0.9)},
	}
	legacy := kvcacheutilization.NewKVCacheUtilizationScorer()
	ea, err := NewEndpointAttributeScorer("ea-kv", fixedParams(extmetrics.KVCacheUsagePercentKey, 0.0, 1.0))
	require.NoError(t, err)

	for i, endpoints := range cases {
		want := legacy.Score(context.Background(), nil, endpoints)
		got := ea.Score(context.Background(), nil, endpoints)
		require.Len(t, got, len(want), "case %d: score map size", i)
		for ep, wantScore := range want {
			assert.InDelta(t, wantScore, got[ep], 1e-9,
				"case %d: kv-cache parity diverges", i)
		}
	}
}
