package activerequest

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/inflightload"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

// Test helper functions

func float64Ptr(v float64) *float64 { return &v }

func newTestEndpoint(name string, queueSize int) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		&fwkdl.Metrics{
			WaitingQueueSize: queueSize,
		},
		nil,
	)
}

func newTestEndpointWithLoad(name string, requests int64) scheduling.Endpoint {
	ep := newTestEndpoint(name, 0)
	ep.Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{Requests: requests})
	return ep
}

func TestActiveRequestScorer_Score(t *testing.T) {
	tests := []struct {
		name      string
		endpoints func() []scheduling.Endpoint
		want      []float64
	}{
		{
			name: "no load attribute set",
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpoint("pod-a", 2),
					newTestEndpoint("pod-b", 0),
					newTestEndpoint("pod-c", 15),
				}
			},
			want: []float64{1.0, 1.0, 1.0},
		},
		{
			name: "all endpoints have different request counts",
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 3),
					newTestEndpointWithLoad("pod-b", 0),
					newTestEndpointWithLoad("pod-c", 6),
				}
			},
			want: []float64{0.5, 1.0, 0.0},
		},
		{
			name: "some endpoints have load data",
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 4),
					newTestEndpoint("pod-b", 0),
					newTestEndpointWithLoad("pod-c", 1),
				}
			},
			want: []float64{0.0, 1.0, 0.75},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			scorer := NewActiveRequest(ctx, nil)
			endpoints := test.endpoints()

			got := scorer.Score(ctx, nil, nil, endpoints)

			for i, endpoint := range endpoints {
				score, ok := got[endpoint]
				assert.True(t, ok, "expected score for endpoint %v", endpoint)
				assert.Equal(t, test.want[i], score)
			}
		})
	}
}

func TestActiveRequestScorer_UsesInFlightLoadProducerLifecycle(t *testing.T) {
	ctx := utils.NewTestContext(t)

	producerPlugin, err := inflightload.InFlightLoadProducerFactory(inflightload.InFlightLoadProducerType, nil, nil)
	require.NoError(t, err)
	producer := producerPlugin.(*inflightload.InFlightLoadProducer)
	scorer := NewActiveRequest(ctx, nil)

	podA := newTestEndpoint("pod-a", 0)
	podB := newTestEndpoint("pod-b", 0)
	endpoints := []scheduling.Endpoint{podA, podB}

	req := &scheduling.InferenceRequest{RequestID: "req-1", RequestSizeBytes: 4}
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{podA}},
		},
	}

	producer.PreRequest(ctx, req, result)
	require.NoError(t, producer.Produce(ctx, req, endpoints))

	require.Equal(t, int64(1), inFlightRequests(t, podA))
	require.Equal(t, int64(0), inFlightRequests(t, podB))
	scores := scorer.Score(ctx, nil, req, endpoints)
	assert.Equal(t, 0.0, scores[podA])
	assert.Equal(t, 1.0, scores[podB])

	req.SchedulingResult = result
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
	require.NoError(t, producer.Produce(ctx, req, endpoints))

	require.Equal(t, int64(0), inFlightRequests(t, podA))
	require.Equal(t, int64(0), inFlightRequests(t, podB))
	scores = scorer.Score(ctx, nil, req, endpoints)
	assert.Equal(t, 1.0, scores[podA])
	assert.Equal(t, 1.0, scores[podB])
}

func TestNewActiveRequestScorer_DeprecatedRequestTimeoutIgnored(t *testing.T) {
	ctx := utils.NewTestContext(t)

	params := &Parameters{RequestTimeout: "invalid"}
	scorer := NewActiveRequest(ctx, params)

	assert.NotNil(t, scorer, "Expected scorer to be created even with deprecated timeout")
}

func TestActiveRequestScorer_Consumes(t *testing.T) {
	ctx := utils.NewTestContext(t)

	scorer := NewActiveRequest(ctx, nil)
	consumes := scorer.Consumes()

	require.Len(t, consumes, 1)
	assert.Equal(t, attrconcurrency.InFlightLoad{}, consumes[attrconcurrency.InFlightLoadKey])
}

func TestActiveRequestScorer_TypedName(t *testing.T) {
	ctx := utils.NewTestContext(t)

	scorer := NewActiveRequest(ctx, nil)

	assert.Equal(t, ActiveRequestType, scorer.TypedName().Type)
}

func TestActiveRequestScorer_WithName(t *testing.T) {
	ctx := utils.NewTestContext(t)

	scorer := NewActiveRequest(ctx, nil)
	testName := "test-scorer"

	scorer = scorer.WithName(testName)

	assert.Equal(t, testName, scorer.TypedName().Name)
}

func TestActiveRequest_IdleThresholdAndMaxBusyScore(t *testing.T) {
	ctx := utils.NewTestContext(t)

	t.Run("binary mode: idleThreshold=0, maxBusyScore=0", func(t *testing.T) {
		params := &Parameters{
			IdleThreshold: 0,
			MaxBusyScore:  float64Ptr(0.0),
		}
		scorer := NewActiveRequest(ctx, params)

		podA := newTestEndpoint("pod-a", 0)
		podB := newTestEndpoint("pod-b", 0)

		// Both idle, so both score 1.0.
		scores := scorer.Score(ctx, nil, nil, []scheduling.Endpoint{podA, podB})
		assert.Equal(t, 1.0, scores[podA])
		assert.Equal(t, 1.0, scores[podB])

		podA.Put(attrconcurrency.InFlightLoadKey, &attrconcurrency.InFlightLoad{Requests: 1})

		scores = scorer.Score(ctx, nil, nil, []scheduling.Endpoint{podA, podB})
		assert.Equal(t, 0.0, scores[podA], "Busy pod scores 0.0 in binary mode")
		assert.Equal(t, 1.0, scores[podB], "Idle pod scores 1.0")
	})

	t.Run("hybrid mode: idleThreshold=1, maxBusyScore=0.5", func(t *testing.T) {
		params := &Parameters{
			IdleThreshold: 1,
			MaxBusyScore:  float64Ptr(0.5),
		}
		scorer := NewActiveRequest(ctx, params)

		podA := newTestEndpointWithLoad("pod-a", 1)
		podB := newTestEndpointWithLoad("pod-b", 2)
		podC := newTestEndpoint("pod-c", 0)

		scores := scorer.Score(ctx, nil, nil, []scheduling.Endpoint{podA, podB, podC})
		assert.Equal(t, 1.0, scores[podA], "Pod with 1 request is idle (threshold=1)")
		assert.Equal(t, 0.0, scores[podB], "Pod with 2 requests (busiest) scores 0.0")
		assert.Equal(t, 1.0, scores[podC], "Pod with 0 requests is idle")
	})
}

// TestActiveRequest_DefaultParamsProduceContinuousScores guards against the
// regression where an unset MaxBusyScore (Go zero-value 0.0) silently put the
// scorer into binary mode, returning 0.0 for every non-idle pod.
func TestActiveRequest_DefaultParamsProduceContinuousScores(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{})

	podLight := newTestEndpointWithLoad("pod-light", 3)
	podHeavy := newTestEndpointWithLoad("pod-heavy", 11)

	scores := scorer.Score(ctx, nil, nil, []scheduling.Endpoint{podLight, podHeavy})

	assert.InDelta(t, 0.7272, scores[podLight], 0.001,
		"light pod must get a non-zero score when no parameters are configured")
	assert.Equal(t, 0.0, scores[podHeavy])
}

func inFlightRequests(t *testing.T, endpoint scheduling.Endpoint) int64 {
	t.Helper()

	val, ok := endpoint.Get(attrconcurrency.InFlightLoadKey)
	require.True(t, ok)
	load, ok := val.(*attrconcurrency.InFlightLoad)
	require.True(t, ok)
	return load.Requests
}
