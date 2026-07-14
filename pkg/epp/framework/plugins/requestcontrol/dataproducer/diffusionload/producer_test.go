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

package diffusionload

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrdiffusion "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/diffusion"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func newTestProducer(t testing.TB, cfg Config) *DiffusionLoadProducer {
	raw, err := json.Marshal(cfg)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	p, err := DiffusionLoadProducerFactory(DiffusionLoadProducerType, json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	return p.(*DiffusionLoadProducer)
}

type stubSchedulingEndpoint struct {
	metadata *datalayer.EndpointMetadata
	attr     datalayer.AttributeMap
}

func newStubSchedulingEndpoint(name string) *stubSchedulingEndpoint {
	return &stubSchedulingEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubSchedulingEndpoint) GetMetadata() *datalayer.EndpointMetadata   { return f.metadata }
func (f *stubSchedulingEndpoint) UpdateMetadata(*datalayer.EndpointMetadata) {}
func (f *stubSchedulingEndpoint) GetMetrics() *datalayer.Metrics             { return nil }
func (f *stubSchedulingEndpoint) UpdateMetrics(*datalayer.Metrics)           {}
func (f *stubSchedulingEndpoint) GetAttributes() datalayer.AttributeMap      { return f.attr }
func (f *stubSchedulingEndpoint) String() string                             { return "" }
func (f *stubSchedulingEndpoint) Put(key string, val datalayer.Cloneable)    { f.attr.Put(key, val) }
func (f *stubSchedulingEndpoint) Get(key string) (datalayer.Cloneable, bool) { return f.attr.Get(key) }
func (f *stubSchedulingEndpoint) Keys() []string                             { return f.attr.Keys() }
func (f *stubSchedulingEndpoint) Clone() datalayer.AttributeMap              { return f.attr.Clone() }

func makeImagesRequest(id string, images *fwkrh.ImagesGenerationsRequest) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: id,
		Body:      &fwkrh.InferenceRequestBody{Images: images},
	}
}

func makeSchedulingResult(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func TestDiffusionLoadProducer_RequestCost(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{})

	tests := []struct {
		name    string
		request *fwksched.InferenceRequest
		want    int64
	}{
		{
			name:    "non-image request costs nothing",
			request: &fwksched.InferenceRequest{RequestID: "r", Body: &fwkrh.InferenceRequestBody{}},
			want:    0,
		},
		{
			name:    "nil body costs nothing",
			request: &fwksched.InferenceRequest{RequestID: "r"},
			want:    0,
		},
		{
			name: "declared fields",
			request: makeImagesRequest("r", &fwkrh.ImagesGenerationsRequest{
				Prompt:            "p",
				NumInferenceSteps: ptr.To[int64](30),
				Size:              "1024x1024",
				N:                 ptr.To[int64](2),
			}),
			want: 60, // 30 steps x 1 MP x 2 images
		},
		{
			name: "all fields defaulted",
			request: makeImagesRequest("r", &fwkrh.ImagesGenerationsRequest{
				Prompt: "p",
			}),
			want: 50, // 50 default steps x 1 MP default size x 1 image
		},
		{
			name: "sub-megapixel size scales down",
			request: makeImagesRequest("r", &fwkrh.ImagesGenerationsRequest{
				Prompt:            "p",
				NumInferenceSteps: ptr.To[int64](20),
				Size:              "512x512",
			}),
			want: 5, // 20 steps x 0.25 MP
		},
		{
			name: "malformed size falls back to default",
			request: makeImagesRequest("r", &fwkrh.ImagesGenerationsRequest{
				Prompt:            "p",
				NumInferenceSteps: ptr.To[int64](10),
				Size:              "huge",
			}),
			want: 10,
		},
		{
			name: "tiny request clamps to one unit",
			request: makeImagesRequest("r", &fwkrh.ImagesGenerationsRequest{
				Prompt:            "p",
				NumInferenceSteps: ptr.To[int64](1),
				Size:              "64x64",
			}),
			want: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, producer.requestCost(tt.request))
		})
	}
}

func TestDiffusionLoadProducer_ConfiguredDefaults(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{DefaultNumInferenceSteps: 8, DefaultSize: "512x512"})

	cost := producer.requestCost(makeImagesRequest("r", &fwkrh.ImagesGenerationsRequest{Prompt: "p"}))
	require.Equal(t, int64(2), cost) // 8 steps x 0.25 MP
}

func TestDiffusionLoadProducerFactory_InvalidConfig(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	raw, err := json.Marshal(Config{DefaultSize: "not-a-size"})
	require.NoError(t, err)
	_, err = DiffusionLoadProducerFactory(DiffusionLoadProducerType, json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
	require.Error(t, err)

	raw, err = json.Marshal(Config{DefaultNumInferenceSteps: -1})
	require.NoError(t, err)
	_, err = DiffusionLoadProducerFactory(DiffusionLoadProducerType, json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
	require.Error(t, err)
}

func TestDiffusionLoadProducer_Lifecycle(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{})
	ctx := context.Background()

	endpoint := newStubSchedulingEndpoint("pod-a")
	eid := endpoint.GetMetadata().NamespacedName.String()
	result := makeSchedulingResult(endpoint)

	request := makeImagesRequest("req-1", &fwkrh.ImagesGenerationsRequest{
		Prompt:            "p",
		NumInferenceSteps: ptr.To[int64](30),
		Size:              "1024x1024",
	})

	// PreRequest commits the declared cost to the chosen endpoint.
	producer.PreRequest(ctx, request, result)
	require.Equal(t, int64(30), producer.GetCostUnits(eid))

	// A second request accumulates.
	request2 := makeImagesRequest("req-2", &fwkrh.ImagesGenerationsRequest{
		Prompt:            "p",
		NumInferenceSteps: ptr.To[int64](10),
	})
	producer.PreRequest(ctx, request2, result)
	require.Equal(t, int64(40), producer.GetCostUnits(eid))

	// Intermediate chunk does not release.
	producer.ResponseBody(ctx, request, &requestcontrol.Response{RequestID: "req-1"}, nil)
	require.Equal(t, int64(40), producer.GetCostUnits(eid))

	// EndOfStream releases exactly the request's own contribution.
	producer.ResponseBody(ctx, request, &requestcontrol.Response{RequestID: "req-1", EndOfStream: true}, nil)
	require.Equal(t, int64(10), producer.GetCostUnits(eid))

	// Releasing the same request again is a no-op.
	producer.ResponseBody(ctx, request, &requestcontrol.Response{RequestID: "req-1", EndOfStream: true}, nil)
	require.Equal(t, int64(10), producer.GetCostUnits(eid))

	producer.ResponseBody(ctx, request2, &requestcontrol.Response{RequestID: "req-2", EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.GetCostUnits(eid))
}

func TestDiffusionLoadProducer_NonImageRequestNotTracked(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{})
	endpoint := newStubSchedulingEndpoint("pod-a")
	eid := endpoint.GetMetadata().NamespacedName.String()

	request := &fwksched.InferenceRequest{RequestID: "req-1", Body: &fwkrh.InferenceRequestBody{}}
	producer.PreRequest(context.Background(), request, makeSchedulingResult(endpoint))
	require.Equal(t, int64(0), producer.GetCostUnits(eid))
}

func TestDiffusionLoadProducer_Extract(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{})
	ctx := context.Background()

	endpoint := newStubSchedulingEndpoint("pod-a")
	eid := endpoint.GetMetadata().NamespacedName.String()

	// Registration does not publish an attribute; Produce does that per request.
	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{Type: datalayer.EventAddOrUpdate, Endpoint: endpoint}))
	_, ok := endpoint.Get(producer.dk.String())
	require.False(t, ok)

	// Commit some cost, then deletion cleans up the tracker.
	request := makeImagesRequest("req-1", &fwkrh.ImagesGenerationsRequest{Prompt: "p", NumInferenceSteps: ptr.To[int64](30)})
	producer.PreRequest(ctx, request, makeSchedulingResult(endpoint))
	require.Equal(t, int64(30), producer.GetCostUnits(eid))

	require.NoError(t, producer.Extract(ctx, datalayer.EndpointEvent{Type: datalayer.EventDelete, Endpoint: endpoint}))
	require.Equal(t, int64(0), producer.GetCostUnits(eid))
}

func TestDiffusionLoadProducer_Produce(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{})
	ctx := context.Background()

	endpoint := newStubSchedulingEndpoint("pod-a")

	// Before any cost is committed, Produce publishes a zero snapshot.
	require.NoError(t, producer.Produce(ctx, nil, []fwksched.Endpoint{endpoint}))
	val, ok := endpoint.Get(producer.dk.String())
	require.True(t, ok)
	require.Equal(t, int64(0), val.(*attrdiffusion.DiffusionLoad).CostUnits)

	// After committing cost, Produce publishes the live snapshot.
	request := makeImagesRequest("req-1", &fwkrh.ImagesGenerationsRequest{Prompt: "p", NumInferenceSteps: ptr.To[int64](30)})
	producer.PreRequest(ctx, request, makeSchedulingResult(endpoint))

	require.NoError(t, producer.Produce(ctx, nil, []fwksched.Endpoint{endpoint}))
	val, ok = endpoint.Get(producer.dk.String())
	require.True(t, ok)
	require.Equal(t, int64(30), val.(*attrdiffusion.DiffusionLoad).CostUnits)
}

func TestDiffusionLoadProducer_Produces(t *testing.T) {
	t.Parallel()

	producer := newTestProducer(t, Config{})
	produces := producer.Produces()
	require.Contains(t, produces, producer.dk)
}
