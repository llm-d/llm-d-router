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

package diffusioncost

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrdiffusion "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/diffusion"
)

type stubEndpoint struct {
	metadata *datalayer.EndpointMetadata
	attr     datalayer.AttributeMap
}

func newStubEndpoint(name string) *stubEndpoint {
	return &stubEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubEndpoint) GetMetadata() *datalayer.EndpointMetadata   { return f.metadata }
func (f *stubEndpoint) UpdateMetadata(*datalayer.EndpointMetadata) {}
func (f *stubEndpoint) GetMetrics() *datalayer.Metrics             { return nil }
func (f *stubEndpoint) UpdateMetrics(*datalayer.Metrics)           {}
func (f *stubEndpoint) GetAttributes() datalayer.AttributeMap      { return f.attr }
func (f *stubEndpoint) String() string                             { return f.metadata.NamespacedName.String() }
func (f *stubEndpoint) Put(key string, val datalayer.Cloneable)    { f.attr.Put(key, val) }
func (f *stubEndpoint) Get(key string) (datalayer.Cloneable, bool) { return f.attr.Get(key) }
func (f *stubEndpoint) Keys() []string                             { return f.attr.Keys() }
func (f *stubEndpoint) Clone() datalayer.AttributeMap              { return f.attr.Clone() }

func newTestEndpoint(name string) scheduling.Endpoint {
	return newStubEndpoint(name)
}

func newTestEndpointWithCost(name string, cost int64) scheduling.Endpoint {
	ep := newStubEndpoint(name)
	ep.Put(attrdiffusion.DiffusionLoadDataKey.String(), &attrdiffusion.DiffusionLoad{Cost: cost})
	return ep
}

func TestDiffusionCostScorer_Score(t *testing.T) {
	tests := []struct {
		name      string
		endpoints []scheduling.Endpoint
		want      []float64
	}{
		{
			name: "no load attribute set",
			endpoints: []scheduling.Endpoint{
				newTestEndpoint("pod-a"),
				newTestEndpoint("pod-b"),
			},
			want: []float64{1.0, 1.0},
		},
		{
			name: "endpoints with different outstanding cost",
			endpoints: []scheduling.Endpoint{
				newTestEndpointWithCost("pod-a", 50),
				newTestEndpointWithCost("pod-b", 0),
				newTestEndpointWithCost("pod-c", 100),
			},
			want: []float64{0.5, 1.0, 0.0},
		},
		{
			name: "some endpoints have cost data",
			endpoints: []scheduling.Endpoint{
				newTestEndpointWithCost("pod-a", 40),
				newTestEndpoint("pod-b"),
				newTestEndpointWithCost("pod-c", 10),
			},
			want: []float64{0.0, 1.0, 0.75},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer := NewDiffusionCost(nil).WithName(DiffusionCostScorerType)
			scores := scorer.Score(context.Background(), nil, tt.endpoints)
			require.Len(t, scores, len(tt.endpoints))
			for i, endpoint := range tt.endpoints {
				assert.InDelta(t, tt.want[i], scores[endpoint], 1e-9, "endpoint %s", endpoint.GetMetadata().NamespacedName.Name)
			}
		})
	}
}

func TestDiffusionCostScorer_ProducerName(t *testing.T) {
	const producerName = "my-diffusion-producer"
	raw, err := json.Marshal(Parameters{DiffusionLoadProducerName: producerName})
	require.NoError(t, err)
	p, err := Factory(DiffusionCostScorerType, json.NewDecoder(bytes.NewReader(raw)), nil)
	require.NoError(t, err)
	scorer := p.(*DiffusionCost)

	namedKey := attrdiffusion.DiffusionLoadDataKey.WithNonEmptyProducerName(producerName)
	require.Contains(t, scorer.Consumes().Required, namedKey)

	// Cost under the named key is read; cost under the default key is ignored.
	ep := newStubEndpoint("pod-a")
	ep.Put(namedKey.String(), &attrdiffusion.DiffusionLoad{Cost: 7})
	require.Equal(t, int64(7), scorer.declaredCost(context.Background(), ep))

	other := newStubEndpoint("pod-b")
	other.Put(attrdiffusion.DiffusionLoadDataKey.String(), &attrdiffusion.DiffusionLoad{Cost: 7})
	require.Equal(t, int64(0), scorer.declaredCost(context.Background(), other))
}
