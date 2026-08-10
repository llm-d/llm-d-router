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

package topologyaffinity

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

func newScorer(t *testing.T, params string) *MultiClusterScorer {
	t.Helper()
	p, err := MultiClusterFactory("test", json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)
	s, ok := p.(*MultiClusterScorer)
	require.True(t, ok)
	return s
}

func TestMultiClusterScorer_WeightByRegion(t *testing.T) {
	s := newScorer(t, `{"weights":{"east":1.0,"west":0.5}}`)

	east := makeEndpoint(t, "east", &attrtopology.Topology{Region: "east"})
	west := makeEndpoint(t, "west", &attrtopology.Topology{Region: "west"})
	unlisted := makeEndpoint(t, "unlisted", &attrtopology.Topology{Region: "central"})
	noTopo := makeEndpoint(t, "no-topo", nil)

	got := s.Score(context.Background(), &fwksched.InferenceRequest{},
		[]fwksched.Endpoint{east, west, unlisted, noTopo})

	assert.Equal(t, 1.0, got[east])
	assert.Equal(t, 0.5, got[west])
	assert.Equal(t, 0.0, got[unlisted], "a region absent from the weights table scores 0, not dropped")
	assert.Equal(t, 0.0, got[noTopo], "an endpoint with no Topology attribute scores 0")
}

func TestMultiClusterScorer_NormalizesWeightsByMax(t *testing.T) {
	// 10 and 5 must not both clamp to 1.0 in the scheduler; normalization by the
	// max keeps the ratio so the preference survives.
	s := newScorer(t, `{"weights":{"east":10,"west":5}}`)

	east := makeEndpoint(t, "east", &attrtopology.Topology{Region: "east"})
	west := makeEndpoint(t, "west", &attrtopology.Topology{Region: "west"})

	got := s.Score(context.Background(), &fwksched.InferenceRequest{},
		[]fwksched.Endpoint{east, west})

	assert.Equal(t, 1.0, got[east])
	assert.Equal(t, 0.5, got[west])
}

func TestMultiClusterScorer_ConfigurableField(t *testing.T) {
	s := newScorer(t, `{"field":"zone","weights":{"z1":1.0}}`)

	inZone := makeEndpoint(t, "in-zone", &attrtopology.Topology{Zone: "z1", Region: "east"})
	otherZone := makeEndpoint(t, "other-zone", &attrtopology.Topology{Zone: "z2", Region: "east"})

	got := s.Score(context.Background(), &fwksched.InferenceRequest{},
		[]fwksched.Endpoint{inZone, otherZone})

	assert.Equal(t, 1.0, got[inZone])
	assert.Equal(t, 0.0, got[otherZone], "weights key on zone, not region, when field is zone")
}

func TestMultiClusterScorer_ProducerScopedKey(t *testing.T) {
	s := newScorer(t, `{"weights":{"east":1.0},"topologyProducerName":"topo2"}`)

	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "east", Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	ep.Put(attrtopology.TopologyAttributeKey.WithNonEmptyProducerName("topo2").String(),
		&attrtopology.Topology{Region: "east"})

	got := s.Score(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{ep})
	assert.Equal(t, 1.0, got[ep], "reads the Topology attribute under the configured producer name")
}

func TestMultiClusterScorer_Category(t *testing.T) {
	s := newScorer(t, `{"weights":{"east":1.0}}`)
	assert.Equal(t, fwksched.Affinity, s.Category())
}

func TestMultiClusterScorer_Consumes(t *testing.T) {
	s := newScorer(t, `{"weights":{"east":1.0}}`)

	consumes := s.Consumes()

	assert.Empty(t, consumes.Required)
	require.Len(t, consumes.Optional, 1)
	assert.Equal(t, attrtopology.Topology{}, consumes.Optional[s.dataKey])
}

func TestMultiClusterFactory_Validation(t *testing.T) {
	tests := []struct {
		name    string
		params  *json.Decoder
		wantErr string
	}{
		{name: "nil params", params: nil, wantErr: "requires parameters"},
		{name: "empty weights", params: json.NewDecoder(strings.NewReader(`{"weights":{}}`)), wantErr: "non-empty weights"},
		{name: "no weights key", params: json.NewDecoder(strings.NewReader(`{"field":"region"}`)), wantErr: "non-empty weights"},
		{name: "negative weight", params: json.NewDecoder(strings.NewReader(`{"weights":{"east":-1}}`)), wantErr: "non-negative"},
		{name: "all-zero weights", params: json.NewDecoder(strings.NewReader(`{"weights":{"east":0}}`)), wantErr: "at least one positive"},
		{name: "bad field", params: json.NewDecoder(strings.NewReader(`{"field":"bogus","weights":{"east":1.0}}`)), wantErr: "must be one of"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := MultiClusterFactory("", tc.params, nil)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestMultiClusterFactory_DefaultFieldAndName(t *testing.T) {
	p, err := MultiClusterFactory("", json.NewDecoder(strings.NewReader(`{"weights":{"east":1.0}}`)), nil)
	require.NoError(t, err)
	s, ok := p.(*MultiClusterScorer)
	require.True(t, ok)
	assert.Equal(t, MultiClusterPluginType, s.TypedName().Name, "name defaults to the type")

	east := makeEndpoint(t, "east", &attrtopology.Topology{Region: "east"})
	got := s.Score(context.Background(), &fwksched.InferenceRequest{}, []fwksched.Endpoint{east})
	assert.Equal(t, 1.0, got[east], "field defaults to region")
}
