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

package mmcacheaffinity

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

func TestFactory(t *testing.T) {
	created, err := Factory("mm-scorer", json.RawMessage(`{}`), nil)
	require.NoError(t, err)
	require.NotNil(t, created)
	assert.Equal(t, "mm-scorer", created.TypedName().Name)

	_, err = Factory("bad", json.RawMessage(`{`), nil)
	require.Error(t, err)
}

func TestScorerConsumesMatchInfo(t *testing.T) {
	scorer := New()

	consumes := scorer.Consumes()
	assert.Contains(t, consumes, attrmm.EncoderCacheMatchInfoKey)
	assert.Equal(t, scheduling.Affinity, scorer.Category())
	assert.Equal(t, Type, scorer.TypedName().Type)
}

func TestScoreFromProducedMatchInfo(t *testing.T) {
	scorer := New()
	endpointA := newEndpoint("default", "pod-a")
	endpointB := newEndpoint("default", "pod-b")
	endpointC := newEndpoint("default", "pod-c")
	requestItems := []attrmm.MatchItem{{Hash: "image", Size: 80}, {Hash: "icon", Size: 20}}
	endpointA.Put(attrmm.EncoderCacheMatchInfoKey, attrmm.NewEncoderCacheMatchInfo([]attrmm.MatchItem{{Hash: "image", Size: 80}}, requestItems))
	endpointB.Put(attrmm.EncoderCacheMatchInfoKey, attrmm.NewEncoderCacheMatchInfo([]attrmm.MatchItem{{Hash: "icon", Size: 20}}, requestItems))
	endpointC.Put(attrmm.EncoderCacheMatchInfoKey, attrmm.NewEncoderCacheMatchInfo(nil, requestItems))

	scores := scorer.Score(context.Background(), scheduling.NewCycleState(), nil, []scheduling.Endpoint{endpointA, endpointB, endpointC})

	assert.Equal(t, 0.8, scores[endpointA])
	assert.Equal(t, 0.2, scores[endpointB])
	assert.Equal(t, 0.0, scores[endpointC])
}

func TestScoreMissingOrInvalidMatchInfoReturnsZero(t *testing.T) {
	scorer := New()
	endpointA := newEndpoint("default", "pod-a")
	endpointB := newEndpoint("default", "pod-b")
	endpointB.Put(attrmm.EncoderCacheMatchInfoKey, attrmm.NewEncoderCacheMatchInfo([]attrmm.MatchItem{{Hash: "image", Size: 1}}, nil))

	scores := scorer.Score(context.Background(), scheduling.NewCycleState(), nil, []scheduling.Endpoint{endpointA, endpointB})

	assert.Equal(t, 0.0, scores[endpointA])
	assert.Equal(t, 0.0, scores[endpointB])
}

func newEndpoint(namespace, name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: namespace, Name: name},
		},
		&fwkdl.Metrics{},
		nil,
	)
}
