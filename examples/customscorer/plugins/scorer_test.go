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

package plugins

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestFactoryDefaultParameters(t *testing.T) {
	created, err := Factory("test-scorer", nil, nil)
	require.NoError(t, err)
	require.NotNil(t, created)
	assert.Equal(t, "test-scorer", created.TypedName().Name)
	assert.Equal(t, Type, created.TypedName().Type)
}

func TestFactoryCustomParameters(t *testing.T) {
	dec := plugin.StrictDecoder([]byte(`{"score": 0.75}`))
	created, err := Factory("custom", dec, nil)
	require.NoError(t, err)
	require.NotNil(t, created)

	scorer, ok := created.(*FixedScorer)
	require.True(t, ok)
	assert.Equal(t, 0.75, scorer.score)
}

func TestFactoryInvalidParameters(t *testing.T) {
	dec := plugin.StrictDecoder([]byte(`{"score": 1.5}`))
	_, err := Factory("invalid", dec, nil)
	assert.Error(t, err)
}

func TestScore(t *testing.T) {
	scorer := New("test", 0.5)
	epA := newEndpoint("pod-a")
	epB := newEndpoint("pod-b")

	scores := scorer.Score(context.Background(), nil, []scheduling.Endpoint{epA, epB})
	assert.Equal(t, 0.5, scores[epA])
	assert.Equal(t, 0.5, scores[epB])
}

func newEndpoint(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: name},
		},
		&fwkdl.Metrics{},
		nil,
	)
}
