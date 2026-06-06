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

package sessionaffinity_test

import (
	"context"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sessionaffinity "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/sessionaffinity"
)

const testProducerName = "test-session-producer"

func newTestEndpoint(name, namespace, address, port string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: namespace},
			Address:        address,
			Port:           port,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func bindingKey() string {
	return attrsession.BoundEndpointDataKey.WithNonEmptyProducerName(testProducerName).String()
}

func boundTo(address, port string) attrsession.BoundEndpoint {
	return attrsession.BoundEndpoint(net.JoinHostPort(address, port))
}

func TestScore(t *testing.T) {
	ep1 := newTestEndpoint("pod-1", "default", "10.0.0.1", "8080")
	ep2 := newTestEndpoint("pod-2", "default", "10.0.0.2", "8080")
	endpoints := []scheduling.Endpoint{ep1, ep2}

	tests := []struct {
		name     string
		bound    attrsession.BoundEndpoint
		expected map[scheduling.Endpoint]float64
	}{
		{
			name:     "no binding scores all zero",
			bound:    "",
			expected: map[scheduling.Endpoint]float64{ep1: 0.0, ep2: 0.0},
		},
		{
			name:     "binding to ep1 scores it 1.0",
			bound:    boundTo("10.0.0.1", "8080"),
			expected: map[scheduling.Endpoint]float64{ep1: 1.0, ep2: 0.0},
		},
		{
			name:     "binding to ep2 scores it 1.0",
			bound:    boundTo("10.0.0.2", "8080"),
			expected: map[scheduling.Endpoint]float64{ep1: 0.0, ep2: 1.0},
		},
		{
			name:     "binding to absent endpoint scores all zero",
			bound:    boundTo("10.0.0.99", "8080"),
			expected: map[scheduling.Endpoint]float64{ep1: 0.0, ep2: 0.0},
		},
		{
			name:     "binding with different port scores all zero",
			bound:    boundTo("10.0.0.1", "9090"),
			expected: map[scheduling.Endpoint]float64{ep1: 0.0, ep2: 0.0},
		},
	}

	scorer := sessionaffinity.NewSessionAffinity(testProducerName)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &scheduling.InferenceRequest{}
			if tt.bound != "" {
				request.PutAttribute(bindingKey(), tt.bound)
			}
			scores := scorer.Score(context.Background(), request, endpoints)
			assert.Equal(t, tt.expected, scores)
		})
	}
}

func TestScoreNilRequest(t *testing.T) {
	ep1 := newTestEndpoint("pod-1", "default", "10.0.0.1", "8080")
	endpoints := []scheduling.Endpoint{ep1}

	scorer := sessionaffinity.NewSessionAffinity(testProducerName)
	scores := scorer.Score(context.Background(), nil, endpoints)

	assert.Equal(t, map[scheduling.Endpoint]float64{ep1: 0.0}, scores)
}
