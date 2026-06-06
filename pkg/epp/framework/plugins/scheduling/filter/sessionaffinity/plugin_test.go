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
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sessionaffinity "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity"
)

const testProducerName = "test-session-producer"

func newTestEndpoint(name, address, port string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"},
			Address:        address,
			Port:           port,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func newFilter(t *testing.T) *sessionaffinity.Plugin {
	t.Helper()
	raw := []byte(`{"sessionIDProducerName":"` + testProducerName + `"}`)
	plugin, err := sessionaffinity.Factory("test-filter", fwkplugin.StrictDecoder(raw), nil)
	require.NoError(t, err)
	return plugin.(*sessionaffinity.Plugin)
}

func bindingKey() string {
	return attrsession.BoundEndpointDataKey.WithNonEmptyProducerName(testProducerName).String()
}

func boundTo(address, port string) attrsession.BoundEndpoint {
	return attrsession.BoundEndpoint(net.JoinHostPort(address, port))
}

func TestFilter(t *testing.T) {
	ep1 := newTestEndpoint("pod-1", "10.0.0.1", "8080")
	ep2 := newTestEndpoint("pod-2", "10.0.0.2", "8080")
	ep3 := newTestEndpoint("pod-3", "10.0.0.3", "8080")
	endpoints := []scheduling.Endpoint{ep1, ep2, ep3}

	tests := []struct {
		name          string
		bound         attrsession.BoundEndpoint
		expectedNames []string
	}{
		{
			name:          "no binding keeps all endpoints",
			bound:         "",
			expectedNames: []string{"pod-1", "pod-2", "pod-3"},
		},
		{
			name:          "binding to ep1 keeps only pod-1",
			bound:         boundTo("10.0.0.1", "8080"),
			expectedNames: []string{"pod-1"},
		},
		{
			name:          "binding to ep2 keeps only pod-2",
			bound:         boundTo("10.0.0.2", "8080"),
			expectedNames: []string{"pod-2"},
		},
		{
			name:          "binding to absent endpoint keeps all",
			bound:         boundTo("10.0.0.99", "8080"),
			expectedNames: []string{"pod-1", "pod-2", "pod-3"},
		},
	}

	filter := newFilter(t)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &scheduling.InferenceRequest{}
			if tt.bound != "" {
				request.PutAttribute(bindingKey(), tt.bound)
			}
			result := filter.Filter(context.Background(), request, endpoints)

			actual := make([]string, 0, len(result))
			for _, ep := range result {
				actual = append(actual, ep.GetMetadata().NamespacedName.Name)
			}
			assert.ElementsMatch(t, tt.expectedNames, actual)
		})
	}
}

func TestFilterSingleEndpointShortcut(t *testing.T) {
	filter := newFilter(t)

	ep1 := newTestEndpoint("pod-1", "10.0.0.1", "8080")
	endpoints := []scheduling.Endpoint{ep1}

	request := &scheduling.InferenceRequest{}
	request.PutAttribute(bindingKey(), boundTo("10.0.0.99", "8080"))

	result := filter.Filter(context.Background(), request, endpoints)
	require.Len(t, result, 1)
	assert.Equal(t, "pod-1", result[0].GetMetadata().NamespacedName.Name)
}

func TestFilterNilRequest(t *testing.T) {
	filter := newFilter(t)

	endpoints := []scheduling.Endpoint{
		newTestEndpoint("pod-1", "10.0.0.1", "8080"),
		newTestEndpoint("pod-2", "10.0.0.2", "8080"),
	}
	result := filter.Filter(context.Background(), nil, endpoints)

	assert.Len(t, result, 2)
}
