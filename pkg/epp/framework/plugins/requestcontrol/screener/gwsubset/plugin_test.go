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

package gwsubset

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func endpoint(name, ip string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Namespace: "default", Name: name},
		Address: ip,
	}, nil, nil)
}

func subsetMetadata(addrs ...string) map[string]any {
	raw := make([]any, 0, len(addrs))
	for _, addr := range addrs {
		raw = append(raw, addr)
	}
	return map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: raw,
		},
	}
}

func TestScreenReturnsAllWhenNoMetadata(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{endpoint("a", "10.0.0.1"), endpoint("b", "10.0.0.2")}

	got := s.Screen(context.Background(), &fwksched.InferenceRequest{}, candidates)
	assert.ElementsMatch(t, candidates, got)
}

func TestScreenReturnsAllWhenNoSubsetKey(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{endpoint("a", "10.0.0.1")}

	got := s.Screen(context.Background(), &fwksched.InferenceRequest{
		Metadata: map[string]any{"other": "value"},
	}, candidates)
	assert.ElementsMatch(t, candidates, got)
}

func TestScreenFiltersBySubsetAddresses(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{
		endpoint("a", "10.0.0.1"),
		endpoint("b", "10.0.0.2"),
		endpoint("c", "10.0.0.3"),
	}

	got := s.Screen(context.Background(), &fwksched.InferenceRequest{
		Metadata: subsetMetadata("10.0.0.1:8080", "10.0.0.3:9090"),
	}, candidates)
	assert.ElementsMatch(t, []fwksched.Endpoint{candidates[0], candidates[2]}, got)
}

func TestScreenEmptySubsetReturnsEmpty(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{endpoint("a", "10.0.0.1")}

	got := s.Screen(context.Background(), &fwksched.InferenceRequest{
		Metadata: subsetMetadata(),
	}, candidates)
	assert.Empty(t, got)
}

func TestScreenIgnoresNonStringEntries(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{endpoint("a", "10.0.0.1"), endpoint("b", "10.0.0.2")}

	got := s.Screen(context.Background(), &fwksched.InferenceRequest{
		Metadata: map[string]any{
			metadata.SubsetFilterNamespace: map[string]any{
				metadata.SubsetFilterKey: []any{"10.0.0.1:80", 12345, "garbage"},
			},
		},
	}, candidates)
	assert.ElementsMatch(t, []fwksched.Endpoint{candidates[0]}, got)
}

func TestScreenSubsetWithOnlyMalformedEntriesIsEmpty(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{endpoint("a", "10.0.0.1")}

	got := s.Screen(context.Background(), &fwksched.InferenceRequest{
		Metadata: map[string]any{
			metadata.SubsetFilterNamespace: map[string]any{
				metadata.SubsetFilterKey: []any{12345},
			},
		},
	}, candidates)
	assert.Empty(t, got)
}

func TestScreenNilRequestReturnsAll(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	candidates := []fwksched.Endpoint{endpoint("a", "10.0.0.1")}

	got := s.Screen(context.Background(), nil, candidates)
	assert.ElementsMatch(t, candidates, got)
}

func TestTypedName(t *testing.T) {
	t.Parallel()

	s := NewScreener()
	assert.Equal(t, PluginType, s.TypedName().Type)
}
