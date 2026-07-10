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

package prefixhash

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestBlockHashesFromMetadata(t *testing.T) {
	// routing wraps a prefix_hashes field value in the llm-d.routing namespace.
	routing := func(field any) *scheduling.InferenceRequest {
		return &scheduling.InferenceRequest{
			Metadata: map[string]any{RoutingMetadataNamespace: map[string]any{prefixHashesMetadataField: field}},
		}
	}

	tests := []struct {
		name            string
		request         *scheduling.InferenceRequest
		maxPrefixBlocks int
		want            [][]BlockHash
	}{
		{
			name:    "valid chain",
			request: routing([]any{"1", "2", "3"}),
			want:    [][]BlockHash{{1, 2, 3}},
		},
		{
			name:            "caps at maxPrefixBlocks",
			request:         routing([]any{"1", "2", "3", "4"}),
			maxPrefixBlocks: 2,
			want:            [][]BlockHash{{1, 2}},
		},
		{
			name:            "cap counts kept hashes, not raw entries",
			request:         routing([]any{"1", "nope", "2", "3"}),
			maxPrefixBlocks: 2,
			want:            [][]BlockHash{{1, 2}},
		},
		{
			name:    "skips non-string and unparsable entries",
			request: routing([]any{"1", 2.0, "nope", "3"}),
			want:    [][]BlockHash{{1, 3}},
		},
		{
			name:    "empty list",
			request: routing([]any{}),
			want:    nil,
		},
		{
			name:    "all entries unparsable",
			request: routing([]any{"nope", "nan"}),
			want:    nil,
		},
		{
			name:    "nil request",
			request: nil,
			want:    nil,
		},
		{
			name:    "nil metadata",
			request: &scheduling.InferenceRequest{},
			want:    nil,
		},
		{
			name:    "namespace absent",
			request: &scheduling.InferenceRequest{Metadata: map[string]any{"other.ns": map[string]any{}}},
			want:    nil,
		},
		{
			name:    "namespace not a map",
			request: &scheduling.InferenceRequest{Metadata: map[string]any{RoutingMetadataNamespace: "not-a-map"}},
			want:    nil,
		},
		{
			name:    "field absent",
			request: &scheduling.InferenceRequest{Metadata: map[string]any{RoutingMetadataNamespace: map[string]any{}}},
			want:    nil,
		},
		{
			name:    "field not a list",
			request: routing("not-a-list"),
			want:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, BlockHashesFromMetadata(tt.request, tt.maxPrefixBlocks))
		})
	}
}
