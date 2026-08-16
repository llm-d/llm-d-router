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

package metrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func ptr(s string) *string { return &s }

func TestResolveConfiguredKey(t *testing.T) {
	tests := []struct {
		name      string
		attribute string
		producer  *string
		want      plugin.DataKey
		wantErr   string
	}{
		{
			name:      "omitted producer selects the core metrics extractor",
			attribute: "GPUUtilization",
			producer:  nil,
			want:      plugin.NewDataKey("GPUUtilization", MetricsExtractorType),
		},
		{
			name:      "named producer is honoured",
			attribute: "GPUUtilization",
			producer:  ptr("dcgm-extractor"),
			want:      plugin.NewDataKey("GPUUtilization", "dcgm-extractor"),
		},
		{
			// The pool aggregates are addressed without a producer, and their
			// names contain "/", so an explicit empty producer must survive.
			name:      "explicit empty producer addresses a producer-agnostic attribute",
			attribute: "llm-d.ai/multicluster-queue-size",
			producer:  ptr(""),
			want:      plugin.NewDataKey("llm-d.ai/multicluster-queue-size", ""),
		},
		{
			// The regression this guards: the combined spelling built a key
			// matching nothing, so the filter passed every endpoint in silence.
			name:      "combined Attribute/Producer spelling is rejected",
			attribute: "GPUUtilization/dcgm-extractor",
			producer:  nil,
			wantErr:   `split it into attribute: "GPUUtilization" and producer: "dcgm-extractor"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveConfiguredKey("test plugin", tc.attribute, tc.producer)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}
