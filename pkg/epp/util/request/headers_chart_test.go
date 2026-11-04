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

package request

import (
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// standaloneValuesPath holds the standalone chart's Envoy config, which
// repeats header names Helm cannot import from Go.
const standaloneValuesPath = "../../../../config/charts/llm-d-router-standalone/values.yaml"

// The fail-open fallback route skips EPP, so it must remove every client
// header EPP removes on the scheduled route.
func TestStandaloneFallbackRouteRemovesRoutingHeaders(t *testing.T) {
	t.Parallel()

	values, err := os.ReadFile(standaloneValuesPath)
	require.NoError(t, err)
	removed, err := fallbackRouteRemovedHeaders(string(values))
	require.NoError(t, err)

	want := InternalRoutingHeaders.Union(ScreeningHeaders)
	assert.Equal(t, sets.List(want), sets.List(removed))
}

// Envoy strips the client's destination header, gates the EPP route on it, and
// routes on it; all three must name the header EPP sets.
func TestStandaloneEnvoyUsesDestinationEndpointHeader(t *testing.T) {
	t.Parallel()

	values, err := os.ReadFile(standaloneValuesPath)
	require.NoError(t, err)

	for _, line := range []string{
		"- remove: " + metadata.DestinationEndpointKey,
		"- name: " + metadata.DestinationEndpointKey,
		"http_header_name: " + metadata.DestinationEndpointKey,
	} {
		assert.Contains(t, string(values), line)
	}
}

func TestFallbackRouteRemovedHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		config  string
		want    []string
		wantErr bool
	}{
		{
			name: "list ends at the next route key",
			config: `
    - match:
        prefix: "/"
      route:
        cluster: inference_pool_fallback
      # comment
      request_headers_to_remove:
        - x-a
        # comment
        - x-b
      typed_per_filter_config: {}
`,
			want: []string{"x-a", "x-b"},
		},
		{
			name: "no fallback route",
			config: `
      route:
        cluster: original_destination_cluster
`,
			wantErr: true,
		},
		{
			name: "fallback route without removal list",
			config: `
      route:
        cluster: inference_pool_fallback
      typed_per_filter_config: {}
    - match:
      request_headers_to_remove:
        - x-a
`,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := fallbackRouteRemovedHeaders(tt.config)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, sets.List(got))
		})
	}
}

// fallbackRouteRemovedHeaders returns the request_headers_to_remove entries of
// the route to inference_pool_fallback. The Envoy config is a Helm template, so
// it is scanned as text rather than parsed as YAML.
func fallbackRouteRemovedHeaders(config string) (sets.Set[string], error) {
	lines := strings.Split(config, "\n")
	start := -1
	for i, line := range lines {
		if strings.TrimSpace(line) == "cluster: inference_pool_fallback" {
			start = i
			break
		}
	}
	if start < 0 {
		return nil, errors.New("no route to inference_pool_fallback")
	}
	for i := start + 1; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		switch {
		case trimmed == "typed_per_filter_config:" || strings.HasPrefix(trimmed, "typed_per_filter_config: ") || strings.HasPrefix(trimmed, "- match:"):
			return nil, errors.New("fallback route has no request_headers_to_remove")
		case trimmed == "request_headers_to_remove:":
			return listItems(lines[i+1:], indent(lines[i])), nil
		}
	}
	return nil, errors.New("fallback route has no request_headers_to_remove")
}

// listItems collects the "- item" entries nested deeper than parentIndent,
// skipping comments, until the first line back at parentIndent or shallower.
func listItems(lines []string, parentIndent int) sets.Set[string] {
	items := sets.New[string]()
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if indent(line) <= parentIndent {
			break
		}
		if item, ok := strings.CutPrefix(trimmed, "- "); ok {
			items.Insert(item)
		}
	}
	return items
}

func indent(line string) int {
	return len(line) - len(strings.TrimLeft(line, " "))
}
