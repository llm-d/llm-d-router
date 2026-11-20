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
	"context"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	"sigs.k8s.io/yaml"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// The Collector configuration reads EPP metric and label names as strings, so
// renaming one breaks the GenAI mapping without a compile error.
var genAICollectorConfigPath = filepath.Join("..", "..", "..", "config", "observability", "otel-collector", "genai-metrics.yaml")

var (
	ottlSourceMetric = regexp.MustCompile(`metric\.name == "(llm_d_epp_[a-z_]+)"`)
	ottlGenAIMetric  = regexp.MustCompile(`metric\.name == "(gen_ai\.[a-z_.]+)"`)
	ottlCopy         = regexp.MustCompile(`copy_metric\(name="(gen_ai\.[a-z_.]+)".*where metric\.name == "(llm_d_epp_[a-z_]+)"`)
	// Source labels have no dot; the GenAI attributes the mapping writes do.
	ottlSourceLabel = regexp.MustCompile(`datapoint\.attributes\["([a-z_]+)"\]`)
	ottlAPISurface  = regexp.MustCompile(`datapoint\.attributes\["api_surface"\] == "([a-z_]+)"`)
)

func TestGenAICollectorConfigMatchesEPPMetrics(t *testing.T) {
	raw, err := os.ReadFile(genAICollectorConfigPath)
	require.NoError(t, err)
	var config struct {
		Processors map[string]any `json:"processors"`
	}
	require.NoError(t, yaml.Unmarshal(raw, &config))
	var ottl []string
	collectStrings(config.Processors, &ottl)
	require.NotEmpty(t, ottl)

	copies := map[string]string{}
	sources := map[string]bool{}
	for _, s := range ottl {
		if m := ottlCopy.FindStringSubmatch(s); m != nil {
			copies[m[1]] = m[2]
		}
		for _, m := range ottlSourceMetric.FindAllStringSubmatch(s, -1) {
			sources[m[1]] = true
		}
	}

	exported := exportedLabels(t)
	for source := range sources {
		_, ok := exported[source]
		require.True(t, ok, "the Collector configuration reads %s, which the EPP does not export", source)
	}

	// A statement that names metrics reads labels only from those metrics;
	// one that names none applies to every copy.
	for _, s := range ottl {
		scope := map[string]bool{}
		for _, m := range ottlSourceMetric.FindAllStringSubmatch(s, -1) {
			scope[m[1]] = true
		}
		for _, m := range ottlGenAIMetric.FindAllStringSubmatch(s, -1) {
			if source, ok := copies[m[1]]; ok {
				scope[source] = true
			}
		}
		if len(scope) == 0 {
			scope = sources
		}
		for _, m := range ottlSourceLabel.FindAllStringSubmatch(s, -1) {
			for source := range scope {
				require.Contains(t, exported[source], m[1], "%q reads label %q, which %s does not have", s, m[1], source)
			}
		}
	}

	surfaces := map[string]bool{}
	for api := reqcommon.APIType(0); api.Surface() != reqcommon.APISurfaceUnknown; api++ {
		surfaces[api.Surface().String()] = true
	}
	mapped := map[string]bool{}
	for _, s := range ottl {
		for _, m := range ottlAPISurface.FindAllStringSubmatch(s, -1) {
			require.True(t, surfaces[m[1]], "the Collector configuration maps api_surface %q, which the EPP never reports", m[1])
			mapped[m[1]] = true
		}
	}
	require.Equal(t, surfaces, mapped, "every api_surface must map to a gen_ai.operation.name")
}

func collectStrings(node any, out *[]string) {
	switch v := node.(type) {
	case string:
		*out = append(*out, v)
	case []any:
		for _, item := range v {
			collectStrings(item, out)
		}
	case map[string]any:
		for _, item := range v {
			collectStrings(item, out)
		}
	}
}

// exportedLabels records one observation in each request metric the mapping
// can read and returns the label names of every gathered family.
func exportedLabels(t *testing.T) map[string]map[string]bool {
	t.Helper()
	ctx := context.Background()
	surface := reqcommon.APITypeChatCompletions.Surface()
	received := time.Now()
	RecordRequestLatencies(ctx, "m", "m", surface, "f", "0", received, received.Add(time.Second))
	RecordRequestTTFT(ctx, "m", "m", surface, "f", "0", true, received, received.Add(time.Millisecond))
	RecordInterTokenLatency(ctx, "m", "m", surface, "f", "0", 0.01)
	RecordInputTokens("m", "m", surface, "f", "0", 1)
	RecordOutputTokens("m", "m", surface, "f", "0", 1)
	RecordPromptCachedTokens("m", "m", surface, "f", "0", 1)

	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	labels := map[string]map[string]bool{}
	for _, family := range families {
		if len(family.GetMetric()) == 0 {
			continue
		}
		names := map[string]bool{}
		for _, pair := range family.GetMetric()[0].GetLabel() {
			names[pair.GetName()] = true
		}
		labels[family.GetName()] = names
	}
	return labels
}
