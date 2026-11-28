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

package contextlengthaware

import (
	"encoding/json"
	"fmt"
	"testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/utilization"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
)

func TestEscapeProbabilityConfiguration(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"always_escape", `{"enableFiltering":true,"escapeProbability":1}`},
		{"scoring_only", `{"escapeProbability":1}`},
		{"disabled_filtering", `{"enableFiltering":false,"escapeProbability":1}`},
		{"disabled_escape", `{"enableFiltering":true,"escapeProbability":0}`},
		{"fractional_escape", `{"enableFiltering":true,"escapeProbability":0.25}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Factory("fallback", fwkplugin.StrictDecoder(json.RawMessage(tc.params)), nil)
			require.NoError(t, err)
		})
	}
}
func TestEscapeProbabilityInvalidConfiguration(t *testing.T) {
	for _, tc := range []struct{ name, params string }{
		{"negative", `{"escapeProbability":-0.1}`},
		{"above_one", `{"escapeProbability":1.1}`},
		{"non_numeric", `{"escapeProbability":true}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Factory("fallback", fwkplugin.StrictDecoder(json.RawMessage(tc.params)), nil)
			require.Error(t, err)
		})
	}
}

func fallbackPlugin(t *testing.T) *ContextLengthAware {
	t.Helper()
	plugin, err := Factory("fallback", fwkplugin.StrictDecoder(json.RawMessage(`{"enableFiltering":true,"escapeProbability":1}`)), nil)
	require.NoError(t, err)
	return plugin.(*ContextLengthAware)
}

func rangeEndpoint(name, rangeValue string, waiting int) scheduling.Endpoint {
	labels := map[string]string{}
	if rangeValue != "" {
		labels[DefaultContextLengthLabel] = rangeValue
	}
	return scheduling.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name}, Labels: labels}, &fwkdl.Metrics{WaitingQueueSize: waiting}, nil)
}

func TestFallbackOnNoMatchSelection(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ranges []string
		want   []int
	}{
		{"matching_range_preferred", []string{"0-128", "129-256", "257-512"}, []int{0}},
		{"only_longer_ranges", []string{"0-64", "129-256", "257-512"}, []int{1, 2}},
		{"no_downward_fallback", []string{"0-64", "65-99"}, nil},
		{"overlapping_matches_retained", []string{"0-128", "64-256", "257-512"}, []int{0, 1}},
		{"inclusive_upper_boundary", []string{"0-100", "101-200"}, []int{0}},
		{"inclusive_lower_boundary", []string{"0-99", "100-200"}, []int{1}},
		{"range_gap_uses_upward_fallback", []string{"0-99", "101-200"}, []int{1}},
		{"invalid_range_excluded_from_fallback", []string{"broken", "129-256"}, []int{1}},
		{"unlabeled_only_fallback", []string{"", "0-128", "129-256"}, []int{1}},
		{"unlabeled_generic_fallback", []string{"", "0-64", "129-256"}, []int{0, 2}},
		{"invalid_range_excluded_when_matched", []string{"broken", "0-128"}, []int{1}},
		{"single_point_range_match", []string{"100-100", "101-200"}, []int{0}},
		{"empty_candidates", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plugin := fallbackPlugin(t)
			endpoints := make([]scheduling.Endpoint, 0, len(tc.ranges))
			for i, value := range tc.ranges {
				endpoints = append(endpoints, rangeEndpoint(string(rune('a'+i)), value, 0))
			}
			want := make([]scheduling.Endpoint, 0, len(tc.want))
			for _, i := range tc.want {
				want = append(want, endpoints[i])
			}
			got := plugin.Filter(t.Context(), createHundredTokenRequest(), endpoints)
			require.ElementsMatch(t, want, got)
		})
	}
}

func TestFallbackAfterUtilizationFilter(t *testing.T) {
	availability, err := utilization.Factory("available", fwkplugin.StrictDecoder(json.RawMessage(`{"fallbackOnEmpty":false,"conditions":[{"metric":"waiting-queue","maxValue":1}]}`)), nil)
	require.NoError(t, err)
	plugin := fallbackPlugin(t)
	request := createHundredTokenRequest()
	for _, tc := range []struct {
		name       string
		shortQueue int
		wantNames  []string
	}{
		{"home_available", 0, []string{"short"}},
		{"home_filtered_out", 2, []string{"medium", "long"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			endpoints := []scheduling.Endpoint{rangeEndpoint("short", "0-128", tc.shortQueue), rangeEndpoint("medium", "129-256", 0), rangeEndpoint("long", "257-512", 0), rangeEndpoint("other-range", "0-64", 0)}
			available := availability.(scheduling.Filter).Filter(t.Context(), request, endpoints)
			got := plugin.Filter(t.Context(), request, available)
			names := make([]string, 0, len(got))
			for _, ep := range got {
				names = append(names, ep.GetMetadata().ID.Name)
			}
			require.ElementsMatch(t, tc.wantNames, names)
			if tc.shortQueue > 1 {
				scores := plugin.Score(t.Context(), request, got)
				require.Greater(t, scores[got[0]], scores[got[1]])
			}
		})
	}
}

func TestFallbackWithoutFilteringPassesThrough(t *testing.T) {
	plugin, err := Factory("disabled", fwkplugin.StrictDecoder(json.RawMessage(`{"escapeProbability":1}`)), nil)
	require.NoError(t, err)
	endpoints := []scheduling.Endpoint{rangeEndpoint("match", "0-128", 0), rangeEndpoint("other", "129-256", 0)}
	require.Equal(t, endpoints, plugin.(*ContextLengthAware).Filter(t.Context(), createHundredTokenRequest(), endpoints))
}

func bucketRequest(tokens int) *scheduling.InferenceRequest {
	request := createHundredTokenRequest()
	request.Body.TokenizedRequest.Prompts[0].TokenIDs = make([]uint32, tokens)
	return request
}

func TestUpwardFallbackLengthAvailabilityMatrix(t *testing.T) {
	availability, err := utilization.Factory("matrix-available", fwkplugin.StrictDecoder(json.RawMessage(`{"fallbackOnEmpty":false,"conditions":[{"metric":"waiting-queue","maxValue":1}]}`)), nil)
	require.NoError(t, err)
	plugin := fallbackPlugin(t)
	ranges := []string{"0-4096", "4097-8192", "8193-32768"}
	for _, tc := range []struct{ tokens, home int }{
		{0, 0}, {1, 0}, {4095, 0}, {4096, 0}, {4097, 1}, {8191, 1}, {8192, 1}, {8193, 2}, {32767, 2}, {32768, 2}, {32769, 3}, {65536, 3},
	} {
		for mask := 0; mask < 8; mask++ {
			t.Run(fmt.Sprintf("tokens_%d_available_%03b", tc.tokens, mask), func(t *testing.T) {
				endpoints := make([]scheduling.Endpoint, 0, len(ranges))
				for i, value := range ranges {
					waiting := 2
					if mask&(1<<i) != 0 {
						waiting = 0
					}
					endpoints = append(endpoints, rangeEndpoint(fmt.Sprintf("bucket-%d", i), value, waiting))
				}
				request := bucketRequest(tc.tokens)
				available := availability.(scheduling.Filter).Filter(t.Context(), request, endpoints)
				got := plugin.Filter(t.Context(), request, available)
				want := make([]scheduling.Endpoint, 0, len(ranges))
				if tc.home < len(ranges) && mask&(1<<tc.home) != 0 {
					want = append(want, endpoints[tc.home])
				} else {
					for i := tc.home + 1; i < len(ranges); i++ {
						if mask&(1<<i) != 0 {
							want = append(want, endpoints[i])
						}
					}
				}
				require.ElementsMatch(t, want, got)
			})
		}
	}
}

func TestUpwardFallbackLabelVariants(t *testing.T) {
	for _, tc := range []struct {
		name     string
		metadata *fwkdl.EndpointMetadata
		generic  bool
	}{
		{"nil_metadata", nil, true},
		{"nil_labels", &fwkdl.EndpointMetadata{}, true},
		{"empty_labels", &fwkdl.EndpointMetadata{Labels: map[string]string{}}, true},
		{"unrelated_label", &fwkdl.EndpointMetadata{Labels: map[string]string{"example.com/other": "0-128"}}, true},
		{"empty_value", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: ""}}, false},
		{"invalid_format", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: "broken"}}, false},
		{"inverted_bounds", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: "4096-0"}}, false},
		{"negative_bound", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: "-1-4096"}}, false},
		{"non_integer", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: "1.5-4096"}}, false},
		{"extra_separator", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: "0-128-4096"}}, false},
		{"overflow", &fwkdl.EndpointMetadata{Labels: map[string]string{DefaultContextLengthLabel: "0-9223372036854775808"}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plugin := fallbackPlugin(t)
			candidate := scheduling.NewEndpoint(tc.metadata, nil, nil)
			longer := rangeEndpoint("longer", "129-256", 0)
			want := []scheduling.Endpoint{longer}
			if tc.generic {
				want = append(want, candidate)
			}
			require.ElementsMatch(t, want, plugin.Filter(t.Context(), bucketRequest(100), []scheduling.Endpoint{candidate, longer}))
			matching := rangeEndpoint("matching", "0-128", 0)
			require.Equal(t, []scheduling.Endpoint{matching}, plugin.Filter(t.Context(), bucketRequest(100), []scheduling.Endpoint{candidate, longer, matching}))
		})
	}
}

func TestUpwardFallbackCustomLabel(t *testing.T) {
	const customLabel = "example.com/input-range"
	plugin, err := Factory("custom-range", fwkplugin.StrictDecoder(json.RawMessage(`{"enableFiltering":true,"escapeProbability":1,"label":"example.com/input-range"}`)), nil)
	require.NoError(t, err)
	first := createEndpoint(types.NamespacedName{Name: "first"}, "", map[string]string{customLabel: "129-256", DefaultContextLengthLabel: "0-128"})
	second := createEndpoint(types.NamespacedName{Name: "second"}, "", map[string]string{customLabel: "0-128", DefaultContextLengthLabel: "129-256"})
	filter := plugin.(*ContextLengthAware)
	require.Equal(t, []scheduling.Endpoint{second}, filter.Filter(t.Context(), bucketRequest(100), []scheduling.Endpoint{first, second}))
	require.Equal(t, []scheduling.Endpoint{first}, filter.Filter(t.Context(), bucketRequest(100), []scheduling.Endpoint{first}))
}

func TestUpwardFallbackWithReusableTokens(t *testing.T) {
	plugin, err := Factory("work-fallback", fwkplugin.StrictDecoder(json.RawMessage(`{"enableFiltering":true,"escapeProbability":1,"label":"llm-d.ai/prefill-work-range","reusableTokensProducerName":"p2p-source"}`)), nil)
	require.NoError(t, err)
	filter := plugin.(*ContextLengthAware)
	request := bucketRequest(10000)
	request.PutAttribute(attrprefix.ReusablePrefixTokensDataKey.WithNonEmptyProducerName(testReusableTokensProducerName), attrprefix.ReusablePrefixTokens(8000))
	short := createEndpoint(types.NamespacedName{Name: "short"}, "", map[string]string{testPrefillWorkRangeLabel: "0-2048"})
	longer := createEndpoint(types.NamespacedName{Name: "longer"}, "", map[string]string{testPrefillWorkRangeLabel: "2049-8192"})
	require.Equal(t, []scheduling.Endpoint{short}, filter.Filter(t.Context(), request, []scheduling.Endpoint{short, longer}))
	require.Greater(t, filter.Score(t.Context(), request, []scheduling.Endpoint{short})[short], 0.3)
	require.Equal(t, []scheduling.Endpoint{longer}, filter.Filter(t.Context(), request, []scheduling.Endpoint{longer}))
}

func TestEscapeProbability(t *testing.T) {
	endpoints := []scheduling.Endpoint{rangeEndpoint("longer", "129-256", 0), rangeEndpoint("longest", "257-512", 0)}
	request := createHundredTokenRequest()
	for _, tc := range []struct {
		probability float64
		want        int
	}{{0, 0}, {1, 2}} {
		t.Run(fmt.Sprint(tc.probability), func(t *testing.T) {
			raw := fmt.Sprintf(`{"enableFiltering":true,"escapeProbability":%g}`, tc.probability)
			plugin, err := Factory("probability", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
			require.NoError(t, err)
			for i := 0; i < 20; i++ {
				require.Len(t, plugin.(*ContextLengthAware).Filter(t.Context(), request, endpoints), tc.want)
			}
		})
	}
}

func TestFractionalEscapeProbability(t *testing.T) {
	plugin, err := Factory("fractional", fwkplugin.StrictDecoder(json.RawMessage(`{"enableFiltering":true,"escapeProbability":0.25}`)), nil)
	require.NoError(t, err)
	filter := plugin.(*ContextLengthAware)
	request := createHundredTokenRequest()
	matching := rangeEndpoint("matching", "0-128", 0)
	longer := rangeEndpoint("longer", "129-256", 0)
	longest := rangeEndpoint("longest", "257-512", 0)
	for i := 0; i < 100; i++ {
		require.Equal(t, []scheduling.Endpoint{matching}, filter.Filter(t.Context(), request, []scheduling.Endpoint{matching, longer, longest}))
	}
	escaped := 0
	for i := 0; i < 100; i++ {
		got := filter.Filter(t.Context(), request, []scheduling.Endpoint{longer, longest})
		if len(got) > 0 {
			require.Equal(t, []scheduling.Endpoint{longer, longest}, got)
			escaped++
		}
	}
	require.Greater(t, escaped, 0)
	require.Less(t, escaped, 100)
	shorter := rangeEndpoint("shorter", "0-64", 0)
	for i := 0; i < 100; i++ {
		require.Empty(t, filter.Filter(t.Context(), request, []scheduling.Endpoint{shorter}))
	}
}

func TestEscapeIncludesOutputCap(t *testing.T) {
	plugin, err := Factory("output-cap", fwkplugin.StrictDecoder(json.RawMessage(`{"enableFiltering":true,"escapeProbability":1,"includeMaxOutputTokens":true}`)), nil)
	require.NoError(t, err)
	request := createHundredTokenRequest()
	outputCap := int64(100)
	request.Body.MaxOutputTokens = &outputCap
	short := rangeEndpoint("short", "0-128", 0)
	match := rangeEndpoint("match", "129-256", 0)
	longer := rangeEndpoint("longer", "257-512", 0)
	filter := plugin.(*ContextLengthAware)
	require.Equal(t, []scheduling.Endpoint{match}, filter.Filter(t.Context(), request, []scheduling.Endpoint{short, match, longer}))
	require.Equal(t, []scheduling.Endpoint{longer}, filter.Filter(t.Context(), request, []scheduling.Endpoint{short, longer}))
}
