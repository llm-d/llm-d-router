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

package composite

import (
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
)

func TestRequestCostRequired(t *testing.T) {
	handle := newHandle(t)
	closed, err := concurrency.ConcurrencyDetectorFactory("closed", plugin.StrictDecoder([]byte(
		`{"concurrencyMode":"tokens","failOpen":false}`)), handle)
	require.NoError(t, err)
	open, err := concurrency.ConcurrencyDetectorFactory("open", plugin.StrictDecoder([]byte(
		`{"concurrencyMode":"tokens","failOpen":true}`)), handle)
	require.NoError(t, err)
	closedDetector := closed.(flowcontrol.SaturationDetector)
	openDetector := open.(flowcontrol.SaturationDetector)
	plain := mockDetector("plain", 0)
	nested := newDetector("nested", []string{"plain", "closed"},
		[]flowcontrol.SaturationDetector{plain, closedDetector}, nil, logr.Discard())
	for _, tc := range []struct {
		name     string
		detector flowcontrol.SaturationDetector
		want     bool
	}{
		{name: "nil"},
		{name: "other-detector", detector: plain},
		{name: "fail-open", detector: openDetector},
		{name: "fail-closed", detector: closedDetector, want: true},
		{name: "no-cost-children", detector: newDetector("no-cost", []string{"plain", "open"},
			[]flowcontrol.SaturationDetector{plain, openDetector}, nil, logr.Discard())},
		{name: "nested-cost-child", detector: newDetector("outer", []string{"open", "nested"},
			[]flowcontrol.SaturationDetector{openDetector, nested}, nil, logr.Discard()), want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, RequestCostRequired(tc.detector))
		})
	}
}

func TestFilterForDispatchStageScopedChildren(t *testing.T) {
	handle := newHandle(t)
	for name, cfg := range map[string]string{
		"prefill": `{"concurrencyMode":"tokens","maxTokenConcurrency":100,"headroom":0,"failOpen":false}`,
		"decode":  `{"concurrencyMode":"tokens","maxTokenConcurrency":1000,"headroom":0,"failOpen":false}`,
	} {
		child, err := concurrency.ConcurrencyDetectorFactory(name, plugin.StrictDecoder([]byte(cfg)), handle)
		require.NoError(t, err)
		handle.AddPlugin(name, child)
	}
	combined, err := MaxSaturationDetectorFactory("combined", plugin.StrictDecoder([]byte(
		`{"detectors":["prefill","decode"],"stages":{"prefill":["prefill"],"decode":["decode"]}}`)), handle)
	require.NoError(t, err)
	_, isFilter := combined.(scheduling.Filter)
	require.False(t, isFilter)

	attrs := datalayer.NewAttributes()
	attrs.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Tokens: 60})
	attrs.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: 50})
	endpoints := []scheduling.Endpoint{scheduling.NewEndpoint(&datalayer.EndpointMetadata{}, nil, attrs)}
	for _, tc := range []struct {
		stage string
		want  int
	}{
		{flowcontrol.SaturationStagePrefill, 0},
		{flowcontrol.SaturationStageDecode, 1},
		{"", 0},
	} {
		t.Run(tc.stage, func(t *testing.T) {
			got := FilterForDispatch(flowcontrol.WithSaturationStage(t.Context(), tc.stage),
				combined.(flowcontrol.SaturationDetector), nil, endpoints)
			require.Len(t, got, tc.want)
		})
	}
}
