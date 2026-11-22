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

package runner

import (
	"testing"

	"github.com/stretchr/testify/assert"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
)

// metricsRegisterOptions only translates an already-validated configuration;
// the edges are rejected in pkg/epp/config/loader, which both startup paths
// reach first. The cases here are the mapping's own branches: a ladder that was
// left out keeps its built-in edges, and a configured ladder is passed through
// without touching the other one.
func TestMetricsRegisterOptionsFromConfig(t *testing.T) {
	tpot := []float64{0.005, 0.01, 0.02}
	itl := []float64{0.001, 0.002}

	testCases := []struct {
		name     string
		config   *configapiv1.EndpointPickerConfig
		wantTPOT []float64
		wantITL  []float64
	}{
		{
			name:   "nil config keeps the built-in edges",
			config: nil,
		},
		{
			name:   "omitted metrics section keeps the built-in edges",
			config: &configapiv1.EndpointPickerConfig{},
		},
		{
			name:   "empty metrics section keeps the built-in edges",
			config: &configapiv1.EndpointPickerConfig{Metrics: &configapiv1.MetricsConfig{}},
		},
		{
			name: "both histograms configured independently",
			config: &configapiv1.EndpointPickerConfig{Metrics: &configapiv1.MetricsConfig{
				RequestTPOTBuckets: tpot,
				RequestITLBuckets:  itl,
			}},
			wantTPOT: tpot,
			wantITL:  itl,
		},
		{
			name: "only TPOT configured leaves ITL on its defaults",
			config: &configapiv1.EndpointPickerConfig{Metrics: &configapiv1.MetricsConfig{
				RequestTPOTBuckets: tpot,
			}},
			wantTPOT: tpot,
		},
		{
			name: "only ITL configured leaves TPOT on its defaults",
			config: &configapiv1.EndpointPickerConfig{Metrics: &configapiv1.MetricsConfig{
				RequestITLBuckets: itl,
			}},
			wantITL: itl,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := metricsRegisterOptions(tc.config)
			assert.Equal(t, tc.wantTPOT, got.RequestTPOTBuckets)
			assert.Equal(t, tc.wantITL, got.RequestITLBuckets)
		})
	}
}
