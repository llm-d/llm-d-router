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

package loader

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/go-logr/logr/testr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func metricsConfig(tpot, itl []float64) *configapiv1.EndpointPickerConfig {
	return &configapiv1.EndpointPickerConfig{Metrics: &configapiv1.MetricsConfig{
		RequestTPOTBuckets: tpot,
		RequestITLBuckets:  itl,
	}}
}

func TestValidateMetrics(t *testing.T) {
	overTheLimit := make([]float64, 0, eppmetrics.MaxHistogramBuckets+1)
	for i := 1; i <= eppmetrics.MaxHistogramBuckets+1; i++ {
		overTheLimit = append(overTheLimit, float64(i)*0.001)
	}
	atTheLimit := overTheLimit[:eppmetrics.MaxHistogramBuckets]

	testCases := []struct {
		name    string
		cfg     *configapiv1.EndpointPickerConfig
		wantErr string
	}{
		{
			name: "section omitted",
			cfg:  &configapiv1.EndpointPickerConfig{},
		},
		{
			name: "section present with neither ladder set",
			cfg:  metricsConfig(nil, nil),
		},
		{
			name: "both ladders configured",
			cfg:  metricsConfig([]float64{0.005, 0.01}, []float64{0.001, 0.002}),
		},
		{
			name: "at the maximum bucket count",
			cfg:  metricsConfig(atTheLimit, nil),
		},
		{
			name:    "one bucket over the maximum",
			cfg:     metricsConfig(overTheLimit, nil),
			wantErr: fmt.Sprintf("metrics.requestTPOTBuckets: %d bucket boundaries exceed the maximum of %d", len(overTheLimit), eppmetrics.MaxHistogramBuckets),
		},
		{
			name:    "an explicitly empty list is rejected rather than ignored",
			cfg:     metricsConfig([]float64{}, nil),
			wantErr: "metrics.requestTPOTBuckets: at least one bucket boundary is required",
		},
		{
			name:    "non-increasing edges are rejected",
			cfg:     metricsConfig([]float64{0.01, 0.01}, nil),
			wantErr: "metrics.requestTPOTBuckets: bucket boundary 1 is 0.01, want a value greater than the preceding 0.01",
		},
		{
			name:    "a non-positive edge is rejected",
			cfg:     metricsConfig(nil, []float64{0, 0.01}),
			wantErr: "metrics.requestITLBuckets: bucket boundary 0 is 0, want a positive value",
		},
		{
			name:    "a non-finite edge is rejected",
			cfg:     metricsConfig(nil, []float64{0.01, math.Inf(1)}),
			wantErr: "metrics.requestITLBuckets: bucket boundary 1 is +Inf, want a finite value",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMetrics(tc.cfg)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.EqualError(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
		})
	}
}

// TestValidateMetricsReportsLaddersInAFixedOrder pins the field order. Ranging
// a map here would make the reported field depend on map iteration order, so a
// configuration with both ladders invalid would name a different one from run
// to run.
func TestValidateMetricsReportsLaddersInAFixedOrder(t *testing.T) {
	cfg := metricsConfig([]float64{}, []float64{})
	for range 32 {
		err := validateMetrics(cfg)
		require.Error(t, err)
		assert.EqualError(t, err, "metrics.requestTPOTBuckets: at least one bucket boundary is required")
	}
}

// The bucket edges arrive in the same configuration file as every other
// EndpointPickerConfig field, so they are checked on the same path. This drives
// the file through the decoder and then validateConfig, which is what
// InstantiateAndConfigure calls on both startup paths.
func TestMetricsBucketsRejectedOnTheConfigFilePath(t *testing.T) {
	testCases := []struct {
		name            string
		metricsFragment string
		wantSectionNil  bool
		wantEdgesNil    bool
		wantErr         string
	}{
		{
			name:           "section omitted",
			wantSectionNil: true,
		},
		{
			name:            "section present but empty",
			metricsFragment: "metrics: {}\n",
			wantEdgesNil:    true,
		},
		{
			name:            "key set to null",
			metricsFragment: "metrics:\n  requestTPOTBuckets: null\n",
			wantEdgesNil:    true,
		},
		{
			name:            "key set to an empty list",
			metricsFragment: "metrics:\n  requestTPOTBuckets: []\n",
			wantErr:         "metrics validation failed: metrics.requestTPOTBuckets: at least one bucket boundary is required",
		},
		{
			name:            "descending edges",
			metricsFragment: "metrics:\n  requestITLBuckets: [0.02, 0.01]\n",
			wantErr:         "metrics validation failed: metrics.requestITLBuckets: bucket boundary 1 is 0.01, want a value greater than the preceding 0.02",
		},
		{
			name:            "too many edges",
			metricsFragment: "metrics:\n  requestTPOTBuckets: [" + strings.Join(edgeLiterals(eppmetrics.MaxHistogramBuckets+1), ", ") + "]\n",
			wantErr: fmt.Sprintf("metrics validation failed: metrics.requestTPOTBuckets: %d bucket boundaries exceed the maximum of %d",
				eppmetrics.MaxHistogramBuckets+1, eppmetrics.MaxHistogramBuckets),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			configText := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n" + tc.metricsFragment +
				"plugins:\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins: []\n"
			rawConfig, _, err := LoadRawConfig([]byte(configText), testr.New(t))
			require.NoError(t, err, "decoding must succeed; the edges are rejected by validation, not by the decoder")

			if tc.wantSectionNil {
				require.Nil(t, rawConfig.Metrics)
			} else {
				require.NotNil(t, rawConfig.Metrics)
			}
			if tc.wantEdgesNil {
				assert.Nil(t, rawConfig.Metrics.RequestTPOTBuckets,
					"an omitted or null key must stay nil so it keeps the built-in edges")
			}

			err = validateConfig(rawConfig)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.EqualError(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
		})
	}
}

// An empty YAML list has to survive the decoder as a non-nil empty slice,
// because that is what separates "left at the defaults" from "configured with
// nothing", and only the second is an error.
func TestMetricsBucketsEmptyListDecodesDistinctlyFromOmitted(t *testing.T) {
	configText := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n" +
		"metrics:\n  requestTPOTBuckets: []\n" +
		"plugins:\n- type: single-profile-handler\nschedulingProfiles:\n- name: default\n  plugins: []\n"
	rawConfig, _, err := LoadRawConfig([]byte(configText), testr.New(t))
	require.NoError(t, err)
	require.NotNil(t, rawConfig.Metrics)
	assert.NotNil(t, rawConfig.Metrics.RequestTPOTBuckets)
	assert.Empty(t, rawConfig.Metrics.RequestTPOTBuckets)
}

func edgeLiterals(n int) []string {
	out := make([]string, 0, n)
	for i := 1; i <= n; i++ {
		out = append(out, fmt.Sprintf("%g", float64(i)*0.001))
	}
	return out
}

// InstantiateAndConfigure is what both startup paths call, so the rejection has
// to surface from there and not only from validateConfig.
func TestInstantiateAndConfigureRejectsInvalidMetricsBuckets(t *testing.T) {
	registerTestPlugins(t)
	configText := "apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\n" +
		"metrics:\n  requestITLBuckets: [0.02, 0.01]\n" +
		"plugins:\n- type: " + testPluginType + "\nschedulingProfiles:\n- name: default\n  plugins: []\n"

	logger := logging.NewTestLogger()
	rawConfig, _, err := LoadRawConfig([]byte(configText), logger)
	require.NoError(t, err)

	_, err = InstantiateAndConfigure(rawConfig, testutils.NewTestHandle(context.Background()), logger)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "metrics.requestITLBuckets: bucket boundary 1 is 0.01, want a value greater than the preceding 0.02")
}
