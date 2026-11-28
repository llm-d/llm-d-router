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
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// fakeSaturationDetector is a saturation detector of any plugin type that reports no load.
type fakeSaturationDetector struct {
	pluginType string
}

func (f *fakeSaturationDetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: f.pluginType, Name: "fake"}
}

func (f *fakeSaturationDetector) Saturation(context.Context, []datalayer.Endpoint) float64 {
	return 0
}

// TestDeriveEvictionConfirmationGrace checks the grace for each detector kind with
// --refresh-metrics-interval 50ms and --metrics-staleness-threshold 2s. The utilization detector's
// grace uses the larger of the flag and its own metricsStalenessThreshold; other scraped detectors
// keep the flag.
func TestDeriveEvictionConfirmationGrace(t *testing.T) {
	tests := []struct {
		name     string
		detector fwkfc.SaturationDetector
		want     time.Duration
	}{
		{
			name: "utilization_default_threshold_keeps_flag",
			detector: utilization.NewDetector("pool-queue",
				utilization.Config{MetricsStalenessThreshold: 200 * time.Millisecond}, logr.Discard()),
			want: 2300 * time.Millisecond,
		},
		{
			name: "utilization_threshold_above_flag",
			detector: utilization.NewDetector("pool-queue",
				utilization.Config{MetricsStalenessThreshold: 5 * time.Second}, logr.Discard()),
			want: 5300 * time.Millisecond,
		},
		{
			name:     "concurrency",
			detector: &fakeSaturationDetector{pluginType: concurrency.ConcurrencyDetectorType},
			want:     10 * time.Millisecond,
		},
		{
			name:     "other_detector_uses_flag",
			detector: &fakeSaturationDetector{pluginType: "custom-detector"},
			want:     2300 * time.Millisecond,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := runserver.NewOptions()
			opts.RefreshMetricsInterval = 50 * time.Millisecond
			opts.MetricsStalenessThreshold = 2 * time.Second
			require.Equal(t, tc.want, deriveEvictionConfirmationGrace(tc.detector, opts))
		})
	}
}
