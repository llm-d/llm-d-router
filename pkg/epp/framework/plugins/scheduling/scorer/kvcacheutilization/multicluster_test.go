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

package kvcacheutilization

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
)

// The factory delegates to the stock scorer, reports the multi-cluster type, and the
// Consumes override advertises the pool key (not the stock per-pod key).
func TestMultiClusterScorerFactory(t *testing.T) {
	h := fwkplugin.NewEppHandle(context.Background(), nil, fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
	p, err := MultiClusterScorerFactory("mc", nil, h)
	require.NoError(t, err)
	require.IsType(t, &MultiClusterScorer{}, p)
	require.Equal(t, MultiClusterScorerType, p.TypedName().Type)
	require.Equal(t, "mc", p.TypedName().Name)

	_, ok := p.(*MultiClusterScorer).Consumes()[attrmetrics.MultiClusterKVCacheUtilizationKey]
	require.True(t, ok, "Consumes must advertise the pool KV-cache utilization key")
}

func TestMultiClusterScorer_Score(t *testing.T) {
	newEP := func(util float64, set bool) fwksched.Endpoint {
		attr := fwkdl.NewAttributes()
		if set {
			attr.Put(attrmetrics.MultiClusterKVCacheUtilizationKey, attrmetrics.ScalarMetricValue(util))
		}
		return fwksched.NewEndpoint(nil, nil, attr)
	}

	tests := []struct {
		name       string
		util       float64
		set        bool
		wantScore  float64
		wantAbsent bool // absent from the result map
	}{
		{name: "low utilization scores high", util: 0.2, set: true, wantScore: 0.8},
		{name: "high utilization scores low", util: 0.9, set: true, wantScore: 0.1},
		{name: "missing attribute is unscored", set: false, wantAbsent: true},
	}

	s := &MultiClusterScorer{name: "mc"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ep := newEP(tt.util, tt.set)
			scores := s.Score(context.Background(), nil, []fwksched.Endpoint{ep})
			got, ok := scores[ep]
			if tt.wantAbsent {
				assert.False(t, ok)
				return
			}
			assert.True(t, ok)
			assert.InDelta(t, tt.wantScore, got, 1e-9)
		})
	}
}
