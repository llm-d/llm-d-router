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

package queuedepth

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

	_, ok := p.(*MultiClusterScorer).Consumes()[attrmetrics.MultiClusterQueueSizeKey]
	require.True(t, ok, "Consumes must advertise the pool queue-size key")
}

func newQueueEP(q float64, set bool) fwksched.Endpoint {
	attr := fwkdl.NewAttributes()
	if set {
		attr.Put(attrmetrics.MultiClusterQueueSizeKey, attrmetrics.ScalarMetricValue(q))
	}
	return fwksched.NewEndpoint(nil, nil, attr)
}

func TestMultiClusterScorer_Score(t *testing.T) {
	q := func(v float64) *float64 { return &v }

	// queues gives each endpoint's pool queue size, nil meaning the attribute is unset.
	// want gives the expected score per endpoint, nil meaning absent from the result.
	tests := []struct {
		name   string
		queues []*float64
		want   []*float64
	}{
		{
			name:   "min-max normalizes across clusters",
			queues: []*float64{q(2), q(6), q(10)},
			want:   []*float64{q(1.0), q(0.5), q(0.0)},
		},
		{
			name:   "equal queues score neutral",
			queues: []*float64{q(4), q(4)},
			want:   []*float64{q(1.0), q(1.0)},
		},
		{
			name:   "missing attribute is unscored",
			queues: []*float64{nil},
			want:   []*float64{nil},
		},
		{
			name:   "unscored endpoint is ignored in normalization",
			queues: []*float64{q(2), nil, q(10)},
			want:   []*float64{q(1.0), nil, q(0.0)},
		},
	}

	s := &MultiClusterScorer{name: "mc"}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eps := make([]fwksched.Endpoint, len(tt.queues))
			for i, v := range tt.queues {
				eps[i] = newQueueEP(deref(v), v != nil)
			}
			scores := s.Score(context.Background(), nil, eps)
			for i, w := range tt.want {
				got, ok := scores[eps[i]]
				if w == nil {
					assert.False(t, ok, "endpoint %d should be unscored", i)
					continue
				}
				assert.True(t, ok, "endpoint %d should be scored", i)
				assert.InDelta(t, *w, got, 1e-9)
			}
		})
	}
}

func deref(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}
