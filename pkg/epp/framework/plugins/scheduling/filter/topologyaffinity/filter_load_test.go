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

package topologyaffinity

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/activerequest"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

func TestFilter_LoadAllowance(t *testing.T) {
	for _, tt := range []struct {
		name   string
		config string
		counts [4]int64
		all    bool
	}{
		{"disabled", `{}`, [4]int64{6, 9, 2, 5}, false},
		{"null disables", `{"loadAllowance":null}`, [4]int64{6, 9, 2, 5}, false},
		{"above allowance", `{"loadAllowance":2}`, [4]int64{6, 9, 2, 5}, true},
		{"at allowance", `{"loadAllowance":4}`, [4]int64{6, 9, 2, 5}, false},
		{"below allowance", `{"loadAllowance":5}`, [4]int64{6, 9, 2, 5}, false},
		{"second local is best", `{"loadAllowance":2}`, [4]int64{9, 1, 2, 5}, false},
		{"second remote is best", `{"loadAllowance":2}`, [4]int64{6, 9, 8, 2}, true},
		{"zero allowance opens", `{"loadAllowance":0}`, [4]int64{3, 9, 2, 5}, true},
		{"zero allowance ties", `{"loadAllowance":0}`, [4]int64{2, 9, 2, 5}, false},
		{"zero counts are valid", `{"loadAllowance":0}`, [4]int64{}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p, err := Factory("test", fwkplugin.StrictDecoder([]byte(tt.config)), nil)
			require.NoError(t, err)
			endpoints := loadTestEndpoints(t, tt.counts)
			got := p.(*Filter).Filter(context.Background(), requestWithPeer(&attrtopology.Topology{Hostname: "h1"}), endpoints)
			want := endpoints[:2]
			if tt.all {
				want = endpoints
			}
			assert.Equal(t, want, got)
		})
	}
}

func loadTestEndpoints(t *testing.T, counts [4]int64) []fwksched.Endpoint {
	t.Helper()
	endpoints := []fwksched.Endpoint{
		makeEndpoint(t, "local-a", &attrtopology.Topology{Hostname: "h1"}),
		makeEndpoint(t, "local-b", &attrtopology.Topology{Hostname: "h1"}),
		makeEndpoint(t, "remote-a", &attrtopology.Topology{Hostname: "h2"}),
		makeEndpoint(t, "remote-b", &attrtopology.Topology{Hostname: "h2"}),
	}
	for i, ep := range endpoints {
		ep.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: counts[i]})
	}
	return endpoints
}

func TestFilter_LoadAllowanceMissingData(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func([]fwksched.Endpoint)
	}{
		{"missing local load", func(e []fwksched.Endpoint) {
			e[0] = makeEndpoint(t, "local-a", &attrtopology.Topology{Hostname: "h1"})
		}},
		{"missing remote load", func(e []fwksched.Endpoint) {
			e[2] = makeEndpoint(t, "remote-a", &attrtopology.Topology{Hostname: "h2"})
		}},
		{"nil load", func(e []fwksched.Endpoint) {
			e[2].Put(attrconcurrency.InFlightLoadDataKey, (*attrconcurrency.InFlightLoad)(nil))
		}},
		{"wrong load type", func(e []fwksched.Endpoint) {
			e[2].Put(attrconcurrency.InFlightLoadDataKey, &attrtopology.Topology{})
		}},
		{"negative load", func(e []fwksched.Endpoint) {
			e[2].Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: -1})
		}},
		{"missing topology", func(e []fwksched.Endpoint) {
			e[2] = makeEndpoint(t, "remote-a", nil)
			e[2].Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Requests: 0})
		}},
		{"missing host is not known remote", func(e []fwksched.Endpoint) {
			e[2].Put(attrtopology.TopologyAttributeKey, &attrtopology.Topology{Rack: "r1"})
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"loadAllowance":0}`)), nil)
			require.NoError(t, err)
			endpoints := loadTestEndpoints(t, [4]int64{6, 9, 2, 5})
			tt.change(endpoints)
			got := p.(*Filter).Filter(context.Background(), requestWithPeer(&attrtopology.Topology{Hostname: "h1"}), endpoints)
			assert.Equal(t, endpoints[:2], got)
		})
	}
}

func TestFilter_LoadAllowanceUnreachableGroups(t *testing.T) {
	p, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"loadAllowance":0}`)), nil)
	require.NoError(t, err)
	f := p.(*Filter)
	endpoints := loadTestEndpoints(t, [4]int64{6, 9, 2, 5})
	req := requestWithPeer(&attrtopology.Topology{Hostname: "h1"})
	assert.Equal(t, endpoints[:2], f.Filter(context.Background(), req, endpoints[:2]), "no remote candidates")
	assert.Equal(t, endpoints[2:], f.Filter(context.Background(), req, endpoints[2:]), "no local candidates")
	assert.Equal(t, endpoints, f.Filter(context.Background(), &fwksched.InferenceRequest{}, endpoints), "no peer")
	assert.Empty(t, f.Filter(context.Background(), req, nil))
}

func TestFactory_LoadAllowanceValidation(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"loadAllowance":-1}`)), nil)
	require.ErrorContains(t, err, "loadAllowance must be non-negative")
	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"loadAllowance":1.5}`)), nil)
	require.Error(t, err)
}

func TestFilter_LoadAllowanceNamedProducer(t *testing.T) {
	p, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"loadAllowance":0,"inFlightLoadProducerName":"custom-load"}`)), nil)
	require.NoError(t, err)
	f := p.(*Filter)
	key := attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("custom-load")
	assert.Equal(t, attrconcurrency.InFlightLoad{}, f.Consumes().Required[key])
	endpoints := loadTestEndpoints(t, [4]int64{})
	for i, count := range [4]int64{6, 9, 2, 5} {
		endpoints[i].Put(key, &attrconcurrency.InFlightLoad{Requests: count})
	}
	assert.Equal(t, endpoints, f.Filter(context.Background(), requestWithPeer(&attrtopology.Topology{Hostname: "h1"}), endpoints))
}

func TestFilter_LoadAllowanceProfileSelection(t *testing.T) {
	for _, tt := range []struct {
		name       string
		config     string
		want       string
		candidates int
	}{
		{"disabled stays local", `{}`, "local-a", 2},
		{"escape reaches picker", `{"loadAllowance":2}`, "remote-a", 4},
		{"boundary stays local", `{"loadAllowance":4}`, "local-a", 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			p, err := Factory("topology", fwkplugin.StrictDecoder([]byte(tt.config)), nil)
			require.NoError(t, err)
			scorer := activerequest.NewActiveRequest(ctx, nil)
			picker := maxscore.NewMaxScorePicker(1)
			datalayer.RegisterScopeSpecs([]fwkplugin.Plugin{p, scorer, picker})
			t.Cleanup(func() { datalayer.RegisterScopeSpecs(nil) })
			profile := scheduling.NewSchedulerProfile().
				WithFilters(p.(*Filter)).
				WithScorers(scheduling.NewWeightedScorer(scorer, 1)).
				WithPicker(picker)
			result, err := profile.Run(ctx, requestWithPeer(&attrtopology.Topology{Hostname: "h1"}), loadTestEndpoints(t, [4]int64{6, 9, 2, 5}))
			require.NoError(t, err)
			require.Len(t, result.TargetEndpoints, 1)
			assert.Equal(t, tt.want, result.TargetEndpoints[0].GetMetadata().ID.Name)
			assert.Len(t, result.ScoredCandidates, tt.candidates)
		})
	}
}
