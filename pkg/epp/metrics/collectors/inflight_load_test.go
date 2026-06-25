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

package collectors

import (
	"strings"
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
)

type fakeInFlightLoad struct {
	requests map[string]int64
	tokens   map[string]int64
}

func (f *fakeInFlightLoad) InFlightRequestsSnapshot() map[string]int64 { return f.requests }
func (f *fakeInFlightLoad) InFlightTokensSnapshot() map[string]int64   { return f.tokens }

func TestInFlightLoadCollectorEmpty(t *testing.T) {
	collector := NewInFlightLoadCollector("default", &fakeInFlightLoad{})
	if err := promtestutil.CollectAndCompare(collector, strings.NewReader("")); err != nil {
		t.Fatal(err)
	}
}

func TestInFlightLoadCollectorPerEndpoint(t *testing.T) {
	collector := NewInFlightLoadCollector("default", &fakeInFlightLoad{
		requests: map[string]int64{
			"ns1/ep1": 3,
			"ns2/ep2": 5,
		},
		tokens: map[string]int64{
			"ns1/ep1": 30,
			"ns2/ep2": 50,
		},
	})

	expected := `
# HELP llm_d_epp_inflight_requests [ALPHA] Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_requests gauge
llm_d_epp_inflight_requests{endpoint_name="ep1",namespace="ns1",producer_name="default"} 3
llm_d_epp_inflight_requests{endpoint_name="ep2",namespace="ns2",producer_name="default"} 5
# HELP llm_d_epp_inflight_tokens [ALPHA] Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.
# TYPE llm_d_epp_inflight_tokens gauge
llm_d_epp_inflight_tokens{endpoint_name="ep1",namespace="ns1",producer_name="default"} 30
llm_d_epp_inflight_tokens{endpoint_name="ep2",namespace="ns2",producer_name="default"} 50
`

	if err := promtestutil.CollectAndCompare(collector, strings.NewReader(expected),
		"llm_d_epp_inflight_requests", "llm_d_epp_inflight_tokens"); err != nil {
		t.Fatal(err)
	}
}

func TestSplitNamespacedName(t *testing.T) {
	cases := []struct {
		id, wantName, wantNS string
	}{
		{"ns/ep", "ep", "ns"},
		{"bare", "bare", ""},
		{"ns/ep/extra", "ep/extra", "ns"},
	}
	for _, c := range cases {
		name, ns := splitNamespacedName(c.id)
		if name != c.wantName || ns != c.wantNS {
			t.Errorf("splitNamespacedName(%q) = (%q,%q), want (%q,%q)", c.id, name, ns, c.wantName, c.wantNS)
		}
	}
}
