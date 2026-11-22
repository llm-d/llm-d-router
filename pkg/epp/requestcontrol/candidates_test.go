/*
Copyright 2025 The Kubernetes Authors.
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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// --- DatastoreEndpointCandidates Tests ---

func TestDatastoreEndpointCandidates_Locate(t *testing.T) {
	t.Parallel()

	endpointA := makeMockEndpoint("pod-a", "10.0.0.1")
	endpointB := makeMockEndpoint("pod-b", "10.0.0.2")
	endpointC := makeMockEndpoint("pod-c", "10.0.0.3")

	allEndpoints := []fwkdl.Endpoint{endpointA, endpointB, endpointC}
	mockDS := &mockDatastore{pods: allEndpoints}

	tests := []struct {
		name                string
		opts                []EndpointCandidatesOption
		metadata            map[string]any
		expectedEndpointIPs []string
	}{
		{
			name:                "Nil metadata returns all pods",
			metadata:            nil,
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name:                "Empty metadata returns all pods",
			metadata:            map[string]any{},
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name: "Metadata without subset namespace returns all pods",
			metadata: map[string]any{
				"other-filter": "value",
			},
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name: "Subset filter with single match",
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
			}),
			expectedEndpointIPs: []string{"10.0.0.1"},
		},
		{
			name: "Subset filter with multiple matches",
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
				"10.0.0.3:9090",
			}),
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.3"},
		},
		{
			name: "Subset filter with no matches (Scale-from-Zero scenario)",
			metadata: makeMetadataWithSubset([]any{
				"192.168.1.1:8080", // Does not exist in mockDS
			}),
			expectedEndpointIPs: []string{},
		},
		{
			name: "Subset filter is present but list is empty",
			metadata: map[string]any{
				metadata.SubsetFilterNamespace: map[string]any{
					metadata.SubsetFilterKey: []any{},
				},
			},
			expectedEndpointIPs: []string{},
		},
		{
			name: "Subset filter contains malformed data (non-string)",
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
				12345, // Should be ignored
			}),
			expectedEndpointIPs: []string{"10.0.0.1"},
		},
		{
			name: "Subset filter with match (filter disabled)",
			opts: []EndpointCandidatesOption{
				WithDisableEndpointSubsetFilter(true),
			},
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
			}),
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name: "Subset filter is present but list is empty (filter disabled)",
			opts: []EndpointCandidatesOption{
				WithDisableEndpointSubsetFilter(true),
			},
			metadata:            makeMetadataWithSubset([]any{}),
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name: "Subset filter with no matches (filter disabled)",
			opts: []EndpointCandidatesOption{
				WithDisableEndpointSubsetFilter(true),
			},
			metadata: makeMetadataWithSubset([]any{
				"192.168.1.1:8080",
			}),
			expectedEndpointIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			endpointCandidates := NewDatastoreEndpointCandidates(mockDS, tc.opts...)
			result := endpointCandidates.Locate(context.Background(), tc.metadata)

			gotIPs := make([]string, len(result))
			for idx, ep := range result {
				gotIPs[idx] = ep.GetMetadata().GetIPAddress()
			}
			assert.ElementsMatch(t, tc.expectedEndpointIPs, gotIPs, "Locate returned unexpected set of endpoint candidates")
		})
	}
}

// --- Helpers & Mocks ---

func makeMockEndpoint(name, ip string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Namespace: "default", Name: name},
		Address: ip,
	}, nil)
}

func makeMetadataWithSubset(endpoints []any) map[string]any {
	return map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: endpoints,
		},
	}
}

func TestDatastoreEndpointCandidates_IPv6Subset(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		ip        string
		subset    string
		wantMatch bool
	}{
		{"IPv4 with port", "10.0.0.1", "10.0.0.1:8080", true},
		{"IPv4 without port", "10.0.0.1", "10.0.0.1", true},
		{"bracketed IPv6 with port", "2001:db8::1", "[2001:db8::1]:8080", true},
		{"IPv6 without port", "2001:db8::1", "2001:db8::1", true},
		{"noncanonical IPv6 with port", "2001:db8::1", "[2001:DB8:0::1]:8080", true},
		{"noncanonical IPv6 without port", "2001:db8::1", "2001:0db8:0:0:0:0:0:1", true},
		{"bracketed IPv6 without port", "2001:db8::1", "[2001:db8::1]", true},
		{"IPv6 zone with port", "fe80::1%eth0", "[FE80:0::1%eth0]:8080", true},
		{"IPv6 zone without port", "fe80::1%eth0", "fe80::1%eth0", true},
		{"bracketed IPv6 zone without port", "fe80::1%eth0", "[fe80::1%eth0]", true},
		{"IPv6 nonnumeric port keeps existing behavior", "2001:db8::1", "[2001:db8::1]:invalid", true},
		{"IPv6 empty port keeps existing behavior", "2001:db8::1", "[2001:db8::1]:", true},
		{"IPv6 final hextet stays address", "2001:db8::1:8080", "2001:db8::1:8080", true},
		{"hostname with port", "worker.example", "worker.example:8080", true},
		{"hostname without port", "worker.example", "worker.example", true},
		{"nonnumeric port keeps existing behavior", "10.0.0.1", "10.0.0.1:invalid", true},
		{"empty port keeps existing behavior", "10.0.0.1", "10.0.0.1:", true},
		{"extra colon stays unmatched", "10.0.0.1", "10.0.0.1:8080:invalid", false},
		{"bracketed hostname stays unmatched", "worker.example", "[worker.example]:8080", false},
		{"bracketed bare hostname stays unmatched", "worker.example", "[worker.example]", false},
		{"missing IPv6 closing bracket stays unmatched", "2001:db8::1", "[2001:db8::1", false},
		{"missing IPv6 opening bracket stays unmatched", "2001:db8::1", "2001:db8::1]", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ep := makeMockEndpoint("target", tc.ip)
			other := makeMockEndpoint("other", "10.0.0.99")
			candidates := NewDatastoreEndpointCandidates(&mockDatastore{pods: []fwkdl.Endpoint{ep, other}})
			got := candidates.Locate(context.Background(), makeMetadataWithSubset([]any{tc.subset}))
			if !tc.wantMatch {
				if len(got) != 0 {
					t.Fatalf("subset %q should remain unmatched, got %v", tc.subset, got)
				}
				return
			}
			if len(got) != 1 || got[0].GetMetadata().GetIPAddress() != tc.ip {
				t.Fatalf("subset %q should retain exactly endpoint %q, got %v", tc.subset, tc.ip, got)
			}
		})
	}
}
