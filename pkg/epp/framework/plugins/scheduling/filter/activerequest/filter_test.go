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

package activerequest

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

type stubEndpoint struct {
	metadata *datalayer.EndpointMetadata
	attr     datalayer.AttributeMap
}

func newStubEndpoint(name string) *stubEndpoint {
	return &stubEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubEndpoint) GetMetadata() *datalayer.EndpointMetadata   { return f.metadata }
func (f *stubEndpoint) UpdateMetadata(*datalayer.EndpointMetadata) {}
func (f *stubEndpoint) GetMetrics() *datalayer.Metrics             { return &datalayer.Metrics{} }
func (f *stubEndpoint) UpdateMetrics(*datalayer.Metrics)           {}
func (f *stubEndpoint) GetAttributes() datalayer.AttributeMap      { return f.attr }
func (f *stubEndpoint) String() string                             { return f.metadata.NamespacedName.String() }
func (f *stubEndpoint) Put(key string, val datalayer.Cloneable)    { f.attr.Put(key, val) }
func (f *stubEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	return f.attr.Get(key)
}
func (f *stubEndpoint) Keys() []string                { return f.attr.Keys() }
func (f *stubEndpoint) Clone() datalayer.AttributeMap { return f.attr.Clone() }

func newTestEndpoint(name string) scheduling.Endpoint {
	return newStubEndpoint(name)
}

func newTestEndpointWithLoad(name string, requests int64) scheduling.Endpoint {
	ep := newStubEndpoint(name)
	ep.Put(attrconcurrency.InFlightLoadDataKey.String(), &attrconcurrency.InFlightLoad{Requests: requests})
	return ep
}

func endpointNames(endpoints []scheduling.Endpoint) []string {
	names := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		names = append(names, ep.GetMetadata().NamespacedName.Name)
	}
	return names
}

func TestActiveRequestFilter_Filter(t *testing.T) {
	tests := []struct {
		name      string
		params    parameters
		endpoints func() []scheduling.Endpoint
		want      []string
	}{
		{
			name:   "keeps endpoints at or below maxRequests",
			params: parameters{MaxRequests: 2},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 1),
					newTestEndpointWithLoad("pod-b", 2),
					newTestEndpointWithLoad("pod-c", 3),
				}
			},
			want: []string{"pod-a", "pod-b"},
		},
		{
			name:   "endpoints without load attribute are kept",
			params: parameters{MaxRequests: 1},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpoint("pod-a"),
					newTestEndpointWithLoad("pod-b", 5),
				}
			},
			want: []string{"pod-a"},
		},
		{
			name:   "all filtered out without fallback returns empty",
			params: parameters{MaxRequests: 1},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 2),
					newTestEndpointWithLoad("pod-b", 3),
				}
			},
			want: []string{},
		},
		{
			name:   "all filtered out with fallback returns original candidates",
			params: parameters{MaxRequests: 1, FallbackOnEmpty: true},
			endpoints: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{
					newTestEndpointWithLoad("pod-a", 2),
					newTestEndpointWithLoad("pod-b", 3),
				}
			},
			want: []string{"pod-a", "pod-b"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := testutils.NewTestContext(t)
			filter, err := NewActiveRequestFilter("", test.params)
			require.NoError(t, err)

			got := filter.Filter(ctx, nil, test.endpoints())

			assert.Equal(t, test.want, endpointNames(got))
		})
	}
}

func TestNewActiveRequestFilter_InvalidMaxRequests(t *testing.T) {
	for _, maxRequests := range []int64{0, -1} {
		_, err := NewActiveRequestFilter("", parameters{MaxRequests: maxRequests})
		assert.Error(t, err, "maxRequests=%d must be rejected", maxRequests)
	}
}

func TestActiveRequestFilter_Factory(t *testing.T) {
	ctx := testutils.NewTestContext(t)

	decoder := json.NewDecoder(bytes.NewBufferString(`{"maxRequests": 5}`))
	p, err := Factory("test-filter", decoder, testutils.NewTestHandle(ctx))
	require.NoError(t, err)

	filter, ok := p.(*ActiveRequestFilter)
	require.True(t, ok)
	assert.Equal(t, ActiveRequestFilterType, filter.TypedName().Type)
	assert.Equal(t, "test-filter", filter.TypedName().Name)
	assert.Equal(t, int64(5), filter.maxRequests)
}

func TestActiveRequestFilter_FactoryRejectsMissingMaxRequests(t *testing.T) {
	ctx := testutils.NewTestContext(t)

	_, err := Factory("test-filter", nil, testutils.NewTestHandle(ctx))
	assert.Error(t, err)
}

func TestActiveRequestFilter_Consumes(t *testing.T) {
	filter, err := NewActiveRequestFilter("", parameters{MaxRequests: 1})
	require.NoError(t, err)

	consumes := filter.Consumes()

	require.Len(t, consumes.Required, 1)
	assert.Equal(t, attrconcurrency.InFlightLoad{}, consumes.Required[attrconcurrency.InFlightLoadDataKey])
}
