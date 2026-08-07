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

package topologystamp

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

func makeEndpoint(t *testing.T, topo *attrtopology.Topology) fwksched.Endpoint {
	t.Helper()
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "prefill-1", Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if topo != nil {
		ep.Put(attrtopology.TopologyAttributeKey.String(), topo)
	}
	return ep
}

func newTestHandler() *Handler {
	return &Handler{
		typedName:  fwkplugin.TypedName{Type: PluginType, Name: "test"},
		headerName: defaultHeaderName,
		dataKey:    attrtopology.TopologyAttributeKey,
	}
}

func schedulingResult(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "prefill",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"prefill": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func TestResponseHeader_StampsFromPrimaryProfile(t *testing.T) {
	ep := makeEndpoint(t, &attrtopology.Topology{Hostname: "node12", Rack: "r7"})
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrc.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, "host=node12,rack=r7", response.Headers[defaultHeaderName])
}

func TestResponseHeader_StampsFromConfiguredProfile(t *testing.T) {
	ep := makeEndpoint(t, &attrtopology.Topology{Hostname: "node12"})
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrc.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.profileName = "prefill"
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, "host=node12", response.Headers[defaultHeaderName])
}

func TestResponseHeader_NoopWhenProfileDidNotRun(t *testing.T) {
	request := &fwksched.InferenceRequest{SchedulingResult: &fwksched.SchedulingResult{PrimaryProfileName: "decode"}}
	response := &fwkrc.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenEndpointMissingAttribute(t *testing.T) {
	ep := makeEndpoint(t, nil)
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrc.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenSchedulingResultNil(t *testing.T) {
	request := &fwksched.InferenceRequest{}
	response := &fwkrc.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenResponseHeadersNil(t *testing.T) {
	ep := makeEndpoint(t, &attrtopology.Topology{Hostname: "node12"})
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrc.Response{}

	h := newTestHandler()
	assert.NotPanics(t, func() {
		h.ResponseHeader(context.Background(), request, response, nil)
	})
}

func TestFactory_Defaults(t *testing.T) {
	p, err := Factory("", nil, nil)
	require.NoError(t, err)
	h, ok := p.(*Handler)
	require.True(t, ok)
	assert.Equal(t, defaultHeaderName, h.headerName)
	assert.Equal(t, PluginType, h.TypedName().Name)
}

func TestFactory_CustomHeaderName(t *testing.T) {
	p, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"headerName": "x-custom-topology"}`)), nil)
	require.NoError(t, err)
	h, ok := p.(*Handler)
	require.True(t, ok)
	assert.Equal(t, "x-custom-topology", h.headerName)
}
