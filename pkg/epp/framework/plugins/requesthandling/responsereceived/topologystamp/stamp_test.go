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

package topologystamp

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
	"github.com/llm-d/llm-d-router/test/utils"
)

func makeEndpoint(t *testing.T, topo *attrtopology.Topology) fwksched.Endpoint {
	t.Helper()
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "prefill-1", Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if topo != nil {
		ep.Put(attrtopology.TopologyAttributeKey, topo)
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
	response := &fwkrh.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, "host=node12,rack=r7", response.Headers[defaultHeaderName])
}

func TestResponseHeader_StampsFromConfiguredProfile(t *testing.T) {
	ep := makeEndpoint(t, &attrtopology.Topology{Hostname: "node12"})
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrh.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.profileName = "prefill"
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, "host=node12", response.Headers[defaultHeaderName])
}

func TestResponseHeader_NoopWhenProfileDidNotRun(t *testing.T) {
	request := &fwksched.InferenceRequest{SchedulingResult: &fwksched.SchedulingResult{PrimaryProfileName: "decode"}}
	response := &fwkrh.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenEndpointMissingAttribute(t *testing.T) {
	ep := makeEndpoint(t, nil)
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrh.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenSchedulingResultNil(t *testing.T) {
	request := &fwksched.InferenceRequest{}
	response := &fwkrh.Response{Headers: map[string]string{}}

	h := newTestHandler()
	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenResponseHeadersNil(t *testing.T) {
	ep := makeEndpoint(t, &attrtopology.Topology{Hostname: "node12"})
	request := &fwksched.InferenceRequest{SchedulingResult: schedulingResult(ep)}
	response := &fwkrh.Response{}

	h := newTestHandler()
	assert.NotPanics(t, func() {
		h.ResponseHeader(context.Background(), request, response, nil)
	})
}

func TestFactory_Defaults(t *testing.T) {
	handle := utils.NewTestHandle(utils.NewTestContext(t))
	p, err := Factory("", nil, handle)
	require.NoError(t, err)
	h, ok := p.(*Handler)
	require.True(t, ok)
	assert.Equal(t, defaultHeaderName, h.headerName)
	assert.Equal(t, PluginType, h.TypedName().Name)
}

func TestFactory_CustomHeaderNameRejected(t *testing.T) {
	handle := utils.NewTestHandle(utils.NewTestContext(t))
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"headerName": "x-custom-topology"}`)), handle)
	require.Error(t, err)
}

func TestFactory_ProfileName(t *testing.T) {
	tests := []struct {
		name      string
		profiles  sets.Set[string]
		params    string
		expectErr bool
	}{
		{name: "defined profile", profiles: sets.New("prefill", "decode"), params: `{"profileName":"prefill"}`},
		{name: "undefined profile rejected", profiles: sets.New("prefill", "decode"), params: `{"profileName":"prefil"}`, expectErr: true},
		{name: "profileName omitted", profiles: sets.New("prefill", "decode"), params: `{}`},
		{name: "profiles unknown to the handle", params: `{"profileName":"prefil"}`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handle := utils.NewTestHandleWithSchedulingProfiles(utils.NewTestContext(t), test.profiles)
			p, err := Factory("test", fwkplugin.StrictDecoder([]byte(test.params)), handle)
			if test.expectErr {
				require.ErrorContains(t, err, "invalid profileName of the 'topology-stamp-handler' handler - scheduling profile 'prefil' is not defined")
				require.Nil(t, p)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, p)
		})
	}
}
