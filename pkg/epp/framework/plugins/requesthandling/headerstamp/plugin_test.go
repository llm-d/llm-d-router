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

package headerstamp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/attribute"
)

func TestFactory_Validation(t *testing.T) {
	cases := []struct {
		name    string
		params  string
		wantErr string
	}{
		{name: "empty stamps rejected", params: `{"stamps": []}`, wantErr: "non-empty"},
		{name: "missing attributeKey rejected", params: `{"stamps": [{"headerName": ":path"}]}`, wantErr: "attributeKey and headerName"},
		{name: "missing headerName rejected", params: `{"stamps": [{"attributeKey": "pathPrefix"}]}`, wantErr: "attributeKey and headerName"},
		{name: "bad mode rejected", params: `{"stamps": [{"attributeKey": "a", "headerName": "h", "mode": "append"}]}`, wantErr: "must be"},
		{name: "valid config", params: `{"stamps": [{"attributeKey": "pathPrefix", "headerName": ":path", "mode": "prepend"}]}`},
		{name: "mode defaults to set", params: `{"stamps": [{"attributeKey": "model", "headerName": "x-llm-d-model"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dec := json.NewDecoder(strings.NewReader(tc.params))
			p, err := Factory("test", dec, nil)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, p)
		})
	}
}

func makeEndpoint(t *testing.T, attrs map[fwkplugin.DataKey]attrstring.Value) fwksched.Endpoint {
	t.Helper()
	meta := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "ep-1", Namespace: "default"}}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	for k, v := range attrs {
		ep.Put(k, v)
	}
	return ep
}

func schedulingResult(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func newTestHandler(rules ...stampRule) *Handler {
	return &Handler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: "test"},
		rules:     rules,
	}
}

func pathPrefixKey() fwkplugin.DataKey {
	return fwkplugin.NewDataKey("pathPrefix", attribute.AttributeProducerType)
}

func TestPreRequest_PrependPath(t *testing.T) {
	ep := makeEndpoint(t, map[fwkplugin.DataKey]attrstring.Value{pathPrefixKey(): "/laneA/"})
	h := newTestHandler(stampRule{headerName: ":path", mode: modePrepend, dataKey: pathPrefixKey()})
	request := &fwksched.InferenceRequest{Headers: map[string]string{":path": "/llm-chat/v1/chat/completions"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(ep)))
	assert.Equal(t, "/laneA/llm-chat/v1/chat/completions", request.Headers[":path"],
		"prefix prepended with trailing slash normalized")
}

func TestPreRequest_SetOverwritesClientValue(t *testing.T) {
	modelKey := fwkplugin.NewDataKey("model", attribute.AttributeProducerType)
	ep := makeEndpoint(t, map[fwkplugin.DataKey]attrstring.Value{modelKey: "nvfp4"})
	h := newTestHandler(stampRule{headerName: "x-llm-d-model", mode: modeSet, dataKey: modelKey})
	request := &fwksched.InferenceRequest{Headers: map[string]string{"x-llm-d-model": "client-spoofed"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(ep)))
	assert.Equal(t, "nvfp4", request.Headers["x-llm-d-model"],
		"endpoint attribute wins over any client-supplied value")
}

func TestPreRequest_MissingAttributeSkips(t *testing.T) {
	ep := makeEndpoint(t, nil)
	h := newTestHandler(stampRule{headerName: ":path", mode: modePrepend, dataKey: pathPrefixKey()})
	request := &fwksched.InferenceRequest{Headers: map[string]string{":path": "/v1/chat/completions"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(ep)))
	assert.Equal(t, "/v1/chat/completions", request.Headers[":path"], "path untouched without the attribute")
}

func TestPreRequest_NoSelectedEndpoint(t *testing.T) {
	h := newTestHandler(stampRule{headerName: ":path", mode: modePrepend, dataKey: pathPrefixKey()})
	request := &fwksched.InferenceRequest{Headers: map[string]string{":path": "/v1/x"}}
	result := &fwksched.SchedulingResult{PrimaryProfileName: "missing"}

	require.NoError(t, h.PreRequest(context.Background(), request, result))
	assert.Equal(t, "/v1/x", request.Headers[":path"])
}

func TestPreRequest_MultipleStamps(t *testing.T) {
	modelKey := fwkplugin.NewDataKey("model", attribute.AttributeProducerType)
	ep := makeEndpoint(t, map[fwkplugin.DataKey]attrstring.Value{
		pathPrefixKey(): "/laneB",
		modelKey:        "fp8",
	})
	h := newTestHandler(
		stampRule{headerName: ":path", mode: modePrepend, dataKey: pathPrefixKey()},
		stampRule{headerName: "x-llm-d-model", mode: modeSet, dataKey: modelKey},
	)
	request := &fwksched.InferenceRequest{Headers: map[string]string{":path": "/v1/x"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(ep)))
	assert.Equal(t, "/laneB/v1/x", request.Headers[":path"])
	assert.Equal(t, "fp8", request.Headers["x-llm-d-model"])
}

func TestPreRequest_NilHeaders(t *testing.T) {
	modelKey := fwkplugin.NewDataKey("model", attribute.AttributeProducerType)
	ep := makeEndpoint(t, map[fwkplugin.DataKey]attrstring.Value{modelKey: "fp8"})
	h := newTestHandler(stampRule{headerName: "x-llm-d-model", mode: modeSet, dataKey: modelKey})
	request := &fwksched.InferenceRequest{}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(ep)))
	assert.Equal(t, "fp8", request.Headers["x-llm-d-model"])
}

func TestFactory_RejectsEPPOwnedHeaderNames(t *testing.T) {
	cases := []struct {
		name       string
		headerName string
	}{
		{name: "system-owned header stripped on the way out", headerName: "x-llm-d-inference-objective"},
		{name: "internal routing header", headerName: "x-llm-d-prefiller-host-port"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := `{"stamps": [{"attributeKey": "a", "headerName": "` + tc.headerName + `"}]}`
			_, err := Factory("test", json.NewDecoder(strings.NewReader(params)), nil)
			require.Error(t, err, "headerName %q must be rejected at factory time", tc.headerName)
		})
	}
}

func TestFactory_NonDefaultProducerName(t *testing.T) {
	params := `{"attributeProducerName": "label-producer", "stamps": [{"attributeKey": "tenant", "headerName": "x-tenant"}]}`
	p, err := Factory("test", json.NewDecoder(strings.NewReader(params)), nil)
	require.NoError(t, err)
	consumes := p.(fwkplugin.ConsumerPlugin).Consumes()
	_, ok := consumes.Optional[fwkplugin.NewDataKey("tenant", "label-producer")]
	assert.True(t, ok, "stamp must read from the configured producer, not the default")
}

func TestPreRequest_PrependNormalizesMissingLeadingSlash(t *testing.T) {
	ep := makeEndpoint(t, map[fwkplugin.DataKey]attrstring.Value{pathPrefixKey(): "laneA"})
	h := newTestHandler(stampRule{headerName: ":path", mode: modePrepend, dataKey: pathPrefixKey()})
	request := &fwksched.InferenceRequest{Headers: map[string]string{":path": "/v1/chat/completions"}}

	require.NoError(t, h.PreRequest(context.Background(), request, schedulingResult(ep)))
	assert.Equal(t, "/laneA/v1/chat/completions", request.Headers[":path"],
		"a slash-less attribute value must still yield a valid prefixed path")
}
