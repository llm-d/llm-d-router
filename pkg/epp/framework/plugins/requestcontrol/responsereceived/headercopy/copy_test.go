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

package headercopy

import (
	"context"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func newTestHandler(headers ...string) *Handler {
	return &Handler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: "test"},
		headers:   headers,
	}
}

func TestResponseHeader_CopiesConfiguredHeaders(t *testing.T) {
	request := &fwksched.InferenceRequest{Headers: map[string]string{
		routing.PrefillEndpointHeader: "10.0.0.2:8000",
		routing.KVCachePrefillHeader:  "10.0.0.2:7777",
		"authorization":               "Bearer secret",
	}}
	response := &fwkrc.Response{Headers: map[string]string{}}
	h := newTestHandler(routing.PrefillEndpointHeader, routing.KVCachePrefillHeader)

	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, map[string]string{
		routing.PrefillEndpointHeader: "10.0.0.2:8000",
		routing.KVCachePrefillHeader:  "10.0.0.2:7777",
	}, response.Headers)
}

func TestResponseHeader_SkipsAbsentHeaders(t *testing.T) {
	request := &fwksched.InferenceRequest{Headers: map[string]string{
		routing.PrefillEndpointHeader: "10.0.0.2:8000",
	}}
	response := &fwkrc.Response{Headers: map[string]string{"x-existing": "kept"}}
	h := newTestHandler(routing.PrefillEndpointHeader, routing.KVCachePrefillHeader)

	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, map[string]string{
		"x-existing":                  "kept",
		routing.PrefillEndpointHeader: "10.0.0.2:8000",
	}, response.Headers)
}

func TestResponseHeader_OverwritesResponseValue(t *testing.T) {
	request := &fwksched.InferenceRequest{Headers: map[string]string{routing.KVCachePrefillHeader: "10.0.0.2:7778"}}
	response := &fwkrc.Response{Headers: map[string]string{routing.KVCachePrefillHeader: "from-upstream"}}
	h := newTestHandler(routing.KVCachePrefillHeader)

	h.ResponseHeader(context.Background(), request, response, nil)

	assert.Equal(t, "10.0.0.2:7778", response.Headers[routing.KVCachePrefillHeader])
}

func TestResponseHeader_NoopWhenRequestHeadersNil(t *testing.T) {
	response := &fwkrc.Response{Headers: map[string]string{}}
	h := newTestHandler(routing.PrefillEndpointHeader)

	h.ResponseHeader(context.Background(), &fwksched.InferenceRequest{}, response, nil)
	h.ResponseHeader(context.Background(), nil, response, nil)

	assert.Empty(t, response.Headers)
}

func TestResponseHeader_NoopWhenResponseHeadersNil(t *testing.T) {
	request := &fwksched.InferenceRequest{Headers: map[string]string{routing.PrefillEndpointHeader: "10.0.0.2:8000"}}
	h := newTestHandler(routing.PrefillEndpointHeader)

	assert.NotPanics(t, func() {
		h.ResponseHeader(context.Background(), request, &fwkrc.Response{}, nil)
		h.ResponseHeader(context.Background(), request, nil, nil)
	})
}

func TestFactory_LowercasesAndTrimsNames(t *testing.T) {
	p, err := Factory("", fwkplugin.StrictDecoder([]byte(`{"headers": [" X-Llm-D-Prefiller-Host-Port ", "x-llm-d-kv-cache-prefill-p2p-host-port"]}`)), nil)
	require.NoError(t, err)
	h, ok := p.(*Handler)
	require.True(t, ok)
	assert.Equal(t, []string{routing.PrefillEndpointHeader, routing.KVCachePrefillHeader}, h.headers)
	assert.Equal(t, PluginType, h.TypedName().Name)
}

func TestFactory_DeduplicatesNamesAndWarns(t *testing.T) {
	var logs []string
	logger := funcr.New(func(_, args string) { logs = append(logs, args) }, funcr.Options{})
	handle := testutils.NewTestHandle(log.IntoContext(t.Context(), logger))

	p, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"headers": ["X-A", "x-b", " x-a ", "x-b"]}`)), handle)
	require.NoError(t, err)
	h, ok := p.(*Handler)
	require.True(t, ok)
	assert.Equal(t, []string{"x-a", "x-b"}, h.headers)
	require.Len(t, logs, 1)
	assert.Contains(t, logs[0], "warning: duplicate header names")
	assert.Contains(t, logs[0], `"duplicates"=["x-a" "x-b"]`)
}

func TestFactory_KeepsName(t *testing.T) {
	p, err := Factory("copy-prefill", fwkplugin.StrictDecoder([]byte(`{"headers": ["x-a"]}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, "copy-prefill", p.TypedName().Name)
}

func TestFactory_RejectsMissingHeaders(t *testing.T) {
	_, err := Factory("test", nil, nil)
	require.Error(t, err)
	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{}`)), nil)
	require.Error(t, err)
	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"headers": []}`)), nil)
	require.Error(t, err)
}

func TestFactory_RejectsEmptyName(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"headers": ["x-a", " "]}`)), nil)
	require.Error(t, err)
}

func TestFactory_RejectsUnknownParameter(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"headers": ["x-a"], "profileName": "prefill"}`)), nil)
	require.Error(t, err)
}
