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

package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
)

// mockModelsDatastore is a minimal Datastore whose AggregateModels returns a fixed result.
// The embedded Datastore satisfies the rest of the interface; only AggregateModels is implemented.
type mockModelsDatastore struct {
	Datastore
	body      json.RawMessage
	collected int
	calls     int
}

func (m *mockModelsDatastore) AggregateModels() (json.RawMessage, int) {
	m.calls++
	return m.body, m.collected
}

func modelsHeaderRequest(method, path string) *extProcPb.ProcessingRequest_RequestHeaders {
	return &extProcPb.ProcessingRequest_RequestHeaders{
		RequestHeaders: &extProcPb.HttpHeaders{
			EndOfStream: true,
			Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{
				{Key: ":method", Value: method},
				{Key: ":path", Value: path},
			}},
		},
	}
}

func TestHandleModelsRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		method      string
		path        string
		wantHandled bool
	}{
		{name: "GET /v1/models is served", method: http.MethodGet, path: "/v1/models", wantHandled: true},
		{name: "trailing slash is served", method: http.MethodGet, path: "/v1/models/", wantHandled: true},
		{name: "query string is ignored", method: http.MethodGet, path: "/v1/models?foo=bar", wantHandled: true},
		{name: "POST is not served", method: http.MethodPost, path: "/v1/models", wantHandled: false},
		{name: "completions path is not served", method: http.MethodGet, path: "/v1/completions", wantHandled: false},
		{name: "single-model path is not served", method: http.MethodGet, path: "/v1/models/gpt", wantHandled: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ds := &mockModelsDatastore{
				body:      json.RawMessage(`{"object":"list","data":[{"id":"base"}]}`),
				collected: 1,
			}
			s := &StreamingServer{datastore: ds}
			reqCtx := &RequestContext{Request: &Request{Headers: make(map[string]string)}}

			handled, err := s.tryServeModelList(context.Background(), reqCtx, modelsHeaderRequest(tc.method, tc.path))
			require.NoError(t, err)

			assert.Equal(t, tc.wantHandled, handled)
			if !tc.wantHandled {
				assert.Nil(t, reqCtx.localResp)
				assert.NotEqual(t, RequestAnsweredLocal, reqCtx.RequestState)
				assert.Equal(t, 0, ds.calls, "AggregateModels should not be called for unhandled requests")
				return
			}

			assert.Equal(t, RequestAnsweredLocal, reqCtx.RequestState)
			require.NotNil(t, reqCtx.localResp)
			ir := reqCtx.localResp.GetImmediateResponse()
			require.NotNil(t, ir)
			assert.Equal(t, envoyTypePb.StatusCode_OK, ir.Status.Code)

			require.NotNil(t, ir.Headers)
			gotHeaders := make(map[string]string, len(ir.Headers.SetHeaders))
			for _, h := range ir.Headers.SetHeaders {
				gotHeaders[h.Header.Key] = string(h.Header.RawValue)
			}
			assert.Equal(t, "application/json", gotHeaders["content-type"])

			// Body is passed through unchanged from the datastore.
			assert.Equal(t, []byte(ds.body), ir.Body)
		})
	}
}

func TestTryServeModelList_ResponsePassesThroughDatalayerBody(t *testing.T) {
	t.Parallel()

	// Verify that the handler places the raw bytes from AggregateModels verbatim into the Envoy
	// response, including the OpenAI-compatible JSON keys.
	const rawJSON = `{"object":"list","data":[{"id":"base","object":"model","created":1699999999,"owned_by":"vllm","parent":"root"}]}`
	ds := &mockModelsDatastore{
		body:      json.RawMessage(rawJSON),
		collected: 1,
	}
	s := &StreamingServer{datastore: ds}
	reqCtx := &RequestContext{Request: &Request{Headers: make(map[string]string)}}

	handled, err := s.tryServeModelList(context.Background(), reqCtx, modelsHeaderRequest(http.MethodGet, "/v1/models"))
	require.NoError(t, err)
	require.True(t, handled)
	require.NotNil(t, reqCtx.localResp)

	assert.Equal(t, []byte(rawJSON), reqCtx.localResp.GetImmediateResponse().Body)
}

func TestTryServeModelList_NotScrapedYetReturns503(t *testing.T) {
	t.Parallel()

	// Endpoints are registered but none have been scraped yet (cold start / scale-up window, since
	// scraping is interval-based). With no model list collected from any endpoint, this must be a
	// 503 so the client retries, not a misleading empty 200.
	ds := &mockModelsDatastore{collected: 0}
	s := &StreamingServer{datastore: ds}
	reqCtx := &RequestContext{Request: &Request{Headers: make(map[string]string)}}

	handled, err := s.tryServeModelList(context.Background(), reqCtx, modelsHeaderRequest(http.MethodGet, "/v1/models"))

	assert.True(t, handled)
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	assert.Equal(t, errcommon.ServiceUnavailable, e.Code)
	assert.Nil(t, reqCtx.localResp)
	assert.NotEqual(t, RequestAnsweredLocal, reqCtx.RequestState)
}

func TestTryServeModelList_NoEndpointsReturns503(t *testing.T) {
	t.Parallel()

	s := &StreamingServer{datastore: &mockModelsDatastore{}}
	reqCtx := &RequestContext{Request: &Request{Headers: make(map[string]string)}}

	handled, err := s.tryServeModelList(context.Background(), reqCtx, modelsHeaderRequest(http.MethodGet, "/v1/models"))

	assert.True(t, handled)
	require.Error(t, err)
	var e errcommon.Error
	require.ErrorAs(t, err, &e)
	assert.Equal(t, errcommon.ServiceUnavailable, e.Code)
	// No local response is stored; the state machine sends the error via the standard error path.
	assert.Nil(t, reqCtx.localResp)
	assert.NotEqual(t, RequestAnsweredLocal, reqCtx.RequestState)
}

func TestHandleRequestHeaders_ModelsShortCircuit(t *testing.T) {
	t.Parallel()

	ds := &mockModelsDatastore{
		body:      json.RawMessage(`{"object":"list","data":[{"id":"base"}]}`),
		collected: 1,
	}
	// A director is wired so a reordering regression (the EndOfStream fallback running before the
	// models short-circuit) routes to a random endpoint and fails these assertions cleanly, rather
	// than nil-panicking on s.director.
	s := &StreamingServer{datastore: ds, director: &mockDirectorRequest{}}
	reqCtx := &RequestContext{
		Request:  &Request{Headers: make(map[string]string)},
		Response: &Response{Headers: make(map[string]string)},
	}

	// modelsHeaderRequest sets EndOfStream: a bodyless GET would otherwise hit the
	// fallback-to-random-endpoint branch, so this guards that the short-circuit runs first.
	err := s.HandleRequestHeaders(context.Background(), reqCtx, modelsHeaderRequest(http.MethodGet, "/v1/models"))
	require.NoError(t, err)

	assert.Equal(t, RequestAnsweredLocal, reqCtx.RequestState)
	require.NotNil(t, reqCtx.localResp)
	// Not routed: no target endpoint chosen and no routing header response built.
	assert.Empty(t, reqCtx.TargetEndpoint)
	assert.Nil(t, reqCtx.reqHeaderResp)
}

func TestUpdateStateAndSendIfNeeded_AnsweredLocally(t *testing.T) {
	t.Parallel()

	srv := &mockProcessServer{}
	logger := logr.Discard()

	want := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_OK},
				Body:   []byte(`{"object":"list","data":[]}`),
			},
		},
	}
	reqCtx := &RequestContext{
		RequestState: RequestAnsweredLocal,
		localResp:    want,
	}

	err := reqCtx.updateStateAndSendIfNeeded(srv, logger)
	require.NoError(t, err)

	require.Len(t, srv.sentResponses, 1)
	assert.Same(t, want, srv.sentResponses[0])
}
