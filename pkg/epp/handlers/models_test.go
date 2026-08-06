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
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
)

// mockModelsDatastore is a minimal Datastore returning a fixed endpoint set. The embedded
// Datastore satisfies the interface; only PodList, the method the models path uses, is implemented.
type mockModelsDatastore struct {
	Datastore
	endpoints []fwkdl.Endpoint
}

func (m *mockModelsDatastore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	out := make([]fwkdl.Endpoint, 0, len(m.endpoints))
	for _, ep := range m.endpoints {
		if predicate(ep) {
			out = append(out, ep)
		}
	}
	return out
}

// endpointWithModels builds a ready endpoint carrying the given models as its collected attribute.
func endpointWithModels(models ...attrmodels.ModelData) fwkdl.Endpoint {
	ep := fwkdl.NewEndpoint(nil, nil)
	ep.GetAttributes().Put(attrmodels.ModelsAttributeKey.String(), attrmodels.ModelDataCollection(models))
	return ep
}

// endpointWithIDAndModels is endpointWithModels with an explicit endpoint identity, used to assert
// the identity-ordered dedup winner when the same model ID is reported by multiple endpoints.
func endpointWithIDAndModels(id string, models ...attrmodels.ModelData) fwkdl.Endpoint {
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: id}}, nil)
	ep.GetAttributes().Put(attrmodels.ModelsAttributeKey.String(), attrmodels.ModelDataCollection(models))
	return ep
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

func modelIDs(data []attrmodels.ModelData) []string {
	ids := make([]string, 0, len(data))
	for _, m := range data {
		ids = append(ids, m.ID)
	}
	return ids
}

func TestAggregateModels(t *testing.T) {
	t.Parallel()

	ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
		endpointWithModels(
			attrmodels.ModelData{ID: "base"},
			attrmodels.ModelData{ID: "legal", Parent: "base"},
		),
		endpointWithModels(
			attrmodels.ModelData{ID: "base"},
			attrmodels.ModelData{ID: "finance", Parent: "base"},
		),
		endpointWithModels(),
	}}
	s := &StreamingServer{datastore: ds}

	got, gotCollected := s.aggregateModels()

	// Three endpoints are registered, but the third reported no model list, so only two count as
	// scraped.
	assert.Equal(t, 2, gotCollected)
	assert.Equal(t, "list", got.Object)
	// union across endpoints, deduplicated by ID, sorted for stable output.
	assert.Equal(t, []string{"base", "finance", "legal"}, modelIDs(got.Data))
	for _, m := range got.Data {
		if m.ID == "legal" || m.ID == "finance" {
			assert.Equal(t, "base", m.Parent, "adapter %q should carry its parent", m.ID)
		}
	}
}

func TestAggregateModels_PreservesOpenAIFields(t *testing.T) {
	t.Parallel()

	ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
		endpointWithModels(attrmodels.ModelData{
			ID:      "base",
			Object:  "model",
			Created: 1699999999,
			OwnedBy: "vllm",
		}),
	}}
	s := &StreamingServer{datastore: ds}

	got, _ := s.aggregateModels()

	require.Len(t, got.Data, 1)
	assert.Equal(t, attrmodels.ModelData{
		ID:      "base",
		Object:  "model",
		Created: 1699999999,
		OwnedBy: "vllm",
	}, got.Data[0])
}

func TestAggregateModels_DeterministicDedupWinner(t *testing.T) {
	t.Parallel()

	// The same model ID is reported by two endpoints with differing metadata. PodList order is
	// unstable, so aggregateModels sorts by endpoint identity and keeps the lower-ID endpoint's
	// entry; the higher-ID endpoint is listed first here to prove ordering, not slice position, wins.
	ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
		endpointWithIDAndModels("ep-b", attrmodels.ModelData{ID: "shared", OwnedBy: "from-b", Created: 200}),
		endpointWithIDAndModels("ep-a", attrmodels.ModelData{ID: "shared", OwnedBy: "from-a", Created: 100}),
	}}
	s := &StreamingServer{datastore: ds}

	got, _ := s.aggregateModels()

	require.Len(t, got.Data, 1)
	assert.Equal(t, attrmodels.ModelData{ID: "shared", OwnedBy: "from-a", Created: 100}, got.Data[0])
}

func TestAggregateModels_NoEndpoints(t *testing.T) {
	t.Parallel()

	s := &StreamingServer{datastore: &mockModelsDatastore{}}

	got, gotEndpoints := s.aggregateModels()

	assert.Equal(t, 0, gotEndpoints)
	assert.Equal(t, "list", got.Object)
	assert.Empty(t, got.Data)
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

			ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
				endpointWithModels(attrmodels.ModelData{ID: "base"}),
			}}
			s := &StreamingServer{datastore: ds}
			reqCtx := &RequestContext{Request: &Request{Headers: make(map[string]string)}}

			handled, err := s.tryServeModelList(context.Background(), reqCtx, modelsHeaderRequest(tc.method, tc.path))
			require.NoError(t, err)

			assert.Equal(t, tc.wantHandled, handled)
			if !tc.wantHandled {
				assert.Nil(t, reqCtx.localResp)
				assert.NotEqual(t, RequestAnsweredLocal, reqCtx.RequestState)
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

			var resp extmodels.ModelResponse
			require.NoError(t, json.Unmarshal(ir.Body, &resp))
			assert.Equal(t, "list", resp.Object)
			assert.Equal(t, []string{"base"}, modelIDs(resp.Data))
		})
	}
}

func TestTryServeModelList_ResponseUsesOpenAIJSONKeys(t *testing.T) {
	t.Parallel()

	ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
		endpointWithModels(attrmodels.ModelData{
			ID:      "base",
			Object:  "model",
			Created: 1699999999,
			OwnedBy: "vllm",
			Parent:  "root",
		}),
	}}
	s := &StreamingServer{datastore: ds}
	reqCtx := &RequestContext{Request: &Request{Headers: make(map[string]string)}}

	handled, err := s.tryServeModelList(context.Background(), reqCtx, modelsHeaderRequest(http.MethodGet, "/v1/models"))
	require.NoError(t, err)
	require.True(t, handled)
	require.NotNil(t, reqCtx.localResp)

	// Assert on the raw JSON keys rather than round-tripping through the same struct tags, so a tag
	// typo (e.g. "ownedby") that still round-trips cleanly is caught. OpenAI compatibility is the point.
	var raw map[string]any
	require.NoError(t, json.Unmarshal(reqCtx.localResp.GetImmediateResponse().Body, &raw))
	assert.Equal(t, "list", raw["object"])

	data, ok := raw["data"].([]any)
	require.True(t, ok, "data must be a JSON array")
	require.Len(t, data, 1)
	model, ok := data[0].(map[string]any)
	require.True(t, ok, "data entry must be a JSON object")

	assert.Equal(t, "base", model["id"])
	assert.Equal(t, "model", model["object"])
	assert.EqualValues(t, 1699999999, model["created"])
	assert.Equal(t, "vllm", model["owned_by"])
	assert.Equal(t, "root", model["parent"])
}

func TestTryServeModelList_NotScrapedYetReturns503(t *testing.T) {
	t.Parallel()

	// Endpoints are registered but none have been scraped yet (cold start / scale-up window, since
	// scraping is interval-based). With no model list collected from any endpoint, this must be a
	// 503 so the client retries, not a misleading empty 200.
	ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
		fwkdl.NewEndpoint(nil, nil), // ready endpoint with no ModelsAttributeKey collected yet
		fwkdl.NewEndpoint(nil, nil), // another endpoint, also not scraped yet
	}}
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

	ds := &mockModelsDatastore{endpoints: []fwkdl.Endpoint{
		endpointWithModels(attrmodels.ModelData{ID: "base"}),
	}}
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
