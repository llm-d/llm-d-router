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

package models

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	srcmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/models"
)

// endpointWithModels builds a scraped endpoint carrying the given models as its collected attribute.
func endpointWithModels(models ...attrmodels.ModelData) fwkdl.Endpoint {
	ep := fwkdl.NewEndpoint(nil, nil)
	ep.GetAttributes().Put(attrmodels.ModelsAttributeKey, attrmodels.ModelDataCollection(models))
	return ep
}

// endpointWithIDAndModels is endpointWithModels with an explicit endpoint identity, used to assert
// the identity-ordered dedup winner when the same model ID is reported by multiple endpoints.
func endpointWithIDAndModels(id string, models ...attrmodels.ModelData) fwkdl.Endpoint {
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: id}}, nil)
	ep.GetAttributes().Put(attrmodels.ModelsAttributeKey, attrmodels.ModelDataCollection(models))
	return ep
}

func modelIDs(data []attrmodels.ModelData) []string {
	ids := make([]string, 0, len(data))
	for _, m := range data {
		ids = append(ids, m.ID)
	}
	return ids
}

func unmarshalModelResponse(t *testing.T, body json.RawMessage) attrmodels.ModelResponse {
	t.Helper()
	var resp attrmodels.ModelResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	return resp
}

func TestAggregate_DuplicateModelFromMultipleEndpoints(t *testing.T) {
	t.Parallel()

	endpoints := []fwkdl.Endpoint{
		endpointWithModels(
			attrmodels.ModelData{ID: "base"},
			attrmodels.ModelData{ID: "legal", Parent: "base"},
		),
		endpointWithModels(
			attrmodels.ModelData{ID: "base"},
			attrmodels.ModelData{ID: "finance", Parent: "base"},
		),
		fwkdl.NewEndpoint(nil, nil), // registered but not yet scraped
	}

	body, gotCollected := aggregate(endpoints)

	// Two of the three endpoints have been scraped; the unscraped one does not count.
	assert.Equal(t, 2, gotCollected)
	resp := unmarshalModelResponse(t, body)
	assert.Equal(t, "list", resp.Object)
	// Union across endpoints, deduplicated by ID, sorted for stable output.
	assert.Equal(t, []string{"base", "finance", "legal"}, modelIDs(resp.Data))
	for _, m := range resp.Data {
		if m.ID == "legal" || m.ID == "finance" {
			assert.Equal(t, "base", m.Parent, "adapter %q should carry its parent", m.ID)
		}
	}
}

func TestAggregate_DuplicateModelOnSameEndpoint(t *testing.T) {
	t.Parallel()

	// A single endpoint reports the same model ID twice; only the first occurrence is kept.
	endpoints := []fwkdl.Endpoint{
		endpointWithModels(
			attrmodels.ModelData{ID: "base", OwnedBy: "second"},
			attrmodels.ModelData{ID: "base", OwnedBy: "first"},
			attrmodels.ModelData{ID: "unique"},
		),
	}

	body, gotCollected := aggregate(endpoints)

	assert.Equal(t, 1, gotCollected)
	resp := unmarshalModelResponse(t, body)
	assert.Equal(t, []string{"base", "unique"}, modelIDs(resp.Data))
	assert.Equal(t, "second", resp.Data[0].OwnedBy)
}

func TestAggregate_PreservesOpenAIFields(t *testing.T) {
	t.Parallel()

	endpoints := []fwkdl.Endpoint{
		endpointWithModels(attrmodels.ModelData{
			ID:      "base",
			Object:  "model",
			Created: 1699999999,
			OwnedBy: "vllm",
		}),
	}

	body, _ := aggregate(endpoints)

	resp := unmarshalModelResponse(t, body)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, attrmodels.ModelData{
		ID:      "base",
		Object:  "model",
		Created: 1699999999,
		OwnedBy: "vllm",
	}, resp.Data[0])
}

func TestAggregate_DeterministicDedupWinner(t *testing.T) {
	t.Parallel()

	// The same model ID is reported by two endpoints with differing metadata. The lower-ID
	// endpoint's entry wins because aggregate sorts by endpoint identity before deduping;
	// ep-b is listed first here to prove ordering, not slice position, determines the winner.
	endpoints := []fwkdl.Endpoint{
		endpointWithIDAndModels("ep-b", attrmodels.ModelData{ID: "shared", OwnedBy: "from-b", Created: 200}),
		endpointWithIDAndModels("ep-a", attrmodels.ModelData{ID: "shared", OwnedBy: "from-a", Created: 100}),
	}

	body, _ := aggregate(endpoints)

	resp := unmarshalModelResponse(t, body)
	require.Len(t, resp.Data, 1)
	assert.Equal(t, attrmodels.ModelData{ID: "shared", OwnedBy: "from-a", Created: 100}, resp.Data[0])
}

func TestAggregate_DoesNotReorderCallerSlice(t *testing.T) {
	t.Parallel()

	// The caller owns the slice it passes; aggregate sorts a copy.
	first := endpointWithIDAndModels("ep-b", attrmodels.ModelData{ID: "b"})
	endpoints := []fwkdl.Endpoint{first, endpointWithIDAndModels("ep-a", attrmodels.ModelData{ID: "a"})}

	aggregate(endpoints)

	assert.Same(t, first, endpoints[0])
}

func TestAggregate_NoEndpoints(t *testing.T) {
	t.Parallel()

	body, gotCollected := aggregate(nil)

	assert.Equal(t, 0, gotCollected)
	assert.Nil(t, body)
}

func TestAggregate_NotScrapedYet(t *testing.T) {
	t.Parallel()

	// Endpoints are registered but none have been scraped yet (no attribute key present).
	endpoints := []fwkdl.Endpoint{
		fwkdl.NewEndpoint(nil, nil),
		fwkdl.NewEndpoint(nil, nil),
	}

	_, gotCollected := aggregate(endpoints)

	assert.Equal(t, 0, gotCollected)
}

func TestRespond_AnswersModelList(t *testing.T) {
	t.Parallel()

	endpoints := []fwkdl.Endpoint{endpointWithModels(attrmodels.ModelData{ID: "base"})}

	resp, err := New().Respond(context.Background(),
		&fwkrc.RequestLine{Method: http.MethodGet, Path: "/v1/models"}, endpoints)

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "application/json", resp.Headers["content-type"])
	assert.Equal(t, []string{"base"}, modelIDs(unmarshalModelResponse(t, resp.Body).Data))
}

func TestRespond_TrailingSlash(t *testing.T) {
	t.Parallel()

	endpoints := []fwkdl.Endpoint{endpointWithModels(attrmodels.ModelData{ID: "base"})}

	resp, err := New().Respond(context.Background(),
		&fwkrc.RequestLine{Method: http.MethodGet, Path: "/v1/models/"}, endpoints)

	require.NoError(t, err)
	assert.NotNil(t, resp)
}

func TestRespond_Declines(t *testing.T) {
	t.Parallel()

	endpoints := []fwkdl.Endpoint{endpointWithModels(attrmodels.ModelData{ID: "base"})}

	tests := []struct {
		name    string
		request *fwkrc.RequestLine
	}{
		{"wrong method", &fwkrc.RequestLine{Method: http.MethodPost, Path: "/v1/models"}},
		{"wrong path", &fwkrc.RequestLine{Method: http.MethodGet, Path: "/v1/chat/completions"}},
		{"path prefix only", &fwkrc.RequestLine{Method: http.MethodGet, Path: "/v1/models/base"}},
		{"nil request", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			resp, err := New().Respond(context.Background(), tt.request, endpoints)
			require.NoError(t, err)
			assert.Nil(t, resp, "responder must decline so the request is routed normally")
		})
	}
}

func TestRespond_UnavailableBeforeFirstScrape(t *testing.T) {
	t.Parallel()

	// Registered but unscraped endpoints must not produce an empty 200.
	endpoints := []fwkdl.Endpoint{fwkdl.NewEndpoint(nil, nil)}

	resp, err := New().Respond(context.Background(),
		&fwkrc.RequestLine{Method: http.MethodGet, Path: "/v1/models"}, endpoints)

	assert.Nil(t, resp)
	require.Error(t, err)
	assert.Equal(t, errcommon.ServiceUnavailable, errcommon.CanonicalCode(err))
}

func TestConsumesModelAttributeOptionally(t *testing.T) {
	t.Parallel()

	// Optional, not required: RegisterDependencies supplies the producer, and required keys
	// are resolved before plugins register their dependencies.
	deps := New().Consumes()

	require.Contains(t, deps.Optional, attrmodels.ModelsAttributeKey)
	assert.Empty(t, deps.Required)
}

func TestRegisterDependenciesCreatesModelsSource(t *testing.T) {
	t.Parallel()

	var recorder fakeRegistrar
	require.NoError(t, New().RegisterDependencies(&recorder))

	require.Len(t, recorder.registered, 1)
	got := recorder.registered[0]
	assert.Equal(t, srcmodels.ModelsDataSourceType, got.SourceType)
	assert.NotNil(t, got.Extractor, "the extractor must be supplied, not left to config")
	assert.NotNil(t, got.DefaultSource, "the source must be auto-created when absent")
}

type fakeRegistrar struct {
	registered []fwkdl.PendingRegistration
}

func (f *fakeRegistrar) Register(reg fwkdl.PendingRegistration) error {
	f.registered = append(f.registered, reg)
	return nil
}
