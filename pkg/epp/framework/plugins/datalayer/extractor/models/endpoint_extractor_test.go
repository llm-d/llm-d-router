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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
)

// modelsServer starts a stub /v1/models server returning body with status code,
// and returns an endpoint whose metrics host points at it.
func modelsServer(t *testing.T, status int, body string) (fwkdl.Endpoint, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	// MetricsHost is host:port without scheme; the fetcher prepends the scheme.
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Name: "pod1"},
		MetricsHost:    strings.TrimPrefix(srv.URL, "http://"),
	}, fwkdl.NewMetrics())
	return ep, srv.Close
}

// readModels returns the /v1/models attribute on ep, or nil when unset.
func readModels(ep fwkdl.Endpoint) attrmodels.ModelDataCollection {
	val, ok := ep.GetAttributes().Get(attrmodels.ModelsAttributeKey.String())
	if !ok {
		return nil
	}
	models, _ := val.(attrmodels.ModelDataCollection)
	return models
}

// newTestExtractor builds an http extractor pointing at an insecure model server.
func newTestExtractor(t *testing.T) *ModelEndpointExtractor {
	t.Helper()
	me, err := NewModelEndpointExtractor("test", "http", "/v1/models", true)
	require.NoError(t, err)
	return me
}

// TestFetchAndStoreRecordsModels verifies a successful fetch records every model
// (and its max_model_len) on the endpoint attribute.
func TestFetchAndStoreRecordsModels(t *testing.T) {
	ep, closeSrv := modelsServer(t, http.StatusOK,
		`{"object":"list","data":[{"id":"m1","max_model_len":131072},{"id":"m2","max_model_len":65536}]}`)
	defer closeSrv()

	newTestExtractor(t).fetchAndStore(context.Background(), ep)

	models := readModels(ep)
	require.Len(t, models, 2)
	assert.Equal(t, "m1", models[0].ID)
	assert.Equal(t, 131072, models[0].MaxModelLen)
	assert.Equal(t, "m2", models[1].ID)
	assert.Equal(t, 65536, models[1].MaxModelLen)
}

// TestFetchAndStoreSwallowsError verifies a failing model server leaves the
// attribute unset rather than recording anything, so the consumer falls back.
func TestFetchAndStoreSwallowsError(t *testing.T) {
	ep, closeSrv := modelsServer(t, http.StatusInternalServerError, "")
	defer closeSrv()

	newTestExtractor(t).fetchAndStore(context.Background(), ep)
	assert.Nil(t, readModels(ep))
}

// TestExtractFetchesOnAdd verifies an add event triggers a background fetch that
// eventually records the model list.
func TestExtractFetchesOnAdd(t *testing.T) {
	ep, closeSrv := modelsServer(t, http.StatusOK,
		`{"object":"list","data":[{"id":"m","max_model_len":131072}]}`)
	defer closeSrv()

	require.NoError(t, newTestExtractor(t).Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))

	// The fetch runs in the background, so wait for the attribute to appear.
	assert.Eventually(t, func() bool {
		m := readModels(ep)
		return len(m) == 1 && m[0].MaxModelLen == 131072
	}, time.Second, 5*time.Millisecond)
}

// TestExtractSkipsWhenAlreadyPresent verifies a second add/update does not
// re-fetch once the attribute is set. The server would error, proving no fetch.
func TestExtractSkipsWhenAlreadyPresent(t *testing.T) {
	ep, closeSrv := modelsServer(t, http.StatusInternalServerError, "")
	defer closeSrv()
	existing := attrmodels.ModelDataCollection{{ID: "m", MaxModelLen: 42}}
	ep.GetAttributes().Put(attrmodels.ModelsAttributeKey.String(), existing)

	require.NoError(t, newTestExtractor(t).Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))

	// No goroutine is spawned, so the pre-existing value must remain untouched.
	assert.Never(t, func() bool {
		m := readModels(ep)
		return len(m) != 1 || m[0].MaxModelLen != 42
	}, 100*time.Millisecond, 10*time.Millisecond)
}

// TestExtractIgnoresDelete verifies a delete event neither fetches nor writes the
// attribute; the attribute leaves with the endpoint.
func TestExtractIgnoresDelete(t *testing.T) {
	// A failing server proves no fetch is attempted on delete.
	ep, closeSrv := modelsServer(t, http.StatusInternalServerError, "")
	defer closeSrv()

	require.NoError(t, newTestExtractor(t).Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: ep}))
	assert.Nil(t, readModels(ep))
}

// TestExtractNilEndpoint verifies missing endpoint or metadata is a no-op and
// does not panic.
func TestExtractNilEndpoint(t *testing.T) {
	me := newTestExtractor(t)
	assert.NoError(t, me.Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: nil}))
	assert.NoError(t, me.Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: fwkdl.NewEndpoint(nil, nil)}))
}

// fakeRegistrar captures the registration a plugin submits.
type fakeRegistrar struct {
	reg fwkdl.PendingRegistration
}

func (f *fakeRegistrar) Register(reg fwkdl.PendingRegistration) error {
	f.reg = reg
	return nil
}

// TestRegisterDependencies verifies the extractor self-wires to an
// endpoint-notification-source with a default source to auto-create.
func TestRegisterDependencies(t *testing.T) {
	me := newTestExtractor(t)
	r := &fakeRegistrar{}
	require.NoError(t, me.RegisterDependencies(r))
	assert.Equal(t, "endpoint-notification-source", r.reg.SourceType)
	assert.Equal(t, fwkplugin.Plugin(me), r.reg.Extractor)
	assert.NotNil(t, r.reg.DefaultSource)
}

// TestProduces verifies the extractor declares the /v1/models attribute so
// consumers of that key are wired to it.
func TestProduces(t *testing.T) {
	produced := newTestExtractor(t).Produces()
	_, ok := produced[attrmodels.ModelsAttributeKey]
	assert.True(t, ok, "must produce the models attribute")
}

// TestFactory verifies the factory builds with defaults, honours custom
// parameters, and rejects an unsupported scheme.
func TestFactory(t *testing.T) {
	// Defaults.
	p, err := ModelEndpointExtractorFactory("models", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, ModelsEndpointExtractorType, p.TypedName().Type)

	// Custom https params are accepted.
	custom := fwkplugin.StrictDecoder([]byte(`{"scheme":"https","path":"/models","insecureSkipVerify":false}`))
	_, err = ModelEndpointExtractorFactory("models", custom, nil)
	require.NoError(t, err)

	// An unsupported scheme is rejected.
	bad := fwkplugin.StrictDecoder([]byte(`{"scheme":"ftp"}`))
	_, err = ModelEndpointExtractorFactory("models", bad, nil)
	assert.Error(t, err)
}
