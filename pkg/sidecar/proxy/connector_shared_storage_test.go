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

package proxy

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// statefulResponsesTestBody is a /v1/responses body carrying the fields
// reqcommon.RejectStatefulResponsesFields refuses, shared by the tests that
// assert such a request is refused before it reaches any upstream.
const statefulResponsesTestBody = `{"model":"m","input":"hi","previous_response_id":"resp-123","conversation":"conv-123","background":true}`

// requireStatefulResponsesRejected asserts the handler answered 400 naming the
// offending field and dispatched nothing upstream. previous_response_id is the
// first field RejectStatefulResponsesFields checks, so it is the one named for
// statefulResponsesTestBody.
func requireStatefulResponsesRejected(t *testing.T, recorder *httptest.ResponseRecorder, dispatched bool) {
	t.Helper()
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	require.Contains(t, recorder.Body.String(), reqcommon.FieldPreviousResponseID)
	require.False(t, dispatched, "request reached an upstream despite an unsupported field")
}

// TestSharedStorage_RejectsStatefulResponsesFields covers handleSharedStorage's
// default path (no cache_hit_threshold): the request is refused in readJSONBody,
// so neither the prefill nor the decode upstream is ever dispatched.
func TestSharedStorage_RejectsStatefulResponsesFields(t *testing.T) {
	var dispatched bool
	prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatched = true
		w.WriteHeader(http.StatusOK)
	}))
	defer prefill.Close()

	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorSharedStorage})
	srv.logger = log.Log
	srv.allowlistValidator = &AllowlistValidator{}
	srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		dispatched = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop"}]}`))
	})

	req := httptest.NewRequest(http.MethodPost, reqcommon.PathResponses, strings.NewReader(statefulResponsesTestBody))
	req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
	recorder := httptest.NewRecorder()
	srv.disaggregatedPrefillHandler(reqcommon.APITypeResponses)(recorder, req)

	requireStatefulResponsesRejected(t, recorder, dispatched)
}
