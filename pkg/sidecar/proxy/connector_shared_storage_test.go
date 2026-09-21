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
	"encoding/json"
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

// requireStatefulResponsesFieldsStripped asserts that body carries none of
// the fields reqcommon.DropStatefulResponsesFields removes, and that store
// was forced to false rather than left absent.
func requireStatefulResponsesFieldsStripped(t *testing.T, body map[string]any) {
	t.Helper()
	require.NotNil(t, body)
	for _, field := range []string{reqcommon.FieldPreviousResponseID, reqcommon.FieldConversation, reqcommon.FieldBackground} {
		_, ok := body[field]
		require.Falsef(t, ok, "expected %q to be dropped, got %v", field, body[field])
	}
	require.Equal(t, false, body[reqcommon.FieldStore])
}

// TestSharedStorage_StripsStatefulResponsesFieldsFromPrefillAndDecode covers
// handleSharedStorage's default path (no cache_hit_threshold): prefill, then
// decode. Both requests are built from the same body readJSONBody already
// stripped, so this guards that shared entry point against a regression that
// would otherwise only be caught for the nixlv2 connector.
func TestSharedStorage_StripsStatefulResponsesFieldsFromPrefillAndDecode(t *testing.T) {
	var prefillBody map[string]any
	prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&prefillBody)
		w.WriteHeader(http.StatusOK)
	}))
	defer prefill.Close()

	var decodeBody map[string]any
	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorSharedStorage})
	srv.logger = log.Log
	srv.allowlistValidator = &AllowlistValidator{}
	srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&decodeBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop"}]}`))
	})

	body := `{"model":"m","input":"hi","previous_response_id":"resp-123","conversation":"conv-123","store":true,"background":true}`
	req := httptest.NewRequest(http.MethodPost, reqcommon.PathResponses, strings.NewReader(body))
	req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
	recorder := httptest.NewRecorder()
	srv.disaggregatedPrefillHandler(reqcommon.APITypeResponses)(recorder, req)
	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())

	requireStatefulResponsesFieldsStripped(t, prefillBody)
	requireStatefulResponsesFieldsStripped(t, decodeBody)
}

// TestSharedStorage_DecodeFirstAttempt_StripsStatefulResponsesFields targets
// the decode-first attempt handleSharedStorage takes when cache_hit_threshold
// is present: that request is built via cloneRequestWithBody from
// readJSONBody's raw bytes, not from the parsed body map, making it the one
// path where a stateful field could leak back in if raw ever stopped being
// re-marshaled after stripping.
func TestSharedStorage_DecodeFirstAttempt_StripsStatefulResponsesFields(t *testing.T) {
	var decodeBody map[string]any
	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)
	srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: KVConnectorSharedStorage})
	srv.logger = log.Log
	srv.allowlistValidator = &AllowlistValidator{}
	srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&decodeBody)
		w.Header().Set("Content-Type", "application/json")
		// No cache_threshold finish_reason, so decode succeeds without ever
		// falling back to prefill.
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop"}]}`))
	})

	body := `{"model":"m","input":"hi","cache_hit_threshold":0.5,"previous_response_id":"resp-123","conversation":"conv-123","store":true,"background":true}`
	req := httptest.NewRequest(http.MethodPost, reqcommon.PathResponses, strings.NewReader(body))
	// Never dialed: decode succeeds on the first attempt, so handleSharedStorage
	// returns before this host would be used.
	req.Header.Set(routing.PrefillEndpointHeader, "unused-prefill-host:9999")
	recorder := httptest.NewRecorder()
	srv.disaggregatedPrefillHandler(reqcommon.APITypeResponses)(recorder, req)
	require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())

	requireStatefulResponsesFieldsStripped(t, decodeBody)
}
