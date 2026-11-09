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
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

// TestStampFailedStage covers the response header that names the stage whose
// failure produced the error response.
func TestStampFailedStage(t *testing.T) {
	respond := func(markStage bool, preset string, writeStatus bool) *httptest.ResponseRecorder {
		ctx, summary := withRequestSummary(context.Background())
		recorder := httptest.NewRecorder()
		if preset != "" {
			recorder.Header().Set(routing.FailedStageHeader, preset)
		}
		w := stampFailedStage(recorder, summary)
		if markStage {
			observeStageError(ctx, routing.StagePrefill)
		}
		if writeStatus {
			w.WriteHeader(http.StatusServiceUnavailable)
		} else {
			_, err := w.Write([]byte("body"))
			assert.NoError(t, err)
		}
		return recorder
	}

	assert.Equal(t, routing.StagePrefill,
		respond(true, "", true).Header().Get(routing.FailedStageHeader),
		"a response written after a stage failed carries the stage")

	assert.Equal(t, routing.StagePrefill,
		respond(true, "", false).Header().Get(routing.FailedStageHeader),
		"a response written without a status carries the stage too")

	assert.Equal(t, routing.StageNone,
		respond(false, "", true).Header().Get(routing.FailedStageHeader),
		"a response written before any stage failed carries none")

	assert.Equal(t, routing.StageNone,
		respond(false, routing.StageDecode, true).Header().Get(routing.FailedStageHeader),
		"a value set before the sidecar wrote the response is dropped")
}

// TestStampFailedStageThroughHandler drives the shared-storage connector
// through disaggregatedPrefillHandler, so the stamp reflects a real connector's
// ordering: a stage is marked before the error response is written.
func TestStampFailedStageThroughHandler(t *testing.T) {
	decodeURL, err := url.Parse("http://decoder:8000")
	require.NoError(t, err)

	t.Run("prefill failure", func(t *testing.T) {
		prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer prefill.Close()

		srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
		srv.logger = log.Log
		srv.allowlistValidator = &AllowlistValidator{}

		req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions,
			strings.NewReader(`{"model":"m","messages":[]}`))
		req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
		recorder := httptest.NewRecorder()
		srv.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(recorder, req)

		require.Equal(t, http.StatusServiceUnavailable, recorder.Code)
		assert.Equal(t, routing.StagePrefill, recorder.Header().Get(routing.FailedStageHeader))
	})

	t.Run("decode-first success drops a model-server value", func(t *testing.T) {
		prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer prefill.Close()

		srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, KVConnector: constants.KVConnectorSharedStorage})
		srv.logger = log.Log
		srv.allowlistValidator = &AllowlistValidator{}
		srv.decoderProxy = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			// A value the decoder set must not survive the sidecar's stamp.
			w.Header().Set(routing.FailedStageHeader, routing.StagePrefill)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop"}]}`))
		})

		req := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions,
			strings.NewReader(`{"model":"m","messages":[],"cache_hit_threshold":0.5}`))
		req.Header.Set(routing.PrefillEndpointHeader, strings.TrimPrefix(prefill.URL, "http://"))
		recorder := httptest.NewRecorder()
		srv.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(recorder, req)

		require.Equal(t, http.StatusOK, recorder.Code)
		assert.Equal(t, routing.StageNone, recorder.Header().Get(routing.FailedStageHeader))
	})
}
