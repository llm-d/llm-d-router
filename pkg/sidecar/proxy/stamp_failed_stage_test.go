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
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// TestStampFailedStage covers the response header that names the stage whose
// failure produced the error response.
func TestStampFailedStage(t *testing.T) {
	respond := func(markStage bool, preset string) *httptest.ResponseRecorder {
		ctx, summary := withRequestSummary(context.Background())
		recorder := httptest.NewRecorder()
		if preset != "" {
			recorder.Header().Set(routing.FailedStageHeader, preset)
		}
		w := stampFailedStage(recorder, summary)
		if markStage {
			observeStageError(ctx, routing.StagePrefill)
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		return recorder
	}

	assert.Equal(t, routing.StagePrefill,
		respond(true, "").Header().Get(routing.FailedStageHeader),
		"a response written after a stage failed carries the stage")

	assert.Empty(t, respond(false, "").Header().Get(routing.FailedStageHeader),
		"a response written before any stage failed carries no stage")

	assert.Empty(t, respond(false, routing.StageDecode).Header().Get(routing.FailedStageHeader),
		"a value set before the sidecar wrote the response is dropped")
}
