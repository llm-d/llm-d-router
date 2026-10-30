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

package asyncbroker

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// Enqueue mode sets no max_seconds, so a client timeout above
// math.MaxInt64/1e9 would overflow time.Duration and produce a deadline the
// producer refuses. The value is clamped so the request enqueues (202) instead
// of failing as a broker transport error.
func TestServeQueued_LargeTimeoutDoesNotOverflow(t *testing.T) {
	step, _ := newAsyncTestStep(t, nil)
	reqCtx, rec := asyncReqCtx(t, `{"model":"test-model"}`, map[string]string{
		defaultModeHeader:    "enqueue",
		defaultTenantHeader:  "team-a",
		defaultTimeoutHeader: "9223372037",
	})

	err := step.Execute(log.IntoContext(t.Context(), log.Log), reqCtx)
	if !errors.Is(err, pipeline.ErrPipelineDone) {
		t.Fatalf("want ErrPipelineDone for a clamped timeout, got %v", err)
	}
	require.Equal(t, http.StatusAccepted, rec.Code, rec.Body.String())
}
