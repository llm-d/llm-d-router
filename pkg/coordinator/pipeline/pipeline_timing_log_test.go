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

package pipeline

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// The summary log line pairs keys and values, so every duration needs its step
// name in front of it. Without the name the durations shift into key
// positions, which makes the per-step timings unreadable and leaves an odd
// number of arguments.
func TestExecute_StepTimingsLogNamesEachDuration(t *testing.T) {
	var logged []string
	logger := funcr.New(func(_, args string) {
		logged = append(logged, args)
	}, funcr.Options{Verbosity: logutil.DEFAULT})
	ctx := log.IntoContext(context.Background(), logger)

	slowStep := func(_ context.Context, _ *RequestContext) error {
		time.Sleep(time.Millisecond)
		return nil
	}
	steps := []Step{
		&mockStep{name: "render", fn: slowStep},
		&mockStep{name: "decode", fn: slowStep},
	}

	require.NoError(t, New(steps).Execute(ctx, &RequestContext{}))

	var timings string
	for _, rec := range logged {
		if strings.Contains(rec, "pipeline step timings") {
			timings = rec
		}
	}
	require.NotEmpty(t, timings, "expected a pipeline step timings record, got %v", logged)
	require.Contains(t, timings, `"render"=`, "render duration must be keyed by its step name")
	require.Contains(t, timings, `"decode"=`, "decode duration must be keyed by its step name")
}
