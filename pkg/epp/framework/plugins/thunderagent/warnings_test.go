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

package thunderagent

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

// A prefill pod logs a prefill/decode disaggregation warning once per live pod,
// and departing or stale prefill pods do not leak entries in prefillSeen.
func TestPrefillPodWarningLoggedOncePerPod(t *testing.T) {
	var lines []string
	ctx := log.IntoContext(context.Background(), funcr.New(func(_, args string) {
		lines = append(lines, args)
	}, funcr.Options{Verbosity: logutil.DEFAULT}))
	l := newTestLedger(testConfig())
	a := l.agent
	prefill := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     types.NamespacedName{Namespace: "default", Name: "pod-p"},
		Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
	}, nil)

	for range 3 {
		a.Saturation(ctx, []fwkdl.Endpoint{prefill})
	}
	all := strings.Join(lines, "\n")
	require.Equal(t, 1, strings.Count(all, "prefill/decode"))

	// Dropping pod-p from an unstaged call prunes prefillSeen.
	a.Saturation(ctx, nil)
	a.mu.Lock()
	require.Empty(t, a.prefillSeen)
	a.mu.Unlock()

	// Expiring a staged prefill pod via sweep also prunes prefillSeen.
	prefillCtx := fwkfc.WithSaturationStage(ctx, fwkfc.SaturationStagePrefill)
	a.Saturation(prefillCtx, []fwkdl.Endpoint{prefill})
	l.step(endpointStaleAfter + time.Second)
	a.sweep()
	a.mu.Lock()
	require.Empty(t, a.prefillSeen)
	a.mu.Unlock()
}
