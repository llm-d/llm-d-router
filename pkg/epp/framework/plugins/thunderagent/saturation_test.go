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
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

// When a pod stops being reported in an unstaged cycle, it drops immediately
// and its sessions re-enter as new.
func TestPodVanishUnbinds(t *testing.T) {
	l := newTestLedger(testConfig())
	podA := l.pod("pod-a", 0, 0)
	podB := l.pod("pod-b", 0, 0)
	l.seed("s1", "pod-a", 300)
	l.pause("s1")
	l.prime(podA)
	require.True(t, l.isPaused("s1"))

	l.removePod("pod-a")
	l.prime(podB)

	require.Equal(t, float64(-1), l.endpointTokens("pod-a"))
	require.False(t, l.isPaused("s1"))

	// s1 re-enters as new on pod-b without counting a resume.
	q := l.makeQueue("s1", l.now(), 400)
	require.Equal(t, q, pick(t, l.agent, q))
	require.Equal(t, float64(0), testutil.ToFloat64(l.agent.metrics.resumes))

	schedB, schedC := schedEndpoint("pod-b"), schedEndpoint("pod-c")
	scores := l.agent.Score(context.Background(), l.request("s1", 400), []fwksched.Endpoint{schedB, schedC})
	require.Equal(t, 1.0, scores[schedB])
	require.Equal(t, 0.0, scores[schedC])
}

// Staged Saturation calls immediately drop absent endpoints from the same stage
// while retaining endpoints from other stages until endpointStaleAfter elapses.
func TestSaturationStageRetentionAndTimeout(t *testing.T) {
	l := newTestLedger(testConfig())
	prefill1 := l.pod("pod-p1", 0, 0)
	prefill2 := l.pod("pod-p2", 0, 0)
	decode1 := l.pod("pod-d1", 0, 0)

	prefillCtx := fwkfc.WithSaturationStage(context.Background(), fwkfc.SaturationStagePrefill)
	decodeCtx := fwkfc.WithSaturationStage(context.Background(), fwkfc.SaturationStageDecode)

	require.Equal(t, 0.0, l.agent.Saturation(prefillCtx, []fwkdl.Endpoint{prefill1, prefill2}))
	require.Equal(t, 0.0, l.agent.Saturation(decodeCtx, []fwkdl.Endpoint{decode1}))

	// Dropping pod-p2 from prefill removes it immediately; decode1 stays live.
	l.step(time.Second)
	require.Equal(t, 0.0, l.agent.Saturation(prefillCtx, []fwkdl.Endpoint{prefill1}))
	require.GreaterOrEqual(t, l.endpointTokens("pod-p1"), 0.0)
	require.Equal(t, float64(-1), l.endpointTokens("pod-p2"))
	require.GreaterOrEqual(t, l.endpointTokens("pod-d1"), 0.0)

	// Past endpointStaleAfter without a decode refresh, decode1 times out.
	l.step(endpointStaleAfter + time.Second)
	require.Equal(t, 0.0, l.agent.Saturation(prefillCtx, []fwkdl.Endpoint{prefill1}))
	require.GreaterOrEqual(t, l.endpointTokens("pod-p1"), 0.0)
	require.Equal(t, float64(-1), l.endpointTokens("pod-d1"))
}

// Replacing an Endpoint object under the same NamespacedName without an absent
// cycle resets old pod overlays and prefillSeen state while preserving in-place
// updates for identical Endpoint pointers.
func TestSaturationReplacementEndpointResetsOldState(t *testing.T) {
	var lines []string
	ctx := log.IntoContext(context.Background(), funcr.New(func(_, args string) {
		lines = append(lines, args)
	}, funcr.Options{Verbosity: logutil.DEFAULT}))

	l := newTestLedger(testConfig())
	podID := types.NamespacedName{Namespace: "default", Name: "pod-a"}
	ep1 := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     podID,
		Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
	}, &fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 100})

	l.agent.Saturation(ctx, []fwkdl.Endpoint{ep1})
	q := l.makeQueue("s1", l.now(), 800) // 200 reservedUnbound on pod-a
	require.Equal(t, q, pick(t, l.agent, q))
	require.Equal(t, float64(200), l.endpointTokens("pod-a"))

	// Updating the same pointer refreshes metrics/timestamp without wiping overlays.
	ep1.UpdateMetrics(&fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 200})
	l.agent.Saturation(ctx, []fwkdl.Endpoint{ep1})
	snap := l.agent.snapshot()
	require.Equal(t, float64(3200), snap.endpoints["default/pod-a"].capacity)
	require.Equal(t, float64(200), snap.endpoints["default/pod-a"].workingSet)
	require.Len(t, lines, 1)

	// Passing a new Endpoint object under the same NamespacedName drops old overlays and prefillSeen.
	ep2 := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:     podID,
		Labels: map[string]string{bylabel.RoleLabel: bylabel.RolePrefill},
	}, &fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 150})
	l.agent.Saturation(ctx, []fwkdl.Endpoint{ep2})
	snap = l.agent.snapshot()
	require.Equal(t, float64(2400), snap.endpoints["default/pod-a"].capacity)
	require.Equal(t, float64(0), snap.endpoints["default/pod-a"].workingSet)
	require.Len(t, lines, 2)
	l.agent.mu.Lock()
	require.Empty(t, l.agent.local)
	l.agent.mu.Unlock()
}
