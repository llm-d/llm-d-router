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

package maxscore

import (
	"context"
	"sync"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// recordingReserver records ReserveFromEndpoint calls.
type recordingReserver struct {
	mu    sync.Mutex
	calls []string
}

func (r *recordingReserver) ReserveFromEndpoint(ep fwksched.Endpoint) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if ep != nil && ep.GetMetadata() != nil {
		r.calls = append(r.calls, ep.GetMetadata().ID.Name)
	}
}

func (r *recordingReserver) Reserve(string, int64) {}

// TypedName makes the reserver usable as a fwkplugin.Plugin in the handle.
func (r *recordingReserver) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-reserver", Name: "test-reserver"}
}

func (r *recordingReserver) reserved() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.calls...)
}

// lateHandle is a plugin handle whose plugin set grows after construction:
// it models the factory-order case where the EndpointReserver (e.g. the
// inflight-load-producer) is instantiated after this picker.
type lateHandle struct {
	mu      sync.Mutex
	plugins []fwkplugin.Plugin
}

func (h *lateHandle) add(p fwkplugin.Plugin) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.plugins = append(h.plugins, p)
}

func (h *lateHandle) GetAllPlugins() []fwkplugin.Plugin {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]fwkplugin.Plugin(nil), h.plugins...)
}

// The remaining fwkplugin.Handle methods are unused here.
func (h *lateHandle) Context() context.Context                            { return context.Background() }
func (h *lateHandle) Plugin(string) fwkplugin.Plugin                      { return nil }
func (h *lateHandle) AddPlugin(string, fwkplugin.Plugin)                  {}
func (h *lateHandle) GetAllPluginsWithNames() map[string]fwkplugin.Plugin { return nil }
func (h *lateHandle) PodList() []k8stypes.NamespacedName                  { return nil }
func (h *lateHandle) Metrics() fwkplugin.MetricsRecorder                  { return nil }
func (h *lateHandle) CrossReplicaSyncer() fwkplugin.Plugin                { return nil }
func (h *lateHandle) SetCrossReplicaSyncer(fwkplugin.Plugin)              {}
func (h *lateHandle) RefreshMetricsInterval() time.Duration               { return 0 }

func (h *lateHandle) SchedulingProfileNames() sets.Set[string] { return nil }

// TestLazyReserverResolution verifies that a picker constructed while no
// EndpointReserver is registered still picks one up once it appears (issue
// #1598: eager resolution at factory time silently disabled the reservation
// path whenever the producer was listed after the picker in the config).
func TestLazyReserverResolution(t *testing.T) {
	h := &lateHandle{}
	pl, err := MaxScorePickerFactory("max-score-picker", nil, h)
	if err != nil {
		t.Fatalf("factory: %v", err)
	}
	picker := pl.(*MaxScorePicker)

	scored := []*fwksched.ScoredEndpoint{
		{Endpoint: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil), Score: 1.0},
		{Endpoint: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil), Score: 0.5},
	}

	// Pick before any reserver exists: must not panic, must not reserve.
	if res := picker.Pick(context.Background(), scored); len(res.TargetEndpoints) != 1 {
		t.Fatalf("expected 1 target, got %d", len(res.TargetEndpoints))
	}

	// The reserver plugin registers late (after the picker's construction).
	reserver := &recordingReserver{}
	h.add(reserver)

	// The next pick must resolve it lazily and reserve the selected endpoint.
	if res := picker.Pick(context.Background(), scored); len(res.TargetEndpoints) != 1 {
		t.Fatalf("expected 1 target, got %d", len(res.TargetEndpoints))
	}
	got := reserver.reserved()
	if len(got) != 1 {
		t.Fatalf("expected 1 reservation after late registration, got %v", got)
	}
	if got[0] != "pod1" {
		t.Errorf("expected the highest-scored endpoint pod1 to be reserved, got %s", got[0])
	}
}
