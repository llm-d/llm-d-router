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

package scheduling_test

// Verifies that the scheduler calls the configured EndpointReserver after
// each successful profile run, passing the request ID and profile name.

import (
	"context"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
)

// recordingReserver records ReserveProfileResult calls.
type recordingReserver struct {
	calls []string
}

func (r *recordingReserver) ReserveProfileResult(_ context.Context, requestID, profileName string, _ *fwksched.ProfileRunResult) {
	r.calls = append(r.calls, requestID+"/"+profileName)
}

var _ fwkrh.EndpointReserver = (*recordingReserver)(nil)

func TestSchedulerReservesAfterProfileRun(t *testing.T) {
	ctx := context.Background()
	reserver := &recordingReserver{}

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: "pod1"}},
		&fwkdl.Metrics{}, fwkdl.NewAttributes())

	// A minimal profile: just the max-score picker (no filters or scorers,
	// so any candidate set passes through to the picker).
	profile := scheduling.NewSchedulerProfile().
		WithPicker(maxscore.NewMaxScorePicker(1))

	handler := single.NewSingleProfileHandler()

	cfg := scheduling.NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{
		"default": profile,
	}).WithReservers([]fwkrh.EndpointReserver{reserver})

	sched := scheduling.NewSchedulerWithConfig(cfg)

	req := &fwksched.InferenceRequest{RequestID: "test-req"}
	result, err := sched.Schedule(ctx, req, []fwksched.Endpoint{endpoint})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	if result == nil || len(result.ProfileResults) == 0 {
		t.Fatal("expected a scheduling result")
	}

	if len(reserver.calls) != 1 {
		t.Fatalf("expected 1 reservation call, got %d: %v", len(reserver.calls), reserver.calls)
	}
	if reserver.calls[0] != "test-req/default" {
		t.Errorf("expected reservation for test-req/default, got %s", reserver.calls[0])
	}
}

// TestSchedulerNoReserverIsNoOp verifies that a scheduler without reservers
// schedules normally (no panics, no behavior change).
func TestSchedulerNoReserverIsNoOp(t *testing.T) {
	ctx := context.Background()

	endpoint := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "default", Name: "pod1"}},
		&fwkdl.Metrics{}, fwkdl.NewAttributes())

	profile := scheduling.NewSchedulerProfile().
		WithPicker(maxscore.NewMaxScorePicker(1))
	handler := single.NewSingleProfileHandler()
	cfg := scheduling.NewSchedulerConfig(handler, map[string]fwksched.SchedulerProfile{
		"default": profile,
	})

	sched := scheduling.NewSchedulerWithConfig(cfg)
	req := &fwksched.InferenceRequest{RequestID: "test-req"}
	result, err := sched.Schedule(ctx, req, []fwksched.Endpoint{endpoint})
	if err != nil {
		t.Fatalf("Schedule without reservers: %v", err)
	}
	if result == nil || len(result.ProfileResults) == 0 {
		t.Fatal("expected a scheduling result")
	}
}

// Verify the config setter compiles and returns the same config.
var _ = fwkplugin.TypedName{} // keep the import used if the others change
