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

package requestcontrol

import (
	"context"
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// stubReserver records ReserveResult calls for the findEndpointReservers test.
type stubReserver struct {
	calls int
}

func (s *stubReserver) ReserveResult(_ context.Context, _ string, _ *fwksched.SchedulingResult) {
	s.calls++
}

// The stub must satisfy the DataProducer list element type as well, so it can
// sit among the producers the director iterates.
func (s *stubReserver) Produce(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	return nil
}

// TypedName and Produces satisfy fwkplugin.ProducerPlugin.
func (s *stubReserver) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "stub-reserver", Name: "stub-reserver"}
}

func (s *stubReserver) Produces() map[fwkplugin.DataKey]any {
	return nil
}

func TestFindEndpointReservers(t *testing.T) {
	reserver := &stubReserver{}
	producers := []fwkrh.DataProducer{reserver}

	got := findEndpointReservers(producers)
	if len(got) != 1 {
		t.Fatalf("expected 1 reserver, got %d", len(got))
	}
	got[0].ReserveResult(context.Background(), "req", nil)
	if reserver.calls != 1 {
		t.Errorf("expected the found reserver to be callable, calls=%d", reserver.calls)
	}

	if got := findEndpointReservers(nil); len(got) != 0 {
		t.Errorf("expected no reservers for an empty producer list, got %d", len(got))
	}
}
