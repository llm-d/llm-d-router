/*
Copyright 2026 The Kubernetes Authors.

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

package statestore

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// FlowControlBackend is the minimal adapter interface that the Local flow
// control provider requires. *controller.FlowController already satisfies
// this directly via its EnqueueAndWait method, so no wrapper code is needed.
//
// Conformance: Implementations MUST be goroutine-safe.
type FlowControlBackend interface {
	// EnqueueAndWait submits a request to flow control and blocks until it
	// reaches a terminal outcome.
	EnqueueAndWait(ctx context.Context, req flowcontrol.FlowControlRequest) (types.QueueOutcome, error)
}

// localFlowControlState wraps a FlowControlBackend to satisfy FlowControlState.
// Release is a no-op: the backend has no separate release call (see
// FlowControlState.Release doc comment).
type localFlowControlState struct {
	backend FlowControlBackend
}

// NewLocalFlowControlState returns a FlowControlState backed by the given
// FlowControlBackend (typically a *controller.FlowController).
func NewLocalFlowControlState(backend FlowControlBackend) FlowControlState {
	return &localFlowControlState{backend: backend}
}

func (s *localFlowControlState) Admit(ctx context.Context, req flowcontrol.FlowControlRequest) (FlowControlOutcome, error) {
	outcome, err := s.backend.EnqueueAndWait(ctx, req)
	if outcome == types.QueueOutcomeDispatched {
		return FlowControlOutcomeAdmitted, nil
	}
	return FlowControlOutcomeRejected, err
}

func (s *localFlowControlState) Release(_ context.Context, _ string, _ FlowControlKey) error {
	return nil
}
