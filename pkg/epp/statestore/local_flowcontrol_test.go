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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// stubFlowControlRequest is a minimal flowcontrol.FlowControlRequest for tests
// that only need FlowKey (and, optionally, a distinct ID) to be meaningful.
type stubFlowControlRequest struct {
	flowKey flowcontrol.FlowKey
	id      string
}

func (r *stubFlowControlRequest) FlowKey() flowcontrol.FlowKey                 { return r.flowKey }
func (r *stubFlowControlRequest) ByteSize() uint64                             { return 0 }
func (r *stubFlowControlRequest) InferenceRequest() *fwksched.InferenceRequest { return nil }
func (r *stubFlowControlRequest) ReceivedTimestamp() time.Time                 { return time.Time{} }
func (r *stubFlowControlRequest) InitialEffectiveTTL() time.Duration           { return 0 }
func (r *stubFlowControlRequest) ID() string {
	if r.id != "" {
		return r.id
	}
	return "req-1"
}
func (r *stubFlowControlRequest) GetMetadata() map[string]any { return nil }
func (r *stubFlowControlRequest) InferencePoolName() string   { return "" }
func (r *stubFlowControlRequest) ModelName() string           { return "" }
func (r *stubFlowControlRequest) TargetModelName() string     { return "" }

var _ flowcontrol.FlowControlRequest = &stubFlowControlRequest{}

type stubFlowControlBackend struct {
	outcome types.QueueOutcome
	err     error
	called  bool
}

func (b *stubFlowControlBackend) EnqueueAndWait(_ context.Context, _ flowcontrol.FlowControlRequest) (types.QueueOutcome, error) {
	b.called = true
	return b.outcome, b.err
}

func TestLocalFlowControlState_Admit_Dispatched(t *testing.T) {
	t.Parallel()
	backend := &stubFlowControlBackend{outcome: types.QueueOutcomeDispatched}
	state := NewLocalFlowControlState(backend)

	outcome, err := state.Admit(context.Background(), &stubFlowControlRequest{})

	require.NoError(t, err)
	require.True(t, backend.called)
	require.Equal(t, FlowControlOutcomeAdmitted, outcome)
}

func TestLocalFlowControlState_Admit_Rejected(t *testing.T) {
	t.Parallel()
	wantErr := errors.New("queue at capacity")
	backend := &stubFlowControlBackend{outcome: types.QueueOutcomeRejectedCapacity, err: wantErr}
	state := NewLocalFlowControlState(backend)

	outcome, err := state.Admit(context.Background(), &stubFlowControlRequest{})

	require.ErrorIs(t, err, wantErr)
	require.Equal(t, FlowControlOutcomeRejected, outcome)
}

func TestLocalFlowControlState_Release_IsNoOp(t *testing.T) {
	t.Parallel()
	backend := &stubFlowControlBackend{}
	state := NewLocalFlowControlState(backend)

	err := state.Release(context.Background(), "req-1", FlowControlKey{ID: "tenant", Priority: 0})

	assert.NoError(t, err)
	assert.False(t, backend.called)
}
