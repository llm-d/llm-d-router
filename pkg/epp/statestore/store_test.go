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
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
)

func TestNewLocalStateStore_ComposesAllThreeCategories(t *testing.T) {
	t.Parallel()
	inflight := &stubInflightBackend{requests: map[string]int64{"ep-1": 1}, tokens: map[string]int64{"ep-1": 10}}
	prefix := &stubPrefixBackend{matches: map[uint64][]string{7: {"ep-1"}}}
	flowControl := &stubFlowControlBackend{outcome: types.QueueOutcomeDispatched}

	store := NewLocalStateStore(inflight, prefix, flowControl)
	ctx := context.Background()

	snap := store.GetInflightSnapshot(ctx, "ep-1")
	require.Equal(t, InflightSnapshot{Requests: 1, Tokens: 10}, snap)

	require.Equal(t, []string{"ep-1"}, store.GetPrefixMatch(ctx, 7))

	outcome, err := store.Admit(ctx, &stubFlowControlRequest{})
	require.NoError(t, err)
	require.Equal(t, FlowControlOutcomeAdmitted, outcome)
}

func TestDefaultAccessModeConfig(t *testing.T) {
	t.Parallel()
	cfg := DefaultAccessModeConfig()
	require.Equal(t, AccessModeLocal, cfg.Inflight)
	require.Equal(t, AccessModeLocal, cfg.Prefix)
	require.Equal(t, AccessModeLocal, cfg.FlowControl)
}
