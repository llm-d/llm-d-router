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

package types

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestOutcomeFromError pins the outcome derived for every error shape a finalization site produces. Most rows are
// the literal expression a producer builds, and the expected outcome is the constant that producer's final state
// carries; a mapping change that would silently reclassify a producer fails here. Rows marked defensive pin shapes
// no producer currently emits, so the mapping stays stable if one appears.
func TestOutcomeFromError(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name          string
		err           error
		expectOutcome QueueOutcome
		expectOK      bool
	}{
		{
			name:          "nil means dispatched",
			err:           nil,
			expectOutcome: QueueOutcomeDispatched,
			expectOK:      true,
		},
		{
			name:          "rejected at capacity",
			err:           fmt.Errorf("%w: %w", ErrRejected, ErrQueueAtCapacity),
			expectOutcome: QueueOutcomeRejectedCapacity,
			expectOK:      true,
		},
		{
			name:          "rejected at capacity with empty pool",
			err:           fmt.Errorf("%w: %w", ErrRejected, ErrNoEndpoints),
			expectOutcome: QueueOutcomeRejectedNoEndpoints,
			expectOK:      true,
		},
		{
			name:          "rejected on registry failure",
			err:           fmt.Errorf("%w: %w", ErrRejected, errors.New("failed to resolve queue")),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      true,
		},
		{
			name:          "rejected on shutdown",
			err:           fmt.Errorf("%w: %w", ErrRejected, ErrFlowControllerNotRunning),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      true,
		},
		{
			// See QueueOutcomeRejectedOther: pre-admission expiries stay in the Rejected family.
			name:          "pre-admission TTL expiry",
			err:           fmt.Errorf("%w: %w", ErrRejected, ErrTTLExpired),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      true,
		},
		{
			name:          "pre-admission context cancellation",
			err:           fmt.Errorf("%w: %w", ErrRejected, fmt.Errorf("%w: %w", ErrContextCancelled, context.Canceled)),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      true,
		},
		{
			// The registry error is flattened with %v at its producer, so only the family sentinel survives.
			name:          "rejected with flattened inner error",
			err:           fmt.Errorf("%w: failed to get ManagedQueue for leased flow: %v", ErrRejected, ErrQueueAtCapacity),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      true,
		},
		{
			// Defensive: no producer double-wraps, but a repeated family sentinel must stay harmless.
			name:          "double-wrapped rejection",
			err:           fmt.Errorf("%w: %w", ErrRejected, fmt.Errorf("%w: %w", ErrRejected, ErrFlowControllerNotRunning)),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      true,
		},
		{
			name:          "evicted on TTL expiry",
			err:           fmt.Errorf("%w: %w", ErrEvicted, ErrTTLExpired),
			expectOutcome: QueueOutcomeEvictedTTL,
			expectOK:      true,
		},
		{
			// The no-endpoint budget expiry wraps both inner sentinels; ErrNoEndpoints takes precedence.
			name:          "evicted on no-endpoint budget expiry",
			err:           fmt.Errorf("%w: %w: %w", ErrEvicted, ErrTTLExpired, ErrNoEndpoints),
			expectOutcome: QueueOutcomeEvictedNoEndpoints,
			expectOK:      true,
		},
		{
			// Defensive: every current no-endpoint eviction also carries ErrTTLExpired.
			name:          "evicted with no endpoints and no TTL sentinel",
			err:           fmt.Errorf("%w: %w", ErrEvicted, ErrNoEndpoints),
			expectOutcome: QueueOutcomeEvictedNoEndpoints,
			expectOK:      true,
		},
		{
			name:          "evicted on context cancellation",
			err:           fmt.Errorf("%w: %w", ErrEvicted, fmt.Errorf("%w: %w", ErrContextCancelled, context.Canceled)),
			expectOutcome: QueueOutcomeEvictedContextCancelled,
			expectOK:      true,
		},
		{
			name:          "evicted on shutdown",
			err:           fmt.Errorf("%w: %w", ErrEvicted, ErrFlowControllerNotRunning),
			expectOutcome: QueueOutcomeEvictedOther,
			expectOK:      true,
		},
		{
			name:          "evicted with unrecognized cause",
			err:           fmt.Errorf("%w: %w", ErrEvicted, errors.New("custom context cause")),
			expectOutcome: QueueOutcomeEvictedOther,
			expectOK:      true,
		},
		{
			name:          "missing family sentinel is an invariant violation",
			err:           errors.New("no family sentinel"),
			expectOutcome: QueueOutcomeRejectedOther,
			expectOK:      false,
		},
	}

	covered := make(map[QueueOutcome]bool)
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			outcome, ok := OutcomeFromError(tc.err)
			assert.Equal(t, tc.expectOutcome, outcome, "unexpected outcome")
			assert.Equal(t, tc.expectOK, ok, "unexpected ok")
		})
		covered[tc.expectOutcome] = true
	}

	// Every reachable outcome must have a producer row, so a new outcome added without extending this table (and the
	// mapping) fails loudly.
	for o := QueueOutcomeNotYetFinalized + 1; o < NumQueueOutcomes; o++ {
		assert.True(t, covered[o], "no test row covers outcome %s", o)
	}
}
