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
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/llm-d/llm-d-router/pkg/epp/statestore/stateapi/proto/gen"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

// ConcurrencyState is the client-side interface to the stateful EPP's fleet-
// wide concurrency lease (pkg/epp/statestore/stateapi.ConcurrencyStore). This
// is a genuinely new primitive, not an extension of FlowControlState/
// FlowController: the existing FlowRegistry counts queue occupancy (items are
// dequeued and finalized the instant they dispatch), not concurrent
// execution, so it cannot itself serve as a remote concurrency cap.
type ConcurrencyState interface {
	Admit(ctx context.Context, key FlowControlKey, requestID string) (FlowControlOutcome, error)
	Release(ctx context.Context, key FlowControlKey, requestID string) error
}

type remoteConcurrencyState struct {
	client pb.StateAPIClient
}

// NewRemoteConcurrencyState returns a ConcurrencyState backed by the given
// gRPC State API client.
func NewRemoteConcurrencyState(client pb.StateAPIClient) ConcurrencyState {
	return &remoteConcurrencyState{client: client}
}

func (r *remoteConcurrencyState) Admit(ctx context.Context, key FlowControlKey, requestID string) (FlowControlOutcome, error) {
	resp, err := r.client.AdmitConcurrency(ctx, &pb.AdmitConcurrencyRequest{
		RequestId: requestID,
		FlowKey:   &pb.FlowKey{Id: key.ID, Priority: int32(key.Priority)},
	})
	if err != nil {
		return FlowControlOutcomeRejected, err
	}
	if resp.GetOutcome() == pb.ConcurrencyOutcome_CONCURRENCY_OUTCOME_ADMITTED {
		return FlowControlOutcomeAdmitted, nil
	}
	return FlowControlOutcomeRejected, nil
}

func (r *remoteConcurrencyState) Release(ctx context.Context, key FlowControlKey, requestID string) error {
	_, err := r.client.ReleaseConcurrency(ctx, &pb.ReleaseConcurrencyRequest{
		RequestId: requestID,
		FlowKey:   &pb.FlowKey{Id: key.ID, Priority: int32(key.Priority)},
	})
	return err
}

// admitRecord tracks, per requestID, which path an Admit call actually took,
// so the later Release call (made independently, without the Admit outcome
// in hand) knows exactly what to undo.
type admitRecord struct {
	usedRemote        bool
	usedLocalFallback bool
}

// localFallbackConcurrencyState implements FlowControlState by composing the
// existing local queue admission (unchanged) with the fleet-wide concurrency
// lease: the LocalFallback degradation strategy.
//
// Composition is sequential and local-first, deliberately:
//   - The local FlowController is a queue (priority, TTL, backpressure); the
//     remote counter is concurrency. Checking remote before local dispatch
//     would hold a fleet concurrency slot for a request that isn't executing
//     yet, conflating queue occupancy with concurrent execution.
//   - Local-first also avoids a compensating-decrement RTT: remote-first with
//     a subsequent local rejection would require rolling back the remote
//     increment, which can itself fail and leak.
//
// On remote failure/timeout, admission falls back to a local concurrency cap
// (localMaxConcurrency) so a single degraded replica cannot exceed the fleet
// limit on its own, and the outcome is reported as Degraded (visible in
// metrics) rather than silently treated as Admitted.
type localFallbackConcurrencyState struct {
	local   FlowControlState
	remote  ConcurrencyState
	timeout time.Duration

	localMaxConcurrency int64
	localFallbackInUse  atomic.Int64

	records sync.Map // requestID -> *admitRecord
}

// NewLocalFallbackFlowControlState returns a FlowControlState that gates on
// the existing local queue admission first, then a fleet-wide concurrency
// lease via remote, falling back to a local concurrency cap on remote
// failure. localMaxConcurrency <= 0 means no local fallback cap (degraded
// requests are always admitted once the local queue admits them).
func NewLocalFallbackFlowControlState(local FlowControlState, remote ConcurrencyState, timeout time.Duration, localMaxConcurrency int64) FlowControlState {
	return &localFallbackConcurrencyState{
		local:               local,
		remote:              remote,
		timeout:             timeout,
		localMaxConcurrency: localMaxConcurrency,
	}
}

func (s *localFallbackConcurrencyState) Admit(ctx context.Context, req flowcontrol.FlowControlRequest) (FlowControlOutcome, error) {
	outcome, err := s.local.Admit(ctx, req)
	if err != nil || outcome != FlowControlOutcomeAdmitted {
		// Local queue itself rejected (or errored): no lease was ever held,
		// so the caller must not call Release for this request.
		return outcome, err
	}

	key := FlowControlKey{ID: req.FlowKey().ID, Priority: req.FlowKey().Priority}
	recordRemoteCall("concurrency_admit")
	rctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	remoteOutcome, rerr := s.remote.Admit(rctx, key, req.ID())

	switch {
	case rerr != nil:
		recordRemoteFallback("concurrency_admit")
		if s.localMaxConcurrency > 0 {
			if s.localFallbackInUse.Add(1) > s.localMaxConcurrency {
				s.localFallbackInUse.Add(-1)
				return FlowControlOutcomeRejected, nil
			}
		}
		s.records.Store(req.ID(), &admitRecord{usedLocalFallback: true})
		return FlowControlOutcomeDegraded, nil
	case remoteOutcome == FlowControlOutcomeRejected:
		return FlowControlOutcomeRejected, nil
	default:
		s.records.Store(req.ID(), &admitRecord{usedRemote: true})
		return FlowControlOutcomeAdmitted, nil
	}
}

func (s *localFallbackConcurrencyState) Release(ctx context.Context, requestID string, flowKey FlowControlKey) error {
	val, ok := s.records.LoadAndDelete(requestID)
	if !ok {
		// No lease was held for this request (rejected earlier, or already released).
		return nil
	}
	rec := val.(*admitRecord)

	if rec.usedLocalFallback {
		decrementClampedInt64(&s.localFallbackInUse)
		return nil
	}
	if rec.usedRemote {
		// Called from response completion (Director.HandleResponseBody at
		// EndOfStream): nothing downstream waits on this outcome, so — like
		// failOpenInflightState/failOpenPrefixState's writes — it runs
		// asynchronously rather than holding up response teardown.
		recordRemoteCall("concurrency_release")
		asyncRemoteWrite(s.timeout, "concurrency_release", func(rctx context.Context) error {
			return s.remote.Release(rctx, flowKey, requestID)
		})
	}
	return nil
}

// decrementClampedInt64 subtracts 1 from counter with a hard floor at zero.
func decrementClampedInt64(counter *atomic.Int64) {
	for {
		current := counter.Load()
		if current <= 0 {
			return
		}
		if counter.CompareAndSwap(current, current-1) {
			return
		}
	}
}
