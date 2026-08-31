/*
Copyright 2025 The Kubernetes Authors.

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
	"errors"
	"strconv"
)

// QueueOutcome represents the high-level final state of a request's lifecycle within the `controller.FlowController`.
//
// It is returned by `FlowController.EnqueueAndWait()` along with a corresponding error. This enum is designed to be a
// low-cardinality label ideal for metrics, while the error provides fine-grained details for non-dispatched outcomes.
//
// The error is the authoritative representation of a final state; the outcome is derived from it via
// `OutcomeFromError`. The per-constant documentation below describes which sentinels each outcome corresponds to.
type QueueOutcome int

const (
	// QueueOutcomeNotYetFinalized indicates the request has not yet been finalized by the `controller.FlowController`.
	// This is an internal default value and should never be returned by `FlowController.EnqueueAndWait()`.
	QueueOutcomeNotYetFinalized QueueOutcome = iota

	// QueueOutcomeDispatched indicates the request was successfully processed by the `controller.FlowController` and
	// unblocked for the caller to proceed.
	// The associated error from `FlowController.EnqueueAndWait()` will be nil.
	QueueOutcomeDispatched

	// --- Pre-Enqueue Rejection Outcomes (request never entered a SafeQueue) ---
	// For these outcomes, the error from `FlowController.EnqueueAndWait()` will wrap `ErrRejected`.

	// QueueOutcomeRejectedCapacity indicates rejection because queue capacity limits were met.
	// The associated error will wrap `ErrQueueAtCapacity` (and `ErrRejected`).
	QueueOutcomeRejectedCapacity

	// QueueOutcomeRejectedNoEndpoints indicates rejection at the queue-capacity boundary while the candidate pool had no
	// endpoints. It is distinguished from `QueueOutcomeRejectedCapacity` so the admission layer can surface genuine
	// unavailability (HTTP 503) instead of backpressure (HTTP 429) when the pool has scaled to zero.
	// The associated error will wrap `ErrNoEndpoints` (and `ErrRejected`).
	QueueOutcomeRejectedNoEndpoints

	// QueueOutcomeRejectedOther indicates rejection for reasons other than capacity before the request was formally
	// enqueued.
	// The specific underlying cause can be determined from the associated error (e.g., a nil request, an unregistered
	// flow ID, or controller shutdown), which will be wrapped by `ErrRejected`.
	// A pre-admission TTL expiry or context cancellation (`ErrRejected` wrapping `ErrTTLExpired` or
	// `ErrContextCancelled`) lands here by design: no dedicated rejection outcome exists for them because rejected
	// items never consumed queue capacity, and the Rejected/Evicted metrics split accounts for exactly that.
	QueueOutcomeRejectedOther

	// --- Post-Enqueue Eviction Outcomes (request was in a SafeQueue but not dispatched) ---
	// For these outcomes, the error from `FlowController.EnqueueAndWait()` will wrap `ErrEvicted`.

	// QueueOutcomeEvictedTTL indicates eviction from a queue because the request's effective Time-To-Live expired.
	// The associated error will wrap `ErrTTLExpired` (and `ErrEvicted`).
	QueueOutcomeEvictedTTL

	// QueueOutcomeEvictedNoEndpoints indicates eviction from a queue because the queue-wait budget expired while the
	// candidate pool had no endpoints. It is distinguished from `QueueOutcomeEvictedTTL` so the admission layer can
	// surface genuine unavailability (HTTP 503) instead of backpressure (HTTP 429), and so the two unavailability
	// regimes remain separable on dashboards.
	// The associated error will wrap `ErrTTLExpired` and `ErrNoEndpoints` (and `ErrEvicted`).
	QueueOutcomeEvictedNoEndpoints

	// QueueOutcomeEvictedContextCancelled indicates eviction from a queue because the request's own context (from
	// `FlowControlRequest.Context()`) was cancelled.
	// The associated error will wrap `ErrContextCancelled` (which may further wrap the underlying `context.Canceled` or
	// `context.DeadlineExceeded` error) (and `ErrEvicted`).
	QueueOutcomeEvictedContextCancelled

	// QueueOutcomeEvictedOther indicates eviction from a queue for reasons not covered by more specific eviction
	// outcomes.
	// The specific underlying cause can be determined from the associated error (e.g., controller shutdown while the item
	// was queued), which will be wrapped by `ErrEvicted`.
	QueueOutcomeEvictedOther

	// NumQueueOutcomes is a sentinel that equals the total number of QueueOutcome values.
	// It is not a valid outcome; it exists to size arrays indexed by QueueOutcome and to allow tests to detect
	// when new values are added without a corresponding update to dependent code.
	NumQueueOutcomes
)

// String returns a human-readable string representation of the QueueOutcome.
func (o QueueOutcome) String() string {
	switch o {
	case QueueOutcomeNotYetFinalized:
		return "NotYetFinalized"
	case QueueOutcomeDispatched:
		return "Dispatched"
	case QueueOutcomeRejectedCapacity:
		return "RejectedCapacity"
	case QueueOutcomeRejectedNoEndpoints:
		return "RejectedNoEndpoints"
	case QueueOutcomeRejectedOther:
		return "RejectedOther"
	case QueueOutcomeEvictedTTL:
		return "EvictedTTL"
	case QueueOutcomeEvictedNoEndpoints:
		return "EvictedNoEndpoints"
	case QueueOutcomeEvictedContextCancelled:
		return "EvictedContextCancelled"
	case QueueOutcomeEvictedOther:
		return "EvictedOther"
	default:
		// Return the integer value for unknown outcomes to aid in debugging.
		return "UnknownOutcome(" + strconv.Itoa(int(o)) + ")"
	}
}

// OutcomeFromError derives the QueueOutcome corresponding to a finalization error. A nil error means
// `QueueOutcomeDispatched`. ok is false when a non-nil error wraps neither `ErrRejected` nor `ErrEvicted`, which is
// an internal invariant violation; callers should log it and use the returned `QueueOutcomeRejectedOther`.
//
// Within the `ErrEvicted` family, `ErrNoEndpoints` is matched before `ErrTTLExpired` because a no-endpoint budget
// expiry wraps both sentinels. The `ErrEvicted` family is matched before `ErrRejected`: no error carries both, but
// if one ever did, eviction is the safer classification because it implies queue capacity was consumed, which is
// what the Rejected/Evicted metrics split accounts for.
func OutcomeFromError(err error) (QueueOutcome, bool) {
	switch {
	case err == nil:
		return QueueOutcomeDispatched, true
	case errors.Is(err, ErrEvicted):
		switch {
		case errors.Is(err, ErrNoEndpoints):
			return QueueOutcomeEvictedNoEndpoints, true
		case errors.Is(err, ErrTTLExpired):
			return QueueOutcomeEvictedTTL, true
		case errors.Is(err, ErrContextCancelled):
			return QueueOutcomeEvictedContextCancelled, true
		default:
			return QueueOutcomeEvictedOther, true
		}
	case errors.Is(err, ErrRejected):
		switch {
		case errors.Is(err, ErrNoEndpoints):
			return QueueOutcomeRejectedNoEndpoints, true
		case errors.Is(err, ErrQueueAtCapacity):
			return QueueOutcomeRejectedCapacity, true
		default:
			// Intentionally includes `ErrTTLExpired` and `ErrContextCancelled`: see `QueueOutcomeRejectedOther`.
			return QueueOutcomeRejectedOther, true
		}
	default:
		return QueueOutcomeRejectedOther, false
	}
}
