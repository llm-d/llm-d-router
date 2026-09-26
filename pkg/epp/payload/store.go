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

// Package payload implements opt-in capture of GenAI request payloads as
// OpenTelemetry span events, following the upstream GenAI semantic conventions
// (gen_ai.input.messages / gen_ai.system_instructions) as specified in the
// llm-d proposal docs/proposals/genai-payload-events.md (llm-d/llm-d).
//
// This package is Phase 1 of the proposal's phased implementation plan: the
// PayloadStore interface, the noop and inline backends, and the gateway
// capture wiring. Object-store backends (gcs, s3, filesystem) and the
// redaction pipeline arrive in later phases.
package payload

import (
	"context"
	"errors"
)

// PayloadKind distinguishes the request (prompt) payload from the response
// (completion) payload of the same span.
type PayloadKind string

const (
	// KindPrompt identifies the request-side payload.
	KindPrompt PayloadKind = "prompt"
	// KindCompletion identifies the response-side payload. Not produced by the
	// gateway capture layer (which sees prompts only); defined here because the
	// store contract is shared with later capture layers.
	KindCompletion PayloadKind = "completion"
)

// PayloadRef identifies one payload part. PartIndex is zero for the text
// payload itself and >= 1 for each non-text content part associated with the
// same span. MediaType is an IANA media type (e.g. "application/json",
// "image/png") and drives both the storage-path extension and Content-Type on
// uploads for URI-producing backends.
type PayloadRef struct {
	TraceID   string
	SpanID    string
	Kind      PayloadKind
	PartIndex int
	MediaType string
}

// ErrPayloadTooLarge is returned by backends that cannot accept a payload of
// the offered size (e.g. InlineStore when the payload exceeds its threshold).
// Callers emit the span event with llm_d.payload.truncated=true instead of
// the payload content.
var ErrPayloadTooLarge = errors.New("payload exceeds backend size limit")

// PayloadStore persists a payload and returns a retrieval URI.
//
// URI-producing backends (gcs, s3, filesystem — later phases) return a
// non-empty URI recorded as llm_d.payload.storage_uri. Backends that do not
// externalise the payload (noop, inline) return an empty URI: NoopStore
// silently discards, and InlineStore signals that the payload may be carried
// inline on the span event itself.
type PayloadStore interface {
	Store(ctx context.Context, ref PayloadRef, data []byte) (uri string, err error)
}

// NoopStore discards every payload. It backs `backend: noop`, the secondary
// kill switch: the capture pipeline short-circuits before any span event is
// emitted, so NoopStore.Store is only reachable through direct use of the
// interface.
type NoopStore struct{}

// Store discards the payload and returns no URI.
func (NoopStore) Store(context.Context, PayloadRef, []byte) (string, error) {
	return "", nil
}

// InlineStore carries payloads inline as span-event attributes, subject to a
// size threshold. It produces no URI; a nil error means the caller may attach
// the payload inline.
type InlineStore struct {
	// MaxBytes is the largest payload accepted for inline attachment
	// (payloadCapture.inlineSizeThresholdBytes).
	MaxBytes int
}

// Store accepts the payload for inline attachment or rejects it with
// ErrPayloadTooLarge. In later phases oversized payloads auto-offload to the
// configured offloadBackend; in Phase 1 there is none, so the caller records
// llm_d.payload.truncated=true instead.
func (s InlineStore) Store(_ context.Context, _ PayloadRef, data []byte) (string, error) {
	if len(data) > s.MaxBytes {
		return "", ErrPayloadTooLarge
	}
	return "", nil
}
