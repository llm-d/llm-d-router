/*
Copyright 2025 The Kubernetes Authors.
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

package requesthandling

import (
	"time"

	"google.golang.org/protobuf/types/known/structpb"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// TerminationCause labels how a dispatched request's response stream ended. Observers that learn
// from response records need it to tell a complete observation from a truncated one: a stream cut
// short by a disconnect or an eviction reports only the tokens emitted before the cut. Requests
// whose response processing is skipped never observe completion, so their records always carry
// TerminationCauseNotObserved.
type TerminationCause string

const (
	// TerminationCauseNatural is a stream the model server ended on its own, including responses
	// that deliver an error status in full.
	TerminationCauseNatural TerminationCause = "natural"
	// TerminationCauseClientDisconnect is a stream torn down under the EPP while the response was
	// in flight. A client disconnect is the most common reason; any teardown the EPP observes only
	// as a cancelled stream (Envoy drain or restart, an upstream reset, the EPP's own drain)
	// produces the same value, because the EPP cannot distinguish the reasons from its side of
	// the stream.
	TerminationCauseClientDisconnect TerminationCause = "client-disconnect"
	// TerminationCauseEvicted is a stream the EPP terminated to reclaim capacity.
	TerminationCauseEvicted TerminationCause = "evicted"
	// TerminationCauseError is a stream that ended without completing for any reason not otherwise
	// classified: a half-close that reaches the EPP before its cancellation propagates, or an
	// EPP-side failure.
	TerminationCauseError TerminationCause = "error"
	// TerminationCauseNotObserved is a request whose route skips response processing: the EPP
	// stops processing the stream after routing and never sees the response.
	TerminationCauseNotObserved TerminationCause = "not-observed"
)

// StopReason is the normalized reason a model response ended. Parsers map their
// provider-specific finish_reason or stop_reason into this vocabulary once, so
// response consumers do not parse provider wire formats. TerminationCause takes
// precedence: when it is not natural, the response did not complete normally and
// the stop reason describes only what was observed before the stream ended.
type StopReason string

const (
	// StopReasonToolCall reports that the model stopped to call a tool.
	StopReasonToolCall StopReason = "tool_call"
	// StopReasonTurnEnded reports that the model finished the turn without a tool call.
	StopReasonTurnEnded StopReason = "turn_ended"
	// StopReasonUnknown reports a missing, malformed, or unsupported reason.
	StopReasonUnknown StopReason = "unknown"
)

// OrUnknown returns the reason when one was observed and unknown otherwise.
// A ParsedResponse leaves StopReason empty when no reason was observed; a
// final Response is always populated with this method.
func (r StopReason) OrUnknown() StopReason {
	if r == "" {
		return StopReasonUnknown
	}
	return r
}

// Response contains information from the response received to be passed to the Response requesthandling plugins
type Response struct {
	// RequestID is the unique identifier for the request; see scheduling.InferenceRequest.RequestID.
	RequestID string
	// Headers is a map of the response headers. Nil during body processing
	Headers map[string]string
	// StartOfStream when true indicates that this invocation contains the first chunk of the response
	StartOfStream bool
	// EndOfStream when true indicates that this invocation contains the last chunk of the response
	EndOfStream bool
	// ReqMetadata is a map of metadata that can be passed from Envoy.
	// It is populated with Envoy's dynamic metadata when ext_proc is processing ProcessingRequest_ResponseHeaders.
	// Currently, this is only used by conformance test.
	ReqMetadata map[string]any
	// Token usage counts parsed from the response body.
	Usage fwkrequest.Usage
	// StreamedEvents is the running count of stream data events observed so far. It is the only
	// length signal this record carries for a truncated stream, since a stream that never
	// completes carries no usage block. Consumers must not treat zero as evidence that nothing
	// was generated; ParsedResponse documents which parsers count and how the
	// count deviates from the token count.
	StreamedEvents int
	// TerminationCause labels how the stream ended.
	TerminationCause TerminationCause
	// StopReason is the normalized reason the response ended. It is set only on the
	// end-of-stream call and is always normalized to unknown when no reason was
	// observed. Check TerminationCause first: a non-natural cause means the stream
	// did not complete normally.
	StopReason StopReason
	// Completion is set only on the end-of-stream call; see ParsedResponse.
	Completion *Completion
	// ReceivedTimestamp is when the EPP received the request. Set only at end of stream.
	ReceivedTimestamp time.Time
	// FirstTokenTimestamp is when the first response body bytes were received; zero if none
	// were. Set only at end of stream.
	FirstTokenTimestamp time.Time
	// CompleteTimestamp is when the response completed; zero for a stream that ended without
	// completing. Set only at end of stream.
	CompleteTimestamp time.Time
	// Streaming reports whether the model server answered with an event stream. Set only at
	// end of stream.
	Streaming bool
	// DynamicMetadata is a map of metadata that can be passed to the Envoy. It is populated into the dynamic
	// metadata when processing ProcessingResponse_RequestHeaders.
	DynamicMetadata *structpb.Struct
}
