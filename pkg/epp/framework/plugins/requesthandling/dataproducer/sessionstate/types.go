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

package sessionstate

import (
	"slices"
	"time"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// SessionStateProducerType is the plugin type registered with the framework.
const SessionStateProducerType = "session-state-producer"

var (
	// SessionStateDataKey identifies session history published on the request
	// attribute store. The default producer is the session-state-producer.
	SessionStateDataKey = plugin.NewDataKey("SessionStateDataKey", SessionStateProducerType)

	// EndpointSessionsDataKey identifies the snapshot of sessions bound to an
	// endpoint published on each endpoint attribute map. The default producer is
	// the session-state-producer.
	EndpointSessionsDataKey = plugin.NewDataKey("EndpointSessionsDataKey", SessionStateProducerType)
)

// SessionState is the session history visible to a request. The producer
// publishes it from RequestHeader, so flow-control and scheduling plugins read
// the latest counters, endpoint binding, and token totals while the request waits.
type SessionState struct {
	SessionID         string
	TurnsTaken        int64
	Duration          time.Duration
	LastSeenAt        time.Time
	LastResponseAt    time.Time
	InFlightRequests  int64
	CompletedRequests int64
	TotalInputTokens  int64
	TotalOutputTokens int64
	TotalCachedTokens int64
	Endpoint          string
	ContextTokens     int64
	InFlightTokens    int64
}

// ContextSize returns the estimated context token footprint for the session.
// Each turn resends the conversation history, so the committed context total and
// the in-flight estimates overlap and the larger value is taken.
func (s SessionState) ContextSize() int64 {
	return max(s.ContextTokens, s.InFlightTokens)
}

// EstimatePromptTokens converts a raw request size in bytes to a prompt token
// estimate, returning at least 1 token.
func EstimatePromptTokens(requestSizeBytes int) int64 {
	return max(int64(max(requestSizeBytes, 0)/fwkrequest.BytesPerToken), 1)
}

// EndpointSessions is the list of sessions bound to an endpoint, stored as a
// dynamic endpoint attribute.
type EndpointSessions []SessionState

// Clone returns an independent copy of the slice.
func (s EndpointSessions) Clone() fwkdl.Cloneable {
	return slices.Clone(s)
}

// ReadSessionState returns the SessionState published by the default producer
// on the request attribute store, or the zero value and false if absent.
func ReadSessionState(r *fwksched.InferenceRequest) (SessionState, bool) {
	key := SessionStateDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	return fwksched.ReadRequestAttribute[SessionState](r, key)
}

// ReadEndpointSessions returns the sessions bound to ep published by the default
// producer, or nil and false if absent.
func ReadEndpointSessions(ep fwkdl.Endpoint) ([]SessionState, bool) {
	if ep == nil {
		return nil, false
	}
	attrs := ep.GetAttributes()
	if attrs == nil {
		return nil, false
	}
	key := EndpointSessionsDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	sessions, ok := fwkdl.ReadAttribute[EndpointSessions](attrs, key)
	if !ok {
		return nil, false
	}
	return []SessionState(sessions), true
}
