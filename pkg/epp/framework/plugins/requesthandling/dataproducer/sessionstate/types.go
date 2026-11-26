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

// EndpointSessions is a read-only view of sessions currently bound to an
// endpoint, published as an endpoint attribute under EndpointSessionsDataKey.
type EndpointSessions struct {
	bucket *endpointBucket
	static []SessionState
}

// NewEndpointSessions builds an EndpointSessions view over a copy of sessions.
func NewEndpointSessions(sessions []SessionState) EndpointSessions {
	if len(sessions) == 0 {
		return EndpointSessions{}
	}
	return EndpointSessions{static: slices.Clone(sessions)}
}

// Clone returns an EndpointSessions value safe for independent storage.
func (s EndpointSessions) Clone() fwkdl.Cloneable {
	if s.bucket != nil {
		return s
	}
	if len(s.static) == 0 {
		return EndpointSessions{}
	}
	return EndpointSessions{static: slices.Clone(s.static)}
}

// TotalContextTokens returns the sum of ContextSize() across sessions currently
// bound to the endpoint.
func (s EndpointSessions) TotalContextTokens() int64 {
	if s.bucket != nil {
		if !s.bucket.alive.Load() {
			return 0
		}
		return s.bucket.totalTokens.Load()
	}
	var total int64
	for i := range s.static {
		total += s.static[i].ContextSize()
	}
	return total
}

// Session returns the current SessionState of id if bound to this endpoint.
func (s EndpointSessions) Session(id string) (SessionState, bool) {
	if id == "" {
		return SessionState{}, false
	}
	if s.bucket != nil {
		return s.bucket.session(id)
	}
	for i := range s.static {
		if s.static[i].SessionID == id {
			return s.static[i], true
		}
	}
	return SessionState{}, false
}

// Sessions returns a read-only slice of sessions currently bound to the
// endpoint; Duration and LastSeenAt reflect when the snapshot was populated.
// Callers must not mutate the returned slice.
func (s EndpointSessions) Sessions() []SessionState {
	if s.bucket != nil {
		return s.bucket.sessions()
	}
	return s.static
}

type attributePeeker interface {
	Peek(plugin.DataKey) (fwkdl.Cloneable, bool)
}

// ReadSessionState returns the SessionState published by the default producer
// on the request attribute store, or the zero value and false if absent.
func ReadSessionState(r *fwksched.InferenceRequest) (SessionState, bool) {
	return fwksched.ReadRequestAttribute[SessionState](r, SessionStateDataKey)
}

// PeekEndpointSessions returns the read-only EndpointSessions view published by
// the default producer on ep without cloning, or the zero value and false if absent.
func PeekEndpointSessions(ep fwkdl.Endpoint) (EndpointSessions, bool) {
	if ep == nil {
		return EndpointSessions{}, false
	}
	attrs := ep.GetAttributes()
	if attrs == nil {
		return EndpointSessions{}, false
	}
	var (
		raw fwkdl.Cloneable
		ok  bool
	)
	if peeker, isPeeker := attrs.(attributePeeker); isPeeker {
		raw, ok = peeker.Peek(EndpointSessionsDataKey)
	} else {
		raw, ok = attrs.Get(EndpointSessionsDataKey)
	}
	if !ok {
		return EndpointSessions{}, false
	}
	view, ok := raw.(EndpointSessions)
	return view, ok
}
