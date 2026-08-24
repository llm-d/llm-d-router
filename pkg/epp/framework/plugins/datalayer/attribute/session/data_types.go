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

// Package session declares the SessionID attribute that carries per-request
// session identity for affinity scoring and filtering. The value is published
// once per request on the InferenceRequest attribute store.
package session

import (
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid/constants"
	sessionstateconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionstate/constants"
)

// SessionIDDataKey identifies the session identifier published on the request
// attribute store. The default producer is the session-id-producer.
var SessionIDDataKey = plugin.NewDataKey("SessionIDDataKey", sessionidconstants.SessionIDProducerType)

// SessionStateDataKey identifies session history published on the request
// attribute store. The default producer is the session-state-producer.
var SessionStateDataKey = plugin.NewDataKey("SessionStateDataKey", sessionstateconstants.SessionStateProducerType)

// SessionID is the session identifier extracted from a request.
type SessionID string

// SessionState is the session history available before the current request is
// dispatched.
type SessionState struct {
	TurnsTaken int64
	Duration   time.Duration
	LastSeenAt time.Time
}

// ReadSessionID returns the SessionID published by the default producer on the
// request attribute store, or "" and false if absent.
//
// Consumers should use this helper rather than reading the attribute directly:
// it encapsulates both the key construction and the type assertion, so a
// future change of storage location or value type does not ripple through
// every reader.
func ReadSessionID(r *fwksched.InferenceRequest) (SessionID, bool) {
	key := SessionIDDataKey.WithNonEmptyProducerName(sessionidconstants.SessionIDProducerType)
	return fwksched.ReadRequestAttribute[SessionID](r, key)
}

// ReadSessionState returns the SessionState published by the default producer
// on the request attribute store, or the zero value and false if absent.
func ReadSessionState(r *fwksched.InferenceRequest) (SessionState, bool) {
	key := SessionStateDataKey.WithNonEmptyProducerName(sessionstateconstants.SessionStateProducerType)
	return fwksched.ReadRequestAttribute[SessionState](r, key)
}
