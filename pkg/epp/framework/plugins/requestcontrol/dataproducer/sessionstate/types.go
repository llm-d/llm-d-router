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
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// SessionStateProducerType is the plugin type registered with the framework.
const SessionStateProducerType = "session-state-producer"

// SessionStateDataKey identifies session history published on the request
// attribute store. The default producer is the session-state-producer.
var SessionStateDataKey = plugin.NewDataKey("SessionStateDataKey", SessionStateProducerType)

// SessionState is the session history available before the current request is
// dispatched.
type SessionState struct {
	TurnsTaken        int64
	Duration          time.Duration
	LastSeenAt        time.Time
	InFlightRequests  int64
	CompletedRequests int64
	TotalInputTokens  int64
	TotalOutputTokens int64
}

// ReadSessionState returns the SessionState published by the default producer
// on the request attribute store, or the zero value and false if absent.
func ReadSessionState(r *fwksched.InferenceRequest) (SessionState, bool) {
	key := SessionStateDataKey.WithNonEmptyProducerName(SessionStateProducerType)
	return fwksched.ReadRequestAttribute[SessionState](r, key)
}
