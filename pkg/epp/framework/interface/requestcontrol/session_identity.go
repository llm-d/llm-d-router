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

package requestcontrol

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// IdentitySource identifies the input or derivation source of a SessionIdentity.
type IdentitySource string

const (
	// IdentitySourceAgentIdentityAttribute means the identity was derived from
	// the request's agent-identity attribute.
	IdentitySourceAgentIdentityAttribute IdentitySource = "AgentIdentityAttribute"
)

// SessionIdentity is the neutral, scoped identity shared by session consumers.
// SessionTag and ScopeVersion are opaque; neither contains the external alias.
type SessionIdentity struct {
	SessionTag     string
	IdentitySource IdentitySource
	ScopeVersion   string
}

// Valid reports whether all v1 identity fields are populated and recognized.
func (i SessionIdentity) Valid() bool {
	return i.SessionTag != "" &&
		i.IdentitySource == IdentitySourceAgentIdentityAttribute &&
		i.ScopeVersion != ""
}

// SessionIdentityDataKey has no default producer. Consumers must name the
// configured session-manager instance explicitly.
var SessionIdentityDataKey = plugin.NewDataKey("SessionIdentity", "")

// ReadSessionIdentity reads and validates identity from a named producer.
func ReadSessionIdentity(request *scheduling.InferenceRequest, producerName string) (SessionIdentity, bool) {
	if request == nil || producerName == "" {
		return SessionIdentity{}, false
	}
	identity, ok := scheduling.ReadRequestAttribute[SessionIdentity](
		request,
		SessionIdentityDataKey.WithNonEmptyProducerName(producerName),
	)
	return identity, ok && identity.Valid()
}
