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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestSessionIdentityValid(t *testing.T) {
	t.Parallel()
	valid := SessionIdentity{
		SessionTag:     "opaque-tag",
		IdentitySource: IdentitySourceAgentIdentityAttribute,
		ScopeVersion:   "opaque-scope",
	}
	assert.True(t, valid.Valid())

	for _, invalid := range []SessionIdentity{
		{},
		{SessionTag: "tag", IdentitySource: IdentitySourceAgentIdentityAttribute},
		{SessionTag: "tag", ScopeVersion: "scope"},
		{IdentitySource: IdentitySourceAgentIdentityAttribute, ScopeVersion: "scope"},
		{SessionTag: "tag", IdentitySource: "unknown", ScopeVersion: "scope"},
	} {
		assert.False(t, invalid.Valid())
	}
}

func TestReadSessionIdentityUsesNamedProducer(t *testing.T) {
	t.Parallel()
	request := &scheduling.InferenceRequest{}
	identity := SessionIdentity{
		SessionTag:     "opaque-tag",
		IdentitySource: IdentitySourceAgentIdentityAttribute,
		ScopeVersion:   "opaque-scope",
	}
	request.PutAttribute(SessionIdentityDataKey.WithNonEmptyProducerName("sessions"), identity)

	got, ok := ReadSessionIdentity(request, "sessions")
	require.True(t, ok)
	assert.Equal(t, identity, got)

	_, ok = ReadSessionIdentity(request, "other")
	assert.False(t, ok)
	_, ok = ReadSessionIdentity(nil, "sessions")
	assert.False(t, ok)
}
