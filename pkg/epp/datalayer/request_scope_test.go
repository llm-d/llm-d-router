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

package datalayer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func newRequestScopePlugin(name string) *producerConsumerPlugin {
	plug := &producerConsumerPlugin{}
	plug.name = name
	plug.produces = map[fwkplugin.DataKey]any{producedKey: nil}
	plug.consumes = &fwkplugin.DataDependencies{Optional: map[fwkplugin.DataKey]any{consumedKey: nil}}
	return plug
}

func TestScopeRequest_AllowsDeclaredAccess(t *testing.T) {
	plug := newRequestScopePlugin("request-declared")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	request := &fwksched.InferenceRequest{RequestID: "req"}
	request.PutAttribute(consumedKey, "consumed-value")

	scoped, violations := ScopeRequest(testLogger(), "test-extension-point", plug, request)

	value, ok := scoped.GetAttribute(consumedKey)
	assert.True(t, ok, "a key in Consumes resolves")
	assert.Equal(t, "consumed-value", value)

	// A producer may read back what it wrote.
	scoped.PutAttribute(producedKey, "produced-value")
	value, ok = scoped.GetAttribute(producedKey)
	assert.True(t, ok)
	assert.Equal(t, "produced-value", value)
	assert.NoError(t, violations.Write())

	// The write reached the store the framework owns, not just the copy.
	value, ok = request.GetAttribute(producedKey)
	assert.True(t, ok, "an allowed write is visible on the unscoped request")
	assert.Equal(t, "produced-value", value)
}

func TestScopeRequest_DropsUndeclaredWrite(t *testing.T) {
	plug := newRequestScopePlugin("request-undeclared-write")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	request := &fwksched.InferenceRequest{RequestID: "req"}
	scoped, violations := ScopeRequest(testLogger(), "test-extension-point", plug, request)

	scoped.PutAttribute(undeclaredKey, "nope")

	_, ok := request.GetAttribute(undeclaredKey)
	assert.False(t, ok, "an undeclared write must not reach the store")
	require.Error(t, violations.Write(), "the rejection is reportable by an extension point with an error return")
	assert.Contains(t, violations.Write().Error(), "add it to Produces()")
}

func TestScopeRequest_UndeclaredReadResolvesAsAbsent(t *testing.T) {
	plug := newRequestScopePlugin("request-undeclared-read")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	request := &fwksched.InferenceRequest{RequestID: "req"}
	request.PutAttribute(undeclaredKey, "secret")

	scoped, violations := ScopeRequest(testLogger(), "test-extension-point", plug, request)

	value, ok := scoped.GetAttribute(undeclaredKey)
	assert.False(t, ok, "an undeclared read resolves as absent")
	assert.Nil(t, value)
	// A read is not reportable: Filter and Score have no error return.
	assert.NoError(t, violations.Write())

	// The value is untouched for the plugin that did declare it.
	value, ok = request.GetAttribute(undeclaredKey)
	assert.True(t, ok)
	assert.Equal(t, "secret", value)
}

// Enumeration lists only the keys the plugin may read, so it cannot be used to
// discover an attribute another plugin owns.
func TestScopeRequest_AttributeKeysListsOnlyDeclaredKeys(t *testing.T) {
	plug := newRequestScopePlugin("request-keys")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	request := &fwksched.InferenceRequest{RequestID: "req"}
	request.PutAttribute(consumedKey, "consumed-value")
	request.PutAttribute(undeclaredKey, "secret")

	scoped, _ := ScopeRequest(testLogger(), "test-extension-point", plug, request)

	assert.ElementsMatch(t, []fwkplugin.DataKey{consumedKey}, scoped.AttributeKeys())
	assert.ElementsMatch(t, []fwkplugin.DataKey{consumedKey, undeclaredKey}, request.AttributeKeys())
}

// The scoped request is a shallow copy, so a lazily allocated store or header
// map must be materialized on the original before the copy is taken. A
// PreRequest plugin initializes Headers before setting one.
func TestScopeRequest_MaterializesLazyFieldsOnTheOriginal(t *testing.T) {
	plug := newRequestScopePlugin("request-lazy-fields")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	request := &fwksched.InferenceRequest{RequestID: "req"}
	require.Nil(t, request.Headers)

	scoped, _ := ScopeRequest(testLogger(), "test-extension-point", plug, request)
	require.NotNil(t, request.Headers, "Headers is materialized so a plugin's write is not lost")

	scoped.Headers["x-set-by-plugin"] = "value"
	assert.Equal(t, "value", request.Headers["x-set-by-plugin"])

	scoped.PutAttribute(producedKey, "written-through-the-copy")
	value, ok := request.GetAttribute(producedKey)
	assert.True(t, ok, "the copy shares the backing store with the original")
	assert.Equal(t, "written-through-the-copy", value)
}

// The framework's own request carries no scope and reaches the whole store,
// which is how the director reads attributes a plugin published.
func TestScopeRequest_UnscopedRequestIsUnconfined(t *testing.T) {
	request := &fwksched.InferenceRequest{RequestID: "req"}
	request.PutAttribute(undeclaredKey, "secret")

	value, ok := request.GetAttribute(undeclaredKey)
	assert.True(t, ok)
	assert.Equal(t, "secret", value)
}

// One invocation reaching outside its declarations on either store reports
// through the same Violations, so a caller with an error return fails once.
func TestScopeInvocation_SharesViolationsAcrossBothStores(t *testing.T) {
	plug := newRequestScopePlugin("invocation-shared")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	request := &fwksched.InferenceRequest{RequestID: "req"}
	endpoint := newEndpoint(t)

	scopedRequest, scopedEndpoints, violations := ScopeInvocation(
		testLogger(), "test-extension-point", plug, request, []fwksched.Endpoint{endpoint})
	require.Len(t, scopedEndpoints, 1)

	scopedRequest.PutAttribute(undeclaredKey, "nope")
	require.Error(t, violations.Write())
	first := violations.Write()

	// The endpoint half records into the same Violations, and the first
	// rejection is the one reported.
	scopedEndpoints[0].Put(undeclaredKey, cloneableStr("nope"))
	assert.Equal(t, first, violations.Write())
}

func TestScopeRequest_NilRequest(t *testing.T) {
	plug := newRequestScopePlugin("request-nil")
	RegisterScopeSpecs([]fwkplugin.Plugin{plug})

	scoped, violations := ScopeRequest(testLogger(), "test-extension-point", plug, nil)
	assert.Nil(t, scoped)
	assert.NoError(t, violations.Write())
}
