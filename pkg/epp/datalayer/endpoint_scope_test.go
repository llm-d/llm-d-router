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
	"context"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type cloneableStr string

func (s cloneableStr) Clone() fwkdl.Cloneable { return s }

var (
	producedKey   = fwkplugin.NewDataKey("produced", "testPlugin")
	consumedKey   = fwkplugin.NewDataKey("consumed", "otherPlugin")
	undeclaredKey = fwkplugin.NewDataKey("undeclared", "otherPlugin")
)

// testPlugin implements Plugin plus whichever of Produces/Consumes the test
// populates, mirroring how real plugins opt into each role.
type testPlugin struct {
	produces map[fwkplugin.DataKey]any
	consumes *fwkplugin.DataDependencies
}

func (p *testPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "testPlugin", Name: "mock"}
}

type producerPlugin struct{ testPlugin }

func (p *producerPlugin) Produces() map[fwkplugin.DataKey]any { return p.produces }

type producerConsumerPlugin struct{ producerPlugin }

func (p *producerConsumerPlugin) Consumes() fwkplugin.DataDependencies { return *p.consumes }

func newEndpoint(t *testing.T) fwksched.Endpoint {
	t.Helper()
	attrs := fwkdl.NewAttributes()
	attrs.Put(consumedKey, cloneableStr("consumed-value"))
	attrs.Put(undeclaredKey, cloneableStr("secret"))
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "ns", Name: "ep"}},
		&fwkdl.Metrics{},
		attrs,
	)
}

func testLogger() logr.Logger { return log.FromContext(context.Background()) }

func scopeProducerConsumer(t *testing.T, endpoint fwksched.Endpoint) (fwksched.Endpoint, *Violations) {
	t.Helper()
	plug := &producerConsumerPlugin{}
	plug.produces = map[fwkplugin.DataKey]any{producedKey: nil}
	plug.consumes = &fwkplugin.DataDependencies{Optional: map[fwkplugin.DataKey]any{consumedKey: nil}}
	scoped, violation := Scope(testLogger(), "test-extension-point", plug, []fwksched.Endpoint{endpoint})
	require.Len(t, scoped, 1)
	return scoped[0], violation
}

func TestScope_PutHonoursProduces(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, violation := scopeProducerConsumer(t, endpoint)

	scoped.Put(producedKey, cloneableStr("written"))
	got, ok := endpoint.Get(producedKey)
	assert.True(t, ok, "a declared write must reach the endpoint")
	assert.Equal(t, cloneableStr("written"), got)
	assert.NoError(t, violation.Write())
}

func TestScope_PutOutsideProducesIsRejectedAndReported(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, violation := scopeProducerConsumer(t, endpoint)

	scoped.Put(undeclaredKey, cloneableStr("overwritten"))

	value, ok := endpoint.Get(undeclaredKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("secret"), value, "an undeclared write must not overwrite another plugin's attribute")
	require.Error(t, violation.Write(), "an undeclared write must be reportable by the caller")
	assert.Contains(t, violation.Write().Error(), "Produces()")
}

// Every wrapper from one Scope call shares its Violations, so a plugin that
// misreaches on each of N candidates is reported once rather than N times.
func TestScope_ViolationsAreSharedAndReportTheFirstWrite(t *testing.T) {
	endpoints := []fwksched.Endpoint{newEndpoint(t), newEndpoint(t), newEndpoint(t)}

	plug := &producerPlugin{}
	plug.produces = map[fwkplugin.DataKey]any{producedKey: nil}
	scoped, violations := Scope(testLogger(), "test-extension-point", plug, endpoints)

	first := fwkplugin.NewDataKey("first-undeclared", "otherPlugin")
	second := fwkplugin.NewDataKey("second-undeclared", "otherPlugin")
	for _, endpoint := range scoped {
		endpoint.Put(first, cloneableStr("a"))
		endpoint.Put(second, cloneableStr("b"))
	}

	require.Error(t, violations.Write())
	assert.Contains(t, violations.Write().Error(), first.String(),
		"the reported violation must be the first one, not the last")
}

// Reads have no error channel, so a denied read must leave Write() nil while
// still resolving as absent.
func TestScope_DeniedReadDoesNotBecomeAWriteViolation(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, violations := scopeProducerConsumer(t, endpoint)

	_, ok := scoped.Get(undeclaredKey)
	assert.False(t, ok)
	assert.NoError(t, violations.Write(), "a denied read must not be reported as a rejected write")
}

func TestScope_GetHonoursConsumesAndProduces(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, _ := scopeProducerConsumer(t, endpoint)

	got, ok := scoped.Get(consumedKey)
	assert.True(t, ok, "a declared read must resolve")
	assert.Equal(t, cloneableStr("consumed-value"), got)

	scoped.Put(producedKey, cloneableStr("self"))
	got, ok = scoped.Get(producedKey)
	assert.True(t, ok, "a producer must be able to read back its own output")
	assert.Equal(t, cloneableStr("self"), got)

	got, ok = scoped.Get(undeclaredKey)
	assert.False(t, ok, "an undeclared read must resolve as absent")
	assert.Nil(t, got)
}

// A plugin that declares nothing exchanges no data, so it reaches nothing --
// the case that previously fell through to unrestricted reads.
func TestScope_PluginDeclaringNothingReachesNothing(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, violation := Scope(testLogger(), "test-extension-point", &testPlugin{}, []fwksched.Endpoint{endpoint})
	require.Len(t, scoped, 1)

	_, ok := scoped[0].Get(consumedKey)
	assert.False(t, ok)

	scoped[0].Put(producedKey, cloneableStr("nope"))
	_, ok = endpoint.Get(producedKey)
	assert.False(t, ok)
	assert.Error(t, violation.Write())
}

// A producer that declares no Consumes still reads nothing beyond its own
// output, rather than being exempted from the scope.
func TestScope_ProducerWithoutConsumesReadsOnlyItsOwnOutput(t *testing.T) {
	endpoint := newEndpoint(t)
	plug := &producerPlugin{}
	plug.produces = map[fwkplugin.DataKey]any{producedKey: nil}

	scoped, _ := Scope(testLogger(), "test-extension-point", plug, []fwksched.Endpoint{endpoint})

	_, ok := scoped[0].Get(consumedKey)
	assert.False(t, ok, "a producer that consumes nothing must not read another plugin's attribute")

	scoped[0].Put(producedKey, cloneableStr("own"))
	got, ok := scoped[0].Get(producedKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("own"), got)
}

func TestScope_KeysAndCloneCannotEscapeTheScope(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, _ := scopeProducerConsumer(t, endpoint)

	assert.Equal(t, []fwkplugin.DataKey{consumedKey}, scoped.Keys(),
		"enumeration must not reveal an attribute the plugin may not read")

	clone := scoped.Clone()
	_, ok := clone.Get(undeclaredKey)
	assert.False(t, ok, "cloning must not hand out an attribute the plugin may not read")
	value, ok := clone.Get(consumedKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("consumed-value"), value)
}

func TestScope_PassesThroughIdentityAndMetadata(t *testing.T) {
	endpoint := newEndpoint(t)
	scoped, _ := scopeProducerConsumer(t, endpoint)

	assert.Equal(t, endpoint.GetMetadata(), scoped.GetMetadata())
	assert.Equal(t, endpoint.GetMetrics(), scoped.GetMetrics())
	assert.Equal(t, endpoint.String(), scoped.String())
	assert.Equal(t, endpoint, scoped.(*ScopedEndpoint).Unwrap())
}

// The scheduler sums scorer results in a map keyed by endpoint, so a scope that
// survived the round trip would split one endpoint's score across entries.
func TestUnscope_RestoresEndpointIdentity(t *testing.T) {
	first, second := newEndpoint(t), newEndpoint(t)
	endpoints := []fwksched.Endpoint{first, second}

	plug := &producerPlugin{}
	plug.produces = map[fwkplugin.DataKey]any{producedKey: nil}
	scopedA, _ := Scope(testLogger(), "test-extension-point", plug, endpoints)
	scopedB, _ := Scope(testLogger(), "test-extension-point", plug, endpoints)

	assert.Equal(t, endpoints, Unscope(scopedA))

	totals := map[fwksched.Endpoint]float64{}
	for endpoint, score := range UnscopeScores(map[fwksched.Endpoint]float64{scopedA[0]: 1, scopedA[1]: 2}) {
		totals[endpoint] += score
	}
	for endpoint, score := range UnscopeScores(map[fwksched.Endpoint]float64{scopedB[0]: 10, scopedB[1]: 20}) {
		totals[endpoint] += score
	}

	require.Len(t, totals, 2, "scores from two scorers must land on the same two endpoints")
	assert.Equal(t, 11.0, totals[first])
	assert.Equal(t, 22.0, totals[second])
}

func TestUnscope_LeavesUnwrappedEndpointsAlone(t *testing.T) {
	endpoints := []fwksched.Endpoint{newEndpoint(t)}
	assert.Equal(t, endpoints, Unscope(endpoints))
}
