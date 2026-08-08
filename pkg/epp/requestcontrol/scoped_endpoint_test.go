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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type cloneableStr string

func (s cloneableStr) Clone() fwkdl.Cloneable { return s }

var (
	declaredKey   = fwkplugin.NewDataKey("declared", "testPlugin")
	undeclaredKey = fwkplugin.NewDataKey("undeclared", "testPlugin")
	consumedKey   = fwkplugin.NewDataKey("consumed", "otherPlugin")
)

func newTestEndpoint(t *testing.T) fwksched.Endpoint {
	t.Helper()
	attrs := fwkdl.NewAttributes()
	attrs.Put(consumedKey, cloneableStr("existing"))
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: types.NamespacedName{Namespace: "ns", Name: "test-ep"}},
		&fwkdl.Metrics{},
		attrs,
	)
}

func testLogger() logr.Logger {
	return log.FromContext(context.Background())
}

func TestScopedEndpoint_PutAllowedKey(t *testing.T) {
	ep := newTestEndpoint(t)
	scoped := &scopedEndpoint{
		inner:      ep,
		allowedPut: map[fwkplugin.DataKey]struct{}{declaredKey: {}},
		pluginName: "testPlugin/mock",
		logger:     testLogger(),
	}

	scoped.Put(declaredKey, cloneableStr("value"))

	got, ok := ep.Get(declaredKey)
	assert.True(t, ok, "declared key should be written to underlying endpoint")
	assert.Equal(t, cloneableStr("value"), got)
}

func TestScopedEndpoint_PutUndeclaredKeyBlocked(t *testing.T) {
	ep := newTestEndpoint(t)
	scoped := &scopedEndpoint{
		inner:      ep,
		allowedPut: map[fwkplugin.DataKey]struct{}{declaredKey: {}},
		pluginName: "testPlugin/mock",
		logger:     testLogger(),
	}

	scoped.Put(undeclaredKey, cloneableStr("bad"))

	_, ok := ep.Get(undeclaredKey)
	assert.False(t, ok, "undeclared key should NOT be written to underlying endpoint")
}

func TestScopedEndpoint_GetAllowedKey(t *testing.T) {
	ep := newTestEndpoint(t)
	scoped := &scopedEndpoint{
		inner:      ep,
		allowedPut: map[fwkplugin.DataKey]struct{}{declaredKey: {}},
		allowedGet: map[fwkplugin.DataKey]struct{}{consumedKey: {}},
		pluginName: "testPlugin/mock",
		logger:     testLogger(),
	}

	got, ok := scoped.Get(consumedKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("existing"), got)
}

func TestScopedEndpoint_GetUndeclaredKeyBlocked(t *testing.T) {
	ep := newTestEndpoint(t)
	scoped := &scopedEndpoint{
		inner:      ep,
		allowedPut: map[fwkplugin.DataKey]struct{}{declaredKey: {}},
		allowedGet: map[fwkplugin.DataKey]struct{}{declaredKey: {}},
		pluginName: "testPlugin/mock",
		logger:     testLogger(),
	}

	got, ok := scoped.Get(consumedKey)
	assert.False(t, ok, "undeclared Get key should be blocked")
	assert.Nil(t, got)
}

func TestScopedEndpoint_GetWithNilAllowedGetPermitsAll(t *testing.T) {
	ep := newTestEndpoint(t)
	scoped := &scopedEndpoint{
		inner:      ep,
		allowedPut: map[fwkplugin.DataKey]struct{}{declaredKey: {}},
		allowedGet: nil,
		pluginName: "testPlugin/mock",
		logger:     testLogger(),
	}

	got, ok := scoped.Get(consumedKey)
	assert.True(t, ok, "nil allowedGet should permit all reads")
	assert.Equal(t, cloneableStr("existing"), got)
}

// producerOnlyPlugin implements ProducerPlugin but NOT ConsumerPlugin.
type producerOnlyPlugin struct {
	executorMockDataProducerPlugin
	produces map[fwkplugin.DataKey]any
}

func (p *producerOnlyPlugin) Produces() map[fwkplugin.DataKey]any { return p.produces }

func TestScopeEndpoints_ProducerOnly(t *testing.T) {
	ep := newTestEndpoint(t)

	p := &producerOnlyPlugin{
		executorMockDataProducerPlugin: executorMockDataProducerPlugin{name: "prod"},
		produces:                       map[fwkplugin.DataKey]any{declaredKey: nil},
	}

	scoped := scopeEndpoints(testLogger(), p, []fwksched.Endpoint{ep})
	assert.Len(t, scoped, 1)

	// Put with declared key succeeds.
	scoped[0].Put(declaredKey, cloneableStr("ok"))
	got, ok := ep.Get(declaredKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("ok"), got)

	// Put with undeclared key is blocked.
	scoped[0].Put(undeclaredKey, cloneableStr("bad"))
	_, ok = ep.Get(undeclaredKey)
	assert.False(t, ok)

	// Get is unrestricted when plugin does not implement ConsumerPlugin.
	got, ok = scoped[0].Get(consumedKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("existing"), got)
}

func TestScopeEndpoints_ProducerAndConsumer(t *testing.T) {
	ep := newTestEndpoint(t)

	p := &dagTestPlugin{
		executorMockDataProducerPlugin: executorMockDataProducerPlugin{name: "both"},
		produces:                       map[fwkplugin.DataKey]any{declaredKey: nil},
		consumes:                       map[fwkplugin.DataKey]any{consumedKey: nil},
	}

	scoped := scopeEndpoints(testLogger(), p, []fwksched.Endpoint{ep})

	// Get with declared consumed key works.
	got, ok := scoped[0].Get(consumedKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("existing"), got)

	// Get with own produced key works (producers may read their own output).
	scoped[0].Put(declaredKey, cloneableStr("self"))
	got, ok = scoped[0].Get(declaredKey)
	assert.True(t, ok)
	assert.Equal(t, cloneableStr("self"), got)

	// Get with undeclared key is blocked.
	got, ok = scoped[0].Get(undeclaredKey)
	assert.False(t, ok)
	assert.Nil(t, got)
}

func TestScopeEndpoints_DelegatesToInner(t *testing.T) {
	ep := newTestEndpoint(t)
	p := &dagTestPlugin{
		executorMockDataProducerPlugin: executorMockDataProducerPlugin{name: "check"},
		produces:                       map[fwkplugin.DataKey]any{declaredKey: nil},
	}

	scoped := scopeEndpoints(testLogger(), p, []fwksched.Endpoint{ep})
	se := scoped[0].(*scopedEndpoint)

	assert.Equal(t, ep.GetMetadata(), se.GetMetadata())
	assert.Equal(t, ep.GetMetrics(), se.GetMetrics())
	assert.Equal(t, ep.String(), se.String())
	assert.Equal(t, ep, se.Inner())
}
