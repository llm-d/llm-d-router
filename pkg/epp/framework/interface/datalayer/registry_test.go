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

package datalayer

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// regProducer is a ProducerPlugin that returns produces for every entry in
// its produces map. It satisfies plugin.Plugin so the registry walk picks it
// up without any framework wiring.
type regProducer struct {
	name     string
	produces map[plugin.DataKey]any
}

func (p *regProducer) TypedName() plugin.TypedName {
	return plugin.TypedName{Name: p.name, Type: "reg"}
}

func (p *regProducer) Produces() map[plugin.DataKey]any { return p.produces }

type regConsumer struct {
	name     string
	consumes map[plugin.DataKey]any
}

func (c *regConsumer) TypedName() plugin.TypedName {
	return plugin.TypedName{Name: c.name, Type: "reg"}
}

func (c *regConsumer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{Required: c.consumes}
}

type regMixedConsumer struct {
	name     string
	required map[plugin.DataKey]any
	optional map[plugin.DataKey]any
}

func (c *regMixedConsumer) TypedName() plugin.TypedName {
	return plugin.TypedName{Name: c.name, Type: "reg"}
}

func (c *regMixedConsumer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{Required: c.required, Optional: c.optional}
}

type regValueA struct{ X int }

func (regValueA) Clone() Cloneable { return regValueA{} }

type regValueB struct{ Y string }

func (regValueB) Clone() Cloneable { return regValueB{} }

func TestBuildRegistry_SingleProducer(t *testing.T) {
	dk := plugin.NewDataKey("alpha", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{dk: regValueA{}}}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	assert.True(t, r.Has(dk))
	assert.Equal(t, "datalayer.regValueA", r.TypeOf(dk).String())
}

func TestBuildRegistry_DuplicateKeySameTypePasses(t *testing.T) {
	// Two producers claiming the same DataKey (e.g. via the same producer
	// name) with the same declared type is consistent and passes.
	dk := plugin.NewDataKey("alpha", "reg")
	p1 := &regProducer{name: "p", produces: map[plugin.DataKey]any{dk: regValueA{}}}
	p2 := &regProducer{name: "p-again", produces: map[plugin.DataKey]any{dk: regValueA{}}}

	_, err := BuildRegistry([]plugin.Plugin{p1, p2})
	assert.NoError(t, err)
}

func TestBuildRegistry_DuplicateKeyDifferentTypeFails(t *testing.T) {
	dk := plugin.NewDataKey("alpha", "reg")
	p1 := &regProducer{name: "p", produces: map[plugin.DataKey]any{dk: regValueA{}}}
	p2 := &regProducer{name: "p-again", produces: map[plugin.DataKey]any{dk: regValueB{}}}

	_, err := BuildRegistry([]plugin.Plugin{p1, p2})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "data key")
	assert.Contains(t, err.Error(), "type")
}

func TestBuildRegistry_NilPlaceholderTreatedAsUntyped(t *testing.T) {
	dk := plugin.NewDataKey("alpha", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{dk: nil}}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	assert.True(t, r.Has(dk))
	assert.Nil(t, r.TypeOf(dk), "nil placeholder should record no type")
}

func TestBuildRegistry_NonProducerSkipped(t *testing.T) {
	// regConsumer is not a ProducerPlugin; BuildRegistry must walk past it.
	c := &regConsumer{name: "c"}
	r, err := BuildRegistry([]plugin.Plugin{c})
	assert.NoError(t, err)
	assert.Empty(t, r.Keys())
}

func TestValidateConsumer_KnownKeyMatchingType(t *testing.T) {
	dk := plugin.NewDataKey("alpha", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{dk: regValueA{}}}
	c := &regConsumer{name: "c", consumes: map[plugin.DataKey]any{dk: regValueA{}}}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	assert.NoError(t, r.ValidateConsumer(c))
}

func TestValidateConsumer_UnknownKeyFails(t *testing.T) {
	// The key the consumer references is not produced by anyone: registry
	// rejects the consumer declaration outright. This is the existence
	// half of the value-safety check.
	known := plugin.NewDataKey("alpha", "reg")
	stranger := plugin.NewDataKey("stranger", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{known: regValueA{}}}
	c := &regConsumer{name: "c", consumes: map[plugin.DataKey]any{stranger: regValueA{}}}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	err = r.ValidateConsumer(c)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not produced by any registered plugin")
	assert.Contains(t, err.Error(), stranger.String())
}

func TestValidateConsumer_TypeMismatchFails(t *testing.T) {
	// The consumer's declared value type does not match what the producer
	// declared. This is the type half of the value-safety check — catches
	// the case where a consumer expects an int but the producer writes a
	// string under the same key.
	dk := plugin.NewDataKey("alpha", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{dk: regValueA{}}}
	c := &regConsumer{name: "c", consumes: map[plugin.DataKey]any{dk: regValueB{}}}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	err = r.ValidateConsumer(c)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "type")
}

func TestValidateConsumer_OptionalKeyUnproducedSkipped(t *testing.T) {
	// An Optional key with no producer is tolerated: CreateMissingDataProducers
	// already allows a consumer to fall back at read time when an Optional
	// dependency goes unproduced, so the registry must not error here either.
	known := plugin.NewDataKey("alpha", "reg")
	stranger := plugin.NewDataKey("stranger", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{known: regValueA{}}}
	c := &regMixedConsumer{
		name:     "c",
		required: map[plugin.DataKey]any{known: regValueA{}},
		optional: map[plugin.DataKey]any{stranger: regValueA{}},
	}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	assert.NoError(t, r.ValidateConsumer(c))
}

func TestValidateConsumer_OptionalKeyProducedTypeMismatchFails(t *testing.T) {
	// An Optional key that IS produced still gets the type check: a producer
	// and consumer disagreeing on the value type is a real bug even when the
	// dependency is optional.
	known := plugin.NewDataKey("alpha", "reg")
	optKey := plugin.NewDataKey("beta", "reg")
	p := &regProducer{name: "p", produces: map[plugin.DataKey]any{known: regValueA{}, optKey: regValueA{}}}
	c := &regMixedConsumer{
		name:     "c",
		required: map[plugin.DataKey]any{known: regValueA{}},
		optional: map[plugin.DataKey]any{optKey: regValueB{}},
	}

	r, err := BuildRegistry([]plugin.Plugin{p})
	assert.NoError(t, err)
	err = r.ValidateConsumer(c)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "type")
}

func TestRegistry_NilSafe(t *testing.T) {
	var r *Registry
	assert.False(t, r.Has(plugin.NewDataKey("x", "y")))
	assert.Nil(t, r.TypeOf(plugin.NewDataKey("x", "y")))
	assert.Nil(t, r.Keys())
	assert.NoError(t, r.ValidateConsumer(&regConsumer{name: "c"}))
}
