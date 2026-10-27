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
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type chainProducer struct {
	name     string
	produces fwkplugin.DataKey
	requires []fwkplugin.DataKey
	err      error
	ran      bool
}

func (p *chainProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "chain", Name: p.name}
}

func (p *chainProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.produces: 0}
}

func (p *chainProducer) Consumes() fwkplugin.DataDependencies {
	required := map[fwkplugin.DataKey]any{}
	for _, key := range p.requires {
		required[key] = 0
	}
	return fwkplugin.DataDependencies{Required: required}
}

func (p *chainProducer) Produce(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) error {
	p.ran = true
	return p.err
}

// A failing producer skips only the producers that require its output, directly
// or through another skipped producer; independent ones still run.
func TestRunDataProducerPluginsContinuesPastFailure(t *testing.T) {
	keyA, keyB, keyC := fwkplugin.NewDataKey("a", "a"), fwkplugin.NewDataKey("b", "b"), fwkplugin.NewDataKey("c", "c")
	a := &chainProducer{name: "a", produces: keyA, err: errors.New("unsupported body")}
	b := &chainProducer{name: "b", produces: keyB}
	c := &chainProducer{name: "c", produces: keyC, requires: []fwkplugin.DataKey{keyA}}
	d := &chainProducer{name: "d", produces: fwkplugin.NewDataKey("d", "d"), requires: []fwkplugin.DataKey{keyC}}
	e := &chainProducer{name: "e", produces: fwkplugin.NewDataKey("e", "e"), requires: []fwkplugin.DataKey{keyB}}
	director := &Director{requestControlPlugins: Config{dataProducerPlugins: []fwkrc.DataProducer{a, b, c, d, e}}}

	err := director.runDataProducerPlugins(context.Background(), &fwksched.InferenceRequest{}, nil)

	require.Error(t, err)
	assert.ErrorContains(t, err, `DataProducer "a/chain" failed: unsupported body`)
	assert.ErrorContains(t, err, `DataProducer "c/chain" skipped`)
	assert.ErrorContains(t, err, `DataProducer "d/chain" skipped`)
	assert.True(t, a.ran)
	assert.True(t, b.ran, "b does not depend on a")
	assert.False(t, c.ran, "c requires a's output")
	assert.False(t, d.ran, "d requires the output of c, which was skipped")
	assert.True(t, e.ran, "e requires only b's output")
}

func TestRunDataProducerPluginsWithoutFailures(t *testing.T) {
	keyA := fwkplugin.NewDataKey("a", "a")
	a := &chainProducer{name: "a", produces: keyA}
	b := &chainProducer{name: "b", produces: fwkplugin.NewDataKey("b", "b"), requires: []fwkplugin.DataKey{keyA}}
	director := &Director{requestControlPlugins: Config{dataProducerPlugins: []fwkrc.DataProducer{a, b}}}

	require.NoError(t, director.runDataProducerPlugins(context.Background(), &fwksched.InferenceRequest{}, nil))
	assert.True(t, a.ran)
	assert.True(t, b.ran)
}
