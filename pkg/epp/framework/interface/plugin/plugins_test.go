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

package plugin

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type basePlugin struct {
	name TypedName
}

func (b *basePlugin) TypedName() TypedName { return b.name }

type consumerImpl struct {
	basePlugin
	consumes map[DataKey]any
}

func (c *consumerImpl) Consumes() map[DataKey]any { return c.consumes }

type producerImpl struct {
	basePlugin
	produces map[DataKey]any
}

func (p *producerImpl) Produces() map[DataKey]any { return p.produces }

func TestPlugin_TypedNameContract(t *testing.T) {
	var p Plugin = &basePlugin{name: TypedName{Type: "scorer", Name: "kv-cache"}}
	assert.Equal(t, "kv-cache/scorer", p.TypedName().String())
}

func TestConsumerPlugin_Contract(t *testing.T) {
	key := NewDataKey("metric.queue", "scraper")
	c := &consumerImpl{
		basePlugin: basePlugin{name: TypedName{Type: "filter", Name: "queue-filter"}},
		consumes:   map[DataKey]any{key: 0},
	}

	var _ Plugin = c
	var cp ConsumerPlugin = c

	got := cp.Consumes()
	assert.Len(t, got, 1)
	_, ok := got[key]
	assert.True(t, ok)
	assert.Equal(t, "queue-filter/filter", cp.TypedName().String())
}

func TestProducerPlugin_Contract(t *testing.T) {
	key := NewDataKey("metric.cache", "scraper")
	p := &producerImpl{
		basePlugin: basePlugin{name: TypedName{Type: "scraper", Name: "cache-scraper"}},
		produces:   map[DataKey]any{key: 0.0},
	}

	var _ Plugin = p
	var pp ProducerPlugin = p

	got := pp.Produces()
	assert.Len(t, got, 1)
	_, ok := got[key]
	assert.True(t, ok)
	assert.Equal(t, "cache-scraper/scraper", pp.TypedName().String())
}

func TestConsumerPlugin_EmptyConsumes(t *testing.T) {
	c := &consumerImpl{
		basePlugin: basePlugin{name: TypedName{Type: "filter", Name: "noop"}},
		consumes:   nil,
	}
	assert.Nil(t, c.Consumes())
}

func TestProducerPlugin_EmptyProduces(t *testing.T) {
	p := &producerImpl{
		basePlugin: basePlugin{name: TypedName{Type: "scraper", Name: "noop"}},
		produces:   map[DataKey]any{},
	}
	assert.Empty(t, p.Produces())
}
