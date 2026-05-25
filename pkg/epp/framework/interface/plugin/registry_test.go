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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

// snapshotRegistries captures the current state of the package-level registries
// so a test can restore them and avoid polluting other tests.
func snapshotRegistries(t *testing.T) {
	t.Helper()
	origRegistry := make(map[string]FactoryFunc, len(Registry))
	for k, v := range Registry {
		origRegistry[k] = v
	}
	origDefaults := make(map[string]string, len(DefaultProducerRegistry))
	for k, v := range DefaultProducerRegistry {
		origDefaults[k] = v
	}
	t.Cleanup(func() {
		Registry = origRegistry
		DefaultProducerRegistry = origDefaults
	})
}

func dummyFactory(name string, parameters *json.Decoder, handle Handle) (Plugin, error) {
	return &basePlugin{name: TypedName{Type: "dummy", Name: name}}, nil
}

func TestRegister(t *testing.T) {
	snapshotRegistries(t)

	Register("my-plugin-type", dummyFactory)

	factory, ok := Registry["my-plugin-type"]
	assert.True(t, ok)
	assert.NotNil(t, factory)

	p, err := factory("instance-a", nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "instance-a/dummy", p.TypedName().String())
}

func TestRegister_Overwrites(t *testing.T) {
	snapshotRegistries(t)

	Register("dup-type", dummyFactory)

	var sentinel Plugin = &basePlugin{name: TypedName{Type: "sentinel", Name: "sentinel"}}
	Register("dup-type", func(name string, parameters *json.Decoder, handle Handle) (Plugin, error) {
		return sentinel, nil
	})

	p, err := Registry["dup-type"]("ignored", nil, nil)
	assert.NoError(t, err)
	assert.Same(t, sentinel, p)
}

func TestRegisterAsDefaultProducer(t *testing.T) {
	snapshotRegistries(t)

	key := NewDataKey("metric.cache", "default-scraper")

	RegisterAsDefaultProducer("cache-scraper", dummyFactory, key)

	_, ok := Registry["cache-scraper"]
	assert.True(t, ok, "factory should be registered")

	pluginType, ok := DefaultProducerRegistry[key.String()]
	assert.True(t, ok, "default producer should be recorded")
	assert.Equal(t, "cache-scraper", pluginType)
}

func TestRegisterAsDefaultProducer_OverwritesDefault(t *testing.T) {
	snapshotRegistries(t)

	key := NewDataKey("metric.queue", "scraper-v1")

	RegisterAsDefaultProducer("scraper-v1", dummyFactory, key)
	RegisterAsDefaultProducer("scraper-v2", dummyFactory, key)

	assert.Equal(t, "scraper-v2", DefaultProducerRegistry[key.String()])
	_, ok := Registry["scraper-v1"]
	assert.True(t, ok, "v1 factory should still be registered")
	_, ok = Registry["scraper-v2"]
	assert.True(t, ok)
}

func TestRegistry_IsolatedBetweenTests(t *testing.T) {
	snapshotRegistries(t)

	const key = "isolation-marker"
	_, exists := Registry[key]
	assert.False(t, exists, "previous test must not have leaked into Registry")
	Register(key, dummyFactory)
	_, exists = Registry[key]
	assert.True(t, exists)
}
