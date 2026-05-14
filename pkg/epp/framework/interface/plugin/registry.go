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
	"bytes"
	"encoding/json"
)

// Factory is the definition of the factory functions that are used to instantiate plugins
// specified in a configuration. The framework provides a strict decoder
// (DisallowUnknownFields) over the plugin's raw parameters, or nil when the plugin was
// instantiated without parameters (e.g., as a default producer). Factories that ignore
// parameters can take the decoder as `_ *json.Decoder`.
type FactoryFunc func(name string, parameters *json.Decoder, handle Handle) (Plugin, error)

// StrictDecoder returns a *json.Decoder configured with DisallowUnknownFields over the
// given raw plugin parameters, or nil when raw is empty. The framework uses this when
// invoking factories so each plugin gets uniform strict parsing; tests use it to
// construct factory arguments without duplicating the decoder boilerplate.
func StrictDecoder(raw json.RawMessage) *json.Decoder {
	if len(raw) == 0 {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	return dec
}

// Register is a static function that can be called to register plugin factory functions.
func Register(pluginType string, factory FactoryFunc) {
	Registry[pluginType] = factory
}

// RegisterAsDefaultProducer registers a factory for the given plugin type and records it as the
// default producer for the given data key. Only one producer may be registered as default per key.
// Out-of-tree projects that extend the EPP can call this to make their producers eligible for
// auto-configuration alongside in-tree producers.
func RegisterAsDefaultProducer(pluginType string, factory FactoryFunc, key string) {
	Register(pluginType, factory)
	DefaultProducerRegistry[key] = pluginType
}

// Registry is a mapping from plugin type to Factory function.
var Registry = map[string]FactoryFunc{}

// DefaultProducerRegistry maps a data key to the plugin type that is the default producer for it.
// Populated via RegisterAsDefaultProducer.
var DefaultProducerRegistry = map[string]string{}
