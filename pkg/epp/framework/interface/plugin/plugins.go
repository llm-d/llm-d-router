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

// Plugin defines the interface for a plugin.
// This interface should be embedded in all plugins across the code.
type Plugin interface {
	// TypedName returns the type and name tuple of this plugin instance.
	TypedName() TypedName
}

// DataDependencies holds the data keys a plugin consumes, split by whether they
// are required (framework errors if no producer exists) or optional (framework
// logs a warning but continues if no producer exists).
type DataDependencies struct {
	// Required keys — the framework will error at init time if no producer exists for any of these.
	Required map[DataKey]any
	// Optional keys — the framework logs a warning at init time but does NOT error if no producer exists.
	// The plugin must handle the case where this data is absent at runtime.
	Optional map[DataKey]any
}

// ConsumerPlugin defines the interface for a consumer.
type ConsumerPlugin interface {
	Plugin
	// Consumes returns the data keys consumed by this plugin, split into Required and Optional.
	// Required keys: the framework errors at init time if no producer exists.
	// Optional keys: the framework logs a warning but does not error; the plugin must handle absence.
	Consumes() DataDependencies
}

// ProducerPlugin defines the interface for a producer.
type ProducerPlugin interface {
	Plugin
	// Produces returns data produced by the producer.
	// This is a map from DataKey produced to
	// the data type of the key (represented as data with default value casted as any field).
	Produces() map[DataKey]any
}
