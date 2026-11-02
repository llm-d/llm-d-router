/*
Copyright 2025 The Kubernetes Authors.
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

package plugin

import (
	"encoding/json"
	"fmt"
	"sort"
)

// Plugin defines the interface for a plugin.
// This interface should be embedded in all plugins across the code.
type Plugin interface {
	// TypedName returns the type and name tuple of this plugin instance.
	TypedName() TypedName
}

// ReadinessChecker is an optional interface for plugins whose asynchronous
// initialization must complete before the EPP can safely receive traffic.
type ReadinessChecker interface {
	Plugin
	// CheckReady returns nil when the plugin is ready. It must be fast and
	// non-blocking, safe for concurrent calls, and report cached state rather
	// than perform external I/O.
	CheckReady() error
}

// DataDependencies holds the data keys a plugin consumes, split by whether they
// are required (framework errors if no producer exists) or optional (framework
// logs a warning but continues if no producer exists).
//
// The declared dependencies give every requestcontrol extension point a
// deterministic order: a plugin runs after the plugins producing the keys it
// consumes. Producer-before-consumer is the intended semantic on RequestHeader,
// Screen, Admit, Produce and PreRequest. ResponseHeader and ResponseBody are
// ordered the same way for determinism, but the correct direction there is
// unsettled - a producer may release consumed state in a response hook - so
// response-hook plugins must not rely on it.
type DataDependencies struct {
	// Required keys — the framework will error at init time if no producer exists for any of these.
	Required map[DataKey]any
	// Optional keys — the framework logs a warning at init time but does NOT error if no producer exists.
	// Configured producers run before their optional consumers.
	// The plugin must handle the case where this data is absent at runtime.
	Optional map[DataKey]any
}

// StateDumper is an optional interface for plugins that can expose sanitized,
// bounded internal state through operational debug endpoints.
//
// DumpState is intended for on-demand debugging snapshots of dynamic runtime
// state that is difficult to understand from metrics alone. Prefer metrics for
// numeric time series, alerting, dashboards, and aggregation over time. Dumped
// state should stay reasonably small; large or high-cardinality state should be
// summarized, capped, or omitted.
type StateDumper interface {
	// DumpState returns a JSON-encoded representation of plugin state.
	// Implementations own serialization and must not include request payloads,
	// credentials, or other sensitive values.
	DumpState() (json.RawMessage, error)
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
	// the data type of the key (represented as data with default value casted as any field,
	// or wrapped in ProducedData to advertise Level-2 configuration attributes).
	Produces() map[DataKey]any
}

// ProducedData optionally wraps a produced value-type witness with Level-2
// attributes derived from the producer's configuration.
// Plugins with no attributes may return the bare value-type witness directly.
type ProducedData struct {
	// Type is the zero-value Go type witness for Level-1 type checking (e.g., fwkrh.TokenizedRequest{}).
	Type any

	// Attributes holds key-value metadata describing the produced data (e.g., {"backend": "vllm"}).
	Attributes map[string]string

	// AllAttributes explicitly declares that this producer dynamically produces any attribute
	// requested by consumers, bypassing specific attribute key/value matching.
	// Must be set explicitly; an empty Attributes map with AllAttributes == false does NOT match all.
	AllAttributes bool
}

// ConsumedData optionally wraps a consumed value-type witness with a Level-2
// attribute matcher function.
// Plugins that accept any producer of the DataKey may return the bare value-type witness directly.
type ConsumedData struct {
	// Type is the zero-value Go type witness for Level-1 type checking.
	Type any

	// Matches validates the producer's published Attributes at startup.
	// Returning a non-nil error fails startup validation with a descriptive message.
	// If nil, any producer of the DataKey (regardless of its Attributes) is accepted.
	Matches func(attrs map[string]string) error
}

// UnwrapDataType returns the underlying Go value-type witness from v,
// unwrapping ProducedData or ConsumedData (or pointers to them) when present.
func UnwrapDataType(v any) any {
	switch d := v.(type) {
	case ProducedData:
		return d.Type
	case *ProducedData:
		if d == nil {
			return nil
		}
		return d.Type
	case ConsumedData:
		return d.Type
	case *ConsumedData:
		if d == nil {
			return nil
		}
		return d.Type
	default:
		return v
	}
}

// MatchKeys returns a Level-2 matcher that verifies all requiredKeys are present
// in the producer's Attributes map.
func MatchKeys(requiredKeys ...string) func(attrs map[string]string) error {
	return func(attrs map[string]string) error {
		for _, k := range requiredKeys {
			if _, ok := attrs[k]; !ok {
				return fmt.Errorf("producer does not declare required attribute key %q", k)
			}
		}
		return nil
	}
}

// MatchKeyValues returns a Level-2 matcher that verifies all expected key-value
// pairs are present and equal in the producer's Attributes map.
func MatchKeyValues(expected map[string]string) func(attrs map[string]string) error {
	keys := make([]string, 0, len(expected))
	for k := range expected {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return func(attrs map[string]string) error {
		for _, k := range keys {
			want := expected[k]
			got, ok := attrs[k]
			if !ok {
				return fmt.Errorf("producer does not declare required attribute %q=%q", k, want)
			}
			if got != want {
				return fmt.Errorf("producer attribute %q=%q does not match required %q", k, got, want)
			}
		}
		return nil
	}
}
