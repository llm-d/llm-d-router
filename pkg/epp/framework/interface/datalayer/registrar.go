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

import "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

// Registrar accepts pending source/extractor dependency declarations from plugins.
// Runtime.Configure() resolves them after processing user config.
// Register returns an error if the registration is invalid (e.g. nil Extractor).
type Registrar interface {
	Register(PendingRegistration) error
}

// Registrant is an optional interface any Plugin may implement to declare its
// datalayer dependencies. The runner calls RegisterDependencies on all eligible
// plugins after instantiation, before Runtime.Configure().
type Registrant interface {
	RegisterDependencies(Registrar) error
}

// PendingRegistration describes a (source-type, extractor) dependency.
// Extractor must be non-nil; registering a DataSource without an Extractor is not supported.
// If DefaultSource is nil, IfMissing governs behavior when no matching source exists.
// If DefaultSource is non-nil, it is registered as a new source when no match is found;
// for NotificationSources its GVK() narrows the match.
type PendingRegistration struct {
	Owner         plugin.TypedName // registering plugin; used as map key + error context
	SourceType    string           // TypedName.Type to match, e.g. "endpoint-notification-source"
	Extractor     ExtractorBase    // required; extractor to wire to the matched source
	DefaultSource DataSource       // nil → no auto-create; non-nil → create if absent
	IfMissing     MissingPolicy    // applies only when DefaultSource is nil
}

// MissingPolicy controls behavior when a required source type is absent from the config.
type MissingPolicy int

const (
	// Fail returns an error if the required source type is not configured (default).
	Fail MissingPolicy = iota
	// Warn logs a warning and skips wiring; the plugin degrades gracefully.
	Warn
)
