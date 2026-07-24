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

package plugin

// StabilityLevel defines the stability level of a plugin (Alpha, Beta, or Stable).
type StabilityLevel string

const (
	// StabilityAlpha indicates an experimental plugin with no backwards-compatibility guarantees.
	StabilityAlpha StabilityLevel = "Alpha"
	// StabilityBeta indicates a feature-complete plugin following the +2 minor release deprecation policy.
	StabilityBeta StabilityLevel = "Beta"
	// StabilityStable indicates a production-grade plugin guaranteed to be backwards compatible.
	StabilityStable StabilityLevel = "Stable"
)

// ExperimentalPluginsFeatureGate is the feature gate name used to enable Alpha plugins.
const ExperimentalPluginsFeatureGate = "experimentalPlugins"

// PluginMetadata holds stability and lifecycle metadata for a registered plugin type.
type PluginMetadata struct {
	Type               string
	Stability          StabilityLevel
	Deprecated         bool
	DeprecatedIn       string
	ScheduledRemovalIn string
	ReplacementType    string
}
