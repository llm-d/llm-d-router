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

package plugin

// StateStore is a pluggable key-value store for cross-plugin state sharing.
// Plugins write endpoint-scoped state here; a sync plugin can swap the
// implementation to aggregate state across replicas (e.g. via Redis).
type StateStore interface {
	Get(key StateKey, id string) (any, bool)
	Set(key StateKey, id string, value any)
	Delete(key StateKey, id string)
}
