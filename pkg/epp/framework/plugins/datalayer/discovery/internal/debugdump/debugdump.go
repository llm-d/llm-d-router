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

// Package debugdump holds the DumpState contract shared by the discovery
// plugins, so every plugin reports the same payload with the same cap.
package debugdump

import (
	"encoding/json"
	"sort"

	"k8s.io/apimachinery/pkg/types"
)

// MaxEndpoints caps the endpoint identities in a dump so the debug payload
// stays bounded.
const MaxEndpoints = 100

// State is the sanitized snapshot returned by DumpState: discovered endpoint
// identities only, never their addresses or labels. The dump is partial when
// TotalEndpoints exceeds MaxEndpoints.
type State struct {
	Endpoints      []string `json:"endpoints"`
	TotalEndpoints int      `json:"totalEndpoints"`
	MaxEndpoints   int      `json:"maxEndpoints"`
}

// Names returns the identities of the endpoints map, unsorted. Callers hold
// whatever lock guards the map.
func Names[V any](endpoints map[types.NamespacedName]V) []string {
	names := make([]string, 0, len(endpoints))
	for id := range endpoints {
		names = append(names, id.String())
	}
	return names
}

// Marshal sorts names, caps them to MaxEndpoints and returns the State as
// JSON. It sorts names in place.
func Marshal(names []string) (json.RawMessage, error) {
	sort.Strings(names)
	state := State{TotalEndpoints: len(names), MaxEndpoints: MaxEndpoints}
	if len(names) > MaxEndpoints {
		names = names[:MaxEndpoints]
	}
	state.Endpoints = names
	return json.Marshal(state)
}
