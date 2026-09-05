/*
Copyright 2026 llm-d Authors.

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

package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// runs is high enough that Go's randomized map iteration would surface a
// map-order dependency in the sort.
const runs = 100

// TestTopologicalSort_Deterministic pins the ordering contract: the result is a
// pure function of the graph, dependencies precede dependents, and nodes with no
// dependency relation come out in ascending name order.
func TestTopologicalSort_Deterministic(t *testing.T) {
	testCases := []struct {
		name    string
		graph   map[string][]string
		want    []string
		wantErr string
	}{
		{
			name:  "empty graph",
			graph: map[string][]string{},
		},
		{
			name:  "single node",
			graph: map[string][]string{"A/mock": {}},
			want:  []string{"A/mock"},
		},
		{
			name:  "independent nodes come out in ascending name order",
			graph: map[string][]string{"B/mock": {}, "A/mock": {}, "C/mock": {}},
			want:  []string{"A/mock", "B/mock", "C/mock"},
		},
		{
			// C depends on A; B and D depend on nothing.
			name: "dependency respected, independent nodes ascending",
			graph: map[string][]string{
				"C/mock": {"A/mock"},
				"A/mock": {},
				"B/mock": {},
				"D/mock": {},
			},
			want: []string{"A/mock", "B/mock", "C/mock", "D/mock"},
		},
		{
			name: "linear chain",
			graph: map[string][]string{
				"C/mock": {"B/mock"},
				"B/mock": {"A/mock"},
				"A/mock": {},
			},
			want: []string{"A/mock", "B/mock", "C/mock"},
		},
		{
			// D depends on B and C, both of which depend on A.
			name: "diamond",
			graph: map[string][]string{
				"D/mock": {"B/mock", "C/mock"},
				"B/mock": {"A/mock"},
				"C/mock": {"A/mock"},
				"A/mock": {},
			},
			want: []string{"A/mock", "B/mock", "C/mock", "D/mock"},
		},
		{
			name: "topological order wins over name order",
			graph: map[string][]string{
				"A/mock": {"Z/mock"},
				"Z/mock": {},
			},
			want: []string{"Z/mock", "A/mock"},
		},
		{
			name:    "cycle is rejected",
			graph:   map[string][]string{"X/mock": {"Y/mock"}, "Y/mock": {"X/mock"}},
			wantErr: "cycle detected",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			for i := range runs {
				got, err := TopologicalSort(tc.graph)
				if tc.wantErr != "" {
					require.Error(t, err)
					assert.Contains(t, err.Error(), tc.wantErr)
					continue
				}
				require.NoError(t, err)
				assert.Equal(t, tc.want, got, "run %d returned a different order", i)
			}
		})
	}
}
