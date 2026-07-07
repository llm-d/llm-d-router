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
	"errors"
	"slices"
)

// TopologicalSort performs Kahn's Algorithm on a DAG.
// It returns the sorted order or an error if a cycle is detected.
func TopologicalSort(graph map[string][]string) ([]string, error) {
	// 1. Initialize in-degree map
	inDegree := make(map[string]int)

	// Ensure all nodes are present in the inDegree map, even those with no dependencies
	for u, neighbors := range graph {
		if _, ok := inDegree[u]; !ok {
			inDegree[u] = 0
		}
		for _, v := range neighbors {
			inDegree[v]++ // Increment in-degree for the destination node
		}
	}

	// 2. Initialize the queue with nodes having 0 in-degree
	var queue []string
	for node, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, node)
		}
	}

	var result []string

	// 3. Process the queue
	for len(queue) > 0 {
		// Dequeue
		u := queue[0]
		queue = queue[1:]

		result = append(result, u)

		// Decrease in-degree of neighbors
		if neighbors, ok := graph[u]; ok {
			for _, v := range neighbors {
				inDegree[v]--
				if inDegree[v] == 0 {
					queue = append(queue, v)
				}
			}
		}
	}

	// 4. Check for cycles
	// If the result size != total nodes, there is a cycle
	if len(result) != len(inDegree) {
		return nil, errors.New("cycle detected: graph is not a DAG")
	}

	// Reverse to get the correct order since edges point from consumer to producer
	slices.Reverse(result)
	return result, nil
}
