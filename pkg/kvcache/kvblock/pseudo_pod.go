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

package kvblock

import (
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

// Pseudo-pod identifiers name KV tiers that are not owned by a single pod.
// A KV-event topic "kv@node:<nodeName>@<model>" credits every candidate
// endpoint scheduled on that node (a node-local cache server shared by all
// pods on the node); "kv@pool:<name>@<model>" credits every candidate endpoint
// (a fleet-wide shared storage tier). The index stores pseudo-pod identifiers
// as opaque pod identifiers; expansion to real endpoints happens at lookup.
const (
	nodePseudoPodPrefix = "node:"
	poolPseudoPodPrefix = "pool:"
)

// NodePseudoPod returns the pseudo-pod identifier for a node-local shared tier.
func NodePseudoPod(nodeName string) string {
	return nodePseudoPodPrefix + nodeName
}

// PoolPseudoPod returns the pseudo-pod identifier for a fleet-wide shared tier.
func PoolPseudoPod(name string) string {
	return poolPseudoPodPrefix + name
}

// IsPoolPseudoPod reports whether podIdentifier names a fleet-wide shared tier.
func IsPoolPseudoPod(podIdentifier string) bool {
	return strings.HasPrefix(podIdentifier, poolPseudoPodPrefix)
}

// IsPseudoPod reports whether podIdentifier is a node: or pool: pseudo-pod.
func IsPseudoPod(podIdentifier string) bool {
	return strings.HasPrefix(podIdentifier, nodePseudoPodPrefix) || IsPoolPseudoPod(podIdentifier)
}

// InPodFilter reports whether an index entry for podIdentifier passes a
// non-empty Lookup filter. Pool pseudo-pods always pass: their names cannot be
// enumerated by the caller and they credit every candidate endpoint.
func InPodFilter(podIdentifierSet sets.Set[string], podIdentifier string) bool {
	return podIdentifierSet.Has(podIdentifier) || IsPoolPseudoPod(podIdentifier)
}

// ResolvePseudoPods rewrites pseudo-pod entries in keyToPods into entries for
// the real endpoints they credit, keeping the device tier. A node:<n> entry
// becomes one entry per endpoint in endpointsByNode[n]; a pool: entry becomes
// one entry per endpoint in allEndpoints. Pseudo-pod entries that resolve to
// no endpoint are dropped. The map is modified in place and returned.
func ResolvePseudoPods(keyToPods map[BlockHash][]PodEntry,
	endpointsByNode map[string][]string, allEndpoints []string,
) map[BlockHash][]PodEntry {
	for key, entries := range keyToPods {
		if !hasPseudoPod(entries) {
			continue
		}
		resolved := make([]PodEntry, 0, len(entries))
		for _, e := range entries {
			switch {
			case IsPoolPseudoPod(e.PodIdentifier):
				resolved = appendFor(resolved, e, allEndpoints)
			case strings.HasPrefix(e.PodIdentifier, nodePseudoPodPrefix):
				node := strings.TrimPrefix(e.PodIdentifier, nodePseudoPodPrefix)
				resolved = appendFor(resolved, e, endpointsByNode[node])
			default:
				resolved = append(resolved, e)
			}
		}
		keyToPods[key] = resolved
	}
	return keyToPods
}

func hasPseudoPod(entries []PodEntry) bool {
	for _, e := range entries {
		if IsPseudoPod(e.PodIdentifier) {
			return true
		}
	}
	return false
}

func appendFor(dst []PodEntry, template PodEntry, endpoints []string) []PodEntry {
	for _, ep := range endpoints {
		e := template
		e.PodIdentifier = ep
		dst = append(dst, e)
	}
	return dst
}
