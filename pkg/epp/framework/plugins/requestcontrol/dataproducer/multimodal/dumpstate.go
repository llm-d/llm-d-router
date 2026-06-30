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

package multimodal

import (
	"encoding/json"
	"sort"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// maxDebugDumpPods bounds the per-pod sample emitted by DumpState so the debug
// payload stays small when many pods are tracked.
const maxDebugDumpPods = 100

var _ plugin.StateDumper = &Producer{}

// encoderCacheState is the sanitized, bounded snapshot returned by DumpState.
// It reports counts only: item hashes and request data are never included.
type encoderCacheState struct {
	CacheSizePerPod int             `json:"cacheSizePerPod"`
	TotalPods       int             `json:"totalPods"`
	TotalEntries    int             `json:"totalEntries"`
	MaxPods         int             `json:"maxPods"`
	Truncated       bool            `json:"truncated"`
	Pods            []podCacheState `json:"pods"`
}

type podCacheState struct {
	Pod     string `json:"pod"`
	Entries int    `json:"entries"`
}

// DumpState implements [plugin.StateDumper] for the /debug/plugins/state
// endpoint, exposing the per-pod encoder-cache entry counts. The pod list is
// sorted by entry count and capped; TotalPods reports the true count so
// operators can tell when the dump is partial.
func (p *Producer) DumpState() (json.RawMessage, error) {
	p.mutex.RLock()
	state := encoderCacheState{
		CacheSizePerPod: p.cacheSize,
		TotalPods:       len(p.caches),
		MaxPods:         maxDebugDumpPods,
		Pods:            make([]podCacheState, 0, len(p.caches)),
	}
	for pod, cache := range p.caches {
		size := cache.Len()
		state.TotalEntries += size
		state.Pods = append(state.Pods, podCacheState{Pod: pod, Entries: size})
	}
	p.mutex.RUnlock()

	sort.Slice(state.Pods, func(a, b int) bool {
		if state.Pods[a].Entries != state.Pods[b].Entries {
			return state.Pods[a].Entries > state.Pods[b].Entries
		}
		return state.Pods[a].Pod < state.Pods[b].Pod
	})
	if len(state.Pods) > maxDebugDumpPods {
		state.Pods = state.Pods[:maxDebugDumpPods]
		state.Truncated = true
	}
	return json.Marshal(state)
}
