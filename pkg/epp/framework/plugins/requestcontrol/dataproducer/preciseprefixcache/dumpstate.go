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

package preciseprefixcache

import (
	"encoding/json"
	"sort"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// maxDebugDumpSpeculativeEntries bounds the per-entry sample emitted by
// DumpState so the debug payload stays small when many speculative entries
// are pending.
const maxDebugDumpSpeculativeEntries = 100

var _ plugin.StateDumper = &Producer{}

// speculativeCacheState is the sanitized, bounded snapshot returned by
// DumpState. Per-entry values are counts only: block-key hashes and request
// payloads are never included.
type speculativeCacheState struct {
	SpeculativeEnabled    bool                    `json:"speculativeEnabled"`
	SpeculativeTTLSeconds float64                 `json:"speculativeTTLSeconds"`
	BlockSizeTokens       int                     `json:"blockSizeTokens"`
	TotalEntries          int                     `json:"totalEntries"`
	MaxEntries            int                     `json:"maxEntries"`
	Truncated             bool                    `json:"truncated"`
	Entries               []speculativeEntryState `json:"entries"`
}

type speculativeEntryState struct {
	RequestID  string `json:"requestID"`
	Prompts    int    `json:"prompts"`
	BlockKeys  int    `json:"blockKeys"`
	PodEntries int    `json:"podEntries"`
}

// DumpState implements [plugin.StateDumper], exposing a sanitized snapshot of
// the speculative index cache for the /debug/plugins/state endpoint. The entry
// list is capped to keep the payload bounded; TotalEntries reports the true
// count so operators can tell when the dump is partial.
func (p *Producer) DumpState() (json.RawMessage, error) {
	return json.Marshal(p.snapshotSpeculativeState())
}

func (p *Producer) snapshotSpeculativeState() speculativeCacheState {
	state := speculativeCacheState{
		SpeculativeEnabled:    p.speculativeEnabled,
		SpeculativeTTLSeconds: p.speculativeTTL.Seconds(),
		BlockSizeTokens:       p.blockSizeTokens,
		MaxEntries:            maxDebugDumpSpeculativeEntries,
	}
	if p.speculativeCache == nil {
		return state
	}

	items := p.speculativeCache.Items()
	state.TotalEntries = len(items)
	state.Entries = make([]speculativeEntryState, 0, len(items))
	for requestID, item := range items {
		entry := item.Value()
		if entry == nil {
			continue
		}
		blockKeys := 0
		for _, keys := range entry.perPromptKeys {
			blockKeys += len(keys)
		}
		state.Entries = append(state.Entries, speculativeEntryState{
			RequestID:  requestID,
			Prompts:    len(entry.perPromptKeys),
			BlockKeys:  blockKeys,
			PodEntries: len(entry.podEntries),
		})
	}

	sort.Slice(state.Entries, func(i, j int) bool {
		return state.Entries[i].RequestID < state.Entries[j].RequestID
	})
	if len(state.Entries) > maxDebugDumpSpeculativeEntries {
		state.Entries = state.Entries[:maxDebugDumpSpeculativeEntries]
		state.Truncated = true
	}
	return state
}
