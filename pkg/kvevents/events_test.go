// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/utils/ptr"
)

func TestIsIndexableLocalGPUStore(t *testing.T) {
	base := BlockStoredEvent{BlockHashes: []uint64{1}, BlockSize: 16}
	assert.True(t, IsIndexableLocalGPUStore(&base), "omitted legacy tier is GPU")

	for name, mutate := range map[string]func(*BlockStoredEvent){
		"cpu":        func(event *BlockStoredEvent) { event.DeviceTier = "cpu" },
		"remote":     func(event *BlockStoredEvent) { event.Locality = "remote" },
		"owned":      func(event *BlockStoredEvent) { event.Ownership = "connector" },
		"empty":      func(event *BlockStoredEvent) { event.BlockHashes = nil },
		"zero block": func(event *BlockStoredEvent) { event.BlockSize = 0 },
		"legacy HMA": func(event *BlockStoredEvent) { event.GroupIdx = ptr.To(0) },
		"unsupported": func(event *BlockStoredEvent) {
			event.KVCacheSpecKind = KVCacheSpecKindSlidingWindow
		},
	} {
		t.Run(name, func(t *testing.T) {
			event := base
			mutate(&event)
			assert.False(t, IsIndexableLocalGPUStore(&event))
		})
	}
}
