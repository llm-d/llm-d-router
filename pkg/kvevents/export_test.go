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

package kvevents

import (
	"context"

	"github.com/cespare/xxhash/v2"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// ModelEngineKey is the index key the pool stores for an engine block hash
// reported by a pod whose KV-events topic names modelName.
func ModelEngineKey(modelName string, hash uint64) kvblock.BlockHash {
	return namespacedEngineKey(xxhash.Sum64String(modelName), hash)
}

// ProcessRawMessageForTest decodes and applies msg synchronously, so external
// tests can drive a real engine adapter, which this package cannot import.
func (p *Pool) ProcessRawMessageForTest(ctx context.Context, msg *RawMessage) {
	p.processRawMessage(ctx, msg)
}
