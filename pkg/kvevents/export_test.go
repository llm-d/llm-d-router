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

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// ReplaySnapshotForTest applies snapshot chunks to a fresh strict generation
// pool, as snapshot bootstrap does, and returns the owned request keys.
func ReplaySnapshotForTest(ctx context.Context, index kvblock.Index, tokens kvblock.TokenProcessor,
	adapter EngineAdapter, topic, generation string, chunks [][]byte, staged bool,
) ([]kvblock.BlockHash, error) {
	pool, err := newGenerationPool(index, tokens, adapter)
	if err != nil {
		return nil, err
	}
	defer pool.queues[0].ShutDown()
	pool.strict = true
	pool.staged = staged
	for _, chunk := range chunks {
		_, model, batch, err := adapter.ParseMessage(&RawMessage{Topic: topic, Payload: chunk})
		if err != nil {
			return nil, err
		}
		if err := pool.processEventBatch(ctx, &batch, generation, model); err != nil {
			return nil, err
		}
	}
	if err := pool.publishStaged(ctx); err != nil {
		return nil, err
	}
	keys := make([]kvblock.BlockHash, 0, len(pool.snapshotEntries))
	for owned := range pool.snapshotEntries {
		keys = append(keys, owned.key)
	}
	return keys, nil
}
