package kvevents

import (
	"context"
	"fmt"
	"sort"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// FollowSnapshotForTest loads snapshot chunks into a strict generation pool
// as bootstrap does, then applies live payloads in the same strict pool, as a
// snapshot publisher's pool does for its whole life. It returns the owned
// index entries and dedup counts as sorted lines, the entry count after the
// snapshot, the number of live payloads applied, and the first live error.
func FollowSnapshotForTest(ctx context.Context, index kvblock.Index, tokens kvblock.TokenProcessor,
	adapter EngineAdapter, topic, pod string, chunks, live [][]byte,
) (lines []string, loaded, applied int, liveErr error, err error) {
	pool, err := newGenerationPool(index, tokens, adapter)
	if err != nil {
		return nil, 0, 0, nil, err
	}
	defer pool.queues[0].ShutDown()
	pool.strict = true
	pool.staged = true
	apply := func(payload []byte) error {
		_, model, batch, err := adapter.ParseMessage(&RawMessage{Topic: topic, Payload: payload})
		if err != nil {
			return err
		}
		return pool.processEventBatch(ctx, &batch, pod, model)
	}
	for i, chunk := range chunks {
		if err := apply(chunk); err != nil {
			return nil, 0, 0, nil, fmt.Errorf("snapshot chunk %d: %w", i, err)
		}
	}
	if err := pool.publishStaged(ctx); err != nil {
		return nil, 0, 0, nil, err
	}
	loaded = len(pool.snapshotEntries)
	for _, payload := range live {
		if liveErr = apply(payload); liveErr != nil {
			break
		}
		applied++
	}
	for owned := range pool.snapshotEntries {
		lines = append(lines, fmt.Sprintf("E %016x %s %v %d", uint64(owned.key), owned.entry.DeviceTier, owned.entry.HasGroup, owned.entry.GroupIdx))
	}
	for key, count := range pool.dedup.refs[pod] {
		lines = append(lines, fmt.Sprintf("R %s %d %016x %d", key.deviceTier, key.groupIdx, key.blockHash, count))
	}
	sort.Strings(lines)
	return lines, loaded, applied, liveErr, nil
}
