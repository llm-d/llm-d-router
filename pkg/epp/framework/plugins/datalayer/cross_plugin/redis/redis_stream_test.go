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

package redis

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/prometheus/client_golang/prometheus/testutil"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const streamStateKey = fwkdl.StateKey("approx-prefix:stream-test")

// commandLog counts commands by name and can fail the next pipelines before or
// after they reach Redis.
type commandLog struct {
	mu         sync.Mutex
	counts     map[string]int
	failBefore int
	failAfter  int
}

func newCommandLog() *commandLog { return &commandLog{counts: map[string]int{}} }

func (l *commandLog) DialHook(next goredis.DialHook) goredis.DialHook {
	return func(ctx context.Context, network, addr string) (net.Conn, error) { return next(ctx, network, addr) }
}

func (l *commandLog) ProcessHook(next goredis.ProcessHook) goredis.ProcessHook {
	return func(ctx context.Context, cmd goredis.Cmder) error {
		l.record([]goredis.Cmder{cmd})
		return next(ctx, cmd)
	}
}

func (l *commandLog) ProcessPipelineHook(next goredis.ProcessPipelineHook) goredis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []goredis.Cmder) error {
		l.mu.Lock()
		before, after := l.failBefore > 0, l.failBefore == 0 && l.failAfter > 0
		if before {
			l.failBefore--
		} else if after {
			l.failAfter--
		}
		l.mu.Unlock()
		if before {
			return errors.New("connection reset")
		}
		l.record(cmds)
		err := next(ctx, cmds)
		if after {
			return errors.New("connection reset")
		}
		return err
	}
}

func (l *commandLog) record(cmds []goredis.Cmder) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, cmd := range cmds {
		l.counts[cmd.Name()]++
	}
}

func (l *commandLog) count(names ...string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, name := range names {
		n += l.counts[name]
	}
	return n
}

func (l *commandLog) reset() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.counts = map[string]int{}
}

// take returns the count for name and resets every count.
func (l *commandLog) take(name string) int {
	n := l.count(name)
	l.reset()
	return n
}

type streamReplica struct {
	syncer *RedisSyncer
	log    *commandLog
	window *attrprefix.PrefixHashSet
}

func newStreamReplica(t *testing.T, server *miniredis.Miniredis, replicaID string, incremental bool) *streamReplica {
	t.Helper()
	syncer, _ := newTestSyncer(t, server, replicaID)
	log := newCommandLog()
	syncer.client.AddHook(log)
	syncer.incremental = incremental
	return &streamReplica{syncer: syncer, log: log, window: &attrprefix.PrefixHashSet{}}
}

func (r *streamReplica) spec() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey: streamStateKey,
		Read:     func(string) fwkdl.Cloneable { return r.window },
		Aggregate: func(values []any) any {
			out := &attrprefix.PrefixHashSet{}
			seen := map[string]bool{}
			for _, v := range values {
				for _, p := range v.(*attrprefix.PrefixHashSet).Prompts {
					if key := fmt.Sprint(p); !seen[key] {
						seen[key] = true
						out.Prompts = append(out.Prompts, p)
					}
				}
			}
			return out
		},
	}
}

func (r *streamReplica) publish(t *testing.T) {
	t.Helper()
	require.NoError(t, r.syncer.Set(context.Background(), r.spec(), testEndpointID))
}

func (r *streamReplica) append(prompts ...[]uint64) {
	r.window = &attrprefix.PrefixHashSet{Base: r.window.Base, Prompts: append(append([][]uint64{}, r.window.Prompts...), prompts...)}
}

func (r *streamReplica) evict(n int) {
	r.window = &attrprefix.PrefixHashSet{Base: r.window.Base + uint64(n), Prompts: r.window.Prompts[n:]}
}

// peers returns the cached peer aggregate's prompts.
func (r *streamReplica) peers(t *testing.T) [][]uint64 {
	t.Helper()
	v, ok := r.syncer.cache.Load(r.syncer.hashKey(streamStateKey, testEndpointID))
	if !ok {
		return nil
	}
	return v.(*aggregateCacheEntry).value.(*attrprefix.PrefixHashSet).Prompts
}

func (r *streamReplica) hashKey() string { return r.syncer.hashKey(streamStateKey, testEndpointID) }

func (r *streamReplica) streamKey() string {
	return r.syncer.streamKey(r.hashKey(), r.syncer.replicaID)
}

func (r *streamReplica) cursor(t *testing.T) streamCursor {
	t.Helper()
	c, ok := r.syncer.loadCursor(r.hashKey())
	require.True(t, ok)
	return c
}

func TestStreamUnchangedWindowRefreshesTTLWithoutAppending(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	a.append([]uint64{1, 2}, []uint64{3})
	a.publish(t)
	require.Equal(t, 2, a.log.take("xadd"))

	server.FastForward(testStateTTL / 2)
	a.publish(t)
	require.Zero(t, a.log.count("xadd"))
	require.Zero(t, a.log.count("xtrim"))
	require.Zero(t, a.log.count("hset"), "an unchanged marker is not rewritten")
	require.Equal(t, testStateTTL, server.TTL(a.streamKey()))
	require.Equal(t, testStateTTL, server.TTL(a.hashKey()))
}

func TestStreamAppendsOnlyNewPrompts(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	b := newStreamReplica(t, server, "epp-b", true)
	a.append([]uint64{1, 2}, []uint64{3})
	a.publish(t)
	b.publish(t)
	require.Equal(t, [][]uint64{{1, 2}, {3}}, b.peers(t))
	a.log.take("xadd")
	b.log.take("xrange")

	a.append([]uint64{4, 5, 6})
	a.publish(t)
	require.Equal(t, 1, a.log.take("xadd"))
	entries, err := a.syncer.client.XRange(context.Background(), a.streamKey(), "-", "+").Result()
	require.NoError(t, err)
	require.Len(t, entries, 3)
	require.Equal(t, streamID(2), entries[2].ID)

	b.publish(t)
	require.Equal(t, 1, b.log.take("xrange"))
	require.Equal(t, [][]uint64{{1, 2}, {3}, {4, 5, 6}}, b.peers(t))

	b.publish(t)
	require.Zero(t, b.log.take("xrange"), "an unchanged peer is not read")
}

func TestStreamEvictionTrimsWithoutRewriting(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	b := newStreamReplica(t, server, "epp-b", true)
	a.append([]uint64{1}, []uint64{2}, []uint64{3})
	a.publish(t)
	b.publish(t)
	a.log.reset()

	a.evict(2)
	a.append([]uint64{4})
	a.publish(t)
	require.Equal(t, 1, a.log.count("xadd"))
	require.Equal(t, 1, a.log.count("xtrim"))
	require.Zero(t, a.log.count("del"))
	entries, err := a.syncer.client.XRange(context.Background(), a.streamKey(), "-", "+").Result()
	require.NoError(t, err)
	require.Equal(t, []string{streamID(2), streamID(3)}, []string{entries[0].ID, entries[1].ID})
	c := a.cursor(t)
	require.Equal(t, uint64(2), c.start)
	require.Equal(t, uint64(4), c.tip)
	require.Equal(t, uint64(16), c.size, "size tracks trimmed and appended prompts without a full window scan")
	require.Equal(t, []uint32{8, 8}, c.sizes)

	b.publish(t)
	require.Equal(t, [][]uint64{{3}, {4}}, b.peers(t))
}

func TestStreamFailedWriteDoesNotAdvanceCursor(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	b := newStreamReplica(t, server, "epp-b", true)
	a.append([]uint64{1}, []uint64{2})
	a.publish(t)
	before := a.cursor(t)

	a.append([]uint64{3})
	a.log.failBefore = 1
	require.Error(t, a.syncer.Set(context.Background(), a.spec(), testEndpointID))
	after := a.cursor(t)
	require.Equal(t, before.tip, after.tip)
	require.NotNil(t, after.unknown)

	a.log.reset()
	a.publish(t)
	require.Zero(t, a.log.take("xadd"), "the next publish only checks the stream length")
	require.Equal(t, before.tip, a.cursor(t).tip)
	require.Nil(t, a.cursor(t).unknown)

	a.publish(t)
	require.Equal(t, 1, a.log.take("xadd"), "only the unpublished prompt is sent")
	b.publish(t)
	require.Equal(t, [][]uint64{{1}, {2}, {3}}, b.peers(t))
}

func TestStreamLostReplyAdoptsStoredState(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	a.append([]uint64{1})
	a.publish(t)

	a.append([]uint64{2})
	a.log.failAfter = 1
	require.Error(t, a.syncer.Set(context.Background(), a.spec(), testEndpointID))
	a.log.reset()
	a.publish(t)
	require.Equal(t, uint64(2), a.cursor(t).tip)

	a.publish(t)
	require.Zero(t, a.log.count("xadd"), "the write that landed is not repeated")
	require.Zero(t, a.log.count("del"))
}

func TestStreamLostStreamStartsNewEpoch(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	b := newStreamReplica(t, server, "epp-b", true)
	a.append([]uint64{1}, []uint64{2}, []uint64{3})
	a.publish(t)
	b.publish(t)
	epoch := a.cursor(t).epoch
	resets := testutil.ToFloat64(streamResets.WithLabelValues(string(streamStateKey)))

	server.Del(a.streamKey())
	a.publish(t)
	require.True(t, a.cursor(t).reset)
	b.publish(t)
	require.Equal(t, [][]uint64{{1}, {2}, {3}}, b.peers(t), "the peer keeps the window it has")

	a.log.reset()
	a.publish(t)
	require.Equal(t, 3, a.log.take("xadd"))
	require.Equal(t, epoch+1, a.cursor(t).epoch)
	require.Equal(t, resets+1, testutil.ToFloat64(streamResets.WithLabelValues(string(streamStateKey))))

	a.evict(1)
	a.publish(t)
	b.publish(t)
	require.Equal(t, [][]uint64{{2}, {3}}, b.peers(t))
}

func TestStreamTruncatedStreamStartsNewEpoch(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	a.append([]uint64{1}, []uint64{2}, []uint64{3})
	a.publish(t)
	require.NoError(t, a.syncer.client.XTrimMaxLen(context.Background(), a.streamKey(), 1).Err())

	a.publish(t)
	require.True(t, a.cursor(t).reset)
}

func TestStreamBudgetSpreadsColdWindowOverTicks(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	b := newStreamReplica(t, server, "epp-b", true)
	now := time.Unix(0, 0)
	// Ten prompts of ten hashes are 80 bytes each; 160 bytes per second sends two.
	a.syncer.publishBudget = &byteBudget{rate: 160, tokens: 160, last: now, now: func() time.Time { return now }}
	for i := range 10 {
		prompt := make([]uint64, 10)
		for j := range prompt {
			prompt[j] = uint64(i*10 + j)
		}
		a.append(prompt)
	}

	ticks := 0
	for a.cursorTip() < 10 {
		a.publish(t)
		require.LessOrEqual(t, a.log.take("xadd"), 2)
		now = now.Add(time.Second)
		ticks++
		require.Less(t, ticks, 10)
	}
	require.Equal(t, 5, ticks)

	a.publish(t)
	require.Zero(t, a.log.count("xadd"))
	b.publish(t)
	require.Equal(t, a.window.Prompts, b.peers(t))
}

func (r *streamReplica) cursorTip() uint64 {
	c, ok := r.syncer.loadCursor(r.hashKey())
	if !ok {
		return 0
	}
	return c.tip
}

func TestStreamSpentBudgetStillRefreshesTTL(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	now := time.Unix(0, 0)
	a.syncer.publishBudget = &byteBudget{rate: 8, tokens: 8, last: now, now: func() time.Time { return now }}
	a.append([]uint64{1}, []uint64{2})
	a.publish(t)
	require.Equal(t, 1, a.log.take("xadd"))

	server.FastForward(testStateTTL / 2)
	a.publish(t)
	require.Zero(t, a.log.count("xadd"))
	require.Equal(t, 1, a.log.count("hpexpire", "hexpire"))
	require.Equal(t, 2, a.log.count("pexpire"))
	require.Equal(t, testStateTTL, server.TTL(a.streamKey()))
}

func TestStreamReadBudgetSpreadsPeerCatchUp(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	b := newStreamReplica(t, server, "epp-b", true)
	now := time.Unix(0, 0)
	b.syncer.readBudget = &byteBudget{rate: 16, tokens: 16, last: now, now: func() time.Time { return now }}
	for i := range 6 {
		a.append([]uint64{uint64(i)})
	}
	a.publish(t)

	b.publish(t)
	require.Len(t, b.peers(t), 2)
	b.publish(t)
	require.Len(t, b.peers(t), 2, "spent budget defers the read")
	now = now.Add(time.Second)
	b.publish(t)
	now = now.Add(time.Second)
	b.publish(t)
	require.Equal(t, a.window.Prompts, b.peers(t))
}

func TestStreamRejectsOversizedPeerEntry(t *testing.T) {
	server := miniredis.RunT(t)
	b := newStreamReplica(t, server, "epp-b", true)
	hashKey := b.hashKey()
	peerStream := b.syncer.streamKey(hashKey, "epp-x")
	ctx := context.Background()
	require.NoError(t, b.syncer.client.XAdd(ctx, &goredis.XAddArgs{Stream: peerStream, ID: streamID(0), Values: []any{streamElementField, string(make([]byte, 8))}}).Err())
	server.HSet(hashKey, markerField("epp-x"), "1 0 1 8")
	b.publish(t)
	require.Equal(t, [][]uint64{{0}}, b.peers(t))

	require.NoError(t, b.syncer.client.XAdd(ctx, &goredis.XAddArgs{Stream: peerStream, ID: streamID(1), Values: []any{streamElementField, strings.Repeat("x", maxStreamElementBytes+8)}}).Err())
	server.HSet(hashKey, markerField("epp-x"), fmt.Sprintf("1 0 2 %d", maxStreamElementBytes+16))
	b.publish(t)
	require.Equal(t, [][]uint64{{0}}, b.peers(t))
	entry, _ := b.syncer.cache.Load(hashKey)
	peer := entry.(*aggregateCacheEntry).streams["epp-x"]
	require.Equal(t, uint64(1), peer.tip)
	require.True(t, peer.stale)
}

func TestStreamReaderAcceptsWholeValuePeers(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	full := newStreamReplica(t, server, "epp-c", false)
	a.append([]uint64{1})
	full.append([]uint64{2})
	full.publish(t)

	a.publish(t)
	require.ElementsMatch(t, [][]uint64{{2}}, a.peers(t))
	full.publish(t)
	require.ElementsMatch(t, [][]uint64{{1}}, full.peers(t), "a whole-value replica reads stream peers")
}

func TestStreamDropsExpiredWholeValuePeers(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	full := newStreamReplica(t, server, "epp-c", false)
	a.append([]uint64{1})
	full.append([]uint64{2})
	full.publish(t)
	a.publish(t)
	require.ElementsMatch(t, [][]uint64{{2}}, a.peers(t))

	require.NoError(t, a.syncer.client.HDel(context.Background(), a.hashKey(), full.syncer.replicaID).Err())
	a.publish(t)
	require.Empty(t, a.peers(t), "an expired whole-value peer must leave the aggregate")
}

func TestStreamRestoresMissingMarker(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	a.append([]uint64{1, 2})
	a.publish(t)
	require.NoError(t, a.syncer.client.HDel(context.Background(), a.hashKey(), markerField("epp-a")).Err())

	a.publish(t)
	exists, err := a.syncer.client.HExists(context.Background(), a.hashKey(), markerField("epp-a")).Result()
	require.NoError(t, err)
	require.True(t, exists, "an idle publish restores a deleted marker")
}

func TestStreamSkipsOversizedPrompt(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	huge := make([]uint64, maxStreamElementBytes/8+1)
	a.window = &attrprefix.PrefixHashSet{Prompts: [][]uint64{huge}}
	a.publish(t)
	require.Zero(t, a.log.take("xadd"))
	require.Equal(t, uint64(0), a.cursor(t).tip)

	a.window = &attrprefix.PrefixHashSet{Base: 1, Prompts: [][]uint64{{7, 8}}}
	a.publish(t)
	require.Equal(t, 1, a.log.take("xadd"))
	require.Equal(t, uint64(2), a.cursor(t).tip)
}

func TestStreamDeleteRemovesStreamAndMarker(t *testing.T) {
	server := miniredis.RunT(t)
	a := newStreamReplica(t, server, "epp-a", true)
	a.append([]uint64{1})
	a.publish(t)

	require.NoError(t, a.syncer.Delete(context.Background(), streamStateKey, testEndpointID))
	require.False(t, server.Exists(a.streamKey()))
	exists, err := a.syncer.client.HExists(context.Background(), a.hashKey(), markerField("epp-a")).Result()
	require.NoError(t, err)
	require.False(t, exists)
	_, ok := a.syncer.loadCursor(a.hashKey())
	require.False(t, ok)
}

func TestFullReplicationCommandsAreUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name        string
		incremental bool
		value       fwkdl.Cloneable
	}{
		{name: "window in full mode", value: &attrprefix.PrefixHashSet{Prompts: [][]uint64{{1}}}},
		{name: "plain value in incremental mode", incremental: true, value: cloneableInt(4)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := &commandRecorder{}
			client := goredis.NewClient(&goredis.Options{})
			client.AddHook(recorder)
			t.Cleanup(func() { require.NoError(t, client.Close()) })
			syncer := &RedisSyncer{replicaID: testReplicaID, client: client, stateTTL: testStateTTL, incremental: tc.incremental}
			spec := fwkdl.CrossReplicaSpec{
				StateKey:  testStateKey,
				Read:      func(string) fwkdl.Cloneable { return tc.value },
				Aggregate: func(values []any) any { return values[0] },
			}

			require.NoError(t, syncer.Set(context.Background(), spec, testEndpointID))

			names := make([]string, 0, len(recorder.commands))
			for _, cmd := range recorder.commands {
				names = append(names, cmd.Name())
			}
			require.Equal(t, []string{"multi", "hset", "hpexpire", "pexpire", "hgetall", "exec"}, names)
		})
	}
}
