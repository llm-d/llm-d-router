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

package kvevents_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	zmq "github.com/go-zeromq/zmq4"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// loadReplyFrames reads a captured snapshot reply stored as length-prefixed
// frames: cut, publisher identity, then the replay chunks.
func loadReplyFrames(tb testing.TB, file string) [][]byte {
	raw, err := os.ReadFile(file)
	require.NoError(tb, err)
	var frames [][]byte
	for len(raw) >= 4 {
		n := int(binary.BigEndian.Uint32(raw[:4]))
		require.LessOrEqual(tb, 4+n, len(raw), "%s: truncated frame", file)
		frames = append(frames, raw[4:4+n])
		raw = raw[4+n:]
	}
	require.GreaterOrEqual(tb, len(frames), 3, "%s: no replay chunks", file)
	require.Len(tb, frames[1], 16, "%s: publisher identity", file)
	return frames
}

// TestSnapshotBootstrapRetention bootstraps real subscribers from captured
// engine snapshots served by a local publisher and reports the heap the
// generations keep once every publisher is ready. It runs only with
// KV_SNAPSHOT_FRAMES set to a glob of captured replies (see hack/snapcount);
// KV_SNAPSHOT_RANKS cycles the captures over more publishers, and
// KV_SNAPSHOT_HEAP_PROFILE writes a heap profile after bootstrap.
func TestSnapshotBootstrapRetention(t *testing.T) {
	pattern := os.Getenv("KV_SNAPSHOT_FRAMES")
	if pattern == "" {
		t.Skip("KV_SNAPSHOT_FRAMES not set")
	}
	files, err := filepath.Glob(pattern)
	require.NoError(t, err)
	require.NotEmpty(t, files, "no captures match %q", pattern)
	ranks := len(files)
	if v := os.Getenv("KV_SNAPSHOT_RANKS"); v != "" {
		ranks, err = strconv.Atoi(v)
		require.NoError(t, err)
	}
	basePort := 25557
	if v := os.Getenv("KV_SNAPSHOT_BASE_PORT"); v != "" {
		basePort, err = strconv.Atoi(v)
		require.NoError(t, err)
	}

	// An unset controller-runtime logger records every WithValues call.
	log.SetLogger(logr.Discard())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cfg := kvevents.DefaultConfig()
	cfg.PodDiscoveryConfig.SocketPort = basePort
	cfg.SnapshotPort = basePort + 1000
	tokens, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 64})
	require.NoError(t, err)
	adapter := engineadapter.NewVLLMAdapter()
	adapter.SnapshotMode = true
	indexCfg := &kvblock.IndexConfig{InMemoryConfig: &kvblock.InMemoryIndexConfig{Size: 10_000_000, PodCacheSize: 1024}}
	manager, err := kvevents.NewSnapshotManager(cfg, indexCfg, tokens, adapter, nil)
	require.NoError(t, err)
	defer manager.Shutdown(ctx)

	heartbeat, err := msgpack.Marshal([]any{float64(time.Now().Unix()), []any{}, nil})
	require.NoError(t, err)
	var wg sync.WaitGroup
	defer func() { cancel(); wg.Wait() }()
	replyBytes := 0
	for i := range ranks {
		frames := loadReplyFrames(t, files[i%len(files)])
		for _, f := range frames[2:] {
			replyBytes += len(f)
		}
		epoch := frames[1]
		livePort := basePort + i
		pub := zmq.NewPub(ctx)
		require.NoError(t, pub.Listen(fmt.Sprintf("tcp://127.0.0.1:%d", livePort)))
		rep := zmq.NewRep(ctx)
		require.NoError(t, rep.Listen(fmt.Sprintf("tcp://127.0.0.1:%d", livePort+1000)))
		t.Cleanup(func() { pub.Close(); rep.Close() })
		source := fmt.Sprintf("10.0.%d.%d:8000", i/256, i%256)
		topic := []byte("kv@" + source + "@zai-org/GLM-5.3")
		var mu sync.Mutex
		var seq uint64
		wg.Add(2)
		go func() {
			// Live heartbeats: sequenced empty batches, as vLLM sends them.
			defer wg.Done()
			ticker := time.NewTicker(100 * time.Millisecond)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
				}
				mu.Lock()
				seq++
				header := binary.BigEndian.AppendUint64(nil, seq)
				mu.Unlock()
				_ = pub.Send(zmq.NewMsgFrom(topic, append(header, epoch...), heartbeat))
			}
		}()
		go func() {
			// Snapshot replies: the cut is the last live sequence sent.
			defer wg.Done()
			for {
				if _, err := rep.Recv(); err != nil {
					return
				}
				mu.Lock()
				cut := seq
				mu.Unlock()
				reply := append([][]byte{binary.BigEndian.AppendUint64(nil, cut), epoch}, frames[2:]...)
				if err := rep.Send(zmq.NewMsgFrom(reply...)); err != nil {
					return
				}
			}
		}()
		require.NoError(t, manager.EnsureSubscriber(ctx, source, source, fmt.Sprintf("tcp://127.0.0.1:%d", livePort), "", "kv@", true))
	}

	started := time.Now()
	require.Eventually(t, func() bool { return manager.Status().Ready == ranks }, 10*time.Minute, 200*time.Millisecond,
		"status %+v", manager.Status())
	t.Logf("%d publishers ready in %s from %.1f MiB of replies", ranks, time.Since(started).Round(time.Millisecond), float64(replyBytes)/(1<<20))
	measure := func(label string) {
		runtime.GC()
		runtime.GC()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		t.Logf("%s: heap live %.1f MiB (%.1f MiB per publisher), inuse %.1f MiB", label,
			float64(ms.HeapAlloc)/(1<<20), float64(ms.HeapAlloc)/(1<<20)/float64(ranks), float64(ms.HeapInuse)/(1<<20))
	}
	measure("after bootstrap")
	time.Sleep(5 * time.Second)
	require.Equal(t, ranks, manager.Status().Ready, "publishers left the ready state on heartbeats")
	measure("after 5 s of heartbeats")
	if path := os.Getenv("KV_SNAPSHOT_HEAP_PROFILE"); path != "" {
		f, err := os.Create(path)
		require.NoError(t, err)
		require.NoError(t, pprof.Lookup("heap").WriteTo(f, 0))
		require.NoError(t, f.Close())
	}
}
