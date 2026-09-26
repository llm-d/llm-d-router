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

package preciseprefixcache

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	scorer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/pkg/kvcache"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/test/utils"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

func snapshotTestPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	port := l.Addr().(*net.TCPAddr).Port
	require.NoError(t, l.Close())
	return port
}

func TestSnapshotVLLMPublisher(t *testing.T) {
	python, script := os.Getenv("VLLM_PYTHON"), os.Getenv("VLLM_SNAPSHOT_SERVER")
	if python == "" {
		t.Skip("set VLLM_PYTHON to the vLLM .venv/bin/python for the real publisher")
	}
	if script == "" {
		script = filepath.Join("testdata", "vllm_snapshot_publisher.py")
	}
	ctx, cancel := context.WithTimeout(utils.NewTestContext(t), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, python, script)
	stdin, err := cmd.StdinPipe()
	require.NoError(t, err)
	stdout, err := cmd.StdoutPipe()
	require.NoError(t, err)
	cmd.Stderr = os.Stderr
	require.NoError(t, cmd.Start())
	defer func() { stdin.Close(); require.NoError(t, cmd.Wait()) }()
	scanner := bufio.NewScanner(stdout)
	read := func() map[string]any {
		for scanner.Scan() {
			line := scanner.Bytes()
			if !bytes.HasPrefix(line, []byte("{\"result\"")) {
				continue
			}
			var out map[string]any
			require.NoError(t, json.Unmarshal(line, &out))
			return out
		}
		t.Fatalf("publisher exited: %v", scanner.Err())
		return nil
	}
	ready := read()
	command := func(action string) {
		_, err := fmt.Fprintln(stdin, action)
		require.NoError(t, err)
		require.Equal(t, "ok", read()["result"])
	}
	cfg := kvevents.DefaultConfig()
	cfg.PodDiscoveryConfig.SocketPort = int(ready["live_port"].(float64))
	cfg.SnapshotPort = int(ready["snapshot_port"].(float64))
	indexCfg, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	start := func() *Producer {
		p, err := New(ctx, "real-vllm", PluginConfig{IndexerConfig: indexCfg, KVEventsConfig: cfg, TokenProcessorConfig: &kvblock.TokenProcessorConfig{BlockSizeTokens: 4}})
		require.NoError(t, err)
		return p
	}
	p := start()
	defer func() { p.subscribersManager.Shutdown(ctx) }()
	ep := scheduling.NewEndpoint(&dl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "real-vllm"}, Address: "127.0.0.1", Port: "8000"}, nil, nil)
	discover := func() {
		require.NoError(t, p.Extract(ctx, dl.EndpointEvent{Type: dl.EventAddOrUpdate, Endpoint: dl.NewEndpoint(ep.GetMetadata(), nil)}))
	}
	discover()
	req := &scheduling.InferenceRequest{RequestID: "real-vllm", TargetModel: "test-model", Body: &rh.InferenceRequestBody{TokenizedRequest: &rh.TokenizedRequest{Prompts: []rh.PromptTokens{{TokenIDs: []uint32{1, 2, 3, 4, 5, 6, 7, 8}}}}}}
	prefix, err := scorer.New(ctx, "prefix", "real-vllm")
	require.NoError(t, err)
	check := func(score float64, gpu, cpu int) {
		t.Helper()
		require.Eventually(t, func() bool {
			if p.Produce(ctx, req, []scheduling.Endpoint{ep}) != nil {
				return false
			}
			raw, _ := ep.Get(p.dk)
			info, ok := raw.(*attrprefix.PrefixCacheMatchInfo)
			if !ok {
				return false
			}
			return prefix.Score(ctx, req, []scheduling.Endpoint{ep})[ep] == score && info.CachedBlocksByTier()["gpu"] == gpu && info.CachedBlocksByTier()["cpu"] == cpu
		}, 8*time.Second, 25*time.Millisecond, "score=%v gpu=%d cpu=%d", score, gpu, cpu)
	}
	check(1, 2, 0)
	command("offload-reset")
	check(0.5, 0, 2)
	// Recreate the actual producer with an empty index while CPU-only blocks exist.
	p.subscribersManager.Shutdown(ctx)
	p = start()
	discover()
	check(0.5, 0, 2)
	command("remove-one-copy")
	check(0.5, 0, 2)
	command("remove-last-copy")
	check(0, 0, 0)
	command("restart")
	check(1, 2, 0)
	command("stop")
	check(0, 0, 0)
	command("restart")
	check(1, 2, 0)
	// A failed recorder answers unavailable until the engine restarts: a new
	// producer indexes the publisher from live events, then recovers snapshots
	// once the restarted engine has a new identity.
	command("fail-recorder")
	p.subscribersManager.Shutdown(ctx)
	p = start()
	discover()
	require.Eventually(t, func() bool { return p.snapshots.Status().LiveOnly == 1 }, 8*time.Second, 25*time.Millisecond)
	check(0, 0, 0)
	command("store-gpu")
	check(1, 2, 0)
	command("restart")
	require.Eventually(t, func() bool { status := p.snapshots.Status(); return status.Ready == 1 && status.LiveOnly == 0 }, 8*time.Second, 25*time.Millisecond)
	check(1, 2, 0)
	require.NoError(t, p.Extract(ctx, dl.EndpointEvent{Type: dl.EventDelete, Endpoint: dl.NewEndpoint(ep.GetMetadata(), nil)}))
	check(0, 0, 0)
}

func TestSnapshotConfigurationRejectsUnsupportedModes(t *testing.T) {
	for _, mode := range []string{"global-socket", "speculation", "redis", "cost-aware", "sglang", "bad-port", "replay"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			indexCfg, err := kvcache.NewDefaultConfig()
			require.NoError(t, err)
			cfg := PluginConfig{IndexerConfig: indexCfg, KVEventsConfig: kvevents.DefaultConfig()}
			cfg.KVEventsConfig.SnapshotPort = 5559
			switch mode {
			case "global-socket":
				cfg.KVEventsConfig.ZMQEndpoint = "tcp://*:5557"
			case "speculation":
				cfg.SpeculativeIndexing = true
			case "redis":
				cfg.IndexerConfig.KVBlockIndexConfig.RedisConfig = &kvblock.RedisIndexConfig{}
			case "cost-aware":
				cfg.IndexerConfig.KVBlockIndexConfig.CostAwareMemoryConfig = &kvblock.CostAwareMemoryIndexConfig{}
			case "sglang":
				cfg.KVEventsConfig.EngineType = "sglang"
			case "replay":
				cfg.KVEventsConfig.PodDiscoveryConfig.ReplaySocketPort = 5558
			case "bad-port":
				cfg.KVEventsConfig.SnapshotPort = 65536
			}
			_, err = New(ctx, "invalid", cfg)
			require.Error(t, err)
		})
	}
}
