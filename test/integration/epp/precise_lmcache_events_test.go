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

package epp

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"syscall"
	"testing"
	"time"

	"github.com/go-zeromq/zmq4"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	fwkepp "github.com/llm-d/llm-d-router/test/framework/epp"
	eppharness "github.com/llm-d/llm-d-router/test/framework/epp/harness"
	fwkk8s "github.com/llm-d/llm-d-router/test/framework/k8s"
)

// TestPreciseRoutingFollowsLMCacheCPUEvents routes requests through an EPP
// whose only scorer reads the precise prefix cache, and feeds that cache the
// KV-event messages captured from a vLLM engine with an LMCache multiprocess
// server (pkg/kvevents/testdata/README.md). Two pods stand in for two
// replicas: each pod's KV-event socket publishes stages of the one capture,
// and the renderer returns the token IDs the engine rendered for the captured
// prompt. No model server runs, so the test ends at the routing decision: it
// does not show the picked replica loading the prompt from LMCache.
func TestPreciseRoutingFollowsLMCacheCPUEvents(t *testing.T) {
	ctx := t.Context()

	data, err := os.ReadFile(filepath.Join(eppharness.RepoRoot(), "pkg", "kvevents", "testdata", "lmcache_mp_kv_events.json"))
	require.NoError(t, err)
	var capture struct {
		Model          string `json:"model"`
		ChunkSize      int    `json:"lmcache_chunk_size"`
		PromptTokenIDs []int  `json:"prompt_token_ids"`
		Messages       []struct {
			Stage  string   `json:"stage"`
			Frames [][]byte `json:"frames"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(data, &capture))

	renderer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"token_ids": capture.PromptTokenIDs})
	}))
	t.Cleanup(renderer.Close)

	// An engine binds its KV-event socket on a fixed port of its pod IP, and
	// the EPP dials every pod on that port. The two replicas need two
	// addresses of the host with one free port.
	const replicaA, replicaB = "127.0.0.1", "127.0.0.2"
	publishers := map[string]zmq4.Socket{}
	var socketPort int
	var listenErr error
	for attempt := 0; len(publishers) == 0 && attempt < 20; attempt++ {
		probe, err := net.Listen("tcp", net.JoinHostPort(replicaB, "0"))
		if errors.Is(err, syscall.EADDRNOTAVAIL) {
			t.Skipf("the second replica needs a second loopback address: %v", err)
		}
		require.NoError(t, err)
		socketPort = probe.Addr().(*net.TCPAddr).Port
		require.NoError(t, probe.Close())
		a, b := zmq4.NewPub(ctx), zmq4.NewPub(ctx)
		listenErr = errors.Join(
			a.Listen("tcp://"+net.JoinHostPort(replicaA, strconv.Itoa(socketPort))),
			b.Listen("tcp://"+net.JoinHostPort(replicaB, strconv.Itoa(socketPort))),
		)
		if listenErr != nil {
			_, _ = a.Close(), b.Close()
			continue
		}
		publishers[replicaA], publishers[replicaB] = a, b
	}
	require.NoError(t, listenErr, "no port is free on both loopback addresses")
	t.Cleanup(func() {
		for _, pub := range publishers {
			_ = pub.Close()
		}
	})

	// The precise prefix cache is the only input to the routing decision: no
	// approximate producer, speculative indexing off and no other scorer. The
	// saturation detector's filter is part of every profile; its staleness
	// threshold is raised so that it keeps both pods, which report no load.
	configText := fmt.Sprintf(`
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
- type: openai-parser
- type: token-producer
  parameters:
    modelName: %s
    vllm:
      url: %s
- type: endpoint-notification-source
- type: precise-prefix-cache-producer
  parameters:
    speculativeIndexing: false
    kvEventsConfig:
      discoverPods: true
      podDiscoveryConfig:
        socketPort: %d
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
- type: max-score-picker
- type: single-profile-handler
- type: mock-metrics-source
- type: utilization-detector
  parameters:
    metricsStalenessThreshold: 1h
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  sources:
  - pluginRef: mock-metrics-source
  - pluginRef: endpoint-notification-source
    extractors:
    - pluginRef: precise-prefix-cache-producer
flowControl:
  saturationDetector:
    pluginRef: utilization-detector
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
  - pluginRef: max-score-picker
`, capture.Model, renderer.URL, socketPort)
	h := eppharness.NewTestHarness(ctx, t, eppharness.WithStandardMode(), eppharness.WithConfigText(configText),
		eppharness.WithEmitEndpointScores()).WithBaseResources()

	// The pods differ in their address only.
	podMetrics := map[types.NamespacedName]*fwkdl.Metrics{}
	for i, address := range []string{replicaA, replicaB} {
		name := fmt.Sprintf("pod-%d", i)
		podMetrics[types.NamespacedName{Namespace: h.Namespace, Name: name + "-rank-0"}] = &fwkdl.Metrics{
			ActiveModels:  map[string]int{},
			WaitingModels: map[string]int{},
		}
		pod := fwkk8s.MakePod(name).
			Namespace(h.Namespace).
			ReadyCondition().
			Labels(map[string]string{"app": eppharness.TestPoolName}).
			IP(address).
			Complete().
			ObjRef()
		status := pod.Status
		require.NoError(t, eppharness.K8sClient().Create(ctx, pod))
		pod.Status = status
		require.NoError(t, eppharness.K8sClient().Status().Update(ctx, pod))
	}
	h.SetPodMetrics(podMetrics)
	h.WaitForSync(2, "").WaitForMetricsDelivery()

	// A PUB socket drops what it sends before a subscription arrives. The EPP
	// subscribes to a pod's socket once it has discovered the pod.
	for replica, pub := range publishers {
		require.Eventually(t, func() bool {
			return slices.Contains(pub.(zmq4.Topics).Topics(), kvevents.DefaultConfig().TopicFilter)
		}, 10*time.Second, 10*time.Millisecond, "the EPP did not subscribe to replica %s", replica)
	}

	// publish sends the captured messages of the named stages from a replica's
	// socket. A socket numbers the messages it sends, as an engine's publisher
	// does, so the captured sequence number is replaced where a replica skips
	// stages of the capture.
	sent := map[string]uint64{}
	publish := func(replica string, stages ...string) {
		t.Helper()
		for _, stage := range stages {
			before := sent[replica]
			for _, message := range capture.Messages {
				if message.Stage != stage {
					continue
				}
				sequence := binary.BigEndian.AppendUint64(nil, sent[replica])
				require.NoError(t, publishers[replica].Send(zmq4.NewMsgFrom(message.Frames[0], sequence, message.Frames[2])))
				sent[replica]++
			}
			require.Greater(t, sent[replica], before, "the capture has no stage %s", stage)
		}
	}

	endpointA := net.JoinHostPort(replicaA, strconv.Itoa(fwkk8s.DefaultTestPort))
	endpointB := net.JoinHostPort(replicaB, strconv.Itoa(fwkk8s.DefaultTestPort))

	// route sends one chat completion for the captured prompt and returns the
	// endpoint the EPP picked and the score of each candidate.
	route := func() (string, map[string]float64) {
		t.Helper()
		body, err := json.Marshal(map[string]any{
			"model":      capture.Model,
			"messages":   []map[string]string{{"role": "user", "content": "the captured prompt"}},
			"max_tokens": 8,
		})
		require.NoError(t, err)
		requests := append(fwkepp.ReqRaw(map[string]string{
			":path":                      reqcommon.PathChatCompletions,
			reqcommon.RequestIDHeaderKey: uuid.NewString(),
		}, string(body)), ReqResponseOnly(
			map[string]string{"content-type": "application/json", ":status": "200"}, `{}`,
		)...)
		responses := streamRequests(t, h, requests, 4)

		envoyLb := responses[0].DynamicMetadata.GetFields()[metadata.DestinationEndpointNamespace].GetStructValue()
		require.NotNil(t, envoyLb)
		scores := map[string]float64{}
		for endpoint, score := range envoyLb.Fields[metadata.DestinationEndpointScoresKey].GetStructValue().GetFields() {
			scores[endpoint] = score.GetNumberValue()
		}
		return envoyLb.Fields[metadata.DestinationEndpointKey].GetStringValue(), scores
	}

	// The capture caches the full LMCache chunks of the prompt. A cached
	// block counts 1 on GPU and 0.8 on CPU. The producer truncates the sum to
	// whole blocks and the scorer divides it by the blocks of the request.
	blockSize := kvblock.DefaultTokenProcessorConfig().BlockSizeTokens
	requestBlocks := float64(len(capture.PromptTokenIDs) / blockSize)
	cachedBlocks := float64(len(capture.PromptTokenIDs) / capture.ChunkSize * (capture.ChunkSize / blockSize))
	onGPU := cachedBlocks / requestBlocks
	onCPU := math.Floor(cachedBlocks*0.8) / requestBlocks

	// expect waits until a request scores the replicas as wanted, then checks
	// that further requests score them the same and go to pick. An empty pick
	// leaves the choice between equal scores open.
	expect := func(want map[string]float64, pick string) {
		t.Helper()
		scored := func(got map[string]float64) bool {
			for endpoint, score := range want {
				if value, ok := got[endpoint]; !ok || math.Abs(value-score) > 1e-9 {
					return false
				}
			}
			return len(got) == len(want)
		}
		deadline := time.Now().Add(10 * time.Second)
		for _, got := route(); !scored(got); _, got = route() {
			require.True(t, time.Now().Before(deadline), "requests score the replicas %v, want %v", got, want)
			time.Sleep(20 * time.Millisecond)
		}
		for range 5 {
			endpoint, got := route()
			assert.InDeltaMapValues(t, want, got, 1e-9)
			if pick != "" {
				assert.Equal(t, pick, endpoint, "scores %v", got)
			}
		}
	}

	// Replica A computed the prompt, stored it in LMCache and evicted it from
	// GPU. Replica B announced nothing.
	publish(replicaA, "warm-up", "cold", "cpu-stored", "gpu-evicted")
	expect(map[string]float64{endpointA: onCPU, endpointB: 0}, endpointA)

	// Replica B computed the prompt and stored it in LMCache. Its GPU blocks
	// outscore replica A's copy in LMCache.
	publish(replicaB, "warm-up", "cold", "cpu-stored")
	expect(map[string]float64{endpointA: onCPU, endpointB: onGPU}, endpointB)

	// Replica B evicts the prompt from GPU. Both replicas hold it in LMCache
	// under the same chunk hashes.
	publish(replicaB, "gpu-evicted")
	expect(map[string]float64{endpointA: onCPU, endpointB: onCPU}, "")

	// LMCache on replica B withdraws the chunks. Replica A keeps its copy.
	publish(replicaB, "cpu-removed")
	expect(map[string]float64{endpointA: onCPU, endpointB: 0}, endpointA)

	// LMCache on replica A withdraws the chunks.
	publish(replicaA, "cpu-removed")
	expect(map[string]float64{endpointA: 0, endpointB: 0}, "")
}
