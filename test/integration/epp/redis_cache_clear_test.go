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
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	miniredisserver "github.com/alicebob/miniredis/v2/server"
	"github.com/go-zeromq/zmq4"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/screener/disaggregatedsetrollout"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

func TestRedisClear_SharedRolloutCoordination(t *testing.T) {
	ctx := t.Context()
	server := miniredis.RunT(t)
	// Let miniredis filter typed scans. Return coordination keys first for
	// untyped scans so an early exit leaves stale pod entries.
	server.Server().SetPreHook(func(peer *miniredisserver.Peer, command string, args ...string) bool {
		if command != "SCAN" || slices.Contains(args, "type") {
			return false
		}
		keys := server.Keys()
		slices.Reverse(keys)
		peer.WriteLen(2)
		peer.WriteBulk("0")
		peer.WriteStrings(keys)
		return true
	})

	renderer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/completions/render" {
			_, _ = w.Write([]byte(`[{"token_ids":[1,2,3,4]}]`))
			return
		}
		_, _ = w.Write([]byte(`{"token_ids":[1,2,3,4]}`))
	}))
	t.Cleanup(renderer.Close)

	port, err := fwknet.GetFreePort()
	require.NoError(t, err)
	endpoint := fmt.Sprintf("tcp://127.0.0.1:%d", port)
	rolloutName := "redis-clear-" + uuid.NewString()
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
- type: precise-prefix-cache-producer
  parameters:
    tokenProcessorConfig:
      blockSizeTokens: 2
    indexerConfig:
      kvBlockIndexConfig:
        redisConfig:
          address: redis://%s/0
    kvEventsConfig:
      discoverPods: false
      zmqEndpoint: %s
- type: redis-syncer
  name: shared-redis
  parameters:
    address: %s
    db: 0
- type: disaggregatedset-rollout-screener
  name: %s
  parameters:
    scope:
      labelSelector: app=%s
    revisionGating:
      mode: sum
      roleLabelKey: %s
      requiredRoles: [prefill, decode]
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
- type: decode-filter
- type: max-score-picker
- type: single-profile-handler
- type: mock-metrics-source
requestHandler:
  parsers:
  - pluginRef: openai-parser
dataLayer:
  injectDefaults: false
  crossReplica:
    syncerPluginRef: shared-redis
  sources:
  - pluginRef: mock-metrics-source
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: decode-filter
  - pluginRef: prefix-cache-scorer
  - pluginRef: max-score-picker
`, modelMyModelTarget, renderer.URL, server.Addr(), endpoint, server.Addr(), rolloutName, testPoolName, bylabel.RoleLabel)
	h := NewTestHarness(ctx, t, WithStandardMode(), WithConfigText(configText)).WithBaseResources()
	h.WithPods([]PodState{P(0, 0, 0.1), P(1, 0, 0.1), P(2, 0, 0.1), P(3, 0, 0.1)})

	// The new revision has decode pods but no prefill pod, so requests stay on
	// v1 while revision decisions are coordinated in Redis.
	for i := range 4 {
		pod := &corev1.Pod{}
		key := types.NamespacedName{Namespace: h.Namespace, Name: fmt.Sprintf("pod-%d", i)}
		require.NoError(t, k8sClient.Get(ctx, key, pod))
		pod.Labels[disaggregatedsetrollout.DefaultRevisionLabel] = "v1"
		pod.Labels[bylabel.RoleLabel] = "decode"
		if i == 2 {
			pod.Labels[bylabel.RoleLabel] = prefillProfile
		}
		if i == 3 {
			pod.Labels[disaggregatedsetrollout.DefaultRevisionLabel] = "v2"
		}
		require.NoError(t, k8sClient.Update(ctx, pod))
	}
	h.WaitForSync(4, modelMyModel)
	require.Eventually(t, func() bool {
		for _, pod := range h.Datastore.PodList(datastore.AllPodsPredicate) {
			if pod.GetMetadata().Labels[disaggregatedsetrollout.DefaultRevisionLabel] == "" {
				return false
			}
		}
		families, err := ctrlmetrics.Registry.Gather()
		if err != nil {
			return false
		}
		for _, family := range families {
			if family.GetName() != "llm_d_epp_disaggregatedset_revision_gating_share" {
				continue
			}
			shares := make(map[string]float64)
			for _, metric := range family.Metric {
				var pluginName, revision string
				for _, label := range metric.Label {
					switch label.GetName() {
					case "plugin_name":
						pluginName = label.GetValue()
					case "revision":
						revision = label.GetValue()
					}
				}
				if pluginName == rolloutName {
					shares[revision] = metric.GetGauge().GetValue()
				}
			}
			return len(shares) == 2 && shares["v1"] == 1 && shares["v2"] == 0
		}
		return false
	}, 10*time.Second, 50*time.Millisecond, "both revisions must reach the rollout screener")

	pub := zmq4.NewPub(ctx)
	require.NoError(t, pub.Dial(endpoint))
	t.Cleanup(func() { _ = pub.Close() })
	var seq uint64
	publish := func(pod string, event []any) {
		t.Helper()
		payload, err := msgpack.Marshal([]any{1234567890.0, []any{event}, nil})
		require.NoError(t, err)
		seq++
		topic := []byte(fmt.Sprintf("kv@%s@%s", pod, modelMyModelTarget))
		require.NoError(t, pub.Send(zmq4.NewMsgFrom(topic, binary.BigEndian.AppendUint64(nil, seq), payload)))
	}

	target := kvblock.PodEntry{PodIdentifier: "192.168.1.1:8000", DeviceTier: "gpu"}
	keeper := kvblock.PodEntry{PodIdentifier: "192.168.1.2:8000", DeviceTier: "gpu"}
	targetField, err := json.Marshal(target)
	require.NoError(t, err)
	keeperField, err := json.Marshal(keeper)
	require.NoError(t, err)
	processor, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 2})
	require.NoError(t, err)
	keys, err := processor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, []uint32{1, 2, 3, 4}, modelMyModelTarget, nil)
	require.NoError(t, err)
	require.Len(t, keys, 2)

	require.Eventually(t, func() bool {
		publish(target.PodIdentifier, []any{string(kvevents.EventTypeBlockStored),
			[]uint64{11, 12}, nil, []uint32{1, 2, 3, 4}, 2, nil, "gpu"})
		publish(keeper.PodIdentifier, []any{string(kvevents.EventTypeBlockStored),
			[]uint64{11}, nil, []uint32{1, 2}, 2, nil, "gpu"})
		first, err := server.HKeys(keys[0].String())
		if err != nil || !slices.Contains(first, string(targetField)) || !slices.Contains(first, string(keeperField)) {
			return false
		}
		second, err := server.HKeys(keys[1].String())
		return err == nil && slices.Contains(second, string(targetField))
	}, 10*time.Second, 50*time.Millisecond, "KV events must populate the shared Redis cache index")

	require.Equal(t, target.PodIdentifier, sendRoutedRequest(t, h, nil))
	var coordinationKeys []string
	for _, key := range server.Keys() {
		if strings.HasPrefix(key, "coord:") {
			coordinationKeys = append(coordinationKeys, key)
		}
	}
	require.Len(t, coordinationKeys, 1, "the request must create a revision coordination record")
	decision, err := server.Get(coordinationKeys[0])
	require.NoError(t, err)

	publish(target.PodIdentifier, []any{string(kvevents.EventTypeAllBlocksCleared)})
	require.Eventually(t, func() bool {
		fields, err := server.HKeys(keys[0].String())
		return err == nil && slices.Equal(fields, []string{string(keeperField)}) && !server.Exists(keys[1].String())
	}, 5*time.Second, 50*time.Millisecond, "AllBlocksCleared must remove every target pod entry despite coordination records")

	actual, err := server.Get(coordinationKeys[0])
	require.NoError(t, err)
	require.Equal(t, decision, actual, "cache cleanup must preserve the revision coordination record")
	require.Equal(t, keeper.PodIdentifier, sendRoutedRequest(t, h, nil),
		"routing must prefer the pod that still caches the prefix")
}
