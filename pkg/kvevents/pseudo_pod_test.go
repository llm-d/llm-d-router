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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
)

// topicAdapter takes the pod identifier and model from the topic
// "kv@<pod>@<model>", as the engine adapters do, and emits one block-stored
// event carrying 16 tokens on the "lmcache-l1" tier.
type topicAdapter struct{}

func (a *topicAdapter) ParseMessage(msg *RawMessage) (string, string, EventBatch, error) {
	parts := strings.Split(msg.Topic, "@")
	return parts[1], parts[2], EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: []uint64{uint64(msg.Payload[0])},
				Tokens:      makeTokens(16),
				DeviceTier:  "lmcache-l1",
			},
		},
	}, nil
}

func (a *topicAdapter) ShardingKey(msg *RawMessage) string {
	return strings.Split(msg.Topic, "@")[1]
}

// A pseudo-pod topic segment is indexed as an opaque pod identifier: it is
// stored, passes a Lookup filter naming it, and is removed by a reset.
func TestPool_PseudoPodTopicIndexedAsOpaqueIdentifier(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tokenProcessor := newTestPool(t, 16)
	pool.adapter = &topicAdapter{}

	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@node:n1@test-model", Payload: []byte{1}})
	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@pool:fs@test-model", Payload: []byte{1}})

	keys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, makeTokens(16), "test-model", nil)
	require.NoError(t, err)
	require.Len(t, keys, 1)

	result, err := idx.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	assert.ElementsMatch(t, []kvblock.PodEntry{
		{PodIdentifier: "node:n1", DeviceTier: "lmcache-l1"},
		{PodIdentifier: "pool:fs", DeviceTier: "lmcache-l1"},
	}, result[keys[0]])

	// A filter naming only real endpoints still returns the pool entry;
	// the node entry needs its identifier in the filter.
	result, err = idx.Lookup(ctx, keys, sets.New("10.0.0.1:8000"))
	require.NoError(t, err)
	assert.Equal(t, []kvblock.PodEntry{{PodIdentifier: "pool:fs", DeviceTier: "lmcache-l1"}}, result[keys[0]])

	result, err = idx.Lookup(ctx, keys, sets.New("10.0.0.1:8000", kvblock.NodePseudoPod("n1")))
	require.NoError(t, err)
	assert.Len(t, result[keys[0]], 2)

	pool.processRawMessage(ctx, &RawMessage{Topic: "kv@node:n1@test-model", reset: true})
	result, err = idx.Lookup(ctx, keys, nil)
	require.NoError(t, err)
	assert.Equal(t, []kvblock.PodEntry{{PodIdentifier: "pool:fs", DeviceTier: "lmcache-l1"}}, result[keys[0]])
}
