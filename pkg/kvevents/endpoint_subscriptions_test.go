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
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func newTestEndpointSubscriptions(t *testing.T, cfg *Config) (*EndpointSubscriptions, *SubscriberManager) {
	t.Helper()
	// Subscriber goroutines outlive the test; a discarded logger avoids a
	// race with the t-bound logger after cleanup.
	ctx := log.IntoContext(context.Background(), logr.Discard())
	pool, err := NewPool(cfg, nil, nil, nil)
	require.NoError(t, err)
	manager := NewSubscriberManager(pool)
	t.Cleanup(func() { manager.Shutdown(ctx) })
	subscriptions, err := NewEndpointSubscriptions(ctx, cfg, manager)
	require.NoError(t, err)
	return subscriptions, manager
}

func discoveryConfig(selector string) *Config {
	cfg := DefaultConfig()
	cfg.DiscoverPods = true
	cfg.PodDiscoveryConfig = DefaultPodReconcilerConfig()
	cfg.PodDiscoveryConfig.SocketPort = 5557
	cfg.PodDiscoveryConfig.PodLabelSelector = selector
	return cfg
}

func TestEndpointSubscriptionsDialsRankOffsetAndServingEndpoint(t *testing.T) {
	cfg := discoveryConfig("")
	cfg.PodDiscoveryConfig.ReplaySocketPort = 5600
	subscriptions, manager := newTestEndpointSubscriptions(t, cfg)
	ctx := context.Background()
	require.True(t, subscriptions.Enabled())

	require.NoError(t, subscriptions.Ensure(ctx, "ns/pod-a-rank-3", "10.0.0.1", "8003", 3))
	require.NoError(t, subscriptions.Ensure(ctx, "ns/pod-v6", "fd00::1", "8080", 0))
	require.NoError(t, subscriptions.Ensure(ctx, "ns/pod-a-rank-3", "10.0.0.1", "8003", 3), "duplicate ensure is idempotent")
	require.NoError(t, subscriptions.Ensure(ctx, "ns/no-address", "", "8080", 0), "no address is a no-op")

	manager.mu.Lock()
	defer manager.mu.Unlock()
	require.Len(t, manager.subscribers, 2)
	rank := manager.subscribers["ns/pod-a-rank-3"]
	assert.Equal(t, "tcp://10.0.0.1:5560", rank.endpoint)
	assert.Equal(t, "tcp://10.0.0.1:5603", rank.replayEndpoint)
	assert.Equal(t, "10.0.0.1:8003", rank.sourceEndpoint)
	assert.Equal(t, "tcp://[fd00::1]:5557", manager.subscribers["ns/pod-v6"].endpoint)
}

func TestEndpointSubscriptionsRemoveReportsExistence(t *testing.T) {
	subscriptions, _ := newTestEndpointSubscriptions(t, discoveryConfig(""))
	ctx := context.Background()
	require.NoError(t, subscriptions.Ensure(ctx, "ns/pod-a", "10.0.0.1", "8080", 0))
	assert.True(t, subscriptions.Remove(ctx, "ns/pod-a"))
	assert.False(t, subscriptions.Remove(ctx, "ns/pod-a"))
}

func TestEndpointSubscriptionsSelector(t *testing.T) {
	subscriptions, _ := newTestEndpointSubscriptions(t, discoveryConfig("llm-d.ai/role=prefill"))
	assert.True(t, subscriptions.Matches(map[string]string{"llm-d.ai/role": "prefill", "app": "vllm"}))
	assert.False(t, subscriptions.Matches(map[string]string{"llm-d.ai/role": "decode"}))
	assert.False(t, subscriptions.Matches(nil))

	unselective, _ := newTestEndpointSubscriptions(t, discoveryConfig(""))
	assert.True(t, unselective.Matches(nil))

	_, err := NewEndpointSubscriptions(context.Background(), discoveryConfig("not a selector!"), nil)
	require.ErrorContains(t, err, "podLabelSelector")
}

func TestEndpointSubscriptionsDisabledWithoutDiscovery(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DiscoverPods = false
	subscriptions, manager := newTestEndpointSubscriptions(t, cfg)
	ctx := context.Background()
	assert.False(t, subscriptions.Enabled())
	require.NoError(t, subscriptions.Ensure(ctx, "ns/pod-a", "10.0.0.1", "8080", 0))
	assert.False(t, subscriptions.Remove(ctx, "ns/pod-a"))
	ids, _ := manager.GetActiveSubscribers()
	assert.Empty(t, ids)

	var none *EndpointSubscriptions
	assert.False(t, none.Enabled())
}
