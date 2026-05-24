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
	"context"
	"reflect"
	"testing"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Avoid a -race ding from subscriber goroutines writing through a t-bound
// logger after t.Run cleanup.
func discardCtx(t *testing.T) context.Context {
	t.Helper()
	return log.IntoContext(context.Background(), logr.Discard())
}

func newExtractorProducer(discoverPods bool) *Producer {
	cfg := kvevents.DefaultConfig()
	cfg.DiscoverPods = discoverPods
	cfg.PodDiscoveryConfig = kvevents.DefaultPodReconcilerConfig()
	cfg.PodDiscoveryConfig.SocketPort = 5557

	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: PluginType},
		subscribersManager: kvevents.NewSubscriberManager(kvevents.NewPool(cfg, nil, nil, nil)),
		kvEventsConfig:     cfg,
		subscriberCtx:      context.Background(),
	}
}

func newEndpoint(name, addr string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: name},
		Address:        addr,
		Port:           "8080",
	}, nil)
}

func TestProducer_EndpointExtractor_InterfaceContract(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(ctx)

	assert.Equal(t, fwkdl.EndpointEventReflectType, p.ExpectedInputType())

	var _ fwkdl.EndpointExtractor = p
	assert.True(t, reflect.TypeOf(p).Implements(reflect.TypeFor[fwkdl.EndpointExtractor]()))
}

func TestProducer_ExtractEndpoint_AddAndDelete(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(ctx)

	ep := newEndpoint("pod-a", "10.0.0.1")
	wantKey := "ns/pod-a"
	wantEndpoint := "tcp://10.0.0.1:5557"

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))

	ids, endpoints := p.subscribersManager.GetActiveSubscribers()
	require.Equal(t, []string{wantKey}, ids)
	require.Equal(t, []string{wantEndpoint}, endpoints)

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))
	ids, _ = p.subscribersManager.GetActiveSubscribers()
	assert.Len(t, ids, 1, "duplicate add must not create a second subscriber")

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: ep,
	}))
	ids, _ = p.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids)
}

// DiscoverPods=false → global-socket mode, per-pod discovery off.
func TestProducer_ExtractEndpoint_DiscoverPodsDisabledIsNoOp(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(false)
	defer p.subscribersManager.Shutdown(ctx)

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: newEndpoint("pod-a", "10.0.0.1"),
	}))

	ids, _ := p.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids)
}

func TestProducer_ExtractEndpoint_IgnoresMissingMetadata(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(ctx)

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
	}, nil)

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))

	ids, _ := p.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids)
}

// Regression: subscribers must survive request-ctx cancellation.
func TestProducer_EnsureSubscriber_SurvivesRequestCtxCancel(t *testing.T) {
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(context.Background())

	reqCtx, cancel := context.WithCancel(context.Background())

	require.NoError(t, p.ensureSubscriber(reqCtx, &fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
		Address:        "10.0.0.1", Port: "8080",
	}))

	cancel()

	ids, _ := p.subscribersManager.GetActiveSubscribers()
	assert.ElementsMatch(t, []string{"ns/pod-a"}, ids)
}

// Per-rank subscribers at SocketPort + RankIndex (vLLM offset_endpoint_port).
func TestProducer_ExtractEndpoint_OffsetsZMQPortByRankIndex(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(ctx)

	endpoints := []struct {
		name    string
		address string
		rank    int
		wantZMQ string
	}{
		{name: "pod-a-rank-0", address: "10.0.0.1", rank: 0, wantZMQ: "tcp://10.0.0.1:5557"},
		{name: "pod-a-rank-1", address: "10.0.0.1", rank: 1, wantZMQ: "tcp://10.0.0.1:5558"},
		{name: "pod-a-rank-2", address: "10.0.0.1", rank: 2, wantZMQ: "tcp://10.0.0.1:5559"},
	}

	for _, ep := range endpoints {
		require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
			Type: fwkdl.EventAddOrUpdate,
			Endpoint: fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: ep.name},
				Address:        ep.address,
				Port:           "8080",
				RankIndex:      ep.rank,
			}, nil),
		}))
	}

	ids, zmqEndpoints := p.subscribersManager.GetActiveSubscribers()
	gotByID := make(map[string]string, len(ids))
	for i, id := range ids {
		gotByID[id] = zmqEndpoints[i]
	}
	for _, ep := range endpoints {
		key := "ns/" + ep.name
		assert.Equal(t, ep.wantZMQ, gotByID[key],
			"rank %d must subscribe at SocketPort + rank", ep.rank)
	}
}

// RankIndex=0 must dial the base SocketPort unchanged.
func TestProducer_ExtractEndpoint_SingleRankUsesBaseSocketPort(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(ctx)

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type: fwkdl.EventAddOrUpdate,
		Endpoint: fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
			Address:        "10.0.0.1",
			Port:           "8080",
			// RankIndex stays at its zero value.
		}, nil),
	}))

	_, zmqEndpoints := p.subscribersManager.GetActiveSubscribers()
	assert.Equal(t, []string{"tcp://10.0.0.1:5557"}, zmqEndpoints,
		"single-rank pod (RankIndex=0) must dial the base SocketPort")
}

// Delete by NamespacedName must work even when the event has no address.
func TestProducer_ExtractEndpoint_DeleteWithMissingAddressRemovesExistingSubscriber(t *testing.T) {
	ctx := discardCtx(t)
	p := newExtractorProducer(true)
	defer p.subscribersManager.Shutdown(ctx)

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: newEndpoint("pod-a", "10.0.0.1"),
	}))

	ids, _ := p.subscribersManager.GetActiveSubscribers()
	require.Len(t, ids, 1)

	deleteEndpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
	}, nil)

	require.NoError(t, p.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: deleteEndpoint,
	}))

	ids, _ = p.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids)
}
