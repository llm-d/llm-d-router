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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// discardCtx returns a context whose logger drops everything. The kvevents
// subscriber spawns a background goroutine that logs via this context's
// logger; under -race a test-bound logger writing after t.Run cleanup races
// with the testing framework. Discarding sidesteps that without losing
// fidelity (we don't assert on log output).
func discardCtx(t *testing.T) context.Context {
	t.Helper()
	return log.IntoContext(context.Background(), logr.Discard())
}

// newExtractorScorer builds a minimal Scorer wired only with the bits the
// EndpointExtractor path touches: a SubscriberManager, the kvevents config,
// and a typed name. Skipping the full New() lets us exercise the data-layer
// callbacks without standing up a tokenizer pool or KV index.
func newExtractorScorer(discoverPods bool) *Scorer {
	cfg := kvevents.DefaultConfig()
	cfg.DiscoverPods = discoverPods
	cfg.PodDiscoveryConfig = kvevents.DefaultPodReconcilerConfig()
	cfg.PodDiscoveryConfig.SocketPort = 5557

	return &Scorer{
		typedName:          plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: PrecisePrefixCachePluginType},
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

func TestScorer_EndpointExtractor_InterfaceContract(t *testing.T) {
	ctx := discardCtx(t)
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(ctx)

	assert.Equal(t, fwkdl.EndpointEventReflectType, s.ExpectedInputType(),
		"ExpectedInputType must report EndpointEvent for data-layer compatibility checks")

	var _ fwkdl.EndpointExtractor = s
	assert.True(t, reflect.TypeOf(s).Implements(reflect.TypeFor[fwkdl.EndpointExtractor]()))
}

func TestScorer_ExtractEndpoint_AddAndDelete(t *testing.T) {
	ctx := discardCtx(t)
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(ctx)

	ep := newEndpoint("pod-a", "10.0.0.1")
	wantKey := "ns/pod-a"
	wantEndpoint := "tcp://10.0.0.1:5557"

	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))

	ids, endpoints := s.subscribersManager.GetActiveSubscribers()
	require.Equal(t, []string{wantKey}, ids, "add/update should register exactly one subscriber")
	require.Equal(t, []string{wantEndpoint}, endpoints, "ZMQ endpoint must derive from address + SocketPort")

	// Re-add is idempotent (EnsureSubscriber dedups on identical endpoint).
	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))
	ids, _ = s.subscribersManager.GetActiveSubscribers()
	assert.Len(t, ids, 1, "duplicate add must not create a second subscriber")

	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: ep,
	}))
	ids, _ = s.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids, "delete should tear down the subscriber")
}

func TestScorer_ExtractEndpoint_DiscoverPodsDisabledIsNoOp(t *testing.T) {
	ctx := discardCtx(t)
	// DiscoverPods=false is the global-socket-mode toggle: per-pod discovery
	// must be skipped so a single shared subscriber drives the index.
	s := newExtractorScorer(false)
	defer s.subscribersManager.Shutdown(ctx)

	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: newEndpoint("pod-a", "10.0.0.1"),
	}))

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids, "no per-pod subscriber should be registered when DiscoverPods is disabled")
}

func TestScorer_ExtractEndpoint_IgnoresMissingMetadata(t *testing.T) {
	ctx := discardCtx(t)
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(ctx)

	// Endpoint with empty address — nothing to subscribe to.
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
	}, nil)

	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	}))

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids, "endpoints without an address must be ignored")
}

// Regression: SubscriberManager binds the subscriber goroutine's lifetime
// to the ctx passed to EnsureSubscriber. If we naively pass the caller's
// (request-scoped) ctx, the subscriber dies when the request ends and we
// have to recreate it on every subsequent request — exactly what showed up
// in production logs after the data-layer refactor. ensureSubscriber must
// use the long-lived subscriberCtx instead.
func TestScorer_EnsureSubscriber_SurvivesRequestCtxCancel(t *testing.T) {
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(context.Background())

	// Simulate a request-scoped ctx that ends as soon as the call returns.
	reqCtx, cancel := context.WithCancel(context.Background())

	require.NoError(t, s.ensureSubscriber(reqCtx, &fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
		Address:        "10.0.0.1", Port: "8080",
	}))

	cancel() // request finishes — must not tear the subscriber down.

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	assert.ElementsMatch(t, []string{"ns/pod-a"}, ids,
		"subscriber must outlive the caller's request-scoped context")
}

// Backwards-compat: configs that don't wire the endpoint-notification-source
// rely on Score()-time subscriber discovery. Verify the legacy path still
// installs a subscriber for each endpoint Score() sees.
func TestScorer_LegacyInScoreDiscovery_EnsuresSubscribers(t *testing.T) {
	ctx := discardCtx(t)
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(ctx)

	endpoints := []scheduling.Endpoint{
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
			Address:        "10.0.0.1", Port: "8080",
		}, nil, nil),
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-b"},
			Address:        "10.0.0.2", Port: "8080",
		}, nil, nil),
	}

	s.ensureSubscribersForEndpoints(ctx, endpoints)

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	assert.ElementsMatch(t, []string{"ns/pod-a", "ns/pod-b"}, ids,
		"legacy in-Score discovery must subscribe to every candidate endpoint")
}

// Once the data layer is wired (ExtractEndpoint has been called even once),
// the legacy in-Score discovery must stop running so the data layer remains
// the sole authority over per-pod subscriber lifecycle.
func TestScorer_LegacyInScoreDiscovery_DisabledOnceExtractorObserved(t *testing.T) {
	ctx := discardCtx(t)
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(ctx)

	// Simulate the data layer dispatching even an unrelated event — the call
	// itself proves the source is wired.
	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: newEndpoint("pod-x", "10.0.0.99"),
	}))

	// Now Score()-time discovery should be a no-op even with fresh endpoints.
	endpoints := []scheduling.Endpoint{
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
			Address:        "10.0.0.1", Port: "8080",
		}, nil, nil),
	}
	s.ensureSubscribersForEndpoints(ctx, endpoints)

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	assert.NotContains(t, ids, "ns/pod-a",
		"legacy path must not subscribe once the data-layer extractor has been observed")
}

func TestScorer_LegacyInScoreDiscovery_DiscoverPodsDisabled(t *testing.T) {
	ctx := discardCtx(t)
	// Global-socket mode: per-pod subscribers must not be opened.
	s := newExtractorScorer(false)
	defer s.subscribersManager.Shutdown(ctx)

	endpoints := []scheduling.Endpoint{
		scheduling.NewEndpoint(&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
			Address:        "10.0.0.1", Port: "8080",
		}, nil, nil),
	}

	s.ensureSubscribersForEndpoints(ctx, endpoints)

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids)
}

// Delete events from the data layer may omit address fields. The subscriber
// is keyed by NamespacedName, so delete must succeed regardless of address
// presence — otherwise stale subscribers leak when pods disappear.
func TestScorer_ExtractEndpoint_DeleteWithMissingAddressRemovesExistingSubscriber(t *testing.T) {
	ctx := discardCtx(t)
	s := newExtractorScorer(true)
	defer s.subscribersManager.Shutdown(ctx)

	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: newEndpoint("pod-a", "10.0.0.1"),
	}))

	ids, _ := s.subscribersManager.GetActiveSubscribers()
	require.Len(t, ids, 1, "sanity check: expected subscriber to be registered before delete")

	deleteEndpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
	}, nil)

	require.NoError(t, s.ExtractEndpoint(ctx, fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: deleteEndpoint,
	}))

	ids, _ = s.subscribersManager.GetActiveSubscribers()
	assert.Empty(t, ids, "delete events must remove an existing subscriber even when the address is missing")
}
