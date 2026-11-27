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
	"fmt"

	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	srckvevents "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/kvevents"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

var (
	_ fwkdl.EndpointExtractor = &Producer{}
	_ fwkdl.Registrant        = &Producer{}
)

// RegisterDependencies requests endpoint events for per-pod subscriptions.
// A global zmqEndpoint can leave discoverPods enabled but needs no such events,
// and a bound kv-events-source follows the endpoints itself.
func (p *Producer) RegisterDependencies(r fwkdl.Registrar) error {
	kc := p.kvEventsConfig
	if p.sharedEvents != nil || !kc.DiscoverPods || kc.PodDiscoveryConfig == nil || kc.ZMQEndpoint != "" {
		return nil
	}
	return r.Register(fwkdl.PendingRegistration{
		Owner:      p.TypedName(),
		SourceType: sourcenotifications.EndpointNotificationSourceType,
		Extractor:  p,
		DefaultSource: sourcenotifications.NewEndpointDataSource(
			sourcenotifications.EndpointNotificationSourceType, sourcenotifications.EndpointNotificationSourceType),
	})
}

// Extract subscribes matching endpoints to per-pod KV events when discovery is
// enabled. Deleted endpoints and endpoints that stop matching lose their
// subscriber and cached index entries.
func (p *Producer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	if p.sharedEvents != nil || !p.kvEventsConfig.DiscoverPods || p.kvEventsConfig.PodDiscoveryConfig == nil {
		return nil
	}
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.ID.Name == "" {
		return nil
	}

	logger := log.FromContext(ctx).WithName(p.typedName.String())
	endpointKey := meta.ID.String()

	matchesSelector := p.podSelector == nil || p.podSelector.Matches(labels.Set(meta.Labels))
	switch {
	case event.Type == fwkdl.EventAddOrUpdate && matchesSelector:
		return p.ensureSubscriber(ctx, meta)
	case event.Type == fwkdl.EventAddOrUpdate || event.Type == fwkdl.EventDelete:
		removed := p.subscribersManager.RemoveSubscriber(ctx, endpointKey)
		if !removed && event.Type == fwkdl.EventAddOrUpdate {
			return nil
		}
		logger.V(logging.DEBUG).Info("Removed KV-events subscriber", "endpoint", endpointKey)
	}
	return nil
}

// ensureSubscriber idempotently installs a KV-events subscriber for the given
// endpoint, dialing SocketPort plus a rank offset to match standard inference-engine port offsetting
// (one ZMQ PUB socket per DP rank on the same pod IP), or the endpoint's
// ZMQPort when discovery sets one. A ZMQPort endpoint receives live events
// only, without replay or snapshot.
func (p *Producer) ensureSubscriber(ctx context.Context, meta *fwkdl.EndpointMetadata) error {
	if meta == nil || meta.Address == "" {
		return nil
	}
	endpointKey := meta.ID.String()
	zmqEndpoint, replayEndpoint, snapshotEndpoint := p.subscriberEndpoints(meta)
	sourceEndpoint := p.indexPodIdentifier(meta)

	logger := log.FromContext(ctx).WithName(p.typedName.String())
	// subscriberCtx is plugin-lifetime; caller ctx would tear subscribers
	// down on request completion.
	if err := p.subscribersManager.EnsureSubscriber(p.subscriberCtx, endpointKey,
		sourceEndpoint, zmqEndpoint, replayEndpoint, snapshotEndpoint, p.kvEventsConfig.TopicFilter, true); err != nil {
		logger.Error(err, "Failed to ensure KV-events subscriber for endpoint",
			"endpoint", endpointKey, "address", meta.Address)
		return fmt.Errorf("ensure subscriber for %s: %w", endpointKey, err)
	}
	logger.V(logging.DEBUG).Info("Ensured KV-events subscriber",
		"endpoint", endpointKey, "zmq", zmqEndpoint, "replay", replayEndpoint, "snapshot", snapshotEndpoint)
	return nil
}

// subscriberEndpoints returns the endpoint's KV-events, replay, and snapshot
// socket addresses.
func (p *Producer) subscriberEndpoints(meta *fwkdl.EndpointMetadata) (zmqEndpoint, replayEndpoint, snapshotEndpoint string) {
	return srckvevents.SubscriberEndpoints(meta, p.kvEventsConfig.PodDiscoveryConfig)
}
