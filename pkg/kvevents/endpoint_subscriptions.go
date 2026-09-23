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
	"fmt"
	"net"
	"strconv"

	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

// EndpointSubscriptions keeps one subscriber per discovered endpoint that
// matches the pod label selector of a per-pod discovery configuration. It is
// disabled, with every method a no-op, when the configuration does not enable
// per-pod discovery. Subscribers are bound to the context given at
// construction, which outlives any request.
type EndpointSubscriptions struct {
	ctx      context.Context
	manager  *SubscriberManager
	config   *PodDiscoveryConfig
	topic    string
	selector labels.Selector // nil matches every endpoint.
}

// NewEndpointSubscriptions parses the discovery configuration's pod label
// selector. The result is disabled when cfg does not enable per-pod discovery.
func NewEndpointSubscriptions(ctx context.Context, cfg *Config, manager *SubscriberManager) (*EndpointSubscriptions, error) {
	s := &EndpointSubscriptions{ctx: ctx, manager: manager}
	if cfg == nil || !cfg.DiscoverPods || cfg.PodDiscoveryConfig == nil {
		return s, nil
	}
	s.config, s.topic = cfg.PodDiscoveryConfig, cfg.TopicFilter
	if s.config.PodLabelSelector != "" {
		selector, err := labels.Parse(s.config.PodLabelSelector)
		if err != nil {
			return nil, fmt.Errorf("invalid kvEventsConfig.podDiscoveryConfig.podLabelSelector %q: %w",
				s.config.PodLabelSelector, err)
		}
		s.selector = selector
	}
	return s, nil
}

// Enabled reports whether per-pod discovery is configured. A nil receiver is
// disabled.
func (s *EndpointSubscriptions) Enabled() bool {
	return s != nil && s.config != nil
}

// Matches reports whether an endpoint with the given labels is subscribed to.
func (s *EndpointSubscriptions) Matches(endpointLabels map[string]string) bool {
	return s.selector == nil || s.selector.Matches(labels.Set(endpointLabels))
}

// Ensure idempotently installs a subscriber for the endpoint, dialing
// SocketPort + rankIndex to match inference-engine port offsetting (one ZMQ
// PUB socket per DP rank on the same pod IP), and the replay port likewise.
// The subscriber's source endpoint is address:port. No-op without an address.
func (s *EndpointSubscriptions) Ensure(ctx context.Context, podIdentifier, address, port string, rankIndex int) error {
	if !s.Enabled() || address == "" {
		return nil
	}
	endpoint := "tcp://" + net.JoinHostPort(address, strconv.Itoa(s.config.SocketPort+rankIndex))
	replayEndpoint := ""
	if replayPort := s.config.EffectiveReplayPort(); replayPort > 0 {
		replayEndpoint = "tcp://" + net.JoinHostPort(address, strconv.Itoa(replayPort+rankIndex))
	}
	logger := log.FromContext(ctx)
	// s.ctx is plugin-lifetime; the caller's ctx would tear the subscriber
	// down on request completion.
	if err := s.manager.EnsureSubscriber(s.ctx, podIdentifier, address+":"+port,
		endpoint, replayEndpoint, s.topic, true); err != nil {
		logger.Error(err, "Failed to ensure KV-events subscriber for endpoint",
			"endpoint", podIdentifier, "address", address)
		return fmt.Errorf("ensure subscriber for %s: %w", podIdentifier, err)
	}
	logger.V(logging.DEBUG).Info("Ensured KV-events subscriber",
		"endpoint", podIdentifier, "zmq", endpoint, "replay", replayEndpoint)
	return nil
}

// Remove drops the endpoint's subscriber and reports whether one existed.
func (s *EndpointSubscriptions) Remove(ctx context.Context, podIdentifier string) bool {
	if !s.Enabled() {
		return false
	}
	removed := s.manager.RemoveSubscriber(ctx, podIdentifier)
	if removed {
		log.FromContext(ctx).V(logging.DEBUG).Info("Removed KV-events subscriber", "endpoint", podIdentifier)
	}
	return removed
}
