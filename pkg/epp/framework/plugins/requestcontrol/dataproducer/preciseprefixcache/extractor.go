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
	"reflect"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

var _ fwkdl.EndpointExtractor = &Producer{}

func (p *Producer) ExpectedInputType() reflect.Type {
	return fwkdl.EndpointEventReflectType
}

// ExtractEndpoint: add/update installs a per-pod ZMQ subscriber, delete tears
// it down. Sole per-pod discovery path; global socket mode is the alternative.
func (p *Producer) ExtractEndpoint(ctx context.Context, event fwkdl.EndpointEvent) error {
	if !p.kvEventsConfig.DiscoverPods || p.kvEventsConfig.PodDiscoveryConfig == nil {
		return nil
	}
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.NamespacedName.Name == "" {
		return nil
	}

	logger := log.FromContext(ctx).WithName(p.typedName.String())
	endpointKey := meta.NamespacedName.String()

	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		if err := p.ensureSubscriber(ctx, meta); err != nil {
			return err
		}
		logger.V(logging.DEBUG).Info("Adding subscriber", "endpoint", endpointKey)
	case fwkdl.EventDelete:
		p.subscribersManager.RemoveSubscriber(ctx, endpointKey)
		logger.V(logging.DEBUG).Info("Removed KV-events subscriber", "endpoint", endpointKey)
	}
	return nil
}

func (p *Producer) ensureSubscriber(ctx context.Context, meta *fwkdl.EndpointMetadata) error {
	if meta == nil || meta.Address == "" {
		return nil
	}
	endpointKey := meta.NamespacedName.String()
	// vLLM offset_endpoint_port: rank N binds at SocketPort + N.
	port := p.kvEventsConfig.PodDiscoveryConfig.SocketPort + meta.GetRankIndex()
	zmqEndpoint := fmt.Sprintf("tcp://%s:%d", meta.Address, port)

	logger := log.FromContext(ctx).WithName(p.typedName.String())
	// Use subscriberCtx (plugin-lifetime), not the caller's ctx.
	if err := p.subscribersManager.EnsureSubscriber(p.subscriberCtx, endpointKey,
		zmqEndpoint, p.kvEventsConfig.TopicFilter, true); err != nil {
		logger.Error(err, "Failed to ensure KV-events subscriber for endpoint",
			"endpoint", endpointKey, "address", meta.Address)
		return fmt.Errorf("ensure subscriber for %s: %w", endpointKey, err)
	}
	logger.V(logging.DEBUG).Info("Ensured KV-events subscriber", "endpoint", endpointKey, "zmq", zmqEndpoint)
	return nil
}
