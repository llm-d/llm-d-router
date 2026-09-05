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

package tokenizer

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sort"
	"strconv"
	"sync"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const roundRobinLoadBalancerType = "round-robin"

// errNoRenderEndpoints indicates that discovery has no selectable render URL.
var errNoRenderEndpoints = errors.New("no vLLM render endpoints discovered")

type loadBalancerConfig struct {
	Type string `json:"type,omitempty"`
}

// endpointPortRule maps matching inference endpoints to a render port range.
type endpointPortRule struct {
	Selector metav1.LabelSelector `json:"selector,omitempty"`
	BasePort int                  `json:"basePort"`
}

type endpointDiscoveryConfig struct {
	LoadBalancer *loadBalancerConfig `json:"loadBalancer,omitempty"`
	// AttemptTimeout optionally caps each HTTP attempt within the request timeout.
	AttemptTimeout string `json:"attemptTimeout,omitempty"`
	// PortRules resolve render ports as BasePort + RankIndex. An empty list uses
	// the inference port published by discovery.
	PortRules []endpointPortRule `json:"portRules,omitempty"`
}

// loadBalancerType returns the configured algorithm or the default round-robin algorithm.
func (c *endpointDiscoveryConfig) loadBalancerType() string {
	if c == nil || c.LoadBalancer == nil || c.LoadBalancer.Type == "" {
		return roundRobinLoadBalancerType
	}
	return c.LoadBalancer.Type
}

// renderEndpointPicker resolves the base URL for one render request.
type renderEndpointPicker interface {
	Pick() (string, error)
}

// retryingRenderEndpointPicker can avoid endpoints already attempted by a request.
type retryingRenderEndpointPicker interface {
	renderEndpointPicker
	PickExcluding(excluded map[string]struct{}) (string, error)
}

// fixedEndpointPicker always returns the statically configured render URL.
type fixedEndpointPicker string

// Pick returns the static render URL.
func (p fixedEndpointPicker) Pick() (string, error) {
	return string(p), nil
}

// endpointLoadBalancer selects a non-excluded URL from the full endpoint snapshot.
type endpointLoadBalancer interface {
	Pick(endpoints []string, excluded map[string]struct{}) (string, error)
}

type endpointLoadBalancerFactory func() endpointLoadBalancer

// endpointLoadBalancerFactories maps configuration names to algorithm constructors.
var endpointLoadBalancerFactories = map[string]endpointLoadBalancerFactory{
	roundRobinLoadBalancerType: func() endpointLoadBalancer { return &roundRobinLoadBalancer{} },
}

// newEndpointLoadBalancer constructs the configured selection algorithm.
func newEndpointLoadBalancer(loadBalancerType string) (endpointLoadBalancer, error) {
	factory, ok := endpointLoadBalancerFactories[loadBalancerType]
	if !ok {
		return nil, fmt.Errorf("unsupported load balancer %q", loadBalancerType)
	}
	return factory(), nil
}

// roundRobinLoadBalancer rotates requests through the endpoint snapshot.
type roundRobinLoadBalancer struct {
	mu   sync.Mutex
	next int
}

// Pick returns the next endpoint and advances the concurrency-safe cursor.
func (b *roundRobinLoadBalancer) Pick(endpoints []string, excluded map[string]struct{}) (string, error) {
	if len(endpoints) == 0 {
		return "", errNoRenderEndpoints
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if b.next >= len(endpoints) {
		b.next = 0
	}
	// Advance across the full snapshot so retry exclusions cannot shift the cursor.
	for range endpoints {
		endpoint := endpoints[b.next]
		b.next = (b.next + 1) % len(endpoints)
		if _, skip := excluded[endpoint]; !skip {
			return endpoint, nil
		}
	}
	return "", errNoRenderEndpoints
}

// compiledEndpointPortRule holds a validated selector and base port.
type compiledEndpointPortRule struct {
	selector labels.Selector
	basePort int
}

// discoveredEndpointPicker maintains render URLs keyed by endpoint identity.
type discoveredEndpointPicker struct {
	mu           sync.RWMutex
	endpoints    map[string]string
	ordered      []string
	loadBalancer endpointLoadBalancer
	portRules    []compiledEndpointPortRule
}

// newDiscoveredEndpointPicker constructs a picker with the configured algorithm.
func newDiscoveredEndpointPicker(config *endpointDiscoveryConfig) (*discoveredEndpointPicker, error) {
	loadBalancer, err := newEndpointLoadBalancer(config.loadBalancerType())
	if err != nil {
		return nil, err
	}
	portRules, err := compileEndpointPortRules(config)
	if err != nil {
		return nil, err
	}
	return &discoveredEndpointPicker{
		endpoints:    map[string]string{},
		loadBalancer: loadBalancer,
		portRules:    portRules,
	}, nil
}

// Pick selects a render URL from a stable endpoint snapshot.
func (p *discoveredEndpointPicker) Pick() (string, error) {
	return p.PickExcluding(nil)
}

// PickExcluding selects a render URL that has not been attempted by the request.
func (p *discoveredEndpointPicker) PickExcluding(excluded map[string]struct{}) (string, error) {
	p.mu.RLock()
	endpoints := append([]string(nil), p.ordered...)
	p.mu.RUnlock()

	if len(endpoints) == 0 {
		return "", errNoRenderEndpoints
	}
	return p.loadBalancer.Pick(endpoints, excluded)
}

// Upsert validates endpoint metadata and stores its HTTP render URL by identity.
func (p *discoveredEndpointPicker) Upsert(meta *fwkdl.EndpointMetadata) error {
	if meta == nil {
		return errors.New("discovered endpoint metadata is nil")
	}
	if meta.ID.Name == "" {
		return errors.New("discovered endpoint ID is empty")
	}
	if meta.Address == "" {
		return fmt.Errorf("discovered endpoint %s has an empty address", meta.ID)
	}
	port, err := p.resolveRenderPort(meta)
	if err != nil {
		return err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.endpoints[meta.ID.String()] = "http://" + net.JoinHostPort(meta.Address, port)
	p.rebuildOrdered()
	return nil
}

// Delete removes an endpoint from render selection by identity.
func (p *discoveredEndpointPicker) Delete(meta *fwkdl.EndpointMetadata) {
	if meta == nil || meta.ID.Name == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.endpoints, meta.ID.String())
	p.rebuildOrdered()
}

// rebuildOrdered sorts by endpoint identity so event arrival order does not affect selection.
func (p *discoveredEndpointPicker) rebuildOrdered() {
	ids := make([]string, 0, len(p.endpoints))
	for id := range p.endpoints {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	ordered := make([]string, 0, len(ids))
	for _, id := range ids {
		ordered = append(ordered, p.endpoints[id])
	}
	p.ordered = ordered
}

// compileEndpointPortRules validates configured selectors and render port ranges.
func compileEndpointPortRules(config *endpointDiscoveryConfig) ([]compiledEndpointPortRule, error) {
	if config == nil || len(config.PortRules) == 0 {
		return nil, nil
	}

	rules := make([]compiledEndpointPortRule, 0, len(config.PortRules))
	for i, rule := range config.PortRules {
		if rule.BasePort < 1 || rule.BasePort > 65535 {
			return nil, fmt.Errorf("port rule %d has invalid base port %d", i, rule.BasePort)
		}
		selector, err := metav1.LabelSelectorAsSelector(&rule.Selector)
		if err != nil {
			return nil, fmt.Errorf("port rule %d has invalid selector: %w", i, err)
		}
		rules = append(rules, compiledEndpointPortRule{selector: selector, basePort: rule.BasePort})
	}
	return rules, nil
}

// resolveRenderPort returns the configured render port for an endpoint.
func (p *discoveredEndpointPicker) resolveRenderPort(meta *fwkdl.EndpointMetadata) (string, error) {
	if len(p.portRules) == 0 {
		port, err := strconv.Atoi(meta.Port)
		if err != nil || port < 1 || port > 65535 {
			return "", fmt.Errorf("discovered endpoint %s has invalid port %q", meta.ID, meta.Port)
		}
		return meta.Port, nil
	}

	for _, rule := range p.portRules {
		if !rule.selector.Matches(labels.Set(meta.Labels)) {
			continue
		}
		if meta.RankIndex < 0 {
			return "", fmt.Errorf("discovered endpoint %s has invalid rank index %d", meta.ID, meta.RankIndex)
		}
		if meta.RankIndex > 65535-rule.basePort {
			return "", fmt.Errorf("discovered endpoint %s resolved render port outside 1-65535", meta.ID)
		}
		return strconv.Itoa(rule.basePort + meta.RankIndex), nil
	}
	return "", fmt.Errorf("discovered endpoint %s does not match any port rule", meta.ID)
}

// endpointDiscoveryHandler gives each token producer an independent extractor identity.
type endpointDiscoveryHandler struct {
	typedName           plugin.TypedName
	picker              *discoveredEndpointPicker
	mu                  sync.Mutex
	registeredEndpoints map[string]fwkdl.Endpoint
}

var _ fwkdl.EndpointExtractor = &endpointDiscoveryHandler{}

// newEndpointDiscoveryHandler binds endpoint events to one token producer's picker.
func newEndpointDiscoveryHandler(owner plugin.TypedName, picker *discoveredEndpointPicker) *endpointDiscoveryHandler {
	return &endpointDiscoveryHandler{
		typedName: plugin.TypedName{
			Type: PluginType + "-endpoint-discovery-" + owner.Name,
			Name: owner.Name,
		},
		picker:              picker,
		registeredEndpoints: make(map[string]fwkdl.Endpoint),
	}
}

// TypedName returns the per-instance extractor identity used by the data layer.
func (h *endpointDiscoveryHandler) TypedName() plugin.TypedName {
	return h.typedName
}

// Extract keeps the picker synchronized with endpoint lifecycle events.
func (h *endpointDiscoveryHandler) Extract(_ context.Context, event fwkdl.EndpointEvent) error {
	if event.Endpoint == nil || event.Endpoint.GetMetadata() == nil {
		return nil
	}

	// Keep generation checks and picker updates atomic across lifecycle callbacks.
	h.mu.Lock()
	defer h.mu.Unlock()
	meta := event.Endpoint.GetMetadata()
	id := meta.ID.String()
	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		h.registeredEndpoints[id] = event.Endpoint
		if err := h.picker.Upsert(meta); err != nil {
			h.picker.Delete(meta)
			return err
		}
	case fwkdl.EventDelete:
		// Delete events carry the Endpoint object used by the corresponding add.
		if registered, ok := h.registeredEndpoints[id]; ok && registered != event.Endpoint {
			return nil
		}
		delete(h.registeredEndpoints, id)
		h.picker.Delete(meta)
	}
	return nil
}
