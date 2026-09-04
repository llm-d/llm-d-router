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
	"errors"
	"fmt"
	"net"
	"sort"
	"strconv"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

const roundRobinLoadBalancerType = "round-robin"

type loadBalancerConfig struct {
	Type string `json:"type,omitempty"`
}

type endpointDiscoveryConfig struct {
	LoadBalancer *loadBalancerConfig `json:"loadBalancer,omitempty"`
}

func (c *endpointDiscoveryConfig) loadBalancerType() string {
	if c == nil || c.LoadBalancer == nil || c.LoadBalancer.Type == "" {
		return roundRobinLoadBalancerType
	}
	return c.LoadBalancer.Type
}

type renderEndpointPicker interface {
	Pick() (string, error)
}

type fixedEndpointPicker string

func (p fixedEndpointPicker) Pick() (string, error) {
	return string(p), nil
}

type endpointLoadBalancer interface {
	Pick(endpoints []string) (string, error)
}

type endpointLoadBalancerFactory func() endpointLoadBalancer

var endpointLoadBalancerFactories = map[string]endpointLoadBalancerFactory{
	roundRobinLoadBalancerType: func() endpointLoadBalancer { return &roundRobinLoadBalancer{} },
}

func newEndpointLoadBalancer(loadBalancerType string) (endpointLoadBalancer, error) {
	factory, ok := endpointLoadBalancerFactories[loadBalancerType]
	if !ok {
		return nil, fmt.Errorf("unsupported load balancer %q", loadBalancerType)
	}
	return factory(), nil
}

type roundRobinLoadBalancer struct {
	mu   sync.Mutex
	next int
}

func (b *roundRobinLoadBalancer) Pick(endpoints []string) (string, error) {
	if len(endpoints) == 0 {
		return "", errors.New("no vLLM render endpoints discovered")
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if b.next >= len(endpoints) {
		b.next = 0
	}
	endpoint := endpoints[b.next]
	b.next = (b.next + 1) % len(endpoints)
	return endpoint, nil
}

type discoveredEndpointPicker struct {
	mu           sync.RWMutex
	endpoints    map[string]string
	ordered      []string
	loadBalancer endpointLoadBalancer
}

func newDiscoveredEndpointPicker(loadBalancerType string) (*discoveredEndpointPicker, error) {
	loadBalancer, err := newEndpointLoadBalancer(loadBalancerType)
	if err != nil {
		return nil, err
	}
	return &discoveredEndpointPicker{
		endpoints:    map[string]string{},
		loadBalancer: loadBalancer,
	}, nil
}

func (p *discoveredEndpointPicker) Pick() (string, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.loadBalancer.Pick(p.ordered)
}

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
	port, err := strconv.Atoi(meta.Port)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("discovered endpoint %s has invalid port %q", meta.ID, meta.Port)
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.endpoints[meta.ID.String()] = "http://" + net.JoinHostPort(meta.Address, meta.Port)
	p.rebuildOrdered()
	return nil
}

func (p *discoveredEndpointPicker) Delete(meta *fwkdl.EndpointMetadata) {
	if meta == nil || meta.ID.Name == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.endpoints, meta.ID.String())
	p.rebuildOrdered()
}

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
