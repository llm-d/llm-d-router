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

// Package peerdemand implements a usage limit policy that holds a replica's lower priority bands while a peer
// replica has requests queued in a higher band.
package peerdemand

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// PolicyType is the registration type for the peer demand holdback usage limit policy.
const PolicyType = "peer-demand-holdback-policy"

type apiConfig struct {
	// Delegate names the usage limit policy whose ceilings this policy tightens. Optional: without it every band
	// starts at 1.0. The pluginRef tag makes the config loader instantiate the delegate first.
	Delegate string `json:"delegate,omitempty" pluginRef:""`
}

// Policy gates every band below the highest band with requests queued on a peer replica, on top of the ceilings
// its delegate computes. Without a current peer view it returns the delegate's ceilings unchanged.
type Policy struct {
	name     string
	delegate flowcontrol.UsageLimitPolicy
	peers    atomic.Pointer[peerSource]
}

type peerSource struct{ flowcontrol.PeerDemand }

var (
	_ flowcontrol.UsageLimitPolicy   = &Policy{}
	_ flowcontrol.PeerDemandConsumer = &Policy{}
)

// PolicyConfigParser parses the policy parameters. It is registered with the plugin registry so the config
// loader can order plugin instantiation by the delegate reference.
func PolicyConfigParser(params *json.Decoder, _ plugin.Handle) (any, error) {
	var cfg apiConfig
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s config: %w", PolicyType, err)
		}
	}
	return cfg, nil
}

// PolicyFactory creates the policy.
func PolicyFactory(name string, params *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	raw, err := PolicyConfigParser(params, handle)
	if err != nil {
		return nil, err
	}
	cfg := raw.(apiConfig)
	var delegate flowcontrol.UsageLimitPolicy
	if cfg.Delegate != "" {
		if handle == nil {
			return nil, errors.New("plugin handle is required to resolve the delegate")
		}
		p := handle.Plugin(cfg.Delegate)
		if p == nil {
			return nil, fmt.Errorf("delegate plugin not found: %s", cfg.Delegate)
		}
		var ok bool
		if delegate, ok = p.(flowcontrol.UsageLimitPolicy); !ok {
			return nil, fmt.Errorf("plugin %s does not implement flowcontrol.UsageLimitPolicy", cfg.Delegate)
		}
	}
	return New(name, delegate), nil
}

// New returns the policy. A nil delegate leaves every band at 1.0 before the peer holdback.
func New(name string, delegate flowcontrol.UsageLimitPolicy) *Policy {
	if name == "" {
		name = PolicyType
	}
	return &Policy{name: name, delegate: delegate}
}

// TypedName implements plugin.Plugin.
func (p *Policy) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: PolicyType, Name: p.name}
}

// SetPeerDemand implements flowcontrol.PeerDemandConsumer.
func (p *Policy) SetPeerDemand(peers flowcontrol.PeerDemand) {
	if peers == nil {
		p.peers.Store(nil)
		return
	}
	p.peers.Store(&peerSource{peers})
}

// ComputeLimit implements flowcontrol.UsageLimitPolicy. The highest band with peer demand keeps its ceiling, so
// the pool's highest queued band always progresses, and gating its tail keeps the ceilings non-increasing.
func (p *Policy) ComputeLimit(ctx context.Context, saturation float64, priorities []int, ceilings []float64) {
	if p.delegate != nil {
		p.delegate.ComputeLimit(ctx, saturation, priorities, ceilings)
	}
	peers := p.peers.Load()
	if peers == nil {
		return
	}
	for i := 0; i < len(priorities)-1; i++ {
		queued, ok := peers.QueuedRequests(priorities[i])
		if !ok {
			return
		}
		if queued > 0 {
			for j := i + 1; j < len(ceilings); j++ {
				ceilings[j] = 0
			}
			return
		}
	}
}
