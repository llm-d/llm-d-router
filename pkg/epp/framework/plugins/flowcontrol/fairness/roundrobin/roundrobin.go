/*
Copyright 2025 The Kubernetes Authors.
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

// Package roundrobin implements a fairness policy that selects queues from a priority band using a
// round-robin strategy, cycling through active flows one by one to guarantee that no single flow
// can starve others.
//
// For detailed documentation, see README.md.
package roundrobin

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// RoundRobinFairnessPolicyType is the registration type for the round-robin fairness policy.
const RoundRobinFairnessPolicyType = "round-robin-fairness-policy"

// RoundRobinFairnessPolicyFactory is the factory function for the round-robin fairness policy.
func RoundRobinFairnessPolicyFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	return newRoundRobin(name), nil
}

// roundRobin implements the FairnessPolicy interface, selecting queues from a priority band
// in a round-robin fashion.
type roundRobin struct {
	name string
}

func newRoundRobin(name string) *roundRobin {
	if name == "" {
		name = RoundRobinFairnessPolicyType
	}
	return &roundRobin{name}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *roundRobin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: RoundRobinFairnessPolicyType,
		Name: p.name,
	}
}

// roundRobinCursor holds the mutable cursor for a specific priority band.
// It is initialized via NewState and stored on the PriorityBandAccessor.
type roundRobinCursor struct {
	mu           sync.Mutex
	lastSelected *flowcontrol.FlowKey
}

// NewState initializes the policy state for a specific priority band.
func (p *roundRobin) NewState(_ context.Context) any {
	return &roundRobinCursor{}
}

// Pick selects the next flow queue in a round-robin fashion from the given priority band.
// It retrieves the band-specific state, locks it, and advances the cursor.
func (p *roundRobin) Pick(
	_ context.Context,
	flowGroup flowcontrol.PriorityBandAccessor,
) (flowcontrol.FlowQueueAccessor, error) {
	if flowGroup == nil {
		// Nothing to pick
		return nil, nil //nolint:nilnil
	}

	v := flowGroup.PolicyState()
	c, ok := v.(*roundRobinCursor)
	if !ok || c == nil {
		return nil, fmt.Errorf("invalid state type for RoundRobin policy: expected *roundRobinCursor, got %T", v)
	}

	var activeQueues []flowcontrol.FlowQueueAccessor
	flowGroup.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool {
		if q != nil && q.Len() > 0 {
			activeQueues = append(activeQueues, q)
		}
		return true
	})
	if len(activeQueues) == 0 {
		c.mu.Lock()
		c.lastSelected = nil
		c.mu.Unlock()
		return nil, nil //nolint:nilnil
	}

	slices.SortFunc(activeQueues, func(a, b flowcontrol.FlowQueueAccessor) int {
		return a.FlowKey().Compare(b.FlowKey())
	})

	c.mu.Lock()
	defer c.mu.Unlock()

	startIdx := 0
	if c.lastSelected != nil {
		idx, found := slices.BinarySearchFunc(
			activeQueues,
			*c.lastSelected,
			func(q flowcontrol.FlowQueueAccessor, target flowcontrol.FlowKey) int {
				return q.FlowKey().Compare(target)
			},
		)
		if found {
			startIdx = (idx + 1) % len(activeQueues)
		} else {
			startIdx = idx % len(activeQueues)
		}
	}

	selectedQueue := activeQueues[startIdx]
	selectedKey := selectedQueue.FlowKey()
	c.lastSelected = &selectedKey
	return selectedQueue, nil
}
