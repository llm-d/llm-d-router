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

// Package wspt implements an ordering policy that favors requests with smaller processing cost
// (Weighted Shortest Processing Time with unit weights), with an optional bound on overtaking.
//
// For detailed documentation, see README.md.
package wspt

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// WSPTOrderingPolicyType is the registration type for the WSPT ordering policy.
//
// By default it selects the request with the smallest request size, using FCFS as a tie-breaker.
// For detailed documentation, see README.md.
const WSPTOrderingPolicyType = "wspt-ordering-policy"

// Config is the JSON parameters block for the policy. The zero value orders by size alone.
type Config struct {
	// MaxDelaySeconds bounds which requests can be dispatched ahead of a request: only those that
	// arrived no more than MaxDelaySeconds after it. A request is ordered as if it arrived
	// MaxDelaySeconds*min(size/ReferenceBytes, 1) later.
	MaxDelaySeconds float64 `json:"maxDelaySeconds,omitempty"`

	// ReferenceBytes is the request size that receives the full MaxDelaySeconds.
	// Required when MaxDelaySeconds is set.
	ReferenceBytes uint64 `json:"referenceBytes,omitempty"`
}

func (c Config) validate() error {
	switch {
	case c.MaxDelaySeconds < 0:
		return fmt.Errorf("maxDelaySeconds must be >= 0, got %v", c.MaxDelaySeconds)
	case c.MaxDelaySeconds > 0 && c.ReferenceBytes == 0:
		return errors.New("referenceBytes must be > 0 when maxDelaySeconds is set")
	}
	return nil
}

func WSPTOrderingPolicyFactory(name string, parameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	var cfg Config
	if parameters != nil {
		if err := parameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("invalid config for %s plugin %q: %w", WSPTOrderingPolicyType, name, err)
		}
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", WSPTOrderingPolicyType, name, err)
	}
	p := newWSPTPolicy().withName(name)
	p.cfg = cfg
	return p, nil
}

// WSPTPolicy favors requests with smaller byte sizes.
// See the documentation for the exported WSPTOrderingPolicyType constant for detailed behavioral guarantees.
type WSPTPolicy struct {
	name string
	cfg  Config
}

var _ flowcontrol.OrderingPolicy = &WSPTPolicy{}

func newWSPTPolicy() *WSPTPolicy {
	return &WSPTPolicy{
		name: WSPTOrderingPolicyType,
	}
}

func (p *WSPTPolicy) withName(name string) *WSPTPolicy {
	if name != "" {
		p.name = name
	}
	return p
}

func (p *WSPTPolicy) Name() string {
	return p.name
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *WSPTPolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: WSPTOrderingPolicyType,
		Name: p.name,
	}
}

// cost is the processing-cost estimate for an item. Items without a request are treated as the
// largest possible request.
func cost(item flowcontrol.QueueItemAccessor) uint64 {
	req := item.OriginalRequest()
	if req == nil {
		return math.MaxUint64
	}
	return req.ByteSize()
}

// delayedArrival is the enqueue time shifted by the size delay. The shift depends only on the item,
// so the order it produces does not change while items wait in the heap.
func (p *WSPTPolicy) delayedArrival(item flowcontrol.QueueItemAccessor) time.Time {
	fraction := float64(min(cost(item), p.cfg.ReferenceBytes)) / float64(p.cfg.ReferenceBytes)
	return item.EnqueueTime().Add(time.Duration(fraction * p.cfg.MaxDelaySeconds * float64(time.Second)))
}

// Less returns true if item 'a' should be dispatched before item 'b'.
// WSPT orders by cost (smallest first), using FCFS as a tie-breaker. With MaxDelaySeconds set, it
// orders by delayed arrival instead, using FCFS as a tie-breaker.
func (p *WSPTPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	if a == nil && b == nil {
		return false
	}
	if a == nil { // Treat nil as lowest priority
		return false
	}
	if b == nil { // Treat non-nil 'a' as higher priority than nil 'b'
		return true
	}
	if p.cfg.MaxDelaySeconds > 0 {
		arrivalA, arrivalB := p.delayedArrival(a), p.delayedArrival(b)
		if !arrivalA.Equal(arrivalB) {
			return arrivalA.Before(arrivalB)
		}
		return a.EnqueueTime().Before(b.EnqueueTime())
	}

	costA := cost(a)
	costB := cost(b)

	if costA != costB {
		return costA < costB
	}

	// Same cost: FCFS (earlier enqueue time = higher priority)
	return a.EnqueueTime().Before(b.EnqueueTime())
}
