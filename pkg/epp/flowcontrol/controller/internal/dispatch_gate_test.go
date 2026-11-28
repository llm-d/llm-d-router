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

package internal

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestGateEligibleBandFiltersHeldHeads(t *testing.T) {
	t.Parallel()

	missKey := flowcontrol.FlowKey{ID: "tenant\x1fcacheclass:miss", Priority: 0}
	hitKey := flowcontrol.FlowKey{ID: "tenant\x1fcacheclass:hit", Priority: 0}
	miss := &fwkfcmocks.MockQueueItemAccessor{}
	hit := &fwkfcmocks.MockQueueItemAccessor{}
	missQ := &fwkfcmocks.MockFlowQueueAccessor{PeekV: miss, FlowKeyV: missKey}
	hitQ := &fwkfcmocks.MockFlowQueueAccessor{PeekV: hit, FlowKeyV: hitKey}

	base := &fwkfcmocks.MockPriorityBandAccessor{
		PriorityV: 0,
		FlowKeysFunc: func() []flowcontrol.FlowKey {
			return []flowcontrol.FlowKey{missKey, hitKey}
		},
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			switch id {
			case missKey.ID:
				return missQ
			case hitKey.ID:
				return hitQ
			default:
				return nil
			}
		},
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			callback(missQ)
			callback(hitQ)
		},
	}
	allowHit := func(item flowcontrol.QueueItemAccessor) bool { return item == hit }
	filtered := &gateEligibleBand{base: base, allow: allowHit}

	keys := filtered.FlowKeys()
	if len(keys) != 1 || keys[0] != hitKey {
		t.Fatalf("FlowKeys = %v, want only hit", keys)
	}
	if got := filtered.Queue(missKey.ID); got != nil {
		t.Fatal("held miss queue should be hidden")
	}
	if got := filtered.Queue(hitKey.ID); got != hitQ {
		t.Fatalf("hit queue = %v, want hitQ", got)
	}

	var seen []flowcontrol.FlowKey
	filtered.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool {
		seen = append(seen, q.FlowKey())
		return true
	})
	if len(seen) != 1 || seen[0] != hitKey {
		t.Fatalf("IterateQueues saw %v, want only hit", seen)
	}
}

func TestApplyDispatchGateNilPickStaysNil(t *testing.T) {
	t.Parallel()

	p := &Processor{dispatchGate: holdAllGate{}}
	got, err := p.applyDispatchGate(context.Background(), nil, &fwkfcmocks.MockPriorityBandAccessor{}, 0.9)
	if err != nil {
		t.Fatalf("applyDispatchGate: %v", err)
	}
	if got != nil {
		t.Fatalf("got %v, want nil when fairness selected nothing", got)
	}
}

func TestSelectForDispatchPrefersEligibleFirst(t *testing.T) {
	t.Parallel()

	missKey := flowcontrol.FlowKey{ID: "tenant\x1fcacheclass:miss", Priority: 0}
	hitKey := flowcontrol.FlowKey{ID: "tenant\x1fcacheclass:hit", Priority: 0}
	miss := &fwkfcmocks.MockQueueItemAccessor{}
	hit := &fwkfcmocks.MockQueueItemAccessor{}
	missQ := &fwkfcmocks.MockFlowQueueAccessor{PeekV: miss, FlowKeyV: missKey, LenV: 1}
	hitQ := &fwkfcmocks.MockFlowQueueAccessor{PeekV: hit, FlowKeyV: hitKey, LenV: 1}
	base := preferTestBand(missKey, hitKey, missQ, hitQ)

	p := &Processor{
		dispatchGate: preferEligibleGate{
			fallback: true,
			allow: func(item flowcontrol.QueueItemAccessor) bool {
				return item == hit
			},
		},
		registry: &preferTestRegistry{},
	}
	got, err := p.selectForDispatch(context.Background(), base, 0.9)
	if err != nil {
		t.Fatalf("selectForDispatch: %v", err)
	}
	if got != hit {
		t.Fatalf("got %v, want hit when prefer-eligible is on", got)
	}
}

func TestSelectForDispatchFallsBackWhenEligibleEmpty(t *testing.T) {
	t.Parallel()

	missKey := flowcontrol.FlowKey{ID: "tenant\x1fcacheclass:miss", Priority: 0}
	miss := &fwkfcmocks.MockQueueItemAccessor{}
	missQ := &fwkfcmocks.MockFlowQueueAccessor{PeekV: miss, FlowKeyV: missKey, LenV: 1}
	base := preferTestBand(missKey, missKey, missQ, missQ)

	p := &Processor{
		dispatchGate: preferEligibleGate{
			fallback: true,
			allow: func(item flowcontrol.QueueItemAccessor) bool {
				return false
			},
		},
		registry: &preferTestRegistry{},
	}
	got, err := p.selectForDispatch(context.Background(), base, 0.9)
	if err != nil {
		t.Fatalf("selectForDispatch: %v", err)
	}
	if got != miss {
		t.Fatalf("got %v, want miss fallback when no hit is waiting", got)
	}
}

func TestSelectForDispatchStrictSkipsEmptyEligibleFallback(t *testing.T) {
	t.Parallel()

	missKey := flowcontrol.FlowKey{ID: "tenant\x1fcacheclass:miss", Priority: 0}
	miss := &fwkfcmocks.MockQueueItemAccessor{}
	missQ := &fwkfcmocks.MockFlowQueueAccessor{PeekV: miss, FlowKeyV: missKey, LenV: 1}
	base := preferTestBand(missKey, missKey, missQ, missQ)

	p := &Processor{
		dispatchGate: preferEligibleGate{
			fallback: false,
			allow: func(item flowcontrol.QueueItemAccessor) bool {
				return false
			},
		},
		registry: &preferTestRegistry{},
	}
	got, err := p.selectForDispatch(context.Background(), base, 0.9)
	if err != nil {
		t.Fatalf("selectForDispatch: %v", err)
	}
	if got != nil {
		t.Fatalf("got %v, want nil when strict and no eligible work", got)
	}
}

type holdAllGate struct{}

func (holdAllGate) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "hold-all", Name: "test"}
}

func (holdAllGate) Allow(context.Context, flowcontrol.QueueItemAccessor, float64) bool {
	return false
}

type preferEligibleGate struct {
	allow    func(flowcontrol.QueueItemAccessor) bool
	fallback bool
}

func (preferEligibleGate) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "prefer-eligible", Name: "test"}
}

func (preferEligibleGate) PreferEligibleFirst(float64) bool { return true }

func (g preferEligibleGate) EmptyEligibleFallback(float64) bool { return g.fallback }

func (g preferEligibleGate) Allow(_ context.Context, item flowcontrol.QueueItemAccessor, _ float64) bool {
	if g.allow == nil {
		return false
	}
	return g.allow(item)
}

type preferTestRegistry struct {
	mocks.MockRegistryDataPlane
}

func (r *preferTestRegistry) FairnessPolicy(int) (flowcontrol.FairnessPolicy, error) {
	policy := &fwkfcmocks.MockFairnessPolicy{}
	policy.PickFunc = func(_ context.Context, band flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
		var selected flowcontrol.FlowQueueAccessor
		band.IterateQueues(func(q flowcontrol.FlowQueueAccessor) bool {
			if q != nil && q.Len() > 0 && q.Peek() != nil {
				selected = q
				return false
			}
			return true
		})
		return selected, nil
	}
	return policy, nil
}

func preferTestBand(
	firstKey, secondKey flowcontrol.FlowKey,
	firstQ, secondQ flowcontrol.FlowQueueAccessor,
) *fwkfcmocks.MockPriorityBandAccessor {
	return &fwkfcmocks.MockPriorityBandAccessor{
		PriorityV: 0,
		FlowKeysFunc: func() []flowcontrol.FlowKey {
			if firstKey == secondKey {
				return []flowcontrol.FlowKey{firstKey}
			}
			return []flowcontrol.FlowKey{firstKey, secondKey}
		},
		QueueFunc: func(id string) flowcontrol.FlowQueueAccessor {
			switch id {
			case firstKey.ID:
				return firstQ
			case secondKey.ID:
				return secondQ
			default:
				return nil
			}
		},
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			if !callback(firstQ) {
				return
			}
			if firstKey != secondKey {
				callback(secondQ)
			}
		},
	}
}
