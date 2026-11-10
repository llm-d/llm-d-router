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

type holdAllGate struct{}

func (holdAllGate) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "hold-all", Name: "test"}
}

func (holdAllGate) Allow(context.Context, flowcontrol.QueueItemAccessor, float64) bool {
	return false
}
