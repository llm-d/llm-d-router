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

package multimodal

import (
	"context"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

type podPlacement struct {
	pod      string
	acquired []string
	pending  []attrmm.MatchItem
}

// placementState owns the references acquired for one routed request. The
// shared completion flag makes cleanup idempotent when response processing and
// PluginState eviction race.
type placementState struct {
	producer   *Producer
	placements []podPlacement
	completed  *atomic.Bool
}

var _ plugin.EvictableStateData = (*placementState)(nil)

func (s *placementState) Clone() plugin.StateData {
	if s == nil {
		return nil
	}
	placements := make([]podPlacement, len(s.placements))
	for i := range s.placements {
		placements[i] = podPlacement{
			pod:      s.placements[i].pod,
			acquired: append([]string(nil), s.placements[i].acquired...),
			pending:  attrmm.CloneMatchItems(s.placements[i].pending),
		}
	}
	return &placementState{
		producer:   s.producer,
		placements: placements,
		completed:  s.completed,
	}
}

func (s *placementState) OnEvicted(requestID string, _ plugin.StateKey) {
	if !s.claimCompletion() {
		return
	}
	go s.producer.completePlacements(requestID, s.placements, false)
}

func (s *placementState) complete(requestID string, commitPending bool) {
	if !s.claimCompletion() {
		return
	}
	s.producer.completePlacements(requestID, s.placements, commitPending)
}

func (s *placementState) claimCompletion() bool {
	return s != nil && s.producer != nil && s.completed != nil && s.completed.CompareAndSwap(false, true)
}

// PreRequest records the selected endpoint(s) for each hash in the current request.
func (p *Producer) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) error {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	if request == nil || request.RequestID == "" {
		return nil
	}

	state, err := plugin.ReadPluginStateKey[*requestState](p.pluginState, request.RequestID, plugin.StateKey(ProducerType))
	if err != nil || len(state.items) == 0 {
		logger.Info("No multimodal request state found, skipping encoder-cache update")
		p.pluginState.Delete(request.RequestID)
		return nil
	}

	targets := targetEndpoints(schedulingResult)
	if len(targets) == 0 {
		logger.Info("No target endpoints found, skipping encoder-cache update")
		p.pluginState.Delete(request.RequestID)
		return nil
	}

	placements := p.acquirePlacements(request.RequestID, targets, state.items)
	if len(placements) == 0 {
		p.pluginState.Delete(request.RequestID)
		return nil
	}
	p.pluginState.Write(request.RequestID, plugin.StateKey(ProducerType), &placementState{
		producer:   p,
		placements: placements,
		completed:  &atomic.Bool{},
	})
	p.pluginState.BindLiveness(ctx, request.RequestID)
	return nil
}

func (p *Producer) acquirePlacements(requestID string, targets []scheduling.Endpoint, items []attrmm.MatchItem) []podPlacement {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	placements := make([]podPlacement, 0, len(targets))
	for _, endpoint := range targets {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			continue
		}
		placement := podPlacement{pod: metadata.ID.String()}
		cache := p.getOrCreatePodCache(placement.pod)
		for _, item := range items {
			item = p.cacheItem(item)
			if cache.acquire(requestID, item) {
				placement.acquired = append(placement.acquired, item.Hash)
			} else {
				placement.pending = append(placement.pending, item)
			}
		}
		placements = append(placements, placement)
	}
	return placements
}

func (p *Producer) completePlacements(requestID string, placements []podPlacement, commitPending bool) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	for _, placement := range placements {
		cache, ok := p.caches[placement.pod]
		if !ok {
			continue
		}
		for _, hash := range placement.acquired {
			cache.release(requestID, hash)
		}
		if commitPending {
			for _, item := range placement.pending {
				cache.commit(item)
			}
		}
	}
}

func (p *Producer) cacheItem(item attrmm.MatchItem) attrmm.MatchItem {
	if p.useItemSize {
		item.Size = normalizedItemSize(item.Size)
	} else {
		item.Size = 1
	}
	return item
}

// ResponseBody releases active encoder-cache references when prefill has
// completed. The director marks an abort before the first response chunk as
// both StartOfStream and EndOfStream, so only a natural one-chunk response may
// commit pending items on that combined callback.
func (p *Producer) ResponseBody(
	_ context.Context,
	request *scheduling.InferenceRequest,
	response *requestcontrol.Response,
	_ *fwkdl.EndpointMetadata,
) {
	if request == nil || response == nil || request.RequestID == "" {
		return
	}
	state, err := plugin.ReadPluginStateKey[*placementState](p.pluginState, request.RequestID, plugin.StateKey(ProducerType))
	if err == nil {
		commitPending := response.StartOfStream &&
			(!response.EndOfStream || response.TerminationCause == requestcontrol.TerminationCauseNatural)
		if commitPending {
			state.complete(request.RequestID, true)
		} else if response.EndOfStream {
			state.complete(request.RequestID, false)
		}
	}
	if response.EndOfStream {
		p.pluginState.Delete(request.RequestID)
	} else {
		p.pluginState.Touch(request.RequestID)
	}
}

func targetEndpoints(schedulingResult *scheduling.SchedulingResult) []scheduling.Endpoint {
	if schedulingResult == nil || schedulingResult.PrimaryProfileName == "" || schedulingResult.ProfileResults == nil {
		return nil
	}
	result := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if result == nil {
		return nil
	}
	return result.TargetEndpoints
}
