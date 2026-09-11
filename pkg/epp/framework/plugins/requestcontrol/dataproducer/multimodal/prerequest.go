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

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PreRequest records the selected endpoint(s) for each hash in the current request.
func (p *Producer) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) error {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	if request == nil || request.RequestID == "" {
		return nil
	}
	defer p.pluginState.Delete(request.RequestID)

	state, err := plugin.ReadPluginStateKey[*requestState](p.pluginState, request.RequestID, plugin.StateKey(ProducerType))
	if err != nil || len(state.items) == 0 {
		logger.Info("No multimodal request state found, skipping encoder-cache update")
		return nil
	}

	targets := p.targetEndpoints(schedulingResult)
	if len(targets) == 0 {
		logger.Info("No target endpoints found, skipping encoder-cache update")
		return nil
	}

	items := state.items
	// Update cache asynchronously to avoid blocking the request path.
	p.wg.Go(func() {
		p.mutex.Lock()
		defer p.mutex.Unlock()
		for _, endpoint := range targets {
			metadata := endpoint.GetMetadata()
			if metadata == nil {
				continue
			}
			podCache := p.getOrCreatePodCache(metadata.ID.String())
			for _, item := range items {
				podCache.Add(item.Hash, struct{}{})
			}
		}
	})
	return nil
}

// targetEndpoints returns the endpoints that computed and hold the encoder cache
// for the request. In disaggregated serving the encode profile selects those
// pods; its target is recorded rather than the primary (decode) profile's, whose
// pod does not hold the encoder cache. In aggregated serving no encode profile
// runs and the primary profile's pod both encodes and serves.
func (p *Producer) targetEndpoints(schedulingResult *scheduling.SchedulingResult) []scheduling.Endpoint {
	if schedulingResult == nil || schedulingResult.ProfileResults == nil {
		return nil
	}
	if result := schedulingResult.ProfileResults[p.encodeProfile]; result != nil {
		return result.TargetEndpoints
	}
	if schedulingResult.PrimaryProfileName == "" {
		return nil
	}
	if result := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]; result != nil {
		return result.TargetEndpoints
	}
	return nil
}
