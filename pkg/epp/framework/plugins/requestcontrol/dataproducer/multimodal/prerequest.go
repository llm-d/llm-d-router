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
	"maps"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

// PreRequest records the selected endpoint(s) for each hash in the current request.
func (p *Producer) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	logger := log.FromContext(ctx).V(logging.DEBUG)
	if request == nil || request.RequestID == "" {
		return
	}
	defer p.pluginState.Delete(request.RequestID)

	state, err := plugin.ReadPluginStateKey[*requestState](p.pluginState, request.RequestID, plugin.StateKey(ProducerType))
	if err != nil || len(state.items) == 0 {
		logger.Info("No multimodal request state found, skipping encoder-cache update")
		return
	}

	targets := targetEndpoints(schedulingResult)
	if len(targets) == 0 {
		logger.Info("No target endpoints found, skipping encoder-cache update")
		return
	}

	p.mutex.Lock()
	defer p.mutex.Unlock()
	for _, item := range state.items {
		pods := map[string]struct{}{}
		if existing, ok := p.cache.Get(item.Hash); ok {
			pods = maps.Clone(existing)
		}
		for _, endpoint := range targets {
			if metadata := endpoint.GetMetadata(); metadata != nil {
				pods[metadata.NamespacedName.String()] = struct{}{}
			}
		}
		if len(pods) > 0 {
			p.cache.Add(item.Hash, pods)
		}
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
