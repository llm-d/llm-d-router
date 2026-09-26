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

package thunderagent

import (
	"context"
	"math"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func (a *ThunderAgent) Category() fwksched.ScorerCategory {
	return fwksched.Balance
}

// Score is piecewise: a program bound to a candidate pod scores that pod 1.0
// and all others 0.0 (sticky), while an unbound or displaced program scores
// pods by free program token capacity (least-loaded placement). Anonymous
// requests are not scored; pair this plugin with a general load scorer to
// account for them.
//
// Keeping the binding and the accounting in one table means re-placement is
// inherent: when a scheduling filter removes the bound pod from the
// candidates, the sticky branch misses, the load branch places the program
// elsewhere, and PreRequest rebinds it to the pod actually picked.
func (a *ThunderAgent) Score(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	id := programID(request)
	if id == "" {
		return nil
	}

	now := time.Now()
	t := a.table
	t.mu.Lock()
	var boundPod string
	if st, ok := t.programs[id]; ok {
		boundPod = st.podName
	}
	loads := t.podLoads(now)
	t.mu.Unlock()

	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	if boundPod != "" {
		for _, endpoint := range endpoints {
			if endpoint.GetMetadata().ID.String() == boundPod {
				for _, e := range endpoints {
					scores[e] = 0.0
				}
				scores[endpoint] = 1.0
				return scores
			}
		}
	}

	logger := log.FromContext(ctx)
	for _, endpoint := range endpoints {
		pod := endpoint.GetMetadata().ID.String()
		capTokens, _ := a.endpointCapacity(endpoint)
		score := 0.0
		if capTokens > 0 {
			score = 1.0 - math.Min(1.0, loads[pod].tokens/capTokens)
		}
		scores[endpoint] = score
		logger.V(logutil.DEBUG).Info("thunderagent scoring", "endpoint", pod, "tokenLoad", loads[pod].tokens, "score", score)
	}
	return scores
}
